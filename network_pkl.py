﻿
import click
import pickle
import re
import copy
import numpy as np
import torch
import dnnlib
from torch_utils import misc

def load_network_pkl(f, force_fp16=False, use_vit_discriminator=False):
    data = _LegacyUnpickler(f).load()

    # Legacy TensorFlow pickle => convert.
    if isinstance(data, tuple) and len(data) == 3 and all(isinstance(net, _TFNetworkStub) for net in data):
        tf_G, tf_D, tf_Gs = data
        G = convert_tf_generator(tf_G)
        
        # Choose the appropriate discriminator converter
        if use_vit_discriminator:
            D = convert_tf_vit_discriminator(tf_D)
  
        G_ema = convert_tf_generator(tf_Gs)
        data = dict(G=G, D=D, G_ema=G_ema)

    # Add missing fields.
    if 'training_set_kwargs' not in data:
        data['training_set_kwargs'] = None
    if 'augment_pipe' not in data:
        data['augment_pipe'] = None

    # Validate contents.
    assert isinstance(data['G'], torch.nn.Module)
    assert isinstance(data['D'], torch.nn.Module)
    assert isinstance(data['G_ema'], torch.nn.Module)
    assert isinstance(data['training_set_kwargs'], (dict, type(None)))
    assert isinstance(data['augment_pipe'], (torch.nn.Module, type(None)))

    if force_fp16:
        for key in ['G', 'D', 'G_ema']:
            old = data[key]
            kwargs = copy.deepcopy(old.init_kwargs)
            if key.startswith('G'):
                kwargs.synthesis_kwargs = dnnlib.EasyDict(kwargs.get('synthesis_kwargs', {}))
                kwargs.synthesis_kwargs.num_fp16_res = 4
                kwargs.synthesis_kwargs.conv_clamp = 256
            if key.startswith('D'):
                kwargs.num_fp16_res = 4
                kwargs.conv_clamp = 256
            if kwargs != old.init_kwargs:
                new = type(old)(**kwargs).eval().requires_grad_(False)
                misc.copy_params_and_buffers(old, new, require_all=True)
                data[key] = new
    return data

class _TFNetworkStub(dnnlib.EasyDict):
    pass

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        return super().find_class(module, name)

def _collect_tf_params(tf_net):
    # pylint: disable=protected-access
    tf_params = dict()
    def recurse(prefix, tf_net):
        for name, value in tf_net.variables:
            tf_params[prefix + name] = value
        for name, comp in tf_net.components.items():
            recurse(prefix + name + '/', comp)
    recurse('', tf_net)
    return tf_params

def _populate_module_params(module, *patterns):
    for name, tensor in misc.named_params_and_buffers(module):
        found = False
        value = None
        for pattern, value_fn in zip(patterns[0::2], patterns[1::2]):
            match = re.fullmatch(pattern, name)
            if match:
                found = True
                if value_fn is not None:
                    value = value_fn(*match.groups())
                break
        try:
            assert found
            if value is not None:
                tensor.copy_(torch.from_numpy(np.array(value)))
        except:
            print(name, list(tensor.shape))
            raise

def convert_tf_generator(tf_G):
    if tf_G.version < 4:
        raise ValueError('TensorFlow pickle version too low')

    # Collect kwargs.
    tf_kwargs = tf_G.static_kwargs
    known_kwargs = set()
    def kwarg(tf_name, default=None, none=None):
        known_kwargs.add(tf_name)
        val = tf_kwargs.get(tf_name, default)
        return val if val is not None else none

    # Convert kwargs.
    kwargs = dnnlib.EasyDict(
        z_dim                   = kwarg('latent_size',          512),
        c_dim                   = kwarg('label_size',           0),
        w_dim                   = kwarg('dlatent_size',         512),
        img_resolution          = kwarg('resolution',           1024),
        img_channels            = kwarg('num_channels',         3),
        mapping_kwargs = dnnlib.EasyDict(
            num_layers          = kwarg('mapping_layers',       8),
            embed_features      = kwarg('label_fmaps',          None),
            layer_features      = kwarg('mapping_fmaps',        None),
            activation          = kwarg('mapping_nonlinearity', 'lrelu'),
            lr_multiplier       = kwarg('mapping_lrmul',        0.01),
            w_avg_beta          = kwarg('w_avg_beta',           0.995,  none=1),
        ),
        synthesis_kwargs = dnnlib.EasyDict(
            channel_base        = kwarg('fmap_base',            16384) * 2,
            channel_max         = kwarg('fmap_max',             512),
            num_fp16_res        = kwarg('num_fp16_res',         0),
            conv_clamp          = kwarg('conv_clamp',           None),
            architecture        = kwarg('architecture',         'skip'),
            resample_filter     = kwarg('resample_kernel',      [1,3,3,1]),
            use_noise           = kwarg('use_noise',            True),
            activation          = kwarg('nonlinearity',         'lrelu'),
        ),
    )

    # Check for unknown kwargs.
    kwarg('truncation_psi')
    kwarg('truncation_cutoff')
    kwarg('style_mixing_prob')
    kwarg('structure')
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError('Unknown TensorFlow kwarg', unknown_kwargs[0])

    # Collect params.
    tf_params = _collect_tf_params(tf_G)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r'ToRGB_lod(\d+)/(.*)', name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f'{r}x{r}/ToRGB/{match.group(2)}'] = value
            kwargs.synthesis.kwargs.architecture = 'orig'

    # Convert params.
    from training import networks
    G = networks.Generator(**kwargs).eval().requires_grad_(False)
    # pylint: disable=unnecessary-lambda
    _populate_module_params(G,
        r'mapping\.w_avg',                                  lambda:     tf_params[f'dlatent_avg'],
        r'mapping\.embed\.weight',                          lambda:     tf_params[f'mapping/LabelEmbed/weight'].transpose(),
        r'mapping\.embed\.bias',                            lambda:     tf_params[f'mapping/LabelEmbed/bias'],
        r'mapping\.fc(\d+)\.weight',                        lambda i:   tf_params[f'mapping/Dense{i}/weight'].transpose(),
        r'mapping\.fc(\d+)\.bias',                          lambda i:   tf_params[f'mapping/Dense{i}/bias'],
        r'synthesis\.b4\.const',                            lambda:     tf_params[f'synthesis/4x4/Const/const'][0],
        r'synthesis\.b4\.conv1\.weight',                    lambda:     tf_params[f'synthesis/4x4/Conv/weight'].transpose(3, 2, 0, 1),
        r'synthesis\.b4\.conv1\.bias',                      lambda:     tf_params[f'synthesis/4x4/Conv/bias'],
        r'synthesis\.b4\.conv1\.noise_const',               lambda:     tf_params[f'synthesis/noise0'][0, 0],
        r'synthesis\.b4\.conv1\.noise_strength',            lambda:     tf_params[f'synthesis/4x4/Conv/noise_strength'],
        r'synthesis\.b4\.conv1\.affine\.weight',            lambda:     tf_params[f'synthesis/4x4/Conv/mod_weight'].transpose(),
        r'synthesis\.b4\.conv1\.affine\.bias',              lambda:     tf_params[f'synthesis/4x4/Conv/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.conv0\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/weight'][::-1, ::-1].transpose(3, 2, 0, 1),
        r'synthesis\.b(\d+)\.conv0\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/bias'],
        r'synthesis\.b(\d+)\.conv0\.noise_const',           lambda r:   tf_params[f'synthesis/noise{int(np.log2(int(r)))*2-5}'][0, 0],
        r'synthesis\.b(\d+)\.conv0\.noise_strength',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/noise_strength'],
        r'synthesis\.b(\d+)\.conv0\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/mod_weight'].transpose(),
        r'synthesis\.b(\d+)\.conv0\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.conv1\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/weight'].transpose(3, 2, 0, 1),
        r'synthesis\.b(\d+)\.conv1\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/bias'],
        r'synthesis\.b(\d+)\.conv1\.noise_const',           lambda r:   tf_params[f'synthesis/noise{int(np.log2(int(r)))*2-4}'][0, 0],
        r'synthesis\.b(\d+)\.conv1\.noise_strength',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/noise_strength'],
        r'synthesis\.b(\d+)\.conv1\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/mod_weight'].transpose(),
        r'synthesis\.b(\d+)\.conv1\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.torgb\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/weight'].transpose(3, 2, 0, 1),
        r'synthesis\.b(\d+)\.torgb\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/bias'],
        r'synthesis\.b(\d+)\.torgb\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/mod_weight'].transpose(),
        r'synthesis\.b(\d+)\.torgb\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.skip\.weight',                 lambda r:   tf_params[f'synthesis/{r}x{r}/Skip/weight'][::-1, ::-1].transpose(3, 2, 0, 1),
        r'.*\.resample_filter',                             None,
    )
    return G

def convert_tf_vit_discriminator(tf_D):
    if tf_D.version < 4:
        raise ValueError('TensorFlow pickle version too low')

    # Collect kwargs.
    tf_kwargs = tf_D.static_kwargs
    known_kwargs = set()
    def kwarg(tf_name, default=None):
        known_kwargs.add(tf_name)
        return tf_kwargs.get(tf_name, default)

    # Convert kwargs.
    # Define ViT-specific kwargs for discriminator
    kwargs = dnnlib.EasyDict(
        img_resolution=kwarg('resolution', 1024),            # Resolution of input images
        img_channels=kwarg('num_channels', 3),               # Number of input channels
        patch_size=kwarg('patch_size', 32),                  # Size of each patch for ViT
        dim=kwarg('dim', 768),                               # Dimension of embeddings
        depth=kwarg('depth', 6),                             # Depth of the transformer (number of encoder layers)
        heads=kwarg('heads', 12),                             # Number of heads in multi-head attention
        mlp_ratio=kwarg('mlp_ratio', 6),                     # Ratio for the hidden size in MLP layers
        attention_dropout=kwarg('attention_dropout', 0.1),   # Dropout rate in attention layers
        proj_dropout=kwarg('proj_dropout', 0.1),             # Dropout rate for projection in attention layers
        drop_rate=kwarg('drop_rate', 0.1),                   # General dropout rate used in MLP and positional embeddings
        spectral_norm=kwarg('spectral_norm', False),         # Whether to use spectral normalization
        num_classes=kwarg('num_classes', 1),                 # Number of output classes (1 for binary classification in discriminator)
        out_channels=kwarg('out_channels', 32),              # Output channels of initial convolution layer
        diff_aug=kwarg('diff_aug', 'color,translation,cutout,rotate')  # Data augmentation policies
    
    )

    # Check for unknown kwargs.
    kwarg('structure')
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError('Unknown TensorFlow kwarg', unknown_kwargs[0])

    # Collect params.
    tf_params = _collect_tf_params(tf_D)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r'FromRGB_lod(\d+)/(.*)', name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f'{r}x{r}/FromRGB/{match.group(2)}'] = value
            kwargs.architecture = 'orig'
    from training import ViT_Discriminator
    D = ViT_Discriminator.HybridDiscriminator(**kwargs).eval().requires_grad_(False)
    _populate_module_params(D,
    r'patch_embed\.weight',         lambda: tf_params.get('ViT/PatchEmbed/weight', None).transpose(3, 2, 0, 1),
    r'patch_embed\.bias',           lambda: tf_params.get('ViT/PatchEmbed/bias', None),
    r'class_embedding',             lambda: tf_params.get('ViT/CLS/weight', None),
    r'positional_embedding',        lambda: tf_params.get('ViT/Position/weight', None),
    r'Encoder_Blocks\.(\d+)\.attn\.qkv\.weight',        lambda i: tf_params.get(f'ViT/Encoder{i}/Attention/qkv/weight', None).transpose(1, 0),
    r'Encoder_Blocks\.(\d+)\.attn\.qkv\.bias',          lambda i: tf_params.get(f'ViT/Encoder{i}/Attention/qkv/bias', None),
    r'Encoder_Blocks\.(\d+)\.attn\.out\.0\.weight',     lambda i: tf_params.get(f'ViT/Encoder{i}/Attention/out/weight', None).transpose(1, 0),
    r'Encoder_Blocks\.(\d+)\.attn\.out\.0\.bias',       lambda i: tf_params.get(f'ViT/Encoder{i}/Attention/out/bias', None),
    r'Encoder_Blocks\.(\d+)\.mlp\.fc1\.weight',         lambda i: tf_params.get(f'ViT/Encoder{i}/MLP/FC1/weight', None).transpose(1, 0),
    r'Encoder_Blocks\.(\d+)\.mlp\.fc1\.bias',           lambda i: tf_params.get(f'ViT/Encoder{i}/MLP/FC1/bias', None),
    r'Encoder_Blocks\.(\d+)\.mlp\.fc2\.weight',         lambda i: tf_params.get(f'ViT/Encoder{i}/MLP/FC2/weight', None).transpose(1, 0),
    r'Encoder_Blocks\.(\d+)\.mlp\.fc2\.bias',           lambda i: tf_params.get(f'ViT/Encoder{i}/MLP/FC2/bias', None),
    r'Encoder_Blocks\.(\d+)\.ln1\.weight',              lambda i: tf_params.get(f'ViT/Encoder{i}/LayerNorm1/weight', None),
    r'Encoder_Blocks\.(\d+)\.ln1\.bias',                lambda i: tf_params.get(f'ViT/Encoder{i}/LayerNorm1/bias', None),
    r'Encoder_Blocks\.(\d+)\.ln2\.weight',              lambda i: tf_params.get(f'ViT/Encoder{i}/LayerNorm2/weight', None),
    r'Encoder_Blocks\.(\d+)\.ln2\.bias',                lambda i: tf_params.get(f'ViT/Encoder{i}/LayerNorm2/bias', None),
    r'norm\.weight',                                    lambda: tf_params.get('ViT/FinalLayerNorm/weight', None),
    r'norm\.bias',                                      lambda: tf_params.get('ViT/FinalLayerNorm/bias', None),
    r'out\.weight',                                     lambda: tf_params.get('ViT/Output/weight', None).transpose(1, 0),
    r'out\.bias',                                       lambda: tf_params.get('ViT/Output/bias', None),
)
    return D

@click.command()
@click.option('--source', help='Input pickle', required=True, metavar='PATH')
@click.option('--dest', help='Output pickle', required=True, metavar='PATH')
@click.option('--force-fp16', help='Force the networks to use FP16', type=bool, default=False, metavar='BOOL', show_default=True)
def convert_network_pickle(source, dest, force_fp16):

    print(f'Loading "{source}"...')
    with dnnlib.util.open_url(source) as f:
        data = load_network_pkl(f, force_fp16=force_fp16)
    print(f'Saving "{dest}"...')
    with open(dest, 'wb') as f:
        pickle.dump(data, f)
    print('Done.')

if __name__ == "__main__":
    convert_network_pickle() # pylint: disable=no-value-for-parameter
