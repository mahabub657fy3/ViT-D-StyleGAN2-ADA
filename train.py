import os
import click
import re
import json
import tempfile
import torch
import dnnlib
from ast import arg
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import network_pkl
from metrics import metric_main
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

class UserError(Exception):
    pass

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def training_loop(  
    run_dir                 = '.',     
    training_set_kwargs     = {},      
    data_loader_kwargs      = {},       
    G_kwargs                = {},       
    D_kwargs                = {},       
    G_opt_kwargs            = {},      
    D_opt_kwargs            = {},      
    augment_kwargs          = None,     
    loss_kwargs             = {},       
    metrics                 = [],       
    random_seed             = 0,        
    num_gpus                = 1,        
    rank                    = 0,       
    batch_size              = 4,        
    batch_gpu               = 4,        
    ema_kimg                = 10,       
    ema_rampup              = None,     
    G_reg_interval          = 4,        
    D_reg_interval          = 16,       
    augment_p               = 0,        
    ada_target              = None,   
    ada_interval            = 4,       
    ada_kimg                = 500,     
    total_kimg              = 25000,    
    kimg_per_tick           = 4,       
    image_snapshot_ticks    = 50,     
    network_snapshot_ticks  = 50,       
    resume_pkl              = None,    
    cudnn_benchmark         = True,     
    allow_tf32              = False,   
    abort_fn                = None,     
    progress_fn             = None,    
):
    
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  
    torch.backends.cudnn.allow_tf32 = allow_tf32        
    conv2d_gradfix.enabled = True                       
    grid_sample_gradfix.enabled = True                
                
    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    d_common_kwargs = dict( img_resolution=training_set.resolution,img_channels=training_set.num_channels)
    
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    # D = load_vit_discriminator()
    D = dnnlib.util.construct_class_by_name(**D_kwargs,**d_common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = network_pkl.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img=misc.print_module_summary(G, [z, c])
        
        misc.print_module_summary(D, [img, c])
        # img = torch.randn(batch_size, 3, 512, 512)
        
    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # ViT-compatible loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else:
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, 
    snap       = None, 
    metrics    = None, 
    seed       = None, 
    # Dataset.
    data       = None, 
    cond       = None, 
    subset     = None, 
    mirror     = None, 
    # Base config.
    cfg        = None, 
    gamma      = None, 
    kimg       = None, 
    batch      = None, 
    # Discriminator augmentation.
    aug        = None,
    p          = None, 
    target     = None, 
    augpipe    = None, 
    # Transfer learning.
    resume     = None, 
    freezed    = None, 
    # Performance options (not included in desc).
    fp32       = None, 
    nhwc       = None, 
    allow_tf32 = None, 
    nobench    = None, 
    workers    = None,
    vit_embedding_dim=768, 
    vit_patch_size=16,             
    vit_num_heads=6,             
    vit_mlp_ratio=2,               
    vit_attention_dropout=0.1,
    vit_proj_dropout=0.1,         
    vit_drop_rate=0.3,
    vit_depth= 6, 
):
    args = dnnlib.EasyDict()

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=2, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) #
        args.training_set_kwargs.resolution = training_set.resolution 
        args.training_set_kwargs.use_labels = training_set.has_labels 
        args.training_set_kwargs.max_size = len(training_set) 
        desc = training_set.name
        del training_set
    except IOError as err:
        raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        'vit':       dict(ref_gpus=4,  kimg=25000,  mb=16, mbstd=4,  fmaps=1,   lrate=0.0005, gamma=10,   ema=10,  ramp=0.1, map=4),
    }   

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)
        spec.mbstd = min(spec.mb // gpus, 4)
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate_G = 0.0025 if res >= 512 else 0.001  # Lower learning rate for ViT
        spec.gamma = 0.0003 * (res ** 2) / spec.mb
        spec.ema = spec.mb * 10 / 32
        spec.lrate_D = 0.0005  # Reduce learning rate for ViT discriminator
        
    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.ViT_Discriminator',
                                        dim=vit_embedding_dim,       
                                        patch_size=vit_patch_size,          
                                        heads=vit_num_heads,              
                                        mlp_ratio=vit_mlp_ratio,             
                                        attention_dropout=vit_attention_dropout,
                                        proj_dropout=vit_proj_dropout,    
                                        drop_rate=vit_drop_rate,             
                                                            
                                        diff_aug='color,translation,cutout,rotate',  
                                        depth = vit_depth,
                                        positional_embedding=True,
                                        class_token=True,
                                    cnn_config=dnnlib.EasyDict(
                                        backbone='resnet50', 
                                        adaptive_pooling=True,
                                        spectral_norm=False
                                    ),
                                    post_transformer_config=dnnlib.EasyDict(
                                        dropout_rate=0.1,
                                        layer_norm=True,
                                        fc_layer=True
                                    ),
                                    fmaps=1,  
                                    block_kwargs=dnnlib.EasyDict(
                                            activation='gelu')
                            )
    args.G_kwargs.synthesis_kwargs.channel_base  = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max  = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res  = 4 
    args.G_kwargs.synthesis_kwargs.conv_clamp  = 256 
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate_G,  betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=spec.lrate_D, betas=[0,0.99], weight_decay=0.01,eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma,phi=10)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 
        args.loss_kwargs.style_mixing_prob = 0 
        args.D_kwargs.architecture = 'orig' 

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.7

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,patch_dropout=1,patch_shuffle=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop(rank=rank, **args)

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Dataset.
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar','vit']))
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', default='2', help='Override batch size', type=int, metavar='INT')

# Discriminator augmentation.
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
@click.option('--target', help='ADA target value for --aug=ada', type=float)
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, dry_run, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data"."""
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Prepare output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
