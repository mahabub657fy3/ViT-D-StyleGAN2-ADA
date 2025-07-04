import torch
import torch.nn.functional as F
import random

def DiffAugment(x, policy='', channels_first=True, **kwargs):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x

def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.2):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_cutout(x, ratio=0.5):
    if random.random() < 0.3:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
    return x

def rand_rotate(x, ratio=0.5):
    k = random.randint(1, 3)
    if random.random() < ratio:
        x = torch.rot90(x, k, [2, 3])
    return x
# New functions for Scaling and Color Jitter
def rand_scaling(x, scale_range=(0.8, 1.2)):
    scale_factor = random.uniform(*scale_range)
    new_size = [int(x.size(2) * scale_factor), int(x.size(3) * scale_factor)]
    x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)
    return F.interpolate(x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

def rand_color_jitter(x, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    # Apply brightness jitter
    x = x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * brightness
    # Apply contrast jitter
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (1 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * contrast) + x_mean
    # Apply saturation jitter
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (1 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * saturation) + x_mean
    # Apply hue jitter
    x = x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * hue
    x = torch.clamp(x, 0, 1)  # Ensure values are in valid range after adjustments
    return x

# Update the AUGMENT_FNS dictionary to include scaling and color jitter
AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast, rand_color_jitter],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'rotate': [rand_rotate],
    'scale': [rand_scaling]  # Add the new scaling function
}