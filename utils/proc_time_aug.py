"""Order-preserving processing-time transforms (Section 4.5.1, Table 5).

Every transform is strictly increasing on the positive processing times and is mapped back to the original range, so
the order of the processing times (e.g. the fastest machine of each operation) is kept. Zeros (non-edges) stay zero.
The transforms are applied only to the inputs of the network, never to the environment.
"""
import torch
import math
import random
from typing import List


def _rescale(t_aug: torch.Tensor, t_orig: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Map the transformed values back to the original range (only where mask is True)."""
    if mask.sum() == 0:
        return t_aug

    orig_min = t_orig[mask].min()
    orig_max = t_orig[mask].max()
    aug_min = t_aug[mask].min()
    aug_max = t_aug[mask].max()

    if (aug_max - aug_min).abs() < 1e-8:
        return t_orig.clone().float()

    t_aug[mask] = (t_aug[mask] - aug_min) / (aug_max - aug_min) * (orig_max - orig_min) + orig_min
    return t_aug


def aug_identity(proc_times: torch.Tensor) -> torch.Tensor:
    """#0 Identity."""
    return proc_times.clone().float()

def aug_sqrt(proc_times: torch.Tensor) -> torch.Tensor:
    """#1 Square root: concave, compresses the differences between long operations."""
    mask = proc_times > 0
    t = proc_times.clone().float()
    t[mask] = t[mask].sqrt()
    return _rescale(t, proc_times.float(), mask)

def aug_square(proc_times: torch.Tensor) -> torch.Tensor:
    """#2 Square: convex, expands the differences between long operations."""
    mask = proc_times > 0
    t = proc_times.clone().float()
    t[mask] = t[mask].pow(2)
    return _rescale(t, proc_times.float(), mask)

def aug_log(proc_times: torch.Tensor) -> torch.Tensor:
    """#3 Logarithm: compresses long operations more strongly."""
    mask = proc_times > 0
    t = proc_times.clone().float()
    t[mask] = torch.log1p(t[mask])
    return _rescale(t, proc_times.float(), mask)

def aug_exp(proc_times: torch.Tensor) -> torch.Tensor:
    """#4 Exponential: expands long operations more strongly."""
    mask = proc_times > 0
    t = proc_times.clone().float()
    vals = t[mask]
    v_min, v_max = vals.min(), vals.max()
    if (v_max - v_min).abs() > 1e-8:
        u = (vals - v_min) / (v_max - v_min)
        t[mask] = torch.exp(u)
    return _rescale(t, proc_times.float(), mask)

def aug_smoothstep(proc_times: torch.Tensor) -> torch.Tensor:
    """#5 Smoothstep 3u^2 - 2u^3: compresses both ends and expands the middle."""
    mask = proc_times > 0
    t = proc_times.clone().float()
    vals = t[mask]
    v_min, v_max = vals.min(), vals.max()
    if (v_max - v_min).abs() > 1e-8:
        u = (vals - v_min) / (v_max - v_min)
        t[mask] = 3 * u.pow(2) - 2 * u.pow(3)
    return _rescale(t, proc_times.float(), mask)

def aug_sin(proc_times: torch.Tensor) -> torch.Tensor:
    """#6 Sine sin(pi u / 2): concave."""
    mask = proc_times > 0
    t = proc_times.clone().float()
    vals = t[mask]
    v_min, v_max = vals.min(), vals.max()
    if (v_max - v_min).abs() > 1e-8:
        u = (vals - v_min) / (v_max - v_min)
        t[mask] = torch.sin(u * math.pi / 2)
    return _rescale(t, proc_times.float(), mask)

def aug_rank(proc_times: torch.Tensor) -> torch.Tensor:
    """#7 Rank: equally spaced values by rank; keeps only the order."""
    mask = proc_times > 0
    t = proc_times.clone().float()
    orig_min = t[mask].min() if mask.any() else t.new_tensor(0)
    orig_max = t[mask].max() if mask.any() else t.new_tensor(1)

    vals = t[mask]
    if vals.numel() <= 1:
        return t
    ranks = vals.argsort().argsort().float()
    ranks = ranks / ranks.max().clamp(min=1) * (orig_max - orig_min) + orig_min
    t[mask] = ranks
    return t


AUGMENTATIONS = [
    ("identity", aug_identity),
    ("sqrt", aug_sqrt),
    ("square", aug_square),
    ("log", aug_log),
    ("exp", aug_exp),
    ("smoothstep", aug_smoothstep),
    ("sin", aug_sin),
    ("rank", aug_rank),
]


def get_all_augmentations() -> List[tuple]:
    """All (name, fn) pairs."""
    return AUGMENTATIONS


def get_random_augmentation() -> tuple:
    """A random transform (name, fn)."""
    return random.choice(AUGMENTATIONS)


def vectorized_augment_all(x: torch.Tensor) -> torch.Tensor:
    """Apply all eight transforms to the processing times x of one instance (any shape); returns [8, ...]."""
    x = x.float()
    mask = x > 0
    result = x.unsqueeze(0).expand(8, *x.shape).clone()

    if not mask.any():
        return result

    vals = x[mask]
    orig_min, orig_max = vals.min(), vals.max()

    if (orig_max - orig_min).abs() < 1e-8:
        return result

    u = (vals - orig_min) / (orig_max - orig_min)

    transforms = torch.stack([
        vals.sqrt(),
        vals.pow(2),
        torch.log1p(vals),
        torch.exp(u),
        3 * u.pow(2) - 2 * u.pow(3),
        torch.sin(u * (math.pi / 2)),
    ])

    t_min = transforms.min(dim=1, keepdim=True).values
    t_max = transforms.max(dim=1, keepdim=True).values
    denom = (t_max - t_min).clamp(min=1e-8)
    transforms = (transforms - t_min) / denom * (orig_max - orig_min) + orig_min

    ranks = vals.argsort().argsort().float()
    ranks = ranks / ranks.max().clamp(min=1) * (orig_max - orig_min) + orig_min

    flat = result.view(8, -1)
    flat_mask = mask.view(-1)
    flat[1:7, flat_mask] = transforms
    flat[7, flat_mask] = ranks

    return result


def vectorized_augment_batch(x: torch.Tensor) -> torch.Tensor:
    """x: [8, ...]; the i-th transform is applied to the i-th batch element."""
    x = x.float()
    B = x.shape[0]
    spatial = x.shape[1:]
    flat_x = x.view(B, -1)
    mask = flat_x > 0

    result = flat_x.clone()

    big = flat_x.clone()
    big[~mask] = float('inf')
    elem_min = big.min(dim=1, keepdim=True).values
    big = flat_x.clone()
    big[~mask] = float('-inf')
    elem_max = big.max(dim=1, keepdim=True).values
    elem_range = (elem_max - elem_min).clamp(min=1e-8)

    u = (flat_x - elem_min) / elem_range * mask.float()

    raw_x = flat_x.clamp(min=0)
    all_t = torch.stack([
        torch.where(mask, raw_x.sqrt(), flat_x),
        torch.where(mask, raw_x.pow(2), flat_x),
        torch.where(mask, torch.log1p(raw_x), flat_x),
        torch.where(mask, torch.exp(u), flat_x),
        torch.where(mask, 3 * u.pow(2) - 2 * u.pow(3), flat_x),
        torch.where(mask, torch.sin(u * (math.pi / 2)), flat_x),
    ])

    mask_3d = mask.unsqueeze(0).expand(6, -1, -1)
    t_for_min = all_t.clone()
    t_for_min[~mask_3d] = float('inf')
    t_min = t_for_min.min(dim=2, keepdim=True).values
    t_for_max = all_t.clone()
    t_for_max[~mask_3d] = float('-inf')
    t_max = t_for_max.max(dim=2, keepdim=True).values
    t_denom = (t_max - t_min).clamp(min=1e-8)

    rescaled = (all_t - t_min) / t_denom * elem_range.unsqueeze(0) + elem_min.unsqueeze(0)
    rescaled = torch.where(mask_3d, rescaled, flat_x.unsqueeze(0).expand(6, -1, -1))

    idx_t = torch.arange(6, device=x.device)
    idx_b = torch.arange(1, 7, device=x.device)
    result[1:7] = rescaled[idx_t, idx_b]

    if mask[7].any():
        vals_7 = flat_x[7, mask[7]]
        if vals_7.numel() > 1:
            ranks = vals_7.argsort().argsort().float()
            ranks = ranks / ranks.max().clamp(min=1) * (elem_max[7] - elem_min[7]).item() + elem_min[7].item()
            result[7, mask[7]] = ranks

    return result.view(B, *spatial)
