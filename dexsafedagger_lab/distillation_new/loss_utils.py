"""Loss/math utilities shared by distillation scripts."""

import torch


def l2(model, target):
    """Compute per-sample L2 norm across the last dimension."""
    return torch.norm(model - target, p=2, dim=-1)


def weighted_l2(model, target, weights):
    """Compute per-sample weighted L2 norm across the last dimension."""
    return torch.sum((model - target) * (weights * (model - target)), dim=-1) ** 0.5
