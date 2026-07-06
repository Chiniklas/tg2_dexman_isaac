"""Loss/math utilities shared by distillation scripts."""

import torch


def l2(model, target):
    """Compute per-sample L2 norm across the last dimension."""
    return torch.norm(model - target, p=2, dim=-1)


def weighted_l2(model, target, weights):
    """Compute per-sample weighted L2 norm across the last dimension."""
    return torch.sum((model - target) * (weights * (model - target)), dim=-1) ** 0.5


def clip_gradients_and_step(parameters, optimizer, max_grad_norm):
    """Clip finite gradients and step without allowing NaNs into parameters.

    Returns ``(stepped, grad_norm)``. A non-finite norm clears the gradients and
    skips the optimizer update, leaving model parameters unchanged.
    """
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        float(max_grad_norm),
        error_if_nonfinite=False,
    )
    grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu().item())
    stepped = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
    if stepped:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return stepped, grad_norm_value
