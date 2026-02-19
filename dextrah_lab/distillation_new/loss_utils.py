"""Loss/math utilities shared by distillation scripts."""

import math

import torch


def l2(model, target):
    """Compute per-sample L2 norm across the last dimension."""
    return torch.norm(model - target, p=2, dim=-1)


def weighted_l2(model, target, weights):
    """Compute per-sample weighted L2 norm across the last dimension."""
    return torch.sum((model - target) * (weights * (model - target)), dim=-1) ** 0.5


def gaussian_kl(mu_student, sigma_student, mu_teacher, sigma_teacher, eps=1e-6):
    """KL(N_teacher || N_student) per sample, summed over action dimensions."""
    sigma_student = torch.clamp(sigma_student, min=eps)
    sigma_teacher = torch.clamp(sigma_teacher, min=eps)
    var_student = sigma_student ** 2
    var_teacher = sigma_teacher ** 2
    mu_term = (mu_teacher - mu_student) ** 2 / (2.0 * var_student)
    sigma_term = torch.log(sigma_student / sigma_teacher) + var_teacher / (2.0 * var_student) - 0.5
    kl = mu_term + sigma_term
    return kl.sum(-1), mu_term.sum(-1), sigma_term.sum(-1)


def gaussian_nll(mu_student, sigma_student, actions, eps=1e-6):
    """Negative log-likelihood under N(mu_student, sigma_student), summed over action dimensions."""
    sigma_student = torch.clamp(sigma_student, min=eps)
    var_student = sigma_student ** 2
    nll = 0.5 * (
        ((actions - mu_student) ** 2) / var_student
        + 2.0 * torch.log(sigma_student)
        + math.log(2.0 * math.pi)
    )
    return nll.sum(-1)
