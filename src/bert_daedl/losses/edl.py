"""Evidential Deep Learning loss (EXP parameterization, annealed KL).

EXP param matches validated DAEDL/ CIFAR: alpha = eps + exp(z).
Bayes-risk digamma NLL + KL to uniform on non-target evidence (alpha_tilde).
"""

from __future__ import annotations

import torch


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ), per-sample."""
    K = alpha.shape[-1]
    a0 = alpha.sum(-1)
    t1 = torch.lgamma(a0) - torch.lgamma(alpha).sum(-1)
    t1 = t1 - torch.lgamma(torch.tensor(float(K), device=alpha.device, dtype=alpha.dtype))
    t2 = (
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(a0).unsqueeze(-1))
    ).sum(-1)
    return t1 + t2


def logits_to_alpha(
    logits: torch.Tensor, clamp: float = 15.0, eps: float = 1e-6
) -> torch.Tensor:
    # EXP parameterization (DAEDL). Clamp guards fresh-head BERT logits
    # from exp overflow in epoch 0; CIFAR didn't need it, BERT does.
    return torch.exp(logits.clamp(-clamp, clamp)) + eps


def edl_loss(logits, targets_onehot, epoch: int, cfg) -> torch.Tensor:
    alpha = logits_to_alpha(logits, cfg.logit_clamp, cfg.alpha_eps)
    a0 = alpha.sum(-1, keepdim=True)
    nll = (targets_onehot * (torch.digamma(a0) - torch.digamma(alpha))).sum(-1)
    lam = min(1.0, (epoch + 1) / max(cfg.kl_anneal_epochs, 1)) * cfg.reg_param
    alpha_tilde = targets_onehot + (1.0 - targets_onehot) * alpha
    return (nll + lam * dirichlet_kl_to_uniform(alpha_tilde)).mean()
