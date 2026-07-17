"""Uncertainty scores. Convention: HIGHER = MORE OOS (unit-tested)."""

from __future__ import annotations

import numpy as np
import torch


def scores_from_logits(logits: torch.Tensor) -> dict:
    p = logits.softmax(-1)
    return {
        "msp": 1.0 - p.max(-1).values,
        "entropy": -(p * (p + 1e-12).log()).sum(-1),
        "energy": -torch.logsumexp(logits, -1),
    }


def scores_from_alpha(alpha: torch.Tensor, prefix: str = "") -> dict:
    K = alpha.shape[-1]
    a0 = alpha.sum(-1)
    pbar = alpha / a0.unsqueeze(-1)
    pred_H = -(pbar * (pbar + 1e-12).log()).sum(-1)
    exp_H = -(
        pbar
        * (torch.digamma(alpha + 1) - torch.digamma(a0 + 1).unsqueeze(-1))
    ).sum(-1)
    lnB = torch.lgamma(alpha).sum(-1) - torch.lgamma(a0)
    diff_H = (
        lnB
        + (a0 - K) * torch.digamma(a0)
        - ((alpha - 1) * torch.digamma(alpha)).sum(-1)
    )
    out = {
        "vacuity": K / a0,
        "one_minus_maxpbar": 1.0 - pbar.max(-1).values,
        "pred_entropy": pred_H,
        "mutual_info": pred_H - exp_H,
        "diff_entropy": diff_H,
        "neg_alpha0": -a0,
    }
    return {prefix + k: v for k, v in out.items()}


def scores_from_density(marginal_logp) -> dict:
    return {"neg_logp": -np.asarray(marginal_logp)}
