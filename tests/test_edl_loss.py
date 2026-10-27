"""EDL loss sanity tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.losses.edl import edl_loss, expected_mse, kl_weight, logits_to_alpha


def _cfg(**kw):
    base = dict(
        logit_clamp=15.0,
        alpha_eps=1e-6,
        kl_anneal_epochs=5,
        reg_param=1e-3,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_loss_finite_at_extreme_logits():
    logits = torch.tensor([[20.0, -20.0, 0.0], [-20.0, 20.0, 5.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = edl_loss(logits, targets, epoch=0, cfg=_cfg())
    assert torch.isfinite(loss)
    alpha = logits_to_alpha(logits)
    assert (alpha > 0).all()


def test_loss_decreases_overfitting_one_batch():
    torch.manual_seed(0)
    logits = torch.nn.Parameter(torch.randn(8, 4) * 0.1)
    targets = torch.nn.functional.one_hot(torch.arange(8) % 4, 4).float()
    opt = torch.optim.Adam([logits], lr=0.2)
    cfg = _cfg(reg_param=1e-4)
    losses = []
    for step in range(50):
        opt.zero_grad()
        loss = edl_loss(logits, targets, epoch=min(step // 10, 4), cfg=cfg)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert (logits_to_alpha(logits.detach()) > 0).all()


def test_mse_loss_finite_and_decreases():
    torch.manual_seed(0)
    logits = torch.nn.Parameter(torch.randn(8, 4) * 0.1)
    targets = torch.nn.functional.one_hot(torch.arange(8) % 4, 4).float()
    opt = torch.optim.Adam([logits], lr=0.2)
    cfg = _cfg(edl_loss="mse", kl_schedule="constant", reg_param=1e-4)
    first = expected_mse(logits_to_alpha(logits.detach()), targets)
    assert torch.isfinite(first).all()
    losses = []
    for step in range(40):
        opt.zero_grad()
        loss = edl_loss(logits, targets, epoch=0, cfg=cfg)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert torch.isfinite(torch.tensor(losses[-1]))
    assert losses[-1] < losses[0]


def test_kl_schedule():
    anneal = _cfg(kl_schedule="anneal", reg_param=1e-3, kl_anneal_epochs=5)
    assert kl_weight(0, anneal) == (1 / 5) * 1e-3
    assert kl_weight(4, anneal) == 1e-3
    constant = _cfg(kl_schedule="constant", reg_param=5e-2)
    assert kl_weight(0, constant) == 5e-2
    assert kl_weight(9, constant) == 5e-2


def test_default_loss_unchanged():
    torch.manual_seed(1)
    logits = torch.randn(6, 3)
    targets = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 0, 1, 2]), 3).float()
    cfg = _cfg()
    alpha = logits_to_alpha(logits, cfg.logit_clamp, cfg.alpha_eps)
    a0 = alpha.sum(-1, keepdim=True)
    nll = (targets * (torch.digamma(a0) - torch.digamma(alpha))).sum(-1)
    lam = min(1.0, (0 + 1) / 5) * cfg.reg_param
    from bert_daedl.losses.edl import dirichlet_kl_to_uniform

    alpha_tilde = targets + (1.0 - targets) * alpha
    expected = (nll + lam * dirichlet_kl_to_uniform(alpha_tilde)).mean()
    assert torch.allclose(edl_loss(logits, targets, epoch=0, cfg=cfg), expected)
