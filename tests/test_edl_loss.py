"""EDL loss sanity tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.losses.edl import edl_loss, logits_to_alpha


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
