"""Score orientation and vacuity unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.uncertainty.scores import scores_from_alpha, scores_from_logits


def test_orientation():
    conf = torch.tensor([[8.0, 0.1, 0.1]])
    unsure = torch.tensor([[0.4, 0.3, 0.3]])
    for k, v in scores_from_logits(unsure).items():
        assert v > scores_from_logits(conf)[k], k
    a_conf, a_un = torch.exp(conf), torch.exp(unsure)
    for k in scores_from_alpha(a_un):
        assert scores_from_alpha(a_un)[k] > scores_from_alpha(a_conf)[k], k


def test_vacuity_by_hand():
    # K=3, alpha=(2,1,1) -> vacuity = 3/4
    alpha = torch.tensor([[2.0, 1.0, 1.0]])
    vac = scores_from_alpha(alpha)["vacuity"]
    assert torch.allclose(vac, torch.tensor([0.75]))
