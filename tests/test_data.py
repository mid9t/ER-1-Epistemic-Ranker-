"""Tests for CLINC150 data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.data.clinc150 import CLINC150Config, PLUS_EXPECTED, build_clinc150_dataloaders


@pytest.fixture(scope="module")
def loaders():
    cfg = CLINC150Config(
        batch_size=16,
        eval_batch_size=64,
        num_workers=0,
        seed=0,
        local_data_dir="data/CLINC150",
    )
    return build_clinc150_dataloaders(cfg)


def test_splits(loaders):
    tr, va, te, ova, ote, meta = loaders
    assert meta["num_classes"] == 150
    # Dataset lengths (loaders may drop_last on train)
    assert len(tr.dataset) == PLUS_EXPECTED["id_train"]
    assert len(va.dataset) == PLUS_EXPECTED["id_val"]
    assert len(te.dataset) == PLUS_EXPECTED["id_test"]
    assert len(ova.dataset) == PLUS_EXPECTED["oos_val"]
    assert len(ote.dataset) == PLUS_EXPECTED["oos_test"]

    def _all_labels(loader):
        return torch.cat([b["labels"] for b in loader])

    for name, loader in [("train", tr), ("val", va), ("test", te)]:
        y = _all_labels(loader)
        assert (y >= 0).all() and (y < 150).all(), name
        assert (y != -1).all(), name

    for name, loader in [("oos_val", ova), ("oos_test", ote)]:
        y = _all_labels(loader)
        assert (y == -1).all(), name
