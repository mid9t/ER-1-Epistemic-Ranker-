"""End-to-end smoke test on a tiny subset (CPU, fast)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.config import ExperimentConfig, GDAConfig
from bert_daedl.data.clinc150 import CLINC150Config, build_clinc150_dataloaders
from bert_daedl.density.diagnostics import density_report
from bert_daedl.density.gda import GaussianDensityModel, NORMALIZERS, scale_alpha
from bert_daedl.eval.protocol import collect, evaluate_arm
from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.models.bert_edl import BertEvidentialClassifier
from bert_daedl.train import train_model
from bert_daedl.utils.io import REQUIRED_RESULT_KEYS, save_json, save_results
from bert_daedl.utils.seed import seed_everything


def _subset_loader(loader, n, batch_size, shuffle=False):
    ds = Subset(loader.dataset, list(range(min(n, len(loader.dataset)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@pytest.mark.slow
def test_smoke(tmp_path):
    seed_everything(0)
    data_cfg = CLINC150Config(
        batch_size=16,
        eval_batch_size=32,
        num_workers=0,
        seed=0,
        local_data_dir="data/CLINC150",
    )
    tr, va, te, ova, ote, meta = build_clinc150_dataloaders(data_cfg)

    # Tiny subsets for speed
    tr = _subset_loader(tr, 512, 16, shuffle=True)
    va = _subset_loader(va, 128, 32)
    te = _subset_loader(te, 128, 32)
    ova = _subset_loader(ova, 64, 32)
    ote = _subset_loader(ote, 64, 32)

    cfg = ExperimentConfig(
        arm="edl",
        seed=0,
        epochs=1,
        device="cpu",
        num_workers=0,
        batch_size=16,
        eval_batch_size=32,
        feat_dim=64,
        gda=GDAConfig(pca_dim=32),
        output_dir=str(tmp_path),
        kl_anneal_epochs=1,
    )
    model = BertEvidentialClassifier(
        cfg.model_name,
        meta["num_classes"],
        cfg.feat_dim,
        cfg.sn_bottleneck,
        cfg.bottleneck_activation,
        cfg.pooling,
        cfg.dropout,
    )
    # ~200 optimizer steps: 512/16 = 32 steps/epoch × a few micro-epochs via manual loop
    # Use train_model with 1 epoch on 512 examples (~32 steps); bump by repeating.
    for _ in range(6):  # ~192 steps
        model, best = train_model(model, tr, va, ova, cfg, meta["num_classes"])

    _, Ftr, Ytr = collect(model, tr, cfg.device)
    gda = GaussianDensityModel(cfg.gda).fit(Ftr.numpy(), Ytr.numpy())
    lp_train = gda.marginal_log_prob(Ftr.numpy())
    _, Fv, _ = collect(model, va, cfg.device)
    _, Fov, _ = collect(model, ova, cfg.device)
    diag = density_report(
        gda,
        lp_train,
        gda.marginal_log_prob(Fv.numpy()),
        gda.marginal_log_prob(Fov.numpy()),
    )
    save_json(tmp_path, "diagnostics.json", diag)

    results = evaluate_arm(model, (tr, va, te, ova, ote), cfg, gda, None)
    results |= {
        "arm": "edl",
        "seed": 0,
        "best_val_acc": best,
        "wall_clock_sec": 0.0,
        "config": {
            "arm": "edl",
            "seed": 0,
            "epochs": 1,
            "device": "cpu",
        },
    }
    path = save_results(tmp_path, results)
    loaded = json.loads(path.read_text())
    for k in REQUIRED_RESULT_KEYS:
        assert k in loaded, k
    assert (tmp_path / "diagnostics.json").exists()
    assert "raw_density_val_auroc" in diag
