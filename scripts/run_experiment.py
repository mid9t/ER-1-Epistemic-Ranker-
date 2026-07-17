#!/usr/bin/env python
"""Run one experiment arm (ce | edl | daedl) end-to-end."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

# Allow `python scripts/run_experiment.py` without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.config import ExperimentConfig
from bert_daedl.data.clinc150 import CLINC150Config, build_clinc150_dataloaders
from bert_daedl.density.diagnostics import density_report
from bert_daedl.density.gda import NORMALIZERS, GaussianDensityModel, scale_alpha
from bert_daedl.eval.protocol import collect, evaluate_arm
from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.models.bert_edl import BertEvidentialClassifier
from bert_daedl.train import train_model
from bert_daedl.utils.io import save_json, save_results
from bert_daedl.utils.seed import seed_everything


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def main(cfg: ExperimentConfig):
    cfg.device = resolve_device(cfg.device)
    if cfg.device == "mps":
        cfg.num_workers = 0

    seed_everything(cfg.seed)
    data_cfg = CLINC150Config(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    loaders = build_clinc150_dataloaders(data_cfg)
    tr, va, te, ova, ote, meta = loaders
    model = BertEvidentialClassifier(
        cfg.model_name,
        meta["num_classes"],
        cfg.feat_dim,
        cfg.sn_bottleneck,
        cfg.bottleneck_activation,
        cfg.pooling,
        cfg.dropout,
    )

    t0 = time.time()
    # daedl reuses edl training; train with edl loss when arm is daedl
    train_cfg = dataclasses.replace(cfg, arm="edl" if cfg.arm == "daedl" else cfg.arm)
    model, best_val_acc = train_model(
        model, tr, va, ova, train_cfg, meta["num_classes"]
    )

    # GDA on TRAIN features (every arm -> Mahalanobis baseline is free)
    _, Ftr, Ytr = collect(model, tr, cfg.device)
    gda = GaussianDensityModel(cfg.gda).fit(Ftr.numpy(), Ytr.numpy())
    lp_train = gda.marginal_log_prob(Ftr.numpy())

    normalizer = None
    diag_alpha = (None, None)
    if cfg.arm == "daedl":
        normalizer = NORMALIZERS[cfg.normalizer]().fit(lp_train)
        Lv, Fv, _ = collect(model, va, cfg.device)
        phat_v = torch.from_numpy(
            normalizer(gda.marginal_log_prob(Fv.numpy()))
        )
        a_plain = logits_to_alpha(Lv, cfg.logit_clamp, cfg.alpha_eps)
        a_scaled = scale_alpha(
            Lv, phat_v, cfg.combine_mode, cfg.lam, cfg.logit_clamp
        )
        diag_alpha = (a_plain, a_scaled)

    _, Fv, _ = collect(model, va, cfg.device)
    _, Fov, _ = collect(model, ova, cfg.device)
    diag = density_report(
        gda,
        lp_train,
        gda.marginal_log_prob(Fv.numpy()),
        gda.marginal_log_prob(Fov.numpy()),
        *diag_alpha,
    )
    save_json(cfg.output_dir, "diagnostics.json", diag)
    if diag.get("materiality", {}).get("density_effect") == "INERT":
        print(
            "WARNING: density term is INERT under current normalizer "
            "(CIFAR failure mode). See diagnostics.json before trusting daedl_* scores."
        )

    results = evaluate_arm(model, (tr, va, te, ova, ote), cfg, gda, normalizer)
    results |= {
        "arm": cfg.arm,
        "seed": cfg.seed,
        "best_val_acc": best_val_acc,
        "wall_clock_sec": round(time.time() - t0, 1),
        "config": dataclasses.asdict(cfg),
    }
    path = save_results(cfg.output_dir, results)
    print(f"Results -> {path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["ce", "edl", "daedl"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--normalizer", default="qsigmoid")
    p.add_argument("--combine_mode", default="mul", choices=["mul", "add_log"])
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=128)
    a = p.parse_args()
    cfg = ExperimentConfig(
        arm=a.arm,
        seed=a.seed,
        epochs=a.epochs,
        normalizer=a.normalizer,
        combine_mode=a.combine_mode,
        lam=a.lam,
        device=a.device,
        batch_size=a.batch_size,
        eval_batch_size=a.eval_batch_size,
        output_dir=f"{a.output_dir}/{a.arm}_seed{a.seed}",
    )
    main(cfg)
