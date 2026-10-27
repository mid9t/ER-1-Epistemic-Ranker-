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
from bert_daedl.data import build_dataloaders
from bert_daedl.density.diagnostics import density_report
from bert_daedl.density.gda import NORMALIZERS, GaussianDensityModel, scale_alpha
from bert_daedl.eval.protocol import collect, evaluate_arm
from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.models.bert_edl import BertEvidentialClassifier
from bert_daedl.posthoc.cache import eval_order_loader, extract_split, save_cache
from bert_daedl.train import train_model
from bert_daedl.utils.io import config_hash, git_sha, save_json, save_results
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


def main(cfg: ExperimentConfig, *, save_cache_arrays: bool = False, save_pooled: bool = False, save_ckpt: bool = False):
    cfg.device = resolve_device(cfg.device)
    if cfg.device == "mps":
        cfg.num_workers = 0

    seed_everything(cfg.seed)
    loaders = build_dataloaders(cfg.dataset, cfg, cfg.data_dir)
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

    # GDA on every training utterance, in file order (no shuffle, no drop_last).
    tr_eval = eval_order_loader(tr, batch_size=cfg.eval_batch_size, num_workers=cfg.num_workers)
    train_arrays = extract_split(model, tr_eval, cfg.device, with_pooled=save_pooled)
    gda = GaussianDensityModel(cfg.gda).fit(train_arrays["feats"], train_arrays["labels"])
    lp_train = gda.marginal_log_prob(train_arrays["feats"])

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
        "tag": cfg.tag,
        "best_val_acc": best_val_acc,
        "wall_clock_sec": round(time.time() - t0, 1),
        "config": dataclasses.asdict(cfg),
    }
    if save_ckpt:
        ckpt_path = Path(cfg.output_dir) / "model.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    path = save_results(cfg.output_dir, results)
    print(f"Results -> {path}")
    if save_cache_arrays:
        arrays = {"train": train_arrays}
        for name, loader in (("val", va), ("test", te), ("oos_val", ova), ("oos_test", ote)):
            arrays[name] = extract_split(model, loader, cfg.device, with_pooled=save_pooled)
        cfg_dict = dataclasses.asdict(cfg)
        save_cache(
            cfg.output_dir,
            arrays,
            meta={
                "arm": cfg.arm,
                "seed": cfg.seed,
                "tag": cfg.tag,
                "dataset": cfg.dataset,
                "model_name": cfg.model_name,
                "max_length": cfg.max_length,
                "data_dir": cfg.data_dir,
                "truncation": meta.get("truncation"),
                "truncation_worst": meta.get("truncation_worst"),
                "config": cfg_dict,
                "config_hash": config_hash(cfg_dict),
                "git_sha": git_sha(),
                "best_val_acc": best_val_acc,
                "label_names": meta.get("id2label"),
            },
        )
        print(f"Cache -> {cfg.output_dir}/cache.npz")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["ce", "edl", "daedl"], required=True)
    p.add_argument("--dataset", default="clinc150", choices=["clinc150", "banking77_oos"])
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--normalizer", default="qsigmoid")
    p.add_argument("--combine_mode", default="mul", choices=["mul", "add_log"])
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--save_cache", action="store_true")
    p.add_argument("--save_pooled", action="store_true")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--sn_bottleneck", choices=["on", "off"], default="on")
    p.add_argument("--edl_loss", choices=["digamma", "mse"], default="digamma")
    p.add_argument("--kl_schedule", choices=["anneal", "constant"], default="anneal")
    p.add_argument("--reg_param", type=float, default=1e-3)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    leaf = f"{a.arm}_seed{a.seed}" if not a.tag else f"{a.arm}_{a.tag}_seed{a.seed}"
    cfg = ExperimentConfig(
        dataset=a.dataset,
        data_dir=a.data_dir,
        max_length=a.max_length,
        arm=a.arm,
        seed=a.seed,
        epochs=a.epochs,
        normalizer=a.normalizer,
        combine_mode=a.combine_mode,
        lam=a.lam,
        device=a.device,
        batch_size=a.batch_size,
        eval_batch_size=a.eval_batch_size,
        model_name=a.model_name,
        sn_bottleneck=a.sn_bottleneck == "on",
        edl_loss=a.edl_loss,
        kl_schedule=a.kl_schedule,
        reg_param=a.reg_param,
        tag=a.tag,
        output_dir=f"{a.output_dir}/{leaf}",
    )
    main(cfg, save_cache_arrays=a.save_cache, save_pooled=a.save_pooled, save_ckpt=a.save_ckpt)
