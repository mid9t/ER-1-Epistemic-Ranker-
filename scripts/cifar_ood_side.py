#!/usr/bin/env python
"""CIFAR OOD-side check against a local DAEDL checkout. Does not write into it."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.density.gda import NORMALIZERS
from bert_daedl.posthoc.gate import pair_flip_gate
from bert_daedl.posthoc.report import write_json, write_md
from bert_daedl.posthoc.stats import auroc_scores


def _jitters():
    vals = [0.0, float(np.finfo(np.float64).tiny)]
    j = 1e-308
    while j <= 0.1:
        vals.append(j)
        j *= 10
    return vals


def _chol(cov):
    eye = np.eye(cov.shape[0])
    for j in _jitters():
        try:
            return np.linalg.cholesky(cov + j * eye), j
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("CIFAR class covariance is not positive definite")


def fit_gda(feats, labels):
    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    mus, chols, logdets = [], [], []
    for c in classes:
        xc = X[y == c]
        mu = xc.mean(0)
        cov = np.cov(xc, rowvar=False)
        chol, _j = _chol(cov)
        mus.append(mu)
        chols.append(chol)
        logdets.append(float(2.0 * np.log(np.diag(chol)).sum()))
    return {"classes": classes, "mu": np.stack(mus), "chol": chols, "logdet": np.array(logdets), "k": X.shape[1]}


def marginal_log_prob(model, feats):
    Z = np.asarray(feats, dtype=np.float64)
    const = model["k"] * np.log(2.0 * np.pi)
    out = np.empty((len(Z), len(model["classes"])))
    for i, chol in enumerate(model["chol"]):
        sol = np.linalg.solve(chol, (Z - model["mu"][i]).T)
        out[:, i] = -0.5 * (const + model["logdet"][i] + np.sum(sol * sol, axis=0))
    return logsumexp(out, axis=1) - np.log(out.shape[1])


def _alpha(logits, phat=None):
    z = np.asarray(logits, dtype=np.float64)
    if phat is None:
        return 1e-6 + np.exp(z)
    p = np.asarray(phat, dtype=np.float64)[:, None]
    return 1e-6 + np.exp(z * p)


def _record(model, loader, device):
    logits, feats, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            z = model(x)
            logits.append(z.detach().cpu().numpy().astype(np.float32))
            feats.append(model.feature.detach().cpu().numpy().astype(np.float32))
            labels.append(y.numpy())
    return {
        "logits": np.concatenate(logits),
        "feats": np.concatenate(feats),
        "labels": np.concatenate(labels),
    }


def extract(args):
    daedl = Path(args.daedl_dir).resolve()
    if not (daedl / "utility.py").exists():
        raise SystemExit(f"DAEDL/utility.py not found at {daedl}. Item 8 needs the local CIFAR checkout.")
    cwd = Path.cwd()
    sys.path.insert(0, str(daedl))
    os.chdir(daedl)
    try:
        if args.np_seed is not None:
            np.random.seed(args.np_seed)
        from utility import load_datasets, vgg16

        train_aug, _val, test_loader, svhn, cifar100 = load_datasets("CIFAR-10", args.batch_size, args.val_size)
        net = vgg16(args.dropout)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            net.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            net.load_state_dict(ckpt)
        else:
            net = ckpt
        net.to(args.device)
        arrays = {
            "train_aug_fit": _record(net, train_aug, args.device),
            "train_aug_score": _record(net, train_aug, args.device),
            "cifar10_test": _record(net, test_loader, args.device),
            "svhn": _record(net, svhn, args.device),
            "cifar100": _record(net, cifar100, args.device),
        }
        if args.np_seed is not None:
            np.random.seed(args.np_seed)
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        clean = datasets.CIFAR10(
            root="./data", train=True, download=False,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        indices = list(range(len(clean)))
        split = int(np.floor(args.val_size * len(clean)))
        np.random.shuffle(indices)
        subset = Subset(clean, indices[split:])
        clean_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
        arrays["train_clean"] = _record(net, clean_loader, args.device)
    finally:
        os.chdir(cwd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cifar_cache.npz", **{f"{k}__{field}": v[field] for k, v in arrays.items() for field in v})
    print(f"wrote {out / 'cifar_cache.npz'}")


def _load(path):
    blob = np.load(path)
    names = sorted({k.split("__", 1)[0] for k in blob.files})
    return {
        name: {field: blob[f"{name}__{field}"] for field in ("logits", "feats", "labels")}
        for name in names
    }


def analyse(args):
    cache = Path(args.out) / "cifar_cache.npz"
    if not cache.exists():
        raise SystemExit(f"missing {cache}; run the extract phase first")
    arrays = _load(cache)
    report = {"fits": {}}
    flips = {}
    for fit_name, fit_key in (("reference", "train_aug_fit"), ("unaugmented", "train_clean")):
        if fit_key not in arrays:
            continue
        model = fit_gda(arrays[fit_key]["feats"], arrays[fit_key]["labels"])
        score_key = "train_aug_score" if fit_name == "reference" and "train_aug_score" in arrays else fit_key
        lp_train = marginal_log_prob(model, arrays[score_key]["feats"])
        block = {
            "train_logp_q05": float(np.quantile(lp_train, 0.05)),
            "train_logp_min": float(lp_train.min()),
            "datasets": {},
        }
        for ood_name in ("svhn", "cifar100"):
            if ood_name not in arrays:
                continue
            lp_id = marginal_log_prob(model, arrays["cifar10_test"]["feats"])
            lp_ood = marginal_log_prob(model, arrays[ood_name]["feats"])
            logits = arrays["cifar10_test"]["logits"]
            ood_logits = arrays[ood_name]["logits"]
            entry = {"raw_density_auroc": auroc_scores(-lp_id, -lp_ood)}
            for norm_name, cls in NORMALIZERS.items():
                norm = cls().fit(lp_train)
                p_id = np.asarray(norm(lp_id), dtype=np.float64)
                p_ood = np.asarray(norm(lp_ood), dtype=np.float64)
                a_id = _alpha(logits)
                a_ood = _alpha(ood_logits)
                a_id_s = _alpha(logits, p_id)
                a_ood_s = _alpha(ood_logits, p_ood)
                def vac(a):
                    return a.shape[-1] / a.sum(-1)
                def ale(a):
                    p = a / a.sum(-1, keepdims=True)
                    return 1.0 - p.max(-1)
                entry[norm_name] = {
                    "epistemic_plain": auroc_scores(vac(a_id), vac(a_ood)),
                    "epistemic_scaled": auroc_scores(vac(a_id_s), vac(a_ood_s)),
                    "aleatoric_plain": auroc_scores(ale(a_id), ale(a_ood)),
                    "aleatoric_scaled": auroc_scores(ale(a_id_s), ale(a_ood_s)),
                    "id_phat_median": float(np.median(p_id)),
                }
                if norm_name == "minmax" and fit_name == "reference":
                    gate = pair_flip_gate(vac(a_id), vac(a_ood), vac(a_id_s), vac(a_ood_s), B=200)
                    entry["minmax_gate"] = {k: gate[k] for k in ("delta_auroc", "flip_rate", "materiality", "benefit", "ci_low", "ci_high")}
                    pred = a_id.argmax(-1)
                    pred_s = a_id_s.argmax(-1)
                    idx = np.flatnonzero(pred != pred_s)
                    flips = {
                        "indices": idx.tolist(),
                        "n": int(len(idx)),
                        "phat": p_id[idx].tolist(),
                        "n_phat_zero": int(np.sum(p_id[idx] == 0)),
                    }
                    q05 = float(np.quantile(lp_train, 0.05))
                    span = float(lp_train.max() - lp_train.min()) + 1e-12
                    block["minmax_bottom5_share"] = float((q05 - lp_train.min()) / span)
            block["datasets"][ood_name] = entry
        if fit_name == "reference":
            block["cifar10_test_phat_median"] = block["datasets"].get("svhn", {}).get("minmax", {}).get("id_phat_median")
        report["fits"][fit_name] = block
    report["flips"] = flips
    out = Path(args.out)
    write_json(out / "cifar_ood_side.json", report)
    lines = ["# CIFAR OOD side", ""]
    ref = report["fits"].get("reference", {})
    lines.append(f"Training log-density 5th percentile: {ref.get('train_logp_q05')}")
    lines.append(f"Flips: {flips.get('n')} (phat = 0: {flips.get('n_phat_zero')})")
    write_md(out / "cifar_ood_side.md", "\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="phase", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--daedl_dir", default="DAEDL")
    ex.add_argument("--ckpt", default="saved_results/model.pt")
    ex.add_argument("--device", default="mps")
    ex.add_argument("--out", default="results/cifar_ood_side")
    ex.add_argument("--batch_size", type=int, default=128)
    ex.add_argument("--val_size", type=float, default=0.05)
    ex.add_argument("--dropout", type=float, default=0.3)
    ex.add_argument("--np_seed", type=int, default=None)
    an = sub.add_parser("analyse")
    an.add_argument("--out", default="results/cifar_ood_side")
    args = p.parse_args()
    if args.phase == "extract":
        extract(args)
    else:
        analyse(args)


if __name__ == "__main__":
    main()
