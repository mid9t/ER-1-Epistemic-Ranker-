#!/usr/bin/env python
"""Which density model and feature space carry the OOS signal?"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.config import GDAConfig
from bert_daedl.posthoc.cache import load_runs
from bert_daedl.posthoc.cli import add_common_args
from bert_daedl.posthoc.gda_variants import (
    mahalanobis_pp,
    partial_mahalanobis,
    relative_mahalanobis,
    spectrum,
)
from bert_daedl.posthoc.report import format_mean_std, write_csv, write_md
from bert_daedl.posthoc.scoring import (
    PCA_UNSET,
    daedl_alpha,
    density_maps,
    detection,
    fit_density,
    gda_from_config,
    parse_spec,
    pca_from_arg,
    phats_for,
    split_logp,
)
from bert_daedl.uncertainty.scores import scores_from_alpha

PCA_GRID = (("none", None), ("128", 128), ("64", 64))


def _neg_logp_metrics(gda, run, feature):
    logp = split_logp(gda, run["splits"], feature)
    neg = {s: -logp[s] for s in logp}
    val = detection(neg["val"], neg["oos_val"])
    test = detection(neg["test"], neg["oos_test"])
    return logp, {
        "val_auroc": val["auroc"],
        "test_auroc": test["auroc"],
        "test_fpr95": test["fpr_at_95tpr"],
    }


def _coupling(run, logp, spec, feature_unused=None):
    phat = phats_for(logp, spec.normalizer, "train")
    clamp = float((run["meta"].get("config") or {}).get("logit_clamp", 15.0))
    scored = {}
    for split, data in run["splits"].items():
        alpha = daedl_alpha(data["logits"], phat[split], spec.mode, spec.lam, clamp)
        scored[split] = scores_from_alpha(alpha)["vacuity"].detach().cpu().numpy()
    return detection(scored["test"], scored["oos_test"])["auroc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--selected", default="results/posthoc/grid/selected.json")
    p.add_argument("--out", default="results/posthoc/gda")
    add_common_args(p, density=False)
    args = p.parse_args()
    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    spec = parse_spec(selected.get("spec") or selected["config"])
    runs = [
        r for r in load_runs(args.root, oos_subset=args.oos_subset)
        if r["meta"].get("tag", "") == "" and r["meta"].get("arm") in ("ce", "edl")
    ]
    pca_grid = PCA_GRID
    requested = pca_from_arg(args.pca_dim)
    if requested is not PCA_UNSET:
        pca_grid = (("none", None),) if requested is None else ((str(requested), requested),)
    if not runs:
        raise SystemExit(f"no cached CE/EDL runs under {args.root}")

    rows = []
    spectra = []
    eigs = []
    norm_rows = []
    base_cfg = GDAConfig()
    for run in runs:
        arm = run["meta"].get("arm")
        seed = int(run["meta"].get("seed", 0))
        spaces = [("feats", run["splits"])]
        if "pooled" in run["splits"]["train"]:
            spaces.append(("pooled", run["splits"]))
        for space, _splits in spaces:
            spec_row = spectrum(run["splits"]["train"][space], run["splits"]["train"]["labels"])
            spectra.append({
                "arm": arm,
                "seed": seed,
                "space": space,
                **{k: v for k, v in spec_row.items() if k != "eigenvalues"},
            })
            for i, value in enumerate(spec_row["eigenvalues"]):
                eigs.append({"arm": arm, "seed": seed, "space": space, "index": i, "eigenvalue": float(value)})
            for pca_name, pca_dim in pca_grid:
                if pca_dim is not None and pca_dim >= run["splits"]["train"][space].shape[1]:
                    continue
                cfg = GDAConfig(
                    cov_mode="tied",
                    pca_dim=pca_dim,
                    whiten=base_cfg.whiten,
                    jitter_grid=base_cfg.jitter_grid,
                    min_var=base_cfg.min_var,
                )
                gda = fit_density(run["splits"]["train"][space], run["splits"]["train"]["labels"], cfg)
                logp, metrics = _neg_logp_metrics(gda, run, space)
                row = {
                    "arm": arm,
                    "seed": seed,
                    "space": space,
                    "pca_dim": pca_name,
                    "variant": "tied",
                    **metrics,
                }
                if arm == "edl" and space == "feats":
                    row["selected_test_auroc"] = _coupling(run, logp, spec)
                rows.append(row)
                train_x = run["splits"]["train"][space]
                train_y = run["splits"]["train"]["labels"]
                for variant, fn in (
                    ("relative", relative_mahalanobis),
                    ("pp", mahalanobis_pp),
                ):
                    id_s = fn(train_x, train_y, run["splits"]["test"][space], pca_dim, True)
                    oos_s = fn(train_x, train_y, run["splits"]["oos_test"][space], pca_dim, True)
                    val_id = fn(train_x, train_y, run["splits"]["val"][space], pca_dim, True)
                    val_oos = fn(train_x, train_y, run["splits"]["oos_val"][space], pca_dim, True)
                    rows.append({
                        "arm": arm,
                        "seed": seed,
                        "space": space,
                        "pca_dim": pca_name,
                        "variant": variant,
                        "val_auroc": detection(val_id, val_oos)["auroc"],
                        "test_auroc": detection(id_s, oos_s)["auroc"],
                        "test_fpr95": detection(id_s, oos_s)["fpr_at_95tpr"],
                    })
            part_id = partial_mahalanobis(run["splits"]["train"][space], run["splits"]["train"]["labels"], run["splits"]["test"][space])
            part_oos = partial_mahalanobis(run["splits"]["train"][space], run["splits"]["train"]["labels"], run["splits"]["oos_test"][space])
            for which in ("top", "bottom"):
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "space": space,
                    "pca_dim": "within",
                    "variant": f"partial_{which}",
                    "test_auroc": detection(part_id[which], part_oos[which])["auroc"],
                })

        # Normalizer fit source on the default bottleneck GDA.
        cfg = gda_from_config((run["meta"].get("config") or {}).get("gda"))
        _gda, logp = density_maps(run, gda_cfg=cfg)
        quant = {}
        for split, lp in logp.items():
            quant[f"lp_{split}_q50"] = float(np.quantile(lp, 0.50))
            quant[f"lp_{split}_q05"] = float(np.quantile(lp, 0.05))
        row = {"arm": arm, "seed": seed, **quant}
        if arm == "edl":
            for fit_on in ("train", "id_val"):
                phat = phats_for(logp, spec.normalizer, "train" if fit_on == "train" else "id_val")
                clamp = float((run["meta"].get("config") or {}).get("logit_clamp", 15.0))
                scored = {}
                for split, data in run["splits"].items():
                    alpha = daedl_alpha(data["logits"], phat[split], spec.mode, spec.lam, clamp)
                    scored[split] = scores_from_alpha(alpha)["vacuity"].detach().cpu().numpy()
                row[f"{fit_on}_test_auroc"] = detection(scored["test"], scored["oos_test"])["auroc"]
        norm_rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["space"], row["pca_dim"], row["variant"])].append(row)
    lines = ["| arm | space | pca | variant | test AUROC |", "|---|---|---|---|---:|"]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        grp = groups[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
            f"{format_mean_std([r['test_auroc'] for r in grp])} |"
        )
    out = Path(args.out)
    write_csv(out / "gda_ablation_runs.csv", rows)
    write_md(out / "gda_ablation_summary.md", "\n".join(lines) + "\n")
    write_csv(out / "spectrum_summary.csv", spectra)
    write_csv(out / "eigenvalues.csv", eigs)
    write_csv(out / "normalizer_fit.csv", norm_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
