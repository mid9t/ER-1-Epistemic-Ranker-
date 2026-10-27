#!/usr/bin/env python
"""Coupling grid: normalizer x combination rule, selected on validation only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.posthoc.cache import load_runs
from bert_daedl.posthoc.cli import add_common_args, density_kwargs
from bert_daedl.posthoc.gate import pair_flip_gate
from bert_daedl.posthoc.report import format_mean_std, sample_std, write_csv, write_json, write_md
from bert_daedl.posthoc.scoring import (
    daedl_alpha,
    logp_from_run,
    detection,
    dirichlet_ece,
    in_scope_accuracy,
    logit_scores,
    operating_point,
    parse_spec,
    phats_for,
)
from bert_daedl.uncertainty.scores import scores_from_alpha

# Pre-registered λ grid (PREREGISTRATION.md). The CLI default stays 1 2 5 10
# so the CLINC150 `make grid` target is unchanged; the replication driver
# passes this full set.
REPLICATION_LAMS = (1, 2, 5, 10, 20, 40, 80, 160)
ECE_MARGIN = 0.01


def _configs(lams):
    cfgs = []
    for norm in ("minmax", "qsigmoid", "ecdf"):
        cfgs.append({"normalizer": norm, "mode": "mul", "lam": None})
        for lam in lams:
            cfgs.append({"normalizer": norm, "mode": "add_log", "lam": float(lam)})
    return cfgs


def _spec(cfg):
    lam = "" if cfg["lam"] is None else f":{cfg['lam']:g}"
    return f"daedl:{cfg['normalizer']}:{cfg['mode']}{lam}/vacuity"


def _eval_config(run, logp, cfg, fit_on, B):
    phat = phats_for(logp, cfg["normalizer"], fit_on=fit_on)
    clamp = float((run["meta"].get("config") or {}).get("logit_clamp", 15.0))
    plain = {}
    scaled = {}
    alphas = {}
    for split, data in run["splits"].items():
        raw = logit_scores(data["logits"], arm="edl", clamp=clamp)
        plain[split] = raw["vacuity"]
        alpha = daedl_alpha(data["logits"], phat[split], cfg["mode"], cfg["lam"], clamp)
        alphas[split] = alpha.detach().cpu().numpy()
        scaled[split] = scores_from_alpha(alpha)["vacuity"].detach().cpu().numpy()
    plain_alpha = {}
    for split, data in run["splits"].items():
        plain_alpha[split] = logits_to_alpha(
            torch.from_numpy(np.asarray(data["logits"], dtype=np.float64)),
            clamp,
        ).detach().cpu().numpy()
    test = run["splits"]["test"]
    labels = test["labels"]
    val_labels = run["splits"]["val"]["labels"]
    pred = alphas["test"].argmax(-1)
    det = detection(scaled["test"], scaled["oos_test"])
    val = detection(scaled["val"], scaled["oos_val"])
    op = operating_point(scaled["val"], scaled["test"], scaled["oos_test"])
    gate_val = pair_flip_gate(plain["val"], plain["oos_val"], scaled["val"], scaled["oos_val"], B=B)
    gate_test = pair_flip_gate(plain["test"], plain["oos_test"], scaled["test"], scaled["oos_test"], B=B)
    return {
        "spec": _spec(cfg),
        "normalizer": cfg["normalizer"],
        "mode": cfg["mode"],
        "lam": cfg["lam"],
        "fit_on": fit_on,
        "val_auroc": val["auroc"],
        "test_auroc": det["auroc"],
        "test_aupr": det["aupr"],
        "test_fpr95": det["fpr_at_95tpr"],
        "oos_recall": op["oos_recall"],
        "config": _spec(cfg),
        "test_acc_scaled": float(np.mean(pred == labels)),
        "test_acc_plain": in_scope_accuracy(test["logits"], labels),
        "val_ece_scaled": dirichlet_ece(alphas["val"], val_labels),
        "val_ece_plain": dirichlet_ece(plain_alpha["val"], val_labels),
        "test_ece_scaled": dirichlet_ece(alphas["test"], labels),
        "test_ece_plain": dirichlet_ece(plain_alpha["test"], labels),
        "val_materiality": gate_val["materiality"],
        "val_benefit": gate_val["benefit"],
        "val_flip_rate": gate_val["flip_rate"],
        "val_delta_auroc": gate_val["delta_auroc"],
        "test_materiality": gate_test["materiality"],
        "test_benefit": gate_test["benefit"],
        "test_delta_auroc": gate_test["delta_auroc"],
        "legacy_id_flip_rate": float(np.mean(pred != np.asarray(test["logits"]).argmax(-1))),
    }


def _on_edge(lams, lam) -> bool:
    if lam is None or not lams:
        return False
    return bool(math.isclose(float(lam), float(min(lams))) or math.isclose(float(lam), float(max(lams))))


def extend_lams(lams, lam) -> list[float]:
    """Two λ values past one edge of the grid. Empty when ``lam`` is interior.

    The high edge grows by 2× and 4×. The low edge steps to half and quarter
    (1 → 0.5 and 0.25).
    """
    if lam is None or not lams:
        return []
    lo, hi = float(min(lams)), float(max(lams))
    if math.isclose(float(lam), hi):
        return [hi * 2, hi * 4]
    if math.isclose(float(lam), lo):
        return [lo / 2, lo / 4]
    return []


def _group(rows):
    by_spec = defaultdict(list)
    for row in rows:
        by_spec[row["spec"]].append(row)
    return by_spec


def _eligible(by_spec) -> list[str]:
    eligible = []
    for spec, group in by_spec.items():
        inert = sum(r["val_materiality"] == "INERT" for r in group)
        if inert >= math.ceil(len(group) / 2):
            continue
        eligible.append(spec)
    return eligible


def _rank(by_spec, pool):
    ranked = []
    for spec in pool:
        group = by_spec[spec]
        ranked.append((float(np.mean([r["val_auroc"] for r in group])), spec, group))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked


def _pack(ranked, lams, rule: str) -> dict:
    best_mean, spec, group = ranked[0]
    sample = group[0]
    edge = sample["mode"] == "add_log" and _on_edge(lams, sample["lam"])
    return {
        "rule": rule,
        "spec": spec,
        "config": spec,
        "normalizer": sample["normalizer"],
        "combine_mode": sample["mode"],
        "lam": sample["lam"],
        "val_auroc_mean": best_mean,
        "val_auroc_std": sample_std([r["val_auroc"] for r in group]),
        "test_auroc_mean": float(np.mean([r["test_auroc"] for r in group])),
        "test_auroc_std": sample_std([r["test_auroc"] for r in group]),
        "test_delta_vs_vacuity": float(np.mean([r["test_delta_auroc"] for r in group])),
        "lam_on_grid_edge": edge,
        "gain_is_lower_bound": False,
        "n_seeds": len(group),
        "runners_up": [
            {"spec": s, "val_auroc_mean": m} for m, s, _g in ranked[1:4]
        ],
    }


def _select(rows, lams):
    by_spec = _group(rows)
    eligible = _eligible(by_spec)
    pool = eligible or list(by_spec)
    return _pack(_rank(by_spec, pool), lams, "primary")


def _select_secondary(rows, lams):
    """Primary rule, after dropping couplings whose validation ECE rises by > 0.01.

    add_log leaves the predictive mean unchanged, so this filter removes
    multiplicative couplings that move ECE and leaves an add_log pick.
    """
    by_spec = _group(rows)
    eligible = _eligible(by_spec)
    pool = []
    for spec in eligible:
        group = by_spec[spec]
        scaled = float(np.mean([r["val_ece_scaled"] for r in group]))
        plain = float(np.mean([r["val_ece_plain"] for r in group]))
        if scaled > plain + ECE_MARGIN:
            continue
        pool.append(spec)
    emptied = not pool
    if not pool:
        pool = [
            spec for spec in (eligible or list(by_spec))
            if by_spec[spec][0]["mode"] == "add_log"
        ] or list(by_spec)
    packed = _pack(_rank(by_spec, pool), lams, "secondary")
    packed["ece_filter_emptied"] = emptied
    return packed


def _summary(rows):
    by_spec = defaultdict(list)
    for row in rows:
        by_spec[row["spec"]].append(row)
    summary = []
    for spec, group in by_spec.items():
        summary.append({
            "spec": spec,
            "normalizer": group[0]["normalizer"],
            "mode": group[0]["mode"],
            "lam": group[0]["lam"],
            "fit_on": group[0]["fit_on"],
            "n": len(group),
            "val_auroc_mean": float(np.mean([r["val_auroc"] for r in group])),
            "val_auroc_std": sample_std([r["val_auroc"] for r in group]),
            "test_auroc_mean": float(np.mean([r["test_auroc"] for r in group])),
            "test_auroc_std": sample_std([r["test_auroc"] for r in group]),
            "test_delta_vs_vacuity": float(np.mean([r["test_delta_auroc"] for r in group])),
            "test_acc_scaled": float(np.mean([r["test_acc_scaled"] for r in group])),
            "test_ece_scaled": float(np.mean([r["test_ece_scaled"] for r in group])),
        })
    summary.sort(key=lambda r: r["val_auroc_mean"], reverse=True)
    lines = [
        "| spec | val AUROC | test AUROC | Δ vs vacuity |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['spec']} | {format_mean_std([row['val_auroc_mean']])} | "
            f"{row['test_auroc_mean']:.4f} ± {row['test_auroc_std']:.4f} | "
            f"{row['test_delta_vs_vacuity']:+.4f} |"
        )
    return summary, "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--out", default="results/posthoc/grid")
    p.add_argument("--tag", default="")
    p.add_argument("--lams", type=float, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--fit_on", nargs="+", default=["train"])
    p.add_argument("--apply-selected", default=None)
    p.add_argument("--B", type=int, default=400)
    p.add_argument("--feature", default="feats", choices=["feats", "pooled"])
    p.add_argument("--variant", default="config", choices=["config", "tied", "lowvar"])
    p.add_argument(
        "--extend-edge",
        action="store_true",
        help="If the selected add_log λ sits on the grid edge, extend once by two values and reselect.",
    )
    add_common_args(p, density=False)
    args = p.parse_args()

    runs = load_runs(args.root, arm="edl", tag=args.tag, oos_subset=args.oos_subset)
    if not runs:
        raise SystemExit(f"no EDL caches under {args.root} tag={args.tag!r}")
    lams = [float(lam) for lam in args.lams]
    if args.apply_selected:
        chosen = json.loads(Path(args.apply_selected).read_text(encoding="utf-8"))
        spec = parse_spec(chosen.get("spec") or chosen["config"])
        cfgs = [{"normalizer": spec.normalizer, "mode": spec.mode, "lam": spec.lam}]
    else:
        cfgs = _configs(lams)

    def _rows_for(configs):
        built = []
        for fit_on in args.fit_on:
            for run in runs:
                logp = logp_from_run(run, **density_kwargs(args))
                for cfg in configs:
                    row = _eval_config(run, logp, cfg, fit_on, args.B)
                    row["seed"] = int(run["meta"].get("seed", 0))
                    row["tag"] = run["meta"].get("tag", "")
                    row["feature"] = args.feature
                    row["variant"] = args.variant
                    row["oos_subset"] = args.oos_subset
                    built.append(row)
        return built

    rows = _rows_for(cfgs)
    out = Path(args.out)
    select_rows = [r for r in rows if r["fit_on"] == args.fit_on[0]]
    do_select = args.oos_subset == "all" and not args.apply_selected
    selected = secondary = None
    if do_select:
        selected = _select(select_rows, lams)
        secondary = _select_secondary(select_rows, lams)
        if args.extend_edge:
            extra = []
            for pick in (selected, secondary):
                if pick["combine_mode"] == "add_log":
                    extra.extend(extend_lams(lams, pick["lam"]))
            extra = [lam for lam in extra if not any(math.isclose(lam, have) for have in lams)]
            # Keep the first occurrence of each new λ.
            deduped = []
            for lam in extra:
                if not any(math.isclose(lam, seen) for seen in deduped):
                    deduped.append(lam)
            if deduped:
                extra_cfgs = []
                for norm in ("minmax", "qsigmoid", "ecdf"):
                    for lam in deduped:
                        extra_cfgs.append({"normalizer": norm, "mode": "add_log", "lam": float(lam)})
                rows.extend(_rows_for(extra_cfgs))
                lams = lams + deduped
                select_rows = [r for r in rows if r["fit_on"] == args.fit_on[0]]
                selected = _select(select_rows, lams)
                secondary = _select_secondary(select_rows, lams)
                selected["extended_lams"] = deduped
                secondary["extended_lams"] = deduped
            selected["gain_is_lower_bound"] = bool(selected["lam_on_grid_edge"])
            secondary["gain_is_lower_bound"] = bool(secondary["lam_on_grid_edge"])

    write_csv(out / "grid_runs.csv", rows)
    if args.apply_selected:
        write_json(out / "applied.json", {"spec": rows[0]["spec"] if rows else None, "n": len(rows)})
        print(f"applied {rows[0]['spec'] if rows else '?'} to {len(rows)} runs -> {out}")
        return
    summary, md = _summary(rows)
    write_csv(out / "grid_summary.csv", summary)
    write_md(out / "grid_summary.md", md)
    if do_select:
        write_json(out / "selected.json", selected)
        write_json(out / "selected_secondary.json", secondary)
        print(json.dumps({"primary": selected, "secondary": secondary}, indent=2))
    else:
        print(f"oos_subset={args.oos_subset}: grid written, selection left to the pooled run")
    print(md)


if __name__ == "__main__":
    main()
