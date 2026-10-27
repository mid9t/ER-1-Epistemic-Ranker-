#!/usr/bin/env python
"""Aggregate results/**/results.json → summary.csv + markdown table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_runs(root: Path):
    runs = []
    for path in sorted(root.glob("**/results.json")):
        with path.open("r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def _group_key(r: dict) -> tuple:
    cfg = r.get("config") or {}
    return (
        r.get("arm", cfg.get("arm")),
        cfg.get("normalizer", ""),
        cfg.get("combine_mode", ""),
        r.get("tag", cfg.get("tag", "")) or "",
    )


def _std(vals, ddof: int) -> float:
    if len(vals) <= ddof:
        return 0.0
    return float(np.std(vals, ddof=ddof))


def aggregate(root: Path, ddof: int = 1) -> tuple[list[dict], str]:
    runs = _load_runs(root)
    if not runs:
        return [], "_No results.json found._\n"

    groups = defaultdict(list)
    for r in runs:
        groups[_group_key(r)].append(r)

    rows = []
    md_lines = [
        "| arm | tag | normalizer | combine | n | in_scope_acc | primary auroc | primary fpr95 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]

    for key in sorted(groups):
        arm, norm, mode, tag = key
        group = groups[key]
        accs = [g["in_scope_acc"] for g in group]
        # pick a representative primary score from thresholded
        primary = group[0]["thresholded"]["score"]
        aurocs, fprs = [], []
        for g in group:
            ps = g["per_score"].get(primary, {})
            if "auroc" in ps:
                aurocs.append(ps["auroc"])
            if "fpr_at_95tpr" in ps:
                fprs.append(ps["fpr_at_95tpr"])

        def fmt(vals):
            if not vals:
                return ""
            return f"{np.mean(vals):.4f}±{_std(vals, ddof):.4f}"

        row = {
            "arm": arm,
            "tag": tag,
            "normalizer": norm,
            "combine_mode": mode,
            "n_seeds": len(group),
            "in_scope_acc_mean": float(np.mean(accs)),
            "in_scope_acc_std": _std(accs, ddof),
            "primary_score": primary,
            "primary_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "primary_auroc_std": _std(aurocs, ddof) if aurocs else None,
            "primary_fpr95_mean": float(np.mean(fprs)) if fprs else None,
            "primary_fpr95_std": _std(fprs, ddof) if fprs else None,
        }
        # flatten per-score aurocs across seeds
        score_names = sorted({s for g in group for s in g["per_score"]})
        for sname in score_names:
            vals = [
                g["per_score"][sname]["auroc"]
                for g in group
                if sname in g["per_score"]
            ]
            if vals:
                row[f"{sname}_auroc_mean"] = float(np.mean(vals))
                row[f"{sname}_auroc_std"] = _std(vals, ddof)
                fvals = [
                    g["per_score"][sname]["fpr_at_95tpr"]
                    for g in group
                    if sname in g["per_score"]
                ]
                row[f"{sname}_fpr95_mean"] = float(np.mean(fvals))
                row[f"{sname}_fpr95_std"] = _std(fvals, ddof)

        rows.append(row)
        md_lines.append(
            f"| {arm} | {tag or '-'} | {norm or '-'} | {mode or '-'} | {len(group)} | "
            f"{fmt(accs)} | {fmt(aurocs)} | {fmt(fprs)} |"
        )

    return rows, "\n".join(md_lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", nargs="?", default="results")
    p.add_argument("--ddof", type=int, default=1, choices=[0, 1])
    args = p.parse_args()
    root = Path(args.results_root)
    rows, md = aggregate(root, ddof=args.ddof)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "summary.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        # put key columns first
        preferred = [
            "arm",
            "tag",
            "normalizer",
            "combine_mode",
            "n_seeds",
            "in_scope_acc_mean",
            "in_scope_acc_std",
            "primary_score",
            "primary_auroc_mean",
            "primary_auroc_std",
            "primary_fpr95_mean",
            "primary_fpr95_std",
        ]
        cols = [c for c in preferred if c in fieldnames] + [
            c for c in fieldnames if c not in preferred
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    md_path = root / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
