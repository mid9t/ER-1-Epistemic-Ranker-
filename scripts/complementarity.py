#!/usr/bin/env python
"""Do the logit score and the density carry different information?"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.posthoc.cache import load_runs
from bert_daedl.posthoc.cli import add_common_args, density_kwargs
from bert_daedl.posthoc.fusion import add_log_sweep, linear_fusion, logistic_fusion, spearman_blocks
from bert_daedl.posthoc.report import format_mean_std, write_csv, write_md
from bert_daedl.posthoc.scoring import logp_from_run, logit_scores

# Same grid the coupling search pre-registers. --lam_max keeps every value
# at or below the cap (160 in the replication).
REPLICATION_LAMS = (1, 2, 5, 10, 20, 40, 80, 160)


def _energy(run, kind):
    clamp = float((run["meta"].get("config") or {}).get("logit_clamp", 15.0))
    arm = "edl" if kind == "energy_clip" else "ce"
    out = {}
    for split, data in run["splits"].items():
        scores = logit_scores(data["logits"], arm=arm, clamp=clamp)
        out[split] = scores[kind]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--out", default="results/posthoc/complementarity")
    p.add_argument("--feature", default="feats", choices=["feats", "pooled"])
    p.add_argument("--variant", default="config", choices=["config", "tied", "lowvar"])
    p.add_argument("--lam_max", type=float, default=None,
                   help="Sweep the pre-registered λ grid up to this value and flag an edge pick.")
    add_common_args(p, density=False)
    args = p.parse_args()
    runs = [
        r for r in load_runs(args.root, oos_subset=args.oos_subset)
        if r["meta"].get("tag", "") == ""
    ]
    sweep_lams = None
    if args.lam_max is not None:
        sweep_lams = [lam for lam in REPLICATION_LAMS if lam <= args.lam_max + 1e-9]
    rows = []
    for run in runs:
        arm = run["meta"].get("arm")
        if arm not in ("ce", "edl"):
            continue
        logp = logp_from_run(run, **density_kwargs(args))
        energy_name = "energy_clip" if arm == "edl" else "energy"
        energy = _energy(run, energy_name)
        neg = {split: -logp[split] for split in logp}
        e_id = {"val": energy["val"], "test": energy["test"]}
        e_oos = {"val": energy["oos_val"], "test": energy["oos_test"]}
        d_id = {"val": neg["val"], "test": neg["test"]}
        d_oos = {"val": neg["oos_val"], "test": neg["oos_test"]}
        row = {
            "arm": arm,
            "seed": int(run["meta"].get("seed", 0)),
            "energy": energy_name,
        }
        row.update(spearman_blocks(e_id, e_oos, d_id, d_oos))
        row.update(linear_fusion(e_id, e_oos, d_id, d_oos))
        row.update(logistic_fusion(e_id, e_oos, d_id, d_oos))
        per_norm = {
            name: {split: logp[split] for split in logp} for name in ("minmax", "qsigmoid", "ecdf")
        }
        # add_log_sweep expects logp[name] to be the full split map
        row.update(add_log_sweep(e_id, e_oos, per_norm, lams=sweep_lams))
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"]].append(row)
    numeric = [k for k in rows[0] if k not in ("arm", "seed", "energy") and isinstance(rows[0][k], float)] if rows else []
    lines = ["| arm | metric | mean ± std |", "|---|---|---:|"]
    for arm, group in sorted(groups.items()):
        for key in numeric:
            lines.append(f"| {arm} | {key} | {format_mean_std([g[key] for g in group])} |")
    out = Path(args.out)
    write_csv(out / "complementarity_runs.csv", rows)
    text = "\n".join(lines) + "\n"
    write_md(out / "complementarity_summary.md", text)
    print(text)


if __name__ == "__main__":
    main()
