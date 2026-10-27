#!/usr/bin/env python
"""Paired stratified bootstrap for headline AUROC differences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.posthoc.cache import load_runs
from bert_daedl.posthoc.cli import add_common_args, density_kwargs
from bert_daedl.posthoc.report import sample_std, write_csv, write_md
from bert_daedl.posthoc.scoring import PCA_UNSET, logp_from_run, parse_spec, phats_for, score_array
from bert_daedl.posthoc.stats import (
    auroc_scores,
    bootstrap_fpr_delta,
    combined_interval,
    fpr_at_95,
    paired_bootstrap,
    paired_bootstrap_did,
)


def _prepare(runs, feature="feats", variant="config", pca_dim=PCA_UNSET):
    key = (feature, variant, None if pca_dim is PCA_UNSET else pca_dim)
    for run in runs:
        if run.get("_prep_key") == key:
            continue
        run["_prep"] = {
            "logp": logp_from_run(run, feature=feature, variant=variant, pca_dim=pca_dim),
            "phat": {},
        }
        run["_prep_key"] = key
    return runs


def _scores(run, spec_text):
    spec = parse_spec(spec_text)
    prep = run["_prep"]
    phat = None
    if spec.kind == "daedl":
        key = (spec.normalizer, "train")
        if key not in prep["phat"]:
            prep["phat"][key] = phats_for(prep["logp"], spec.normalizer, "train")
        phat = prep["phat"][key]
    scored = score_array(run, spec, logp=prep["logp"], phat=phat)
    return scored["test"], scored["oos_test"]


def _by_seed(runs, spec_text, seeds=None):
    chosen = []
    for run in runs:
        seed = int(run["meta"].get("seed", 0))
        if seeds is not None and seed not in seeds:
            continue
        id_s, oos_s = _scores(run, spec_text)
        chosen.append((seed, id_s, oos_s))
    chosen.sort(key=lambda t: t[0])
    return chosen


def _align(a, b):
    bmap = {seed: (id_s, oos_s) for seed, id_s, oos_s in b}
    seeds, id_a, oos_a, id_b, oos_b = [], [], [], [], []
    for seed, id_s, oos_s in a:
        if seed not in bmap:
            continue
        seeds.append(seed)
        id_a.append(id_s)
        oos_a.append(oos_s)
        id_b.append(bmap[seed][0])
        oos_b.append(bmap[seed][1])
    return seeds, id_a, oos_a, id_b, oos_b


def _compare(name, runs_a, spec_a, runs_b, spec_b, args, seeds=None):
    density = density_kwargs(args)
    _prepare(runs_a, **density)
    if runs_b is not runs_a:
        _prepare(runs_b, **density)
    left = _by_seed(runs_a, spec_a, seeds)
    right = _by_seed(runs_b, spec_b, seeds)
    seeds, id_a, oos_a, id_b, oos_b = _align(left, right)
    if not seeds:
        raise SystemExit(f"no paired seeds for {name}")
    boot = paired_bootstrap(
        id_a, oos_a, id_b, oos_b,
        B=args.B, seed=0, hierarchical=args.hierarchical,
    )
    per_seed = [
        float(auroc_scores(id_a[i], oos_a[i]) - auroc_scores(id_b[i], oos_b[i]))
        for i in range(len(seeds))
    ]
    row = {
        "comparison": name,
        "score_a": spec_a,
        "score_b": spec_b,
        "seeds": " ".join(str(s) for s in seeds),
        "n_seeds": len(seeds),
        "auroc_a": float(sum(auroc_scores(id_a[i], oos_a[i]) for i in range(len(seeds))) / len(seeds)),
        "auroc_b": float(sum(auroc_scores(id_b[i], oos_b[i]) for i in range(len(seeds))) / len(seeds)),
        "delta": boot["delta"],
        "per_seed_delta": " ".join(f"{d:.6f}" for d in per_seed),
        "per_seed_delta_std": sample_std(per_seed),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "se": boot["se"],
        "se_boot": boot["se"],
        "p": boot["p"],
    }
    combined = combined_interval(boot["delta"], boot["se"], per_seed)
    row["se_combined"] = combined["se_combined"]
    row["combined_ci95_low"] = combined["ci95_low"]
    row["combined_ci95_high"] = combined["ci95_high"]
    row["combined_ci90_low"] = combined["ci90_low"]
    row["combined_ci90_high"] = combined["ci90_high"]
    if args.fpr:
        fpr = bootstrap_fpr_delta(id_a, oos_a, id_b, oos_b, B=min(args.B, 1000), seed=1)
        row["fpr95_a"] = float(sum(fpr_at_95(id_a[i], oos_a[i]) for i in range(len(seeds))) / len(seeds))
        row["fpr95_b"] = float(sum(fpr_at_95(id_b[i], oos_b[i]) for i in range(len(seeds))) / len(seeds))
        row["fpr95_delta"] = fpr["delta"]
        row["fpr95_ci_low"] = fpr["ci_low"]
        row["fpr95_ci_high"] = fpr["ci_high"]
    return row


def _load_arm(args, arm):
    return load_runs(args.root, arm=arm, tag="", oos_subset=args.oos_subset)


def _item5(args):
    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    spec = selected.get("spec") or selected["config"]
    edl = _load_arm(args, "edl")
    ce = _load_arm(args, "ce")
    return [
        ("selected − vacuity", edl, spec, edl, "vacuity"),
        ("readme qsigmoid×mul − vacuity", edl, "daedl:qsigmoid:mul/vacuity", edl, "vacuity"),
        ("selected − EDL neg_logp", edl, spec, edl, "neg_logp"),
        ("CE energy − EDL vacuity", ce, "energy", edl, "vacuity"),
        ("EDL vacuity − CE MSP", edl, "vacuity", ce, "msp"),
        ("EDL vacuity − EDL neg_logp", edl, "vacuity", edl, "neg_logp"),
        ("CE neg_logp − CE MSP", ce, "neg_logp", ce, "msp"),
    ]


def _replication(args):
    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    spec = selected.get("spec") or selected["config"]
    edl = _load_arm(args, "edl")
    ce = _load_arm(args, "ce")
    return [
        ("H1: CE energy − EDL vacuity", ce, "energy", edl, "vacuity", None),
        ("H2: qsigmoid×mul − vacuity", edl, "daedl:qsigmoid:mul/vacuity", edl, "vacuity", None),
        ("Positive: primary pick − vacuity", edl, spec, edl, "vacuity", None),
        ("Positive: primary pick − density", edl, spec, edl, "neg_logp", None),
    ]


def _did_row(args):
    path = args.selected_secondary or args.selected
    chosen = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = chosen.get("spec") or chosen["config"]
    near = load_runs(args.root, arm="edl", tag="", oos_subset="id")
    far = load_runs(args.root, arm="edl", tag="", oos_subset="ood")
    density = density_kwargs(args)
    _prepare(near, **density)
    _prepare(far, **density)
    near_a = _by_seed(near, spec)
    near_b = _by_seed(near, "vacuity")
    far_a = _by_seed(far, spec)
    far_b = _by_seed(far, "vacuity")
    seeds, id_a_n, oos_a_n, id_b_n, oos_b_n = _align(near_a, near_b)
    seeds_f, id_a_f, oos_a_f, id_b_f, oos_b_f = _align(far_a, far_b)
    if seeds != seeds_f:
        raise SystemExit(f"ID-OOS seeds {seeds} do not match OOD-OOS seeds {seeds_f}")
    boot = paired_bootstrap_did(
        id_a_n, oos_a_n, id_b_n, oos_b_n,
        id_a_f, oos_a_f, id_b_f, oos_b_f,
        B=args.B, seed=0,
    )
    combined = combined_interval(boot["delta"], boot["se"], boot["per_seed_delta"])
    return {
        "comparison": "H3: DiD (ID-OOS − OOD-OOS)",
        "score_a": spec,
        "score_b": "vacuity",
        "seeds": " ".join(str(s) for s in seeds),
        "n_seeds": len(seeds),
        "a": boot["a"],
        "b": boot["b"],
        "delta": boot["delta"],
        "per_seed_a": " ".join(f"{d:.6f}" for d in boot["per_seed_a"]),
        "per_seed_b": " ".join(f"{d:.6f}" for d in boot["per_seed_b"]),
        "per_seed_delta": " ".join(f"{d:.6f}" for d in boot["per_seed_delta"]),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "se": boot["se"],
        "se_boot": boot["se"],
        "se_combined": combined["se_combined"],
        "combined_ci95_low": combined["ci95_low"],
        "combined_ci95_high": combined["ci95_high"],
        "combined_ci90_low": combined["ci90_low"],
        "combined_ci90_high": combined["ci90_high"],
    }


def _item7(args):
    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    spec = selected.get("spec") or selected["config"]
    main = load_runs(args.root, arm="edl", tag="", oos_subset=args.oos_subset)
    nosn = load_runs(args.ablation_root, arm="edl", tag="nosn")
    mse = load_runs(args.ablation_root, arm="edl", tag="mse")
    seeds = {0, 1, 2}
    rows_spec = []
    for tag, runs in (("nosn", nosn), ("mse", mse)):
        for score in ("vacuity", "neg_logp", spec):
            rows_spec.append((f"{tag} {score} − main {score}", runs, score, main, score, seeds))
    return rows_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["item5", "item7", "replication"], default=None)
    p.add_argument("--root", default="results")
    p.add_argument("--ablation-root", default="results/ablations")
    p.add_argument("--selected", default="results/posthoc/grid/selected.json")
    p.add_argument("--out", required=True)
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--fpr", action="store_true")
    p.add_argument("--hierarchical", action="store_true")
    p.add_argument("--cmp", nargs=5, action="append", default=[], metavar=("NAME", "RUN_A", "SCORE_A", "RUN_B", "SCORE_B"))
    p.add_argument("--feature", default="feats", choices=["feats", "pooled"])
    p.add_argument("--variant", default="config", choices=["config", "tied", "lowvar"])
    p.add_argument("--did", action="store_true",
                   help="H3 difference-in-differences on ID-OOS versus OOD-OOS.")
    p.add_argument("--selected-secondary", default=None,
                   help="Secondary-rule pick used as score A for --did.")
    add_common_args(p, density=False)
    args = p.parse_args()

    if args.did:
        row = _did_row(args)
        out = Path(args.out)
        write_csv(out, [row])
        lines = [
            "| comparison | a (ID-OOS) | b (OOD-OOS) | DiD | combined 95% CI |",
            "|---|---:|---:|---:|---|",
            f"| {row['comparison']} | {row['a']:+.4f} | {row['b']:+.4f} | {row['delta']:+.4f} | "
            f"[{row['combined_ci95_low']:+.4f}, {row['combined_ci95_high']:+.4f}] |",
        ]
        write_md(out.with_suffix(".md"), "\n".join(lines) + "\n")
        print("\n".join(lines))
        if args.preset is None and not args.cmp:
            return

    jobs = []
    if args.preset == "item5":
        jobs.extend((name, a, sa, b, sb, None) for name, a, sa, b, sb in _item5(args))
    elif args.preset == "item7":
        jobs.extend(_item7(args))
    elif args.preset == "replication":
        jobs.extend(_replication(args))
    for name, run_a, score_a, run_b, score_b in args.cmp:
        jobs.append((name, _resolve(run_a, args.oos_subset), score_a, _resolve(run_b, args.oos_subset), score_b, None))
    if not jobs:
        raise SystemExit("pass --preset or --cmp")

    rows = []
    for name, runs_a, spec_a, runs_b, spec_b, seeds in jobs:
        rows.append(_compare(name, runs_a, spec_a, runs_b, spec_b, args, seeds))
    out = Path(args.out)
    write_csv(out, rows)
    if args.preset == "replication":
        lines = [
            "| comparison | ΔAUROC | combined 95% CI | combined 90% CI | seeds |",
            "|---|---:|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['comparison']} | {row['delta']:+.4f} | "
                f"[{row['combined_ci95_low']:+.4f}, {row['combined_ci95_high']:+.4f}] | "
                f"[{row['combined_ci90_low']:+.4f}, {row['combined_ci90_high']:+.4f}] | {row['seeds']} |"
            )
    else:
        lines = ["| comparison | ΔAUROC | 95% CI | seeds |", "|---|---:|---|---|"]
        for row in rows:
            lines.append(
                f"| {row['comparison']} | {row['delta']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['seeds']} |"
            )
    md_path = out.with_suffix(".md")
    write_md(md_path, "\n".join(lines) + "\n")
    print("\n".join(lines))


def _resolve(token: str, oos_subset: str = "all"):
    # ROOT:ARM[:TAG] or a path
    if ":" not in token:
        return load_runs(token, oos_subset=oos_subset)
    root, rest = token.split(":", 1)
    parts = rest.split(":")
    arm = parts[0]
    tag = parts[1] if len(parts) > 1 else ""
    return load_runs(root, arm=arm, tag=tag, oos_subset=oos_subset)


if __name__ == "__main__":
    main()
