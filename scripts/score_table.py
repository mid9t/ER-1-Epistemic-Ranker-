#!/usr/bin/env python
"""Score-matched table: every arm gets every score it supports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.posthoc.cache import load_runs
from bert_daedl.posthoc.cli import add_common_args, density_kwargs
from bert_daedl.posthoc.report import format_mean_std, write_csv, write_json, write_md
from bert_daedl.posthoc.scoring import (
    CE_SCORES,
    EDL_SCORES,
    EXACT_VERIFY,
    VACUITY_FAMILY,
    clamp_share,
    detection,
    in_scope_accuracy,
    logit_scores,
    logp_from_run,
    operating_point,
    softmax_ece,
    spearman,
)


def _scores_for_run(run, density):
    arm = run["meta"].get("arm", "edl")
    cfg = run["meta"].get("config") or {}
    clamp = float(cfg.get("logit_clamp", 15.0))
    names = CE_SCORES if arm == "ce" else EDL_SCORES
    logp = logp_from_run(run, **density)
    per_split = {}
    for split, data in run["splits"].items():
        raw = logit_scores(data["logits"], arm=arm, clamp=clamp, eps=float(cfg.get("alpha_eps", 1e-6)))
        per_split[split] = {name: np.asarray(raw[name], dtype=np.float64) for name in names if name != "neg_logp"}
        per_split[split]["neg_logp"] = -np.asarray(logp[split], dtype=np.float64)
    return per_split


def _metrics(scores):
    det = detection(scores["test"], scores["oos_test"])
    val = detection(scores["val"], scores["oos_val"])
    op = operating_point(scores["val"], scores["test"], scores["oos_test"])
    return {
        "val_auroc": val["auroc"],
        "test_auroc": det["auroc"],
        "test_aupr": det["aupr"],
        "test_fpr95": det["fpr_at_95tpr"],
        "oos_recall": op["oos_recall"],
        "id_reject": op["id_reject_rate"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--out", default="results/posthoc/scores")
    p.add_argument("--tag", default=None)
    p.add_argument("--verify", action="store_true")
    add_common_args(p, density=True)
    args = p.parse_args()

    runs = load_runs(args.root, tag=args.tag, oos_subset=args.oos_subset)
    density = density_kwargs(args)
    if not runs:
        raise SystemExit(f"no cached runs under {args.root}")

    run_rows = []
    score_rows = []
    checks = []
    for run in runs:
        meta = run["meta"]
        arm = meta.get("arm")
        seed = int(meta.get("seed", 0))
        tag = meta.get("tag", "")
        test = run["splits"]["test"]
        cfg = meta.get("config") or {}
        clamp = float(cfg.get("logit_clamp", 15.0))
        scored = _scores_for_run(run, density)
        run_rows.append({
            "arm": arm,
            "seed": seed,
            "tag": tag,
            "in_scope_acc": in_scope_accuracy(test["logits"], test["labels"]),
            "ece": softmax_ece(test["logits"], test["labels"]),
            "clamp_share": clamp_share(test["logits"], clamp),
        })
        family = {}
        for name, by_split in _iter_scores(scored):
            metrics = _metrics(by_split)
            family[name] = metrics["test_auroc"]
            score_rows.append({"arm": arm, "seed": seed, "tag": tag, "score": name, **metrics})
        check = {"arm": arm, "seed": seed, "tag": tag}
        if arm != "ce" and "vacuity" in scored["test"]:
            rho = spearman(scored["test"]["vacuity"], scored["test"]["energy_clip"])
            gaps = [family[n] for n in VACUITY_FAMILY if n in family]
            check["spearman_vacuity_energy_clip"] = rho
            check["max_auroc_gap_vacuity_family"] = float(max(gaps) - min(gaps)) if gaps else None
        if args.verify:
            result_path = Path(run["path"]) / "results.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8")).get("per_score", {})
                gaps = []
                for name in EXACT_VERIFY:
                    if name in saved and name in family:
                        gaps.append(abs(family[name] - float(saved[name]["auroc"])))
                check["verify_max_gap_exact_scores"] = float(max(gaps)) if gaps else None
        checks.append(check)

    summary = _summarize(score_rows)
    out = Path(args.out)
    write_csv(out / "runs.csv", run_rows)
    write_csv(out / "score_runs.csv", score_rows)
    write_csv(out / "score_table.csv", summary["rows"])
    write_md(out / "score_table.md", summary["markdown"])
    payload = {"runs": checks}
    exact = [c["verify_max_gap_exact_scores"] for c in checks if c.get("verify_max_gap_exact_scores") is not None]
    rhos = [c["spearman_vacuity_energy_clip"] for c in checks if "spearman_vacuity_energy_clip" in c]
    fam = [c["max_auroc_gap_vacuity_family"] for c in checks if c.get("max_auroc_gap_vacuity_family") is not None]
    if exact:
        payload["verify_max_gap_exact_scores"] = float(max(exact))
    if rhos:
        payload["spearman_vacuity_energy_clip_min"] = float(min(rhos))
    if fam:
        payload["max_auroc_gap_vacuity_family"] = float(max(fam))
    write_json(out / "checks.json", payload)
    print(summary["markdown"])


def _iter_scores(scored):
    names = list(next(iter(scored.values())))
    for name in names:
        yield name, {split: scored[split][name] for split in scored}


def _summarize(score_rows):
    groups = defaultdict(list)
    for row in score_rows:
        groups[(row["score"], row["arm"])].append(row)
    scores = []
    for score, _arm in groups:
        if score not in scores:
            scores.append(score)
    arms = []
    for _score, arm in groups:
        if arm not in arms:
            arms.append(arm)
    lines = ["| score | " + " | ".join(f"{a} AUROC | {a} FPR@95" for a in arms) + " |",
             "|---|" + "|".join("---|" for _ in arms) + "|".join("---|" for _ in arms)]
    # header was wrong. rebuild.
    header = ["score"]
    for arm in arms:
        header += [f"{arm} AUROC", f"{arm} FPR@95"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    rows = []
    for score in scores:
        cells = [score]
        row = {"score": score}
        for arm in arms:
            grp = groups.get((score, arm), [])
            if not grp:
                cells += ["—", "—"]
                continue
            cells.append(format_mean_std([g["test_auroc"] for g in grp]))
            cells.append(format_mean_std([g["test_fpr95"] for g in grp]))
            row[f"{arm}_auroc"] = float(np.mean([g["test_auroc"] for g in grp]))
            row[f"{arm}_fpr95"] = float(np.mean([g["test_fpr95"] for g in grp]))
        rows.append(row)
        lines.append("| " + " | ".join(cells) + " |")
    return {"rows": rows, "markdown": "\n".join(lines) + "\n"}


if __name__ == "__main__":
    main()
