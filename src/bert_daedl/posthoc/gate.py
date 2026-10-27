"""Pair-flip materiality and the benefit gate on ID–OOS pairs."""

from __future__ import annotations

import numpy as np

from bert_daedl.posthoc.stats import auroc_scores, paired_bootstrap


def pair_credits(id_scores, oos_scores) -> np.ndarray:
    """Credit 1 if the OOS score is higher, 0.5 on a tie, else 0. Shape (n_oos, n_id)."""
    diff = np.asarray(oos_scores, dtype=np.float64)[:, None] - np.asarray(id_scores, dtype=np.float64)[None, :]
    return np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))


def pair_decomposition(s0_id, s0_oos, s1_id, s1_oos) -> dict:
    diff = pair_credits(s1_id, s1_oos) - pair_credits(s0_id, s0_oos)
    fixed = float(np.maximum(diff, 0).mean())
    broken = float(np.maximum(-diff, 0).mean())
    delta = fixed - broken
    flip = fixed + broken
    return {
        "delta_auroc": delta,
        "flip_rate": flip,
        "fixed": fixed,
        "broken": broken,
        "auroc_gap": auroc_scores(s1_id, s1_oos) - auroc_scores(s0_id, s0_oos),
    }


def materiality(flip_rate: float, thresh: float = 0.001) -> str:
    return "INERT" if flip_rate < thresh else "ACTIVE"


def benefit_verdict(ci_low: float, ci_high: float, band: float = 0.001) -> str:
    if ci_low >= -band and ci_high <= band:
        return "NEUTRAL"
    if ci_low > 0:
        return "USEFUL"
    if ci_high < 0:
        return "HARMFUL"
    return "INCONCLUSIVE"


def pair_flip_gate(s0_id, s0_oos, s1_id, s1_oos, *, B: int = 2000, seed: int = 0) -> dict:
    parts = pair_decomposition(s0_id, s0_oos, s1_id, s1_oos)
    boot = paired_bootstrap(
        [np.asarray(s1_id, dtype=np.float64)],
        [np.asarray(s1_oos, dtype=np.float64)],
        [np.asarray(s0_id, dtype=np.float64)],
        [np.asarray(s0_oos, dtype=np.float64)],
        B=B,
        seed=seed,
    )
    return {
        **parts,
        "materiality": materiality(parts["flip_rate"]),
        "benefit": benefit_verdict(boot["ci_low"], boot["ci_high"]),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p": boot["p"],
    }
