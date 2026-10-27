"""AUROC, FPR@95, and a paired stratified bootstrap."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def auroc_scores(id_scores, oos_scores) -> float:
    id_scores = np.asarray(id_scores, dtype=np.float64)
    oos_scores = np.asarray(oos_scores, dtype=np.float64)
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(oos_scores))])
    s = np.concatenate([id_scores, oos_scores])
    return float(roc_auc_score(y, s))


def fpr_at_95(id_scores, oos_scores) -> float:
    id_scores = np.asarray(id_scores, dtype=np.float64)
    oos_scores = np.asarray(oos_scores, dtype=np.float64)
    thr = np.quantile(oos_scores, 0.05)
    return float(np.mean(id_scores >= thr))


class WeightedAUROC:
    """AUROC from multiplicity weights on scores sorted once.

    Matches ``roc_auc_score`` on the arrays expanded by those integer weights,
    including ties (half credit).
    """

    def __init__(self, id_scores, oos_scores):
        id_scores = np.asarray(id_scores, dtype=np.float64)
        oos_scores = np.asarray(oos_scores, dtype=np.float64)
        self.n_id = len(id_scores)
        scores = np.concatenate([id_scores, oos_scores])
        labels = np.concatenate([
            np.zeros(self.n_id, dtype=np.int8),
            np.ones(len(oos_scores), dtype=np.int8),
        ])
        order = np.argsort(scores, kind="mergesort")
        self.scores_sorted = scores[order]
        self.labels_sorted = labels[order]
        self.order = order

    def auroc(self, w_id, w_oos) -> float:
        weights = np.concatenate([
            np.asarray(w_id, dtype=np.float64),
            np.asarray(w_oos, dtype=np.float64),
        ])
        w = weights[self.order]
        labels = self.labels_sorted
        scores = self.scores_sorted
        total_pos = float(np.asarray(w_oos, dtype=np.float64).sum())
        total_neg = float(np.asarray(w_id, dtype=np.float64).sum())
        if total_pos <= 0 or total_neg <= 0:
            return float("nan")
        numer = 0.0
        neg_below = 0.0
        i = 0
        n = len(scores)
        while i < n:
            j = i + 1
            while j < n and scores[j] == scores[i]:
                j += 1
            w_pos = float(w[i:j][labels[i:j] == 1].sum())
            w_neg = float(w[i:j][labels[i:j] == 0].sum())
            numer += w_pos * (neg_below + 0.5 * w_neg)
            neg_below += w_neg
            i = j
        return numer / (total_pos * total_neg)


def paired_bootstrap(
    id_a,
    oos_a,
    id_b,
    oos_b,
    *,
    B: int = 5000,
    seed: int = 0,
    hierarchical: bool = False,
    alpha: float = 0.05,
) -> dict:
    """Seed-mean of AUROC(A) − AUROC(B), with one shared utterance draw."""
    n_seeds = len(id_a)
    n_id = len(id_a[0])
    n_oos = len(oos_a[0])
    eng_a = [WeightedAUROC(id_a[i], oos_a[i]) for i in range(n_seeds)]
    eng_b = [WeightedAUROC(id_b[i], oos_b[i]) for i in range(n_seeds)]
    observed = float(np.mean([
        auroc_scores(id_a[i], oos_a[i]) - auroc_scores(id_b[i], oos_b[i])
        for i in range(n_seeds)
    ]))
    rng = np.random.default_rng(seed)
    reps = np.empty(B, dtype=np.float64)
    for b in range(B):
        w_id = np.bincount(rng.integers(0, n_id, size=n_id), minlength=n_id).astype(np.float64)
        w_oos = np.bincount(rng.integers(0, n_oos, size=n_oos), minlength=n_oos).astype(np.float64)
        seed_idx = rng.integers(0, n_seeds, size=n_seeds) if hierarchical else np.arange(n_seeds)
        reps[b] = np.mean([
            eng_a[i].auroc(w_id, w_oos) - eng_b[i].auroc(w_id, w_oos) for i in seed_idx
        ])
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    p_le = float(np.mean(reps <= 0))
    p_ge = float(np.mean(reps >= 0))
    p = max(2 * min(p_le, p_ge), 1.0 / B)
    return {
        "delta": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(np.std(reps, ddof=1)) if B > 1 else 0.0,
        "p": float(p),
    }


def bootstrap_fpr_delta(id_a, oos_a, id_b, oos_b, *, B: int = 1000, seed: int = 0) -> dict:
    n_seeds = len(id_a)
    n_id = len(id_a[0])
    n_oos = len(oos_a[0])
    observed = float(np.mean([
        fpr_at_95(id_a[i], oos_a[i]) - fpr_at_95(id_b[i], oos_b[i]) for i in range(n_seeds)
    ]))
    rng = np.random.default_rng(seed)
    reps = np.empty(B, dtype=np.float64)
    for b in range(B):
        id_idx = rng.integers(0, n_id, size=n_id)
        oos_idx = rng.integers(0, n_oos, size=n_oos)
        deltas = []
        for i in range(n_seeds):
            deltas.append(
                fpr_at_95(id_a[i][id_idx], oos_a[i][oos_idx])
                - fpr_at_95(id_b[i][id_idx], oos_b[i][oos_idx])
            )
        reps[b] = np.mean(deltas)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"delta": observed, "ci_low": float(lo), "ci_high": float(hi)}


def combined_interval(delta, se_boot, seed_deltas) -> dict:
    """Normal interval that adds the seed-level SE to the utterance bootstrap SE.

    SE = √(SE²_boot + s²_seed / n), with s_seed the sample standard deviation
    of the per-seed estimates. Returns the 95% and 90% intervals.
    """
    seed_deltas = np.asarray(list(seed_deltas), dtype=np.float64)
    n = int(seed_deltas.size)
    if n > 1:
        s_seed = float(np.std(seed_deltas, ddof=1))
    else:
        s_seed = 0.0
    se = float(math.sqrt(float(se_boot) ** 2 + (s_seed ** 2) / max(n, 1)))
    z95 = float(norm.ppf(0.975))
    z90 = float(norm.ppf(0.95))
    delta = float(delta)
    return {
        "delta": delta,
        "se_boot": float(se_boot),
        "s_seed": s_seed,
        "se_seed": float(s_seed / math.sqrt(max(n, 1))),
        "n_seeds": n,
        "se_combined": se,
        "ci95_low": delta - z95 * se,
        "ci95_high": delta + z95 * se,
        "ci90_low": delta - z90 * se,
        "ci90_high": delta + z90 * se,
    }


def holm(p_values: dict, alpha: float = 0.05) -> dict:
    """Holm step-down adjusted p-values.

    ``p_values`` maps a claim name to a raw p-value. Adjusted p-values are
    monotone in the sorted raw p-values. ``reject`` is adjusted p < ``alpha``.
    """
    items = sorted(p_values.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        val = min(1.0, float(p) * (m - i))
        running = max(running, val)
        adjusted[name] = running
    return {
        name: {
            "p": float(p_values[name]),
            "p_holm": adjusted[name],
            "reject": adjusted[name] <= alpha,
        }
        for name in p_values
    }


def one_sided_p(delta: float, se: float) -> float:
    """Normal one-sided p for the claim delta > 0."""
    delta = float(delta)
    se = float(se)
    if se <= 0.0:
        return 0.0 if delta > 0.0 else 1.0
    return float(1.0 - norm.cdf(delta / se))


def paired_bootstrap_did(
    id_a_near,
    oos_a_near,
    id_b_near,
    oos_b_near,
    id_a_far,
    oos_a_far,
    id_b_far,
    oos_b_far,
    *,
    B: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Difference in differences of AUROC.

    DiD = (AUROC(A) − AUROC(B) on near / ID-OOS) − (the same on far / OOD-OOS).

    In-scope utterances are resampled once and shared. Each OOS subset is
    resampled on its own. Seeds stay paired (the same seed index on both sides).
    """
    n_seeds = len(id_a_near)
    if len(id_a_far) != n_seeds:
        raise ValueError("near and far arms need the same number of seeds")
    n_id = len(id_a_near[0])
    n_near = len(oos_a_near[0])
    n_far = len(oos_a_far[0])
    if len(id_a_far[0]) != n_id:
        raise ValueError("near and far in-scope sets need the same length")

    def _engines(id_scores, oos_scores):
        return [WeightedAUROC(id_scores[i], oos_scores[i]) for i in range(n_seeds)]

    eng = {
        "a_near": _engines(id_a_near, oos_a_near),
        "b_near": _engines(id_b_near, oos_b_near),
        "a_far": _engines(id_a_far, oos_a_far),
        "b_far": _engines(id_b_far, oos_b_far),
    }

    def _gains(i, w_id, w_near, w_far):
        gain_near = eng["a_near"][i].auroc(w_id, w_near) - eng["b_near"][i].auroc(w_id, w_near)
        gain_far = eng["a_far"][i].auroc(w_id, w_far) - eng["b_far"][i].auroc(w_id, w_far)
        return gain_near, gain_far

    ones_id = np.ones(n_id, dtype=np.float64)
    ones_near = np.ones(n_near, dtype=np.float64)
    ones_far = np.ones(n_far, dtype=np.float64)
    per_a, per_b, per_d = [], [], []
    for i in range(n_seeds):
        gain_near, gain_far = _gains(i, ones_id, ones_near, ones_far)
        per_a.append(float(gain_near))
        per_b.append(float(gain_far))
        per_d.append(float(gain_near - gain_far))
    observed = float(np.mean(per_d))
    rng = np.random.default_rng(seed)
    reps = np.empty(B, dtype=np.float64)
    for b in range(B):
        w_id = np.bincount(rng.integers(0, n_id, size=n_id), minlength=n_id).astype(np.float64)
        w_near = np.bincount(rng.integers(0, n_near, size=n_near), minlength=n_near).astype(np.float64)
        w_far = np.bincount(rng.integers(0, n_far, size=n_far), minlength=n_far).astype(np.float64)
        gains = [_gains(i, w_id, w_near, w_far) for i in range(n_seeds)]
        reps[b] = np.mean([near_gain - far_gain for near_gain, far_gain in gains])
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": observed,
        "a": float(np.mean(per_a)),
        "b": float(np.mean(per_b)),
        "per_seed_a": per_a,
        "per_seed_b": per_b,
        "per_seed_delta": per_d,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(np.std(reps, ddof=1)) if B > 1 else 0.0,
    }
