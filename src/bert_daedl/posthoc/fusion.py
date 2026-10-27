"""Correlations, linear-fusion angles, and the add_log family."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from bert_daedl.posthoc.scoring import phats_for, spearman
from bert_daedl.posthoc.stats import auroc_scores


def _pool(id_scores, oos_scores):
    return np.concatenate([np.asarray(id_scores, np.float64), np.asarray(oos_scores, np.float64)])


def _labels(n_id, n_oos):
    return np.concatenate([np.zeros(n_id), np.ones(n_oos)])


def spearman_blocks(e_id, e_oos, d_id, d_oos) -> dict:
    return {
        "rho_pooled_val": spearman(_pool(e_id["val"], e_oos["val"]), _pool(d_id["val"], d_oos["val"])),
        "rho_pooled_test": spearman(_pool(e_id["test"], e_oos["test"]), _pool(d_id["test"], d_oos["test"])),
        "rho_id_test": spearman(e_id["test"], d_id["test"]),
        "rho_oos_test": spearman(e_oos["test"], d_oos["test"]),
    }


def linear_fusion(e_id, e_oos, d_id, d_oos, n_angles: int = 720) -> dict:
    theta = np.linspace(0, np.pi, n_angles, endpoint=False)

    def sweep(split):
        e = _pool(e_id[split], e_oos[split])
        d = _pool(d_id[split], d_oos[split])
        y_id = len(e_id[split])
        y_oos = len(e_oos[split])
        aurocs = np.empty(n_angles)
        for i, th in enumerate(theta):
            fused = np.cos(th) * e + np.sin(th) * d
            aurocs[i] = auroc_scores(fused[:y_id], fused[y_id:])
        return aurocs

    val = sweep("val")
    test = sweep("test")
    chosen = int(np.argmax(val))
    e_only = auroc_scores(e_id["test"], e_oos["test"])
    d_only = auroc_scores(d_id["test"], d_oos["test"])
    return {
        "fusion_val_auroc": float(val[chosen]),
        "fusion_test_auroc": float(test[chosen]),
        "fusion_test_auroc_oracle": float(test.max()),
        "fusion_theta": float(theta[chosen]),
        "test_auroc_E": float(e_only),
        "test_auroc_D": float(d_only),
    }


def add_log_sweep(energy_id, energy_oos, logp, lams=None) -> dict:
    # An explicit grid (the replication λ set) is flagged when the validation
    # pick sits on either end. The default fine grid keeps its old columns.
    flag_edge = lams is not None
    if lams is None:
        lams = np.arange(0.0, 20.0 + 1e-9, 0.25)
    lams = np.asarray(lams, dtype=np.float64)
    out = {}
    for name, lp in logp.items():
        phat = phats_for(lp, name, fit_on="train")
        val = np.array([
            auroc_scores(
                energy_id["val"] + lam * (1.0 - phat["val"]),
                energy_oos["val"] + lam * (1.0 - phat["oos_val"]),
            )
            for lam in lams
        ])
        test = np.array([
            auroc_scores(
                energy_id["test"] + lam * (1.0 - phat["test"]),
                energy_oos["test"] + lam * (1.0 - phat["oos_test"]),
            )
            for lam in lams
        ])
        chosen = int(np.argmax(val))
        chosen_lam = float(lams[chosen])
        out[f"{name}_add_log_val_auroc"] = float(val[chosen])
        out[f"{name}_add_log_test_auroc"] = float(test[chosen])
        out[f"{name}_add_log_test_oracle"] = float(test.max())
        out[f"{name}_add_log_lam"] = chosen_lam
        if flag_edge:
            lo = float(np.min(lams))
            hi = float(np.max(lams))
            out[f"{name}_add_log_on_edge"] = float(
                bool(np.isclose(chosen_lam, lo) or np.isclose(chosen_lam, hi))
            )
    return out


def logistic_fusion(e_id, e_oos, d_id, d_oos) -> dict:
    def pack(split):
        e = _pool(e_id[split], e_oos[split])
        d = _pool(d_id[split], d_oos[split])
        x = np.column_stack([e, d])
        y = _labels(len(e_id[split]), len(e_oos[split]))
        return x, y

    x_val, y_val = pack("val")
    x_test, _ = pack("test")
    mu, sd = x_val.mean(0), x_val.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    clf = LogisticRegression(max_iter=200)
    clf.fit((x_val - mu) / sd, y_val)
    proba = clf.predict_proba((x_test - mu) / sd)[:, 1]
    n_id = len(e_id["test"])
    return {"logistic_test_auroc": auroc_scores(proba[:n_id], proba[n_id:])}
