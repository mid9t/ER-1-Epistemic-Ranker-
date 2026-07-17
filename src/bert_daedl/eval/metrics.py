"""OOD / misclassification / calibration metrics. OOS is the positive class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ood_metrics(unc_id: np.ndarray, unc_oos: np.ndarray) -> dict:
    y = np.r_[np.zeros(len(unc_id)), np.ones(len(unc_oos))]
    s = np.r_[unc_id, unc_oos]
    thr = np.quantile(unc_oos, 0.05)  # keep 95% of OOS above thr
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),  # OOS positive
        "fpr_at_95tpr": float((unc_id >= thr).mean()),
    }


def misclf_metrics(unc: np.ndarray, correct: np.ndarray) -> dict:
    err = (~correct).astype(int)
    return {
        "err_auroc": float(roc_auc_score(err, unc)),
        "err_aupr": float(average_precision_score(err, unc)),
        "succ_aupr": float(average_precision_score(correct.astype(int), -unc)),
    }


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)
