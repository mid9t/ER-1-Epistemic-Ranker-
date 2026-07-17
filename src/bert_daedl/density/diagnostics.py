"""GDA / density diagnostics including the materiality gate for inert density."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def density_report(
    gda,
    lp_train,
    lp_id_val,
    lp_oos_val,
    alpha_plain=None,
    alpha_scaled=None,
) -> dict:
    jitter = gda.jitter_
    if not np.isscalar(jitter):
        jitter = list(jitter)

    rep = {
        "cov_mode": gda.cfg.cov_mode,
        "pca_dim": gda.k_,
        "jitter": jitter,
        "raw_logp": {
            "train_q": np.quantile(lp_train, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
            "id_val_median": float(np.median(lp_id_val)),
            "oos_val_median": float(np.median(lp_oos_val)),
        },
        # Can raw density alone separate, before any normalization?
        "raw_density_val_auroc": float(
            roc_auc_score(
                np.r_[np.zeros(len(lp_id_val)), np.ones(len(lp_oos_val))],
                np.r_[-lp_id_val, -lp_oos_val],
            )
        ),
    }
    if alpha_plain is not None and alpha_scaled is not None:
        a0p = alpha_plain.sum(-1).detach().cpu().numpy()
        a0s = alpha_scaled.sum(-1).detach().cpu().numpy()
        flip = (
            (alpha_plain.argmax(-1) != alpha_scaled.argmax(-1)).float().mean().item()
        )
        rel = np.median(np.abs(a0s - a0p) / (a0p + 1e-12))
        rho = float(spearmanr(a0p, a0s).statistic)
        rep["materiality"] = {
            "flip_rate": flip,
            "median_rel_delta_alpha0": float(rel),
            "spearman_alpha0": rho,
            # INERT == the CIFAR failure mode: scaled alpha ranks identically
            "density_effect": "INERT" if (rho >= 0.999 and rel < 0.05) else "ACTIVE",
        }
    return rep
