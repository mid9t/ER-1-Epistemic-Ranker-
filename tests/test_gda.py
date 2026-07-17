"""GDA synthetic tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.config import GDAConfig
from bert_daedl.density.gda import GaussianDensityModel


def _synth(n_per=80, d=16, seed=0):
    rng = np.random.default_rng(seed)
    mu0, mu1 = np.zeros(d), np.ones(d) * 3.0
    X0 = rng.normal(mu0, 0.5, size=(n_per, d))
    X1 = rng.normal(mu1, 0.5, size=(n_per, d))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per + [1] * n_per)
    ood = rng.normal(np.ones(d) * -5.0, 0.5, size=(20, d))
    return X, y, ood


def test_gda_tied_separates_ood():
    X, y, ood = _synth()
    cfg = GDAConfig(cov_mode="tied", pca_dim=8, whiten=True)
    gda = GaussianDensityModel(cfg).fit(X, y)
    lp_id = gda.marginal_log_prob(X)
    lp_ood = gda.marginal_log_prob(ood)
    assert lp_ood.mean() < lp_id.mean()
    # Cholesky succeeded (PD)
    assert gda.chol_.shape[0] == gda.k_


def test_gda_per_class_fits_pd():
    X, y, ood = _synth()
    cfg = GDAConfig(cov_mode="per_class", pca_dim=8, whiten=True)
    gda = GaussianDensityModel(cfg).fit(X, y)
    lp_id = gda.marginal_log_prob(X)
    lp_ood = gda.marginal_log_prob(ood)
    assert lp_ood.mean() < lp_id.mean()
    assert len(gda.chol_) == 2
