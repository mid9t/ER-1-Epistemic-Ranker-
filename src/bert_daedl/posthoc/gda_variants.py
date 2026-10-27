"""Relative, partial, and l2-normalized Mahalanobis scores, plus the spectrum."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from bert_daedl.config import GDAConfig
from bert_daedl.density.gda import GaussianDensityModel


def _chol(S, jitter_grid=(0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2)):
    eye = np.eye(S.shape[0])
    for j in jitter_grid:
        try:
            return np.linalg.cholesky(S + j * eye)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("background covariance is not positive definite")


def _l2(X):
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def min_class_sq(train_x, train_y, query_x, pca_dim, whiten=True) -> np.ndarray:
    cfg = GDAConfig(cov_mode="tied", pca_dim=pca_dim, whiten=whiten)
    gda = GaussianDensityModel(cfg).fit(train_x, train_y)
    return gda.class_sq_mahalanobis(query_x).min(axis=1), gda


def relative_mahalanobis(train_x, train_y, query_x, pca_dim=None, whiten=True) -> np.ndarray:
    """Class Mahalanobis minus one background Gaussian, in the GDA feature space."""
    class_sq, gda = min_class_sq(train_x, train_y, query_x, pca_dim, whiten)
    z_train = gda._transform(train_x)
    z_query = gda._transform(query_x)
    mu = z_train.mean(0)
    cov = np.cov(z_train, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    chol = _chol(cov)
    sol = solve_triangular(chol, (z_query - mu).T, lower=True)
    background = np.sum(sol * sol, axis=0)
    return class_sq - background


def mahalanobis_pp(train_x, train_y, query_x, pca_dim=None, whiten=True) -> np.ndarray:
    """Mahalanobis++: l2-normalize, then tied GDA distance."""
    sq, _ = min_class_sq(_l2(train_x), train_y, _l2(query_x), pca_dim, whiten)
    return sq


def partial_basis(train_x, train_y, top_k: int = 128):
    """Tied within-class covariance, eigenvalues descending. Shared by every query split."""
    X = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y)
    classes = np.unique(y)
    mus = np.stack([X[y == c].mean(0) for c in classes])
    pieces = [X[y == c] - mus[i] for i, c in enumerate(classes)]
    R = np.concatenate(pieces)
    cov = (R.T @ R) / max(len(R) - len(classes), 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 1e-8, None)
    evecs = evecs[:, order]
    k = min(top_k, len(evals))
    return mus, evecs, evals, k


def partial_sq(mus, evecs, evals, query_x, idx) -> np.ndarray:
    """Min over classes of squared Mahalanobis distance on eigen-directions ``idx``."""
    Q = np.asarray(query_x, dtype=np.float64)
    if len(idx) == 0:
        return np.zeros(len(Q))
    V = evecs[:, idx]
    w = evals[idx]
    q = (Q @ V) / np.sqrt(w)
    mu = (mus @ V) / np.sqrt(w)
    q2 = np.sum(q * q, axis=1, keepdims=True)
    mu2 = np.sum(mu * mu, axis=1)
    return (q2 + mu2 - 2.0 * (q @ mu.T)).min(axis=1)


def partial_mahalanobis(train_x, train_y, query_x, top_k: int = 128) -> dict[str, np.ndarray]:
    """Distance on the top eigen-directions of the tied within-class covariance, and the rest."""
    mus, evecs, evals, k = partial_basis(train_x, train_y, top_k)
    return {
        "top": partial_sq(mus, evecs, evals, query_x, np.arange(k)),
        "bottom": partial_sq(mus, evecs, evals, query_x, np.arange(k, len(evals))),
    }


def spectrum(feats, labels) -> dict:
    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels)
    mu = X.mean(0)
    xc = X - mu
    cov = (xc.T @ xc) / max(len(X) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    w = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]
    total = float(w.sum()) + 1e-30
    classes = np.unique(y)
    mus = np.stack([X[y == c].mean(0) for c in classes])
    ns = np.array([(y == c).sum() for c in classes], dtype=np.float64)
    centered = mus - mu
    trace_sb = float(np.sum(ns * np.sum(centered * centered, axis=1)))
    k = min(128, evecs.shape[1])
    proj = centered @ evecs[:, :k]
    trace_top = float(np.sum(ns * np.sum(proj * proj, axis=1)))
    return {
        "dim": int(X.shape[1]),
        "participation_ratio": float((w.sum() ** 2) / np.sum(w * w)),
        "condition_number": float(w[0] / max(w[-1], 1e-30)),
        "var_top64": float(w[: min(64, len(w))].sum() / total),
        "var_top128": float(w[:k].sum() / total),
        "between_var_in_top128": float(trace_top / trace_sb) if trace_sb > 0 else 1.0,
        "eigenvalues": w,
    }
