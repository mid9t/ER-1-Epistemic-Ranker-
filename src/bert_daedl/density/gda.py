"""Gaussian discriminant density over encoder features (float64, CPU).

NLP-specific choices (see plan §3.1): tied covariance default,
center -> PCA(k) -> whiten preprocessing, jitter escalation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.special import logsumexp


class GaussianDensityModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, feats: np.ndarray, labels: np.ndarray):
        X = np.asarray(feats, dtype=np.float64)
        y = np.asarray(labels)
        self.classes_ = np.unique(y)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w, V = np.clip(w[order], self.cfg.min_var, None), V[:, order]
        k = self.cfg.pca_dim or X.shape[1]
        k = min(k, X.shape[1])
        self.T_ = V[:, :k] / np.sqrt(w[:k]) if self.cfg.whiten else V[:, :k]
        Z = Xc @ self.T_
        self.k_ = k
        self.mu_ = np.stack([Z[y == c].mean(0) for c in self.classes_])

        if self.cfg.cov_mode == "tied":
            R = np.concatenate(
                [Z[y == c] - self.mu_[i] for i, c in enumerate(self.classes_)]
            )
            S = (R.T @ R) / max(len(R) - len(self.classes_), 1)
            self.chol_, self.jitter_ = self._chol(S)
            self.logdet_ = 2.0 * np.log(np.diag(self.chol_)).sum()
        else:  # per_class (QDA) — ablation only
            self.chol_, self.logdet_, self.jitter_ = [], [], []
            for i, c in enumerate(self.classes_):
                Zc = Z[y == c] - self.mu_[i]
                S = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
                L, j = self._chol(S)
                self.chol_.append(L)
                self.jitter_.append(j)
                self.logdet_.append(2.0 * np.log(np.diag(L)).sum())
        return self

    def _chol(self, S):
        for j in self.cfg.jitter_grid:
            try:
                return np.linalg.cholesky(S + j * np.eye(S.shape[0])), j
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError(
            "Covariance not PD at max jitter — reduce pca_dim or check features."
        )

    def _transform(self, feats):
        return (np.asarray(feats, np.float64) - self.mean_) @ self.T_

    def class_log_prob(self, feats) -> np.ndarray:
        """(N, C) log-density per class."""
        Z = self._transform(feats)
        C = len(self.classes_)
        out = np.empty((len(Z), C))
        const = self.k_ * np.log(2.0 * np.pi)
        for i in range(C):
            L = self.chol_ if self.cfg.cov_mode == "tied" else self.chol_[i]
            ld = self.logdet_ if self.cfg.cov_mode == "tied" else self.logdet_[i]
            sol = solve_triangular(L, (Z - self.mu_[i]).T, lower=True)
            out[:, i] = -0.5 * (const + ld + (sol ** 2).sum(0))
        return out

    def marginal_log_prob(self, feats) -> np.ndarray:
        lp = self.class_log_prob(feats)
        return logsumexp(lp, axis=1) - np.log(lp.shape[1])  # uniform prior


# ---------------- normalizers (fit on TRAIN log-densities only) --------------
class MinMaxNorm:
    """Reference-repo behavior; keep for comparison."""

    def fit(self, lp):
        self.lo, self.hi = float(lp.min()), float(lp.max())
        return self

    def __call__(self, lp):
        return np.clip((lp - self.lo) / (self.hi - self.lo + 1e-12), 0.0, 1.0)


class QuantileSigmoidNorm:
    """Recommended for combine_mode='mul'."""

    def __init__(self, center_q=0.05, scale_q=0.50):
        self.center_q, self.scale_q = center_q, scale_q

    def fit(self, lp):
        self.c = float(np.quantile(lp, self.center_q))
        self.s = (float(np.quantile(lp, self.scale_q)) - self.c) / 2.0 + 1e-12
        return self

    def __call__(self, lp):
        return 1.0 / (1.0 + np.exp(-(lp - self.c) / self.s))


class ECDFNorm:
    """Recommended for combine_mode='add_log'."""

    def fit(self, lp):
        self.sorted_ = np.sort(np.asarray(lp, np.float64))
        return self

    def __call__(self, lp):
        r = np.searchsorted(self.sorted_, lp, side="right") / len(self.sorted_)
        return np.clip(r, 1.0 / len(self.sorted_), 1.0)


NORMALIZERS = {"minmax": MinMaxNorm, "qsigmoid": QuantileSigmoidNorm, "ecdf": ECDFNorm}


def scale_alpha(
    logits: torch.Tensor,
    phat: torch.Tensor,
    mode: str = "mul",
    lam: float = 5.0,
    clamp: float = 15.0,
):
    z = logits.clamp(-clamp, clamp)
    p = phat.to(z.dtype)
    if p.ndim == 1:
        p = p.unsqueeze(-1)
    if mode == "mul":  # DAEDL-faithful: can flip predictions
        return torch.exp(z * p)
    if mode == "add_log":  # prediction-preserving: scales alpha0 only
        return torch.exp(z + lam * (p - 1.0))
    raise ValueError(mode)
