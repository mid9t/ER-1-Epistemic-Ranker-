"""Scores from cached float32 arrays, in float64, through the repo's own code."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import spearmanr

from bert_daedl.config import GDAConfig
from bert_daedl.density.gda import NORMALIZERS, GaussianDensityModel, scale_alpha
from bert_daedl.eval.metrics import ece, ood_metrics
from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.uncertainty.scores import scores_from_alpha, scores_from_logits

LOGIT_CLAMP = 15.0
ALPHA_EPS = 1e-6
CE_SCORES = ("msp", "entropy", "energy", "energy_clip", "neg_logp")
EDL_SCORES = CE_SCORES + (
    "vacuity",
    "neg_alpha0",
    "one_minus_maxpbar",
    "pred_entropy",
    "mutual_info",
    "diff_entropy",
)
VACUITY_FAMILY = ("vacuity", "neg_alpha0", "energy_clip")
EXACT_VERIFY = ("energy", "vacuity", "neg_alpha0", "neg_logp")


@dataclass(frozen=True)
class ScoreSpec:
    kind: str
    name: str
    normalizer: str | None = None
    mode: str | None = None
    lam: float | None = None

    def __str__(self) -> str:
        return format_spec(self)


def parse_spec(text: str) -> ScoreSpec:
    text = text.strip()
    if not text.startswith("daedl:"):
        return ScoreSpec("plain", text)
    body = text[len("daedl:") :]
    score = "vacuity"
    if "/" in body:
        body, score = body.split("/", 1)
    parts = body.split(":")
    if len(parts) < 2:
        raise ValueError(f"bad daedl spec: {text}")
    normalizer, mode = parts[0], parts[1]
    lam = float(parts[2]) if len(parts) > 2 else None
    return ScoreSpec("daedl", score, normalizer, mode, lam)


def format_spec(spec: ScoreSpec) -> str:
    if spec.kind == "plain":
        return spec.name
    lam = ""
    if spec.mode == "add_log":
        value = 5.0 if spec.lam is None else spec.lam
        lam = f":{value:g}"
    elif spec.lam is not None:
        lam = f":{spec.lam:g}"
    return f"daedl:{spec.normalizer}:{spec.mode}{lam}/{spec.name}"


def _as_f64(logits) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64)


def logit_scores(logits, *, arm: str = "edl", clamp: float = LOGIT_CLAMP, eps: float = ALPHA_EPS) -> dict:
    z = torch.from_numpy(_as_f64(logits))
    out = {k: v.detach().cpu().numpy() for k, v in scores_from_logits(z).items()}
    clipped = np.clip(_as_f64(logits), -clamp, clamp)
    out["energy_clip"] = -logsumexp(clipped, axis=-1)
    if arm in ("edl", "daedl"):
        alpha = logits_to_alpha(z, clamp, eps)
        for k, v in scores_from_alpha(alpha).items():
            out[k] = v.detach().cpu().numpy()
        out["_alpha"] = alpha.detach().cpu().numpy()
    return out


def gda_from_config(raw) -> GDAConfig:
    if not isinstance(raw, dict):
        return GDAConfig()
    allowed = {f.name for f in dataclasses.fields(GDAConfig)}
    kw = {k: v for k, v in raw.items() if k in allowed}
    if kw.get("jitter_grid") is not None:
        kw["jitter_grid"] = tuple(kw["jitter_grid"])
    return GDAConfig(**kw)


def fit_density(train_feats, train_labels, gda_cfg: GDAConfig | None = None):
    gda = GaussianDensityModel(gda_cfg or GDAConfig()).fit(train_feats, train_labels)
    return gda


def density_maps(run, feature: str = "feats", gda_cfg: GDAConfig | None = None):
    cfg = gda_cfg or gda_from_config((run["meta"].get("config") or {}).get("gda"))
    train = run["splits"]["train"]
    gda = fit_density(train[feature], train["labels"], cfg)
    return gda, split_logp(gda, run["splits"], feature)


PCA_UNSET = object()


def pca_from_arg(text):
    """Map a ``--pca_dim`` string to a dimension.

    ``None`` means the flag was omitted (``PCA_UNSET``). ``'none'`` means no
    PCA. Anything else is an integer dimension.
    """
    if text is None:
        return PCA_UNSET
    if str(text).strip().lower() in {"none", "null"}:
        return None
    return int(text)


def logp_from_run(
    run,
    *,
    feature: str = "feats",
    variant: str = "config",
    top_k: int = 128,
    pca_dim=PCA_UNSET,
):
    """Log-density maps for a cached run.

    ``config`` uses the GDA saved with the run. ``tied`` is tied covariance on
    ``feature`` with no PCA truncation. ``lowvar`` uses only the bottom
    eigen-directions of that tied within-class covariance; the returned values
    are negated distances, so higher still means more in-distribution.

    ``pca_dim`` overrides the variant with a tied, whitened Gaussian at that
    dimension (``None`` disables PCA). The low-variance density ignores it.
    """
    if feature not in run["splits"]["train"]:
        raise ValueError(f"{feature} is not in the cache for {run.get('path')}")
    if pca_dim is not PCA_UNSET:
        if variant == "lowvar":
            raise ValueError("pca_dim does not apply to the low-variance density")
        cfg = GDAConfig(cov_mode="tied", pca_dim=pca_dim, whiten=True)
        return density_maps(run, feature=feature, gda_cfg=cfg)[1]
    if variant == "config":
        return density_maps(run, feature=feature)[1]
    if variant == "tied":
        cfg = GDAConfig(cov_mode="tied", pca_dim=None, whiten=True)
        return density_maps(run, feature=feature, gda_cfg=cfg)[1]
    if variant == "lowvar":
        from bert_daedl.posthoc.gda_variants import partial_basis, partial_sq

        train = run["splits"]["train"]
        mus, evecs, evals, k = partial_basis(train[feature], train["labels"], top_k)
        idx = np.arange(k, len(evals))
        return {
            name: -partial_sq(mus, evecs, evals, data[feature], idx)
            for name, data in run["splits"].items()
        }
    raise ValueError(variant)


def split_logp(gda, splits, feature: str = "feats") -> dict[str, np.ndarray]:
    return {name: gda.marginal_log_prob(data[feature]) for name, data in splits.items()}


def phats_for(logp: dict[str, np.ndarray], normalizer: str, fit_on: str = "train") -> dict[str, np.ndarray]:
    if fit_on == "train":
        norm = NORMALIZERS[normalizer]().fit(logp["train"])
        return {name: np.asarray(norm(lp), dtype=np.float64) for name, lp in logp.items()}
    norm = NORMALIZERS[normalizer]().fit(logp["val"])
    out = {name: np.asarray(norm(lp), dtype=np.float64) for name, lp in logp.items()}
    out["val"] = cross_fit_phat(logp["val"], normalizer)
    return out


def cross_fit_phat(lp, normalizer: str, n_folds: int = 2) -> np.ndarray:
    lp = np.asarray(lp, dtype=np.float64)
    folds = np.array_split(np.arange(len(lp)), n_folds)
    out = np.empty(len(lp), dtype=np.float64)
    for i, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        norm = NORMALIZERS[normalizer]().fit(lp[train])
        out[test] = np.asarray(norm(lp[test]), dtype=np.float64)
    return out


def daedl_alpha(logits, phat, mode: str, lam: float | None, clamp: float = LOGIT_CLAMP):
    z = torch.from_numpy(_as_f64(logits))
    p = torch.from_numpy(np.asarray(phat, dtype=np.float64))
    return scale_alpha(z, p, mode, 5.0 if lam is None else lam, clamp)


def score_array(run, spec: ScoreSpec, *, logp=None, phat=None, arm: str | None = None) -> dict[str, np.ndarray]:
    """Score every cached split. ``logp`` / ``phat`` are optional precomputed maps."""
    meta = run["meta"]
    arm = arm or meta.get("arm", "edl")
    cfg = meta.get("config") or {}
    clamp = float(cfg.get("logit_clamp", LOGIT_CLAMP))
    eps = float(cfg.get("alpha_eps", ALPHA_EPS))
    out = {}
    for name, data in run["splits"].items():
        scores = logit_scores(data["logits"], arm=arm if arm != "ce" else "ce", clamp=clamp, eps=eps)
        if spec.kind == "plain":
            if spec.name == "neg_logp":
                if logp is None:
                    raise ValueError("neg_logp requires a fitted density")
                out[name] = -np.asarray(logp[name], dtype=np.float64)
            else:
                out[name] = np.asarray(scores[spec.name], dtype=np.float64)
        else:
            if phat is None:
                raise ValueError("daedl specs require phat")
            alpha = daedl_alpha(data["logits"], phat[name], spec.mode, spec.lam, clamp)
            named = scores_from_alpha(alpha)
            out[name] = named[spec.name].detach().cpu().numpy()
    return out


def operating_point(id_val, id_test, oos_test) -> dict:
    thr = float(np.quantile(id_val, 0.95))
    return {
        "threshold": thr,
        "oos_recall": float(np.mean(oos_test > thr)),
        "id_reject_rate": float(np.mean(id_test > thr)),
    }


def detection(id_scores, oos_scores) -> dict:
    return ood_metrics(np.asarray(id_scores), np.asarray(oos_scores))


def in_scope_accuracy(logits, labels) -> float:
    pred = np.asarray(logits).argmax(-1)
    return float(np.mean(pred == np.asarray(labels)))


def clamp_share(logits, clamp: float = LOGIT_CLAMP) -> float:
    return float(np.mean(np.asarray(logits).max(-1) >= clamp))


def ece_from_probs(conf, correct) -> float:
    return ece(np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=bool))


def softmax_ece(logits, labels) -> float:
    z = torch.from_numpy(_as_f64(logits))
    prob = torch.softmax(z, dim=-1).numpy()
    conf = prob.max(-1)
    correct = prob.argmax(-1) == np.asarray(labels)
    return ece_from_probs(conf, correct)


def dirichlet_ece(alpha, labels) -> float:
    alpha = np.asarray(alpha, dtype=np.float64)
    p = alpha / alpha.sum(-1, keepdims=True)
    correct = p.argmax(-1) == np.asarray(labels)
    return ece_from_probs(p.max(-1), correct)


def spearman(a, b) -> float:
    rho = spearmanr(a, b).statistic
    if rho is None or not np.isfinite(rho):
        return float("nan")
    return float(rho)
