# Implementation Plan — Evidential BERT on CLINC150 (DAEDL Port, Milestone 1)

**File:** `bert_edl_clinc150_implementation_plan.md`
**Milestone scope:** CLINC150 data pipeline + a replicable, leak-free evaluation path for three arms (CE baseline, vanilla EDL, DAEDL-style density-scaled EDL), with diagnostics that make an inert density term impossible to ship silently.
**Explicitly deferred:** early-exit heads and threshold gating (Milestone 2). The architecture below reserves the interfaces so Milestone 2 requires no rework.

---

## 1. Review of current code

### 1.1 `clinc150_pipeline.py` — keep, with 6 revisions

The pipeline is fundamentally sound: correct ID/OOS separation, HF + local fallback, `-1` sentinel for OOS labels, deterministic label maps. Required changes:

1. **`oos_val` is silently discarded (critical).** Only `oos_test` becomes a loader. You need an `oos_val_loader` for (a) normalizer/temperature selection in the density phase, (b) rejection-threshold calibration, (c) per-epoch "is uncertainty learning?" sanity checks — all *without touching test*. This is the single most important fix.
2. **`include_oos_in_train=True` crashes.** `to_id_dataset` does `label2id[l]` and `"oos"` is not in the map → `KeyError`. Remove the flag entirely (outlier-exposure baselines are out of scope; `oos_train` (250 ex.) stays unused by design — log it).
3. **Add split-size validation.** For config `plus`: ID train 15,000 / OOS train 250 / ID val 3,000 / OOS val 100 / ID test 4,500 / OOS test 1,000. Warn (don't crash) on mismatch.
4. **Centralize seeding** into `utils/seed.py::seed_everything(seed)` (torch, numpy, random, cudnn flags). Remove the in-function `manual_seed` calls.
5. **Pin `datasets<3.0`** or handle the fallback: `clinc_oos` is a script-based HF dataset; `datasets>=3` drops script support. Your local `data_full.json` fallback already covers this — add one comment.
6. **Return signature change:** `(train, val, test, oos_val, oos_test, meta)` — update all callers and the smoke test. Optionally add `return_texts` for error analysis later.

Replacement tail of `build_clinc150_dataloaders` (only the changed part):

```python
def _validate_plus_counts(id_train, oos_train, id_val, oos_val, id_test, oos_test):
    expected = {"id_train": 15000, "oos_train": 250, "id_val": 3000,
                "oos_val": 100, "id_test": 4500, "oos_test": 1000}
    got = {"id_train": len(id_train), "oos_train": len(oos_train),
           "id_val": len(id_val), "oos_val": len(oos_val),
           "id_test": len(id_test), "oos_test": len(oos_test)}
    for k, v in expected.items():
        if got[k] != v:
            logger.warning("Split size mismatch %s: expected %d got %d", k, v, got[k])

    # ... inside build_clinc150_dataloaders, after _split_id_oos calls:
    _validate_plus_counts(id_train, oos_train, id_val, oos_val, id_test, oos_test)
    logger.info("oos_train (n=%d) intentionally unused (no outlier exposure).", len(oos_train))

    train_ds    = to_id_dataset(id_train)
    val_ds      = to_id_dataset(id_val)
    test_ds     = to_id_dataset(id_test)
    oos_val_ds  = to_oos_dataset(oos_val)     # NEW
    oos_test_ds = to_oos_dataset(oos_test)

    # loaders: train shuffled w/ generator; all eval loaders shuffle=False
    ...
    return train_loader, val_loader, test_loader, oos_val_loader, oos_test_loader, meta
```

### 1.2 `bert_daedl.py` — replace entirely

It's a skeleton with blocking bugs; replaced by `scripts/run_experiment.py` (§5.10). Bug inventory so nothing is lost:

- `main()` takes no parameters but is invoked as `main(args)`; inside, `args` is used as an implicit global.
- `save_result[output_dir, result]` — square brackets (indexing) instead of a call → `TypeError`.
- Two near-duplicate savers (`save_result` local, `save_results` imported from `DAEDL.main`) used interchangeably.
- `load_model` returns `None` (calls an empty `bert()` stub); typos `args.indeex`, `args.akip_train`, `args.chekcpoint`, `args.checkpoibt`, `model.to(args,device)`.
- `num_classes` is read from `meta` and then ignored in favor of `args.num_classes` (CIFAR-oriented `parse_args` from `DAEDL.main` doesn't define BERT-relevant args).
- Wildcard imports (`from clinc150_pipeline import *` etc.) alongside explicit imports.
- Result keys drift between `"OOS AUROC"`/`"OOD AUROC"` across your files — standardize on **OOS** for CLINC150 (schema in §5.8).

---

## 2. Target repository layout

```
project/
├── DAEDL/                          # existing CIFAR replication — READ-ONLY reference
├── configs/
│   └── default.yaml
├── src/bert_daedl/
│   ├── __init__.py
│   ├── config.py                   # §5.1 dataclasses (single source of truth)
│   ├── data/
│   │   └── clinc150.py             # revised pipeline (§1.1)
│   ├── models/
│   │   ├── bert_edl.py             # §5.3 encoder + SN bottleneck + evidential head
│   │   └── exit_heads.py           # (Milestone 2 placeholder — empty)
│   ├── losses/
│   │   └── edl.py                  # §5.4 — mirrors DAEDL/train.py loss
│   ├── density/
│   │   ├── gda.py                  # §5.5 tied/per-class GDA, PCA+whitening, normalizers
│   │   └── diagnostics.py          # §5.6 port of your verify_gda + materiality gate
│   ├── uncertainty/
│   │   └── scores.py               # §5.7 all scores, single orientation convention
│   ├── eval/
│   │   ├── metrics.py              # §5.8 AUROC/AUPR/FPR95/ECE
│   │   ├── protocol.py             # §5.8 one entrypoint → results.json
│   │   └── exit_calibration.py     # (Milestone 2 placeholder — empty)
│   ├── train.py                    # §5.9
│   └── utils/
│       ├── seed.py
│       └── io.py                   # save_json, config_hash, git_sha
├── scripts/
│   ├── run_experiment.py           # §5.10 replaces bert_daedl.py
│   └── aggregate_results.py        # §5.11
├── tests/                          # §5.12
├── results/                        # per-run JSON + summary.csv (gitignored)
├── requirements.txt
├── Makefile
└── README.md
```

Mapping from your validated CIFAR code (convert, don't rewrite blindly):

| DAEDL/ (CIFAR, validated) | New module | What changes |
|---|---|---|
| `train.py` loss + `alpha = 1e-6 + exp(z)` | `losses/edl.py` | same formula & annealing; add logit clamp for BERT stability |
| `density_estimation.py` (`fit_gda`) | `density/gda.py` | float64, tied-cov default, PCA→whiten preprocessing, jitter escalation |
| `oos_detection.py`, `conf_calibration.py` | `eval/protocol.py` + `uncertainty/scores.py` | one orientation convention; fixes the aleatoric/epistemic swap class of bug |
| `verify_gda.py` | `density/diagnostics.py` | + materiality gate (rank-corr / flip-rate / Δα) |

`requirements.txt`:

```
torch>=2.2
transformers>=4.40
datasets>=2.16,<3.0
scikit-learn>=1.3
scipy>=1.11
numpy>=1.26
pyyaml
pytest
```

---

## 3. Design decisions for the NLP port (the "genuine ideas")

These are the decisions that make this a real adaptation rather than a copy-paste, each tied to a known transformer issue and to what your CIFAR diagnostics already proved.

**3.1 Anisotropy → tied covariance + PCA + whitening, fit in float64.** Your CIFAR run hit condition numbers ~3.5×10¹⁴ fitting per-class 512-d covariances. BERT pooled embeddings are worse: anisotropic (representation degeneration), 768-d, with only ~100 examples/class on CLINC150 — per-class QDA at 768-d is unfittable (100 samples cannot estimate a 768×768 covariance). Design: default to **tied (shared) covariance** — the Lee et al. (2018) Mahalanobis detector, which pools all 15k training samples into one covariance and is the standard, well-conditioned choice in NLP OOD — with preprocessing `center → PCA(k=128) → whiten`, and per-class QDA only as an ablation at k≤64. All density math in numpy float64 on CPU (your MPS float32 stiffness disappears).

**3.2 Bi-Lipschitz → spectral norm on a bottleneck head only, not on BERT.** Full distance-preservation through attention+LayerNorm+residual stacks is an open problem; naive SN inside BERT can tank accuracy. Compromise: a `Linear(768→256)` bottleneck with `torch.nn.utils.parametrizations.spectral_norm`, **no nonlinearity by default** (keeps features an affine map of the pooled embedding — Gaussian-friendlier for GDA; ReLU/tanh as ablations). The classifier head stays un-normalized so logit scale is free — your CIFAR diagnosis showed the `exp(z·p̂)` mechanism is hostage to logit magnitude, so we do not constrain it and we *measure* it (§5.6). Precedent to cite and benchmark against in the writeup: the SNGP paper (Liu et al., NeurIPS 2020) includes a BERT-on-CLINC150-OOS experiment — pull their exact numbers when writing; it is the canonical distance-aware-transformer anchor on this very dataset. Ablation D: SN on the FFN output dense layers (SNGP-BERT style) only if ahead of schedule.

**3.3 Near-OOD realism.** CIFAR→SVHN is far-OOD; CLINC150 OOS utterances are fluent English chit-chat — *near*-OOD relative to a pretrained encoder that has seen the whole internet. Expect smaller raw density gaps than you saw on images. Consequence: normalization design (3.4) and val-based selection matter more, and honest expectations: beating MSP by a lot is not guaranteed; beating it *reliably across seeds* is the claim to target.

**3.4 Density normalization redesigned around your CIFAR finding.** The reference repo's min-max normalization squashed ID `p̂` into [0.90, 1.00] → `exp(z·p̂) ≈ exp(z)` → inert density. Three pluggable normalizers (fit on **train** log-densities only):

- `minmax` — reference-faithful arm (keep for comparability).
- `qsigmoid(center=q05, scale=(q50−q05)/2)` — recommended for the multiplicative DAEDL rule: bulk ID sits near 1, low-density inputs fall toward 0; robust to the heavy left tail that breaks min-max.
- `ecdf` — rank against train log-densities; pairs with the additive rule below.

And two combination rules:

- `mul` (DAEDL-faithful): `α = exp(z ⊙ p̂)` — can flip predictions.
- `add_log` (our controlled variant): `α = exp(z + λ(p̂−1))` — multiplies α₀ by `e^{λ(p̂−1)}`, injecting density into *evidence mass only*; argmax is provably unchanged (prediction-preserving), so ID accuracy cannot degrade. Clean epistemic injection; default λ=5, selected on val.

**3.5 One orientation convention, unit-tested.** Every uncertainty score is defined so **higher = more OOS**; OOS is the positive class everywhere. Your CIFAR `auroc_epis`-used-aleatoric bug is exactly the class of error this kills — the convention is pinned by tests (§5.12), not by discipline.

**3.6 Sequence-level only.** [CLS] pooling default, mean-pooling as a cheap ablation; `max_length=64` (CLINC utterances are short). Token-level evidential uncertainty is a different project — out of scope, say so in the writeup.

---

## 4. Experiment arms

All arms share the encoder, data, training loop, and eval protocol. GDA is fit for **every** arm (so the Mahalanobis baseline comes free on the CE model).

| Arm | Train loss | Uncertainty scores computed |
|---|---|---|
| `ce` | cross-entropy | msp, entropy, energy, neg_logp (Mahalanobis) |
| `edl` | EDL (EXP param, KL-annealed) | + vacuity, pred_entropy, mutual_info, diff_entropy, neg_alpha0 |
| `daedl` | same weights as `edl` (post-hoc scaling) | + all `daedl_*` scores from density-scaled α |

Seed protocol: 3 seeds {0,1,2} during development/selection; 5 seeds {0..4} for any table that goes in the writeup. Report mean ± std. `daedl` is *evaluation-time* on the `edl` checkpoint (matches the reference repo's post-hoc design you verified), so it costs no extra training.

---

## 5. Module-by-module implementation

### 5.1 `src/bert_daedl/config.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GDAConfig:
    cov_mode: str = "tied"            # "tied" | "per_class"
    pca_dim: Optional[int] = 128      # None = no reduction
    whiten: bool = True
    jitter_grid: tuple = (1e-6, 1e-5, 1e-4, 1e-3)
    min_var: float = 1e-8

@dataclass
class ExperimentConfig:
    # data
    model_name: str = "bert-base-uncased"
    max_length: int = 64
    batch_size: int = 32
    eval_batch_size: int = 128
    num_workers: int = 4              # set 0 on MPS
    # model
    feat_dim: int = 256
    sn_bottleneck: bool = True
    bottleneck_activation: str = "none"   # "none" | "relu" | "tanh"
    pooling: str = "cls"                  # "cls" | "mean"
    dropout: float = 0.1
    # training
    arm: str = "edl"                      # "ce" | "edl" | "daedl"
    epochs: int = 5
    lr_encoder: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    # EDL — mirror the values from your validated DAEDL/ CIFAR run
    reg_param: float = 1e-3
    kl_anneal_epochs: int = 5             # = epochs
    alpha_eps: float = 1e-6
    logit_clamp: float = 15.0
    # density
    gda: GDAConfig = field(default_factory=GDAConfig)
    normalizer: str = "qsigmoid"          # "minmax" | "qsigmoid" | "ecdf"
    combine_mode: str = "mul"             # "mul" | "add_log"
    lam: float = 5.0                      # add_log strength
    # bookkeeping
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "results"
```

### 5.2 Data — as revised in §1.1 (file moves to `src/bert_daedl/data/clinc150.py`).

### 5.3 `src/bert_daedl/models/bert_edl.py`

```python
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from transformers import AutoModel

_ACT = {"none": nn.Identity, "relu": nn.ReLU, "tanh": nn.Tanh}

class BertEvidentialClassifier(nn.Module):
    """Encoder -> pooled -> (SN) bottleneck feature -> logits.

    GDA fits on `features`. `alpha = exp(logits)` (EXP parameterization,
    DAEDL) is computed in the loss / scores, NOT here — forward stays
    loss-agnostic so all three arms share this class.

    Milestone 2 note: pass `output_hidden_states=True` to the encoder and
    attach `exit_heads` per layer; the forward signature already reserves
    the flag so no rework is needed.
    """

    def __init__(self, model_name: str, num_classes: int, feat_dim: int = 256,
                 sn_bottleneck: bool = True, bottleneck_activation: str = "none",
                 pooling: str = "cls", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        bottleneck = nn.Linear(hidden, feat_dim)
        self.bottleneck = spectral_norm(bottleneck) if sn_bottleneck else bottleneck
        self.act = _ACT[bottleneck_activation]()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def _pool(self, last_hidden, attention_mask):
        if self.pooling == "cls":
            return last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                return_features: bool = False, return_hidden_states: bool = False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           output_hidden_states=return_hidden_states)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        feats = self.act(self.bottleneck(self.dropout(pooled)))
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits
```

### 5.4 `src/bert_daedl/losses/edl.py`

Source of truth is **your validated `DAEDL/train.py`** — the formula below is the standard Bayes-risk (digamma) EDL loss with annealed KL to a uniform Dirichlet and EXP parameterization; diff it against your CIFAR file and reconcile any difference *in favor of the CIFAR file* (it's the one whose numbers you reproduced).

```python
import torch

def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ), per-sample."""
    K = alpha.shape[-1]
    a0 = alpha.sum(-1)
    t1 = torch.lgamma(a0) - torch.lgamma(alpha).sum(-1)
    t1 = t1 - torch.lgamma(torch.tensor(float(K), device=alpha.device))
    t2 = ((alpha - 1.0) *
          (torch.digamma(alpha) - torch.digamma(a0).unsqueeze(-1))).sum(-1)
    return t1 + t2

def logits_to_alpha(logits, clamp: float = 15.0, eps: float = 1e-6):
    # EXP parameterization (DAEDL). Clamp guards fresh-head BERT logits
    # from exp overflow in epoch 0; CIFAR didn't need it, BERT does.
    return torch.exp(logits.clamp(-clamp, clamp)) + eps

def edl_loss(logits, targets_onehot, epoch: int, cfg) -> torch.Tensor:
    alpha = logits_to_alpha(logits, cfg.logit_clamp, cfg.alpha_eps)
    a0 = alpha.sum(-1, keepdim=True)
    nll = (targets_onehot * (torch.digamma(a0) - torch.digamma(alpha))).sum(-1)
    lam = min(1.0, (epoch + 1) / cfg.kl_anneal_epochs) * cfg.reg_param
    alpha_tilde = targets_onehot + (1.0 - targets_onehot) * alpha
    return (nll + lam * dirichlet_kl_to_uniform(alpha_tilde)).mean()
```

### 5.5 `src/bert_daedl/density/gda.py`

```python
"""Gaussian discriminant density over encoder features (float64, CPU).

NLP-specific choices (see plan §3.1): tied covariance default,
center -> PCA(k) -> whiten preprocessing, jitter escalation.
"""
from __future__ import annotations
import numpy as np
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
        self.T_ = V[:, :k] / np.sqrt(w[:k]) if self.cfg.whiten else V[:, :k]
        Z = Xc @ self.T_
        self.k_ = k
        self.mu_ = np.stack([Z[y == c].mean(0) for c in self.classes_])

        if self.cfg.cov_mode == "tied":
            R = np.concatenate([Z[y == c] - self.mu_[i]
                                for i, c in enumerate(self.classes_)])
            S = (R.T @ R) / max(len(R) - len(self.classes_), 1)
            self.chol_, self.jitter_ = self._chol(S)
            self.logdet_ = 2.0 * np.log(np.diag(self.chol_)).sum()
        else:                                   # per_class (QDA) — ablation only
            self.chol_, self.logdet_, self.jitter_ = [], [], []
            for i, c in enumerate(self.classes_):
                Zc = Z[y == c] - self.mu_[i]
                S = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
                L, j = self._chol(S)
                self.chol_.append(L); self.jitter_.append(j)
                self.logdet_.append(2.0 * np.log(np.diag(L)).sum())
        return self

    def _chol(self, S):
        for j in self.cfg.jitter_grid:
            try:
                return np.linalg.cholesky(S + j * np.eye(S.shape[0])), j
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError(
            "Covariance not PD at max jitter — reduce pca_dim or check features.")

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
        return logsumexp(lp, axis=1) - np.log(lp.shape[1])   # uniform prior


# ---------------- normalizers (fit on TRAIN log-densities only) --------------
class MinMaxNorm:                       # reference-repo behavior; keep for comparison
    def fit(self, lp):
        self.lo, self.hi = float(lp.min()), float(lp.max()); return self
    def __call__(self, lp):
        return np.clip((lp - self.lo) / (self.hi - self.lo + 1e-12), 0.0, 1.0)

class QuantileSigmoidNorm:              # recommended for combine_mode="mul"
    def __init__(self, center_q=0.05, scale_q=0.50):
        self.center_q, self.scale_q = center_q, scale_q
    def fit(self, lp):
        self.c = float(np.quantile(lp, self.center_q))
        self.s = (float(np.quantile(lp, self.scale_q)) - self.c) / 2.0 + 1e-12
        return self
    def __call__(self, lp):
        return 1.0 / (1.0 + np.exp(-(lp - self.c) / self.s))

class ECDFNorm:                         # recommended for combine_mode="add_log"
    def fit(self, lp):
        self.sorted_ = np.sort(np.asarray(lp, np.float64)); return self
    def __call__(self, lp):
        r = np.searchsorted(self.sorted_, lp, side="right") / len(self.sorted_)
        return np.clip(r, 1.0 / len(self.sorted_), 1.0)

NORMALIZERS = {"minmax": MinMaxNorm, "qsigmoid": QuantileSigmoidNorm, "ecdf": ECDFNorm}

# ---------------- density -> alpha combination -------------------------------
import torch

def scale_alpha(logits: torch.Tensor, phat: torch.Tensor,
                mode: str = "mul", lam: float = 5.0, clamp: float = 15.0):
    z = logits.clamp(-clamp, clamp)
    p = phat.to(z.dtype).unsqueeze(-1)
    if mode == "mul":            # DAEDL-faithful: can flip predictions
        return torch.exp(z * p)
    if mode == "add_log":        # prediction-preserving: scales alpha0 only
        return torch.exp(z + lam * (p - 1.0))
    raise ValueError(mode)
```

### 5.6 `src/bert_daedl/density/diagnostics.py`

Port of your `verify_gda.py`, plus the **materiality gate** that would have caught the CIFAR inert-density issue automatically. Emit `diagnostics.json` next to every `results.json`.

```python
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def density_report(gda, lp_train, lp_id_val, lp_oos_val,
                   alpha_plain=None, alpha_scaled=None) -> dict:
    rep = {
        "cov_mode": gda.cfg.cov_mode, "pca_dim": gda.k_,
        "jitter": gda.jitter_ if np.isscalar(gda.jitter_) else list(gda.jitter_),
        "raw_logp": {
            "train_q": np.quantile(lp_train, [.01, .05, .5, .95, .99]).tolist(),
            "id_val_median": float(np.median(lp_id_val)),
            "oos_val_median": float(np.median(lp_oos_val)),
        },
        # Can raw density alone separate, before any normalization?
        "raw_density_val_auroc": float(roc_auc_score(
            np.r_[np.zeros(len(lp_id_val)), np.ones(len(lp_oos_val))],
            np.r_[-lp_id_val, -lp_oos_val])),
    }
    if alpha_plain is not None and alpha_scaled is not None:
        a0p = alpha_plain.sum(-1).cpu().numpy()
        a0s = alpha_scaled.sum(-1).cpu().numpy()
        flip = (alpha_plain.argmax(-1) != alpha_scaled.argmax(-1)).float().mean().item()
        rel = np.median(np.abs(a0s - a0p) / (a0p + 1e-12))
        rho = float(spearmanr(a0p, a0s).statistic)
        rep["materiality"] = {
            "flip_rate": flip,
            "median_rel_delta_alpha0": float(rel),
            "spearman_alpha0": rho,
            # INERT == the CIFAR failure mode: scaled alpha ranks identically
            "density_effect": "INERT" if (rho >= 0.999 and rel < 0.05)
                              else "ACTIVE",
        }
    return rep
```

### 5.7 `src/bert_daedl/uncertainty/scores.py`

```python
import torch

# CONVENTION (unit-tested): every score is oriented HIGHER = MORE OOS.
# eval treats OOS as the positive class everywhere.

def scores_from_logits(logits: torch.Tensor) -> dict:
    p = logits.softmax(-1)
    return {
        "msp": 1.0 - p.max(-1).values,
        "entropy": -(p * (p + 1e-12).log()).sum(-1),
        "energy": -torch.logsumexp(logits, -1),
    }

def scores_from_alpha(alpha: torch.Tensor, prefix: str = "") -> dict:
    K = alpha.shape[-1]
    a0 = alpha.sum(-1)
    pbar = alpha / a0.unsqueeze(-1)
    pred_H = -(pbar * (pbar + 1e-12).log()).sum(-1)
    exp_H = -(pbar * (torch.digamma(alpha + 1)
                      - torch.digamma(a0 + 1).unsqueeze(-1))).sum(-1)
    lnB = torch.lgamma(alpha).sum(-1) - torch.lgamma(a0)
    diff_H = (lnB + (a0 - K) * torch.digamma(a0)
              - ((alpha - 1) * torch.digamma(alpha)).sum(-1))
    out = {
        "vacuity": K / a0,
        "one_minus_maxpbar": 1.0 - pbar.max(-1).values,
        "pred_entropy": pred_H,
        "mutual_info": pred_H - exp_H,
        "diff_entropy": diff_H,
        "neg_alpha0": -a0,
    }
    return {prefix + k: v for k, v in out.items()}

def scores_from_density(marginal_logp) -> dict:
    import numpy as np
    return {"neg_logp": -np.asarray(marginal_logp)}
```

### 5.8 `src/bert_daedl/eval/` — metrics, protocol, schema

`metrics.py`:

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def ood_metrics(unc_id: np.ndarray, unc_oos: np.ndarray) -> dict:
    y = np.r_[np.zeros(len(unc_id)), np.ones(len(unc_oos))]
    s = np.r_[unc_id, unc_oos]
    thr = np.quantile(unc_oos, 0.05)          # keep 95% of OOS above thr
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),   # OOS positive; base rate 1000/5500
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
```

`protocol.py` (single entrypoint; one forward pass per split, everything downstream reuses cached logits/features):

```python
import numpy as np, torch
from ..losses.edl import logits_to_alpha
from ..uncertainty.scores import scores_from_logits, scores_from_alpha, scores_from_density
from ..density.gda import scale_alpha
from .metrics import ood_metrics, misclf_metrics, ece

@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    L, F, Y = [], [], []
    for b in loader:
        ids = b["input_ids"].to(device); am = b["attention_mask"].to(device)
        tt = b.get("token_type_ids"); tt = tt.to(device) if tt is not None else None
        logits, feats = model(ids, am, tt, return_features=True)
        L.append(logits.cpu()); F.append(feats.cpu()); Y.append(b["labels"])
    return torch.cat(L), torch.cat(F), torch.cat(Y)

def evaluate_arm(model, loaders, cfg, gda=None, normalizer=None) -> dict:
    dev = cfg.device
    (tr, va, te, ova, ote) = loaders          # train excluded here; gda pre-fit
    Lv, Fv, Yv = collect(model, va, dev)
    Lt, Ft, Yt = collect(model, te, dev)
    Lov, Fov, _ = collect(model, ova, dev)
    Lot, Fot, _ = collect(model, ote, dev)

    def score_split(logits, feats):
        s = scores_from_logits(logits)
        if cfg.arm in ("edl", "daedl"):
            alpha = logits_to_alpha(logits, cfg.logit_clamp, cfg.alpha_eps)
            s |= scores_from_alpha(alpha)
        if gda is not None:
            lp = gda.marginal_log_prob(feats.numpy())
            s |= scores_from_density(lp)
            if cfg.arm == "daedl":
                phat = torch.from_numpy(normalizer(lp))
                a_s = scale_alpha(logits, phat, cfg.combine_mode, cfg.lam,
                                  cfg.logit_clamp)
                s |= scores_from_alpha(a_s, prefix="daedl_")
        return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in s.items()}

    S_val, S_test = score_split(Lv, Fv), score_split(Lt, Ft)
    S_oval, S_otest = score_split(Lov, Fov), score_split(Lot, Fot)

    pred = Lt.argmax(-1); correct = (pred == Yt).numpy()
    conf = Lt.softmax(-1).max(-1).values.numpy()

    res = {"in_scope_acc": float(correct.mean()), "ece": ece(conf, correct),
           "per_score": {}}
    primary = cfg.primary_score if hasattr(cfg, "primary_score") else \
              ("daedl_vacuity" if cfg.arm == "daedl"
               else "vacuity" if cfg.arm == "edl" else "msp")
    for name in S_test:
        res["per_score"][name] = {
            **ood_metrics(S_test[name], S_otest[name]),
            **misclf_metrics(S_test[name], correct),
        }
    # Larson-style thresholded operating point, threshold from VAL only:
    thr = float(np.quantile(S_val[primary], 0.95))       # accept 95% of ID val
    res["thresholded"] = {
        "score": primary, "threshold": thr,
        "oos_recall": float((S_otest[primary] > thr).mean()),
        "id_reject_rate": float((S_test[primary] > thr).mean()),
    }
    return res
```

`results.json` schema (enforced by `utils/io.save_results`):

```json
{
  "run_id": "edl_seed0_2026-07-18T09-11",
  "arm": "edl", "seed": 0,
  "git_sha": "abc1234", "config_hash": "9f3e...", "config": { "...": "full dump" },
  "in_scope_acc": 0.966, "ece": 0.031,
  "per_score": { "vacuity": {"auroc": 0.958, "aupr": 0.87, "fpr_at_95tpr": 0.21,
                              "err_auroc": 0.89, "err_aupr": 0.31, "succ_aupr": 0.995 } },
  "thresholded": { "score": "vacuity", "threshold": 1.83,
                    "oos_recall": 0.71, "id_reject_rate": 0.05 },
  "diagnostics_file": "diagnostics.json", "wall_clock_sec": 1440
}
```

Naming is standardized: **OOS** everywhere (this schema supersedes the `"OOD AUROC"`/`"OOS AUROC"` drift in your current files). AUPR here uses OOS-positive with the natural 1000/4500 base rate — do not compare raw AUPR against papers using different base rates; AUROC and FPR@95 are the cross-paper-comparable numbers.

### 5.9 `src/bert_daedl/train.py`

```python
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from .losses.edl import edl_loss
from .data.clinc150 import labels_to_one_hot

def build_optimizer(model, cfg):
    enc, head = [], []
    for n, p in model.named_parameters():
        (enc if n.startswith("encoder") else head).append(p)
    opt = AdamW([{"params": enc, "lr": cfg.lr_encoder},
                 {"params": head, "lr": cfg.lr_head}],
                weight_decay=cfg.weight_decay)
    return opt

def train_model(model, train_loader, val_loader, oos_val_loader, cfg, num_classes):
    dev = cfg.device; model.to(dev)
    opt = build_optimizer(model, cfg)
    total = len(train_loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * total), total)
    ce = torch.nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None

    for epoch in range(cfg.epochs):
        model.train()
        for b in train_loader:
            ids = b["input_ids"].to(dev); am = b["attention_mask"].to(dev)
            tt = b.get("token_type_ids"); tt = tt.to(dev) if tt is not None else None
            y = b["labels"].to(dev)
            logits = model(ids, am, tt)
            loss = (ce(logits, y) if cfg.arm == "ce"
                    else edl_loss(logits, labels_to_one_hot(y, num_classes),
                                  epoch, cfg))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()

        acc, vac_gap = quick_val(model, val_loader, oos_val_loader, cfg)
        print(f"epoch {epoch}: val_acc={acc:.4f} vacuity_gap(oos-id)={vac_gap:+.3f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc
```

`quick_val` computes ID-val accuracy plus `mean vacuity(oos_val) − mean vacuity(id_val)` (CE arm: entropy gap instead). A negative gap after epoch 1 means the uncertainty signal is broken — fail fast, don't wait for final eval.

### 5.10 `scripts/run_experiment.py` — replaces `bert_daedl.py`

```python
import argparse, dataclasses, time
import numpy as np, torch
from bert_daedl.config import ExperimentConfig
from bert_daedl.data.clinc150 import CLINC150Config, build_clinc150_dataloaders
from bert_daedl.models.bert_edl import BertEvidentialClassifier
from bert_daedl.train import train_model
from bert_daedl.eval.protocol import collect, evaluate_arm
from bert_daedl.density.gda import GaussianDensityModel, NORMALIZERS
from bert_daedl.density.diagnostics import density_report
from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.density.gda import scale_alpha
from bert_daedl.utils.seed import seed_everything
from bert_daedl.utils.io import save_results, save_json

def main(cfg: ExperimentConfig):
    seed_everything(cfg.seed)
    data_cfg = CLINC150Config(model_name=cfg.model_name, max_length=cfg.max_length,
                              batch_size=cfg.batch_size,
                              eval_batch_size=cfg.eval_batch_size,
                              num_workers=cfg.num_workers, seed=cfg.seed)
    loaders = build_clinc150_dataloaders(data_cfg)      # 6-tuple, see §1.1
    tr, va, te, ova, ote, meta = loaders
    model = BertEvidentialClassifier(
        cfg.model_name, meta["num_classes"], cfg.feat_dim,
        cfg.sn_bottleneck, cfg.bottleneck_activation, cfg.pooling, cfg.dropout)

    t0 = time.time()
    model, best_val_acc = train_model(model, tr, va, ova, cfg, meta["num_classes"])

    # GDA on TRAIN features (every arm -> Mahalanobis baseline is free)
    Ltr, Ftr, Ytr = collect(model, tr, cfg.device)
    gda = GaussianDensityModel(cfg.gda).fit(Ftr.numpy(), Ytr.numpy())
    lp_train = gda.marginal_log_prob(Ftr.numpy())

    normalizer = None
    diag_alpha = (None, None)
    if cfg.arm == "daedl":
        normalizer = NORMALIZERS[cfg.normalizer]().fit(lp_train)
        Lv, Fv, _ = collect(model, va, cfg.device)
        phat_v = torch.from_numpy(normalizer(gda.marginal_log_prob(Fv.numpy())))
        a_plain = logits_to_alpha(Lv, cfg.logit_clamp, cfg.alpha_eps)
        a_scaled = scale_alpha(Lv, phat_v, cfg.combine_mode, cfg.lam, cfg.logit_clamp)
        diag_alpha = (a_plain, a_scaled)

    _, Fv, _ = collect(model, va, cfg.device)
    _, Fov, _ = collect(model, ova, cfg.device)
    diag = density_report(gda, lp_train,
                          gda.marginal_log_prob(Fv.numpy()),
                          gda.marginal_log_prob(Fov.numpy()),
                          *diag_alpha)
    save_json(cfg.output_dir, "diagnostics.json", diag)
    if diag.get("materiality", {}).get("density_effect") == "INERT":
        print("WARNING: density term is INERT under current normalizer "
              "(CIFAR failure mode). See diagnostics.json before trusting daedl_* scores.")

    results = evaluate_arm(model, (tr, va, te, ova, ote), cfg, gda, normalizer)
    results |= {"arm": cfg.arm, "seed": cfg.seed, "best_val_acc": best_val_acc,
                "wall_clock_sec": round(time.time() - t0, 1),
                "config": dataclasses.asdict(cfg)}
    path = save_results(cfg.output_dir, results)
    print(f"Results -> {path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["ce", "edl", "daedl"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--normalizer", default="qsigmoid")
    p.add_argument("--combine_mode", default="mul", choices=["mul", "add_log"])
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="results")
    a = p.parse_args()
    cfg = ExperimentConfig(arm=a.arm, seed=a.seed, epochs=a.epochs,
                           normalizer=a.normalizer, combine_mode=a.combine_mode,
                           lam=a.lam, device=a.device,
                           output_dir=f"{a.output_dir}/{a.arm}_seed{a.seed}")
    main(cfg)
```

### 5.11 `scripts/aggregate_results.py`

~30 lines: glob `results/**/results.json`, group by `(arm, normalizer, combine_mode)`, emit `results/summary.csv` and a markdown table of mean ± std over seeds for `in_scope_acc` and each score's `auroc`/`fpr_at_95tpr`. Every number in the writeup comes from this file, never from a single run's stdout.

### 5.12 `tests/` — what must be pinned

```python
# test_data.py
def test_splits(loaders):              # sizes per §1.1; no label==-1 in train/val/test;
    ...                                # labels in [0,150); -1 only in oos loaders

# test_scores.py  — kills the orientation-bug class
def test_orientation():
    conf  = torch.tensor([[8.0, 0.1, 0.1]])   # confident logits
    unsure = torch.tensor([[0.4, 0.3, 0.3]])
    for k, v in scores_from_logits(unsure).items():
        assert v > scores_from_logits(conf)[k], k          # higher = more OOS
    a_conf, a_un = torch.exp(conf), torch.exp(unsure)
    for k in scores_from_alpha(a_un):
        assert scores_from_alpha(a_un)[k] > scores_from_alpha(a_conf)[k], k

def test_vacuity_by_hand():            # K=3, alpha=(2,1,1) -> vacuity = 3/4
    ...

# test_gda.py — synthetic 2-Gaussian: OOD point far from both means gets
# lower marginal_log_prob than ID points; tied & per_class both fit PD.

# test_edl_loss.py — loss finite at logits=±20 (clamp works); loss decreases
# over 50 steps overfitting one batch; alpha > 0 everywhere.

# test_smoke.py — 200 train steps on a 512-example subset, arm=edl, cpu:
# runs end-to-end and writes a schema-valid results.json + diagnostics.json.
```

`Makefile`:

```make
test:      ; pytest -q
ce:        ; python scripts/run_experiment.py --arm ce   --seed $(SEED)
edl:       ; python scripts/run_experiment.py --arm edl  --seed $(SEED)
daedl:     ; python scripts/run_experiment.py --arm daedl --seed $(SEED) \
              --normalizer $(NORM) --combine_mode $(MODE)
seeds:     ; for s in 0 1 2 3 4; do \
              python scripts/run_experiment.py --arm $(ARM) --seed $$s; done
aggregate: ; python scripts/aggregate_results.py results/
```

---

## 6. Iteration design — phases, gates, decision rules, kill criteria

Rule for the whole milestone: **all selection happens on `id_val` + `oos_val`; `id_test` + `oos_test` are touched only by the final `evaluate_arm` call of a chosen configuration.** 3 seeds during selection, 5 seeds for finals.

### Phase A — harness + CE baseline (Jul 17–23)

Do: §1.1 pipeline revisions, model, train loop, protocol, tests, CE arm × 5 seeds.
**Exit gates:**
- All tests green; smoke test produces schema-valid JSON.
- CE in-scope test acc (5-seed mean) **≥ 0.960**. External anchor: fully fine-tuned BERT-base on CLINC150 in-scope is ≈0.965–0.97 in the literature (also cross-check the SNGP paper's deterministic-BERT row when writing up).
- Record `msp`, `energy`, `neg_logp` (Mahalanobis) AUROCs — these are your house baselines; every later claim is relative to them.

**If acc < 0.95**, debug in this order before touching anything else: (1) label-map consistency train↔eval, (2) pooling (`cls`→`mean`), (3) head LR (1e-3→5e-4), (4) epochs 5→10. Do not proceed to Phase B on a broken baseline — every downstream number would be uninterpretable.

### Phase B — vanilla EDL arm (Jul 24–29)

Do: `edl` arm × 3 seeds with CIFAR-mirrored loss.
**Exit gates:**
- In-scope acc within **1.5 pts** of CE mean.
- `vacuity_gap(oos_val − id_val) > 0` from epoch 1 (else it's a bug: check score orientation tests, one-hot construction, KL target `alpha_tilde`).
- `vacuity` val AUROC ≥ `msp` val AUROC − 0.01 (EDL is allowed to merely match MSP here; the density phase is where lift is claimed).

**Decision rule on accuracy gap:** if gap > 1.5 pts, try at most three configs in one day — `reg_param/10`, `lr_head/3`, `kl_anneal_epochs=2×epochs` — then keep the best and move on, reporting the gap honestly. Do not spend the week tuning EDL accuracy; DAEDL's own motivation is that EDL costs accuracy and EXP parameterization mitigates it.

### Phase C — density (Jul 30 – Aug 7, hard timebox; kill decision Aug 5)

Three nested selection loops, each consuming the previous loop's winner, all judged on **val**:

**C1 — representation (1–2 days).** Grid: `pca_dim ∈ {None, 128, 64}` × `cov_mode ∈ {tied, per_class(k≤64 only)}`, whiten=True. Selector: `raw_density_val_auroc` from `diagnostics.json` — *raw* log-density separation, before any normalizer, so normalization can't mask a dead representation.
Gate: best raw-density val AUROC ≥ 0.85 → continue. Between 0.75–0.85 → continue but flag expectations (near-OOD, §3.3). **< 0.75 → density cannot help; skip to the kill branch.**

**C2 — normalizer × combination (1–2 days).** Grid: `{minmax, qsigmoid, ecdf}` × `{mul, add_log(λ ∈ {2, 5, 10})}` on the C1 winner. Selector: `daedl_vacuity` val AUROC. **Materiality constraint:** configurations whose `diagnostics.json` says `density_effect: INERT` are disqualified *even if their AUROC looks fine* — an inert config's AUROC is just vanilla EDL's AUROC wearing a costume, which is precisely the CIFAR trap.

**C3 — inclusion gate (final, 5 seeds).** DAEDL enters the headline table iff
`mean(daedl val AUROC) − mean(edl val AUROC) > pooled std across 5 seeds`.
Then, and only then, run the single test-set evaluation for the writeup table.

**Kill branch (triggered by C1 < 0.75 or C3 failure or the Aug 5 clock):** headline becomes CE+MSP / CE+Mahalanobis / EDL; DAEDL is written up as a rigorously-diagnosed negative result — "the density term is inert/ineffective on near-OOD intent detection under X, Y, Z normalizations; diagnostics attribute this to …" with the `diagnostics.json` numbers. This mirrors your CIFAR finding and is a *strength* of the writeup, not a failure: you will have shown the same forensic discipline twice, on two modalities.

### Phase D — optional SN ablations (Aug 5–8, only if C finished early)

`sn_bottleneck ∈ {on, off}` × best arm, 3 seeds. Stretch: SNGP-style SN on encoder FFN dense layers — pre-declared as experimental; abandon at the first sign of accuracy collapse.

### Phase E — early-exit (Milestone 2, Aug 8–24; **not in this milestone**)

Reserved interfaces already in place: `return_hidden_states=True` passthrough in the model, empty `models/exit_heads.py` and `eval/exit_calibration.py`, and — the reason `oos_val` had to exist — threshold calibration for exit gating will use `id_val`+`oos_val` exactly like the rejection threshold in §5.8. Design sketch to hold in mind (do not build yet): exit heads at layers {3, 6, 9, 12}, per-exit EDL losses (weighted sum), gate on per-exit vacuity, report exit distribution + FLOPs saved + the compute–accuracy–OOS-safety Pareto curve. That curve is the writeup's centerpiece; everything in Milestone 1 exists to make it trustworthy.

### Calendar reality check

You are ~2 weeks behind the June–July sketch we made, which is absorbed as follows: Milestone 1 compresses into Jul 17–Aug 7 with the C-phase timebox doing the compressing; early-exit runs Aug 8–24; writeup drafting starts in parallel ~Aug 15; preprint/blog + repo public by ~Aug 31. Applications proceed from early August with the Milestone-1 results table as the in-progress artifact — a clean 3-arm table with diagnostics is already presentable in interviews.

---

## 7. Milestone acceptance checklist

- [ ] Pipeline returns 6-tuple incl. `oos_val_loader`; split-size validation logs clean; `include_oos_in_train` removed.
- [ ] `pytest -q` green, including orientation tests and GDA synthetic test.
- [ ] CE arm, 5 seeds: in-scope acc ≥ 0.960 mean; `summary.csv` generated by aggregator.
- [ ] EDL arm, 5 seeds: gates of Phase B met (or deviations documented in results).
- [ ] `diagnostics.json` produced for every run; `density_effect` field present; no INERT config in any headline number.
- [ ] DAEDL either passes C3 across 5 seeds or is documented as a negative result with diagnostics attached.
- [ ] Every number in the draft table traces to a `results.json` with `git_sha` + `config_hash`.
- [ ] `DAEDL/` folder untouched (reference stays reproducible).
