# src/models/phase2.py
# Epistemic Ranker — Phase 2  (fully fixed)
#
# Bugs fixed vs previous version:
#   FIX-1  OOD eval collapse (n=4): OOD rows all shared query_id=-1 →
#          collapsed to 1 quad group → only 4 samples reached profile_uncertainty.
#          Solution: dedicated OodFlatDataset + ood_eval_loader bypasses quad grouping.
#   FIX-2  Filename mismatch: phase2 looked for train_pairs_clean.parquet but
#          mine.py produces train_pairs.parquet.
#   FIX-3  Vacuity loss: ood_alpha.mean() doesn't target S≈2.
#          Replaced with (S_ood - K)² to explicitly minimise distance from vacuity.
#   FIX-4  OOD shares label=0 → Brier loss fought vacuity loss on same samples.
#          OOD rows are now excluded from Brier/KL; only vacuity loss applies to them.
#   FIX-5  Label imbalance: easy_negatives provided 2× gradient signal vs positives.
#          Per-sample loss weighting: positive×4, hard_neg×2, easy_neg×0.5.
#   FIX-6  S-explosion on easy_negatives (S=30): no ceiling on total strength.
#          Added S-regularisation term capping easy_neg at S_TARGET=10.
#   FIX-7  Insufficient BERT unfreezing: only layer 11 → couldn't learn fine-grained
#          relevance boundary for hard negatives.
#          Unfrozen layers 9–11 with layerwise learning-rate decay.
#
# Dataset size check (from mine.py output):
#   positive:      31,112  (1 per query)
#   hard_negative: 31,105  (≈1 per query; ~7 queries had no hard neg → padded)
#   easy_negative: 100,000 (2 per query used in quads; 37,776 discarded — by design)
#   Ratio enforced per batch: 1:1:2 ✓  (25% / 25% / 50% of sequences)
#
# Run: python -m src.models.phase2

import os
import json
import random
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, set_seed

# ==========================================
# 1. Environment & Paths
# ==========================================
def detect_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    return here.parent


REPO_ROOT   = detect_repo_root()
DATA_DIR    = REPO_ROOT / "data"
PROCESSED   = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports" / "phase2"

NUM_CLASSES      = 2
BATCH_SIZE       = 32        # 32 quads × 4 seqs = 128 sequences/forward pass
MAX_LENGTH       = 128
EPOCHS           = 5

# Optimizer — layerwise LR decay across unfrozen BERT layers (FIX-7)
LR_HEAD          = 5e-4
LR_LAYER11       = 1e-5
LR_LAYER10       = 5e-6
LR_LAYER9        = 1e-6
WEIGHT_DECAY     = 1e-3

LABEL_SMOOTH_EPS = 0.02
OOD_EXPOSE_EVERY = 3
OOD_LOSS_WEIGHT  = 0.5      # Increased from 0.1 — vacuity signal needs more weight (FIX-3)
KL_WEIGHT        = 0.4
S_REG_WEIGHT     = 0.01     # S-regularisation weight (FIX-6)
S_TARGET         = 10.0     # Allowed S ceiling for easy_negatives (FIX-6)

# Per-sample loss weights to counterbalance 1:1:2 gradient imbalance (FIX-5)
PAIR_WEIGHTS = {
    "positive":      4.0,
    "hard_negative": 2.0,
    "easy_negative": 0.5,
}


def setup_environment():
    set_seed(42)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Environment setup complete. Seed=42.")
    print(f"Python executable: {sys.executable}")
    print(f"Processed data dir: {PROCESSED}")


# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ==========================================
# 2. Data Pipeline
# ==========================================
class EpicRankerQuadDataset(Dataset):
    """
    In-distribution only (positive / hard_negative / easy_negative).
    One index = one query's quad of 4 rows in canonical 1:1:2 order.

    OOD rows must NOT be passed here — they all share query_id=-1 and
    would collapse into a single group of 4 repeated samples. (FIX-1)
    """
    TYPE_ORDER = ["positive", "hard_negative", "easy_negative", "easy_negative"]

    def __init__(self, parquet_path: Path, tokenizer, max_length: int = 128):
        df = pd.read_parquet(parquet_path)

        # FIX-1: strip OOD rows unconditionally — they must never enter quad grouping
        self.df        = df[df["pair_type"] != "ood"].copy().reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

        self.grouped   = self.df.groupby("query_id")
        self.query_ids = list(self.grouped.groups.keys())

        counts = self.df["pair_type"].value_counts().to_dict()
        print(f"[QuadDataset] {parquet_path.name}: {counts}")

    def __len__(self) -> int:
        return len(self.query_ids)

    def _get_row_for_type(self, group_df: pd.DataFrame, pair_type: str) -> pd.Series:
        subset = group_df[group_df["pair_type"] == pair_type]
        return subset.iloc[0] if len(subset) > 0 else group_df.iloc[-1]

    def __getitem__(self, idx: int) -> dict:
        qid      = self.query_ids[idx]
        group_df = self.grouped.get_group(qid)

        seen_easy = 0
        rows = []
        for t in self.TYPE_ORDER:
            if t == "easy_negative":
                easy_rows = group_df[group_df["pair_type"] == "easy_negative"]
                rows.append(
                    easy_rows.iloc[seen_easy] if len(easy_rows) > seen_easy
                    else group_df.iloc[-1]
                )
                seen_easy += 1
            else:
                rows.append(self._get_row_for_type(group_df, t))

        quad_df = pd.DataFrame(rows)
        enc = self.tokenizer(
            quad_df["query_text"].tolist(),
            quad_df["passage_text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
            "labels":         torch.tensor(
                                  quad_df["label"].astype(int).values,
                                  dtype=torch.long),
            "pair_types":     quad_df["pair_type"].tolist(),
        }


class OodFlatDataset(Dataset):
    """
    FIX-1: Flat dataset for OOD rows — no query_id grouping.
    Each OOD row becomes one independent sample.
    Used for both:
      - training vacuity exposure (ood_slice.parquet, all 5,000 rows)
      - evaluation profiling    (test_4bucket.parquet, 500 OOD rows)
    """
    def __init__(self, parquet_path: Path, tokenizer, max_length: int = 128):
        df = pd.read_parquet(parquet_path)
        if "pair_type" in df.columns:
            df = df[df["pair_type"] == "ood"].copy().reset_index(drop=True)

        self.passages   = df["passage_text"].fillna("").astype(str).tolist()
        # OOD rows have no meaningful query — empty string keeps token_type_ids
        # marking all tokens as passage tokens (type 1), which is correct
        self.queries    = (df["query_text"].fillna("").astype(str).tolist()
                           if "query_text" in df.columns
                           else [""] * len(self.passages))
        self.pair_types = (df["pair_type"].tolist()
                           if "pair_type" in df.columns
                           else ["ood"] * len(self.passages))
        self.tokenizer  = tokenizer
        self.max_length = max_length

        print(f"[OodFlatDataset] {parquet_path.name}: {len(self.passages)} OOD rows")

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.queries[idx],
            self.passages[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "pair_types":     self.pair_types[idx],
        }


def collate_quads(batch: list) -> dict:
    """Flatten a list of quad dicts → single batch tensors."""
    return {
        "input_ids":      torch.cat([b["input_ids"]      for b in batch], dim=0),
        "attention_mask": torch.cat([b["attention_mask"]  for b in batch], dim=0),
        "token_type_ids": torch.cat([b["token_type_ids"]  for b in batch], dim=0),
        "labels":         torch.cat([b["labels"]          for b in batch], dim=0),
        "pair_types":     [pt for b in batch for pt in b["pair_types"]],
    }


def collate_flat(batch: list) -> dict:
    """Collate for OodFlatDataset — simple stack, no flattening needed."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"]  for b in batch]),
        "pair_types":     [b["pair_types"] for b in batch],
    }


def ensure_phase2_inputs() -> tuple:
    # FIX-2: correct filename — mine.py produces train_pairs.parquet,
    # not train_pairs_clean.parquet
    train_path = PROCESSED / "train_pairs.parquet"
    test_path  = PROCESSED / "test_4bucket.parquet"
    ood_path   = PROCESSED / "ood_slice.parquet"

    missing = [p for p in [train_path, ood_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files: {[str(p) for p in missing]}. "
            "Run src/data/mine.py first."
        )

    if not test_path.exists():
        print(f"Rebuilding {test_path.name} from processed data...")
        df_pairs = pd.read_parquet(train_path)
        df_ood   = pd.read_parquet(ood_path)

        for frame in (df_pairs, df_ood):
            for col in ("query_text", "passage_text"):
                if col in frame.columns:
                    frame[col] = frame[col].fillna("").astype(str)
            if "passage_id" in frame.columns:
                frame["passage_id"] = frame["passage_id"].astype(str)
            if "pair_type" in frame.columns:
                frame["pair_type"] = frame["pair_type"].astype(str)

        # In-distribution: 500 rows per pair_type
        indist_sample = (
            df_pairs[df_pairs["pair_type"] != "ood"]
            .groupby("pair_type", group_keys=False)
            .head(500)
        )

        # FIX-1: Give each OOD row a unique negative query_id so they are
        # treated as independent samples and not collapsed into one quad group.
        ood_sample = df_ood.head(500).copy()
        ood_sample["query_id"] = range(-1, -len(ood_sample) - 1, -1)  # -1, -2, …, -500

        test_4bucket = pd.concat([indist_sample, ood_sample], ignore_index=True)
        test_4bucket.to_parquet(test_path, index=False, compression="snappy")

        counts = test_4bucket["pair_type"].value_counts().to_dict()
        print(f"Created test_4bucket.parquet: {counts}")

    return train_path, test_path, ood_path


# ==========================================
# 3. Model Architecture
# ==========================================
class EvidentialHead(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.register_buffer("temperature", torch.ones(1))

    def set_temperature(self, temp: float):
        if temp <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature.fill_(float(temp))

    def forward(self, features: torch.Tensor, temp=None):
        t        = temp if temp is not None else self.temperature
        evidence = F.softplus(self.linear(features) / t)
        alpha    = evidence + 1.0
        return evidence, alpha


class BertWithEvidentialHead(nn.Module):
    """
    BERT Cross-Encoder with binary Evidential head.
    token_type_ids passed through for query/passage distinction.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()
        self.head = EvidentialHead(768, num_classes)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, temp=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.head(cls, temp=temp)


# ==========================================
# 4. Loss Functions
# ==========================================
def get_expected_probs(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)


def get_smoothed_one_hot(labels: torch.Tensor, num_classes: int = 2,
                         epsilon: float = 0.02) -> torch.Tensor:
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return (1 - epsilon) * one_hot + (epsilon / num_classes)


def brier_score_loss(alpha: torch.Tensor,
                     target_or_onehot: torch.Tensor,
                     num_classes: int = 2,
                     sample_weights: torch.Tensor = None) -> torch.Tensor:
    """
    Brier Score with optional per-sample weights (FIX-5).
    Accepts integer labels (1-D) or pre-smoothed one-hot (2-D).
    """
    p_hat = get_expected_probs(alpha)
    if target_or_onehot.dim() == 1:
        one_hot = torch.zeros(target_or_onehot.size(0), num_classes,
                              device=target_or_onehot.device)
        one_hot.scatter_(1, target_or_onehot.unsqueeze(1), 1.0)
    else:
        one_hot = target_or_onehot

    per_sample = torch.sum((p_hat - one_hot) ** 2, dim=1)
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    return per_sample.mean()


def binary_kl_penalty(alpha: torch.Tensor, labels: torch.Tensor,
                      eps: float = 1e-12) -> torch.Tensor:
    """
    Binary KL: penalise evidence on the one incorrect class only.
    Correct-class alpha is fixed at 1 (unpunished).
    """
    one_hot    = F.one_hot(labels, num_classes=NUM_CLASSES).float()
    wrong_mask = 1.0 - one_hot
    beta       = one_hot + wrong_mask * alpha
    S_beta     = beta.sum(dim=1, keepdim=True).clamp_min(eps)

    term1 = torch.lgamma(S_beta) - torch.lgamma(
        torch.tensor(float(NUM_CLASSES), device=alpha.device))
    term2 = -torch.sum(torch.lgamma(beta.clamp_min(eps)), dim=1, keepdim=True)
    term3 = torch.sum(
        (beta - 1.0) * (torch.digamma(beta.clamp_min(eps)) - torch.digamma(S_beta)),
        dim=1, keepdim=True,
    )
    return (term1 + term2 + term3).mean()


def s_regularisation_loss(alpha: torch.Tensor, pair_types: list,
                           s_target: float = S_TARGET) -> torch.Tensor:
    """
    FIX-6: Squared-hinge loss penalising S > s_target on easy_negatives.
    Prevents the Brier+KL feedback loop from driving easy_neg S to 30+.
    Hard negatives and positives are intentionally uncapped — their S
    carries meaningful aleatoric signal.
    """
    easy_mask = torch.tensor(
        [pt == "easy_negative" for pt in pair_types],
        dtype=torch.bool, device=alpha.device,
    )
    if not easy_mask.any():
        return torch.tensor(0.0, device=alpha.device)

    S_easy = alpha[easy_mask].sum(dim=1)
    excess = F.relu(S_easy - s_target)
    return excess.pow(2).mean()


def vacuity_loss_fn(ood_alpha: torch.Tensor) -> torch.Tensor:
    """
    FIX-3: Target S≈K=2 explicitly via squared-error on total strength.
    Previous implementation (ood_alpha.mean()) was too weak and did not
    impose a directional target — it only minimised alpha values globally.
    """
    S_ood = ood_alpha.sum(dim=1)
    return (S_ood - NUM_CLASSES).pow(2).mean()


# ==========================================
# 5. Metrics
# ==========================================
def base_accuracy(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (torch.argmax(get_expected_probs(alpha), dim=1) == target).float().mean()


def eru_mutual_information(alpha: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    S        = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    p_mean   = alpha / S
    pred_ent = -torch.sum(p_mean * torch.log(p_mean.clamp_min(eps)), dim=1)
    exp_ent  = -torch.sum(
        (alpha / S) * (torch.digamma(alpha + 1) - torch.digamma(S + 1)), dim=1)
    return pred_ent - exp_ent


def _ece_from_probs(probs: torch.Tensor, target: torch.Tensor,
                    n_bins: int = 15) -> torch.Tensor:
    conf, preds = probs.max(dim=1)
    acc         = preds.eq(target).float()
    bins        = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece         = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - acc[mask].mean()).abs()
    return ece.squeeze(0)


# ==========================================
# 6. Uncertainty Profiling
# ==========================================
def profile_uncertainty(all_alphas: torch.Tensor,
                        all_pair_types: list) -> dict:
    """
    Groups Dirichlet Total Strength S by pair_type.
    Expected ordering after all fixes:
      easy_negative ≈ positive (high S, confident) >>
      hard_negative (moderate S, aleatoric) >>
      ood ≈ 2.0 (near-vacuous, epistemic)
    """
    profiles = {k: [] for k in
                ("positive", "hard_negative", "easy_negative", "ood")}
    S = all_alphas.sum(dim=1)

    for i, pt in enumerate(all_pair_types):
        if pt in profiles:
            profiles[pt].append(S[i].item())

    print("\n--- Phase 2 Uncertainty Profiles (Mean Strength S) ---")
    results = {}
    for pt, vals in profiles.items():
        if vals:
            mean_s = sum(vals) / len(vals)
            results[pt] = round(mean_s, 4)
            print(f"  {pt.capitalize():>15}: {mean_s:.4f}  (n={len(vals)})")
        else:
            results[pt] = None
            print(f"  {pt.capitalize():>15}: no samples")
    return results


# ==========================================
# 7. Utility Helpers
# ==========================================
def get_ood_batch(ood_iter, ood_loader):
    try:
        batch = next(ood_iter)
    except StopIteration:
        ood_iter = iter(ood_loader)
        batch    = next(ood_iter)
    return batch, ood_iter


def quick_gradient_check(model: nn.Module):
    model.train()
    ids  = torch.randint(0, 30522, (4, MAX_LENGTH), device=device)
    mask = torch.ones_like(ids)
    tids = torch.zeros_like(ids)
    tgt  = torch.randint(0, NUM_CLASSES, (4,), device=device)

    _, alpha = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
    loss     = brier_score_loss(alpha, tgt, num_classes=NUM_CLASSES)

    assert torch.isfinite(loss), "Loss is NaN/Inf"
    model.zero_grad()
    loss.backward()

    head_grads = [p.grad for p in model.head.parameters() if p.requires_grad]
    assert all(g is not None for g in head_grads), "No gradients on head"
    assert all(torch.isfinite(g).all() for g in head_grads), "NaN/Inf in grads"
    print("Gradient check passed.")


# ==========================================
# 8. Training & Evaluation
# ==========================================
def train_and_evaluate():
    setup_environment()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_path, test_path, ood_path = ensure_phase2_inputs()

    # --- In-distribution loaders (quad structure) ---
    train_ds = EpicRankerQuadDataset(train_path, tokenizer, MAX_LENGTH)
    test_ds  = EpicRankerQuadDataset(test_path,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_quads)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_quads)

    # FIX-1: dedicated flat OOD loaders — never touch quad grouping
    ood_train_ds = OodFlatDataset(ood_path,  tokenizer, MAX_LENGTH)  # 5,000 rows
    ood_eval_ds  = OodFlatDataset(test_path, tokenizer, MAX_LENGTH)  # 500 rows

    ood_train_loader = DataLoader(ood_train_ds, batch_size=BATCH_SIZE * 4,
                                  shuffle=True,  num_workers=0,
                                  collate_fn=collate_flat)
    ood_eval_loader  = DataLoader(ood_eval_ds,  batch_size=BATCH_SIZE * 4,
                                  shuffle=False, num_workers=0,
                                  collate_fn=collate_flat)
    ood_iter = iter(ood_train_loader)

    # --- Model ---
    model = BertWithEvidentialHead(num_classes=NUM_CLASSES).to(device)

    # FIX-7: unfreeze last 3 BERT layers with layerwise LR decay
    for layer_idx in (9, 10, 11):
        for p in model.bert.encoder.layer[layer_idx].parameters():
            p.requires_grad = True
        model.bert.encoder.layer[layer_idx].train()

    model.head.set_temperature(1.0)
    quick_gradient_check(model)

    # FIX-7: four param groups
    optimizer = optim.AdamW(
        [
            {"params": model.head.parameters(),                   "lr": LR_HEAD},
            {"params": model.bert.encoder.layer[11].parameters(), "lr": LR_LAYER11},
            {"params": model.bert.encoder.layer[10].parameters(), "lr": LR_LAYER10},
            {"params": model.bert.encoder.layer[9].parameters(),  "lr": LR_LAYER9},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # --- Training Loop ---
    train_losses, val_accs, epoch_times = [], [], []

    print("\n--- Starting Phase 2 Training ---")
    print(f"    Epochs      : {EPOCHS}")
    print(f"    Batch size  : {BATCH_SIZE} quads ({BATCH_SIZE * 4} seqs/step)")
    print(f"    Unfrozen    : BERT layers 9, 10, 11")
    print(f"    Loss weights: pos={PAIR_WEIGHTS['positive']} "
          f"hard={PAIR_WEIGHTS['hard_negative']} "
          f"easy={PAIR_WEIGHTS['easy_negative']}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for layer_idx in (9, 10, 11):
            model.bert.encoder.layer[layer_idx].train()

        total_loss, total_count = 0.0, 0
        batch_idx   = 0
        epoch_start = time.time()

        for batch in train_loader:
            batch_idx  += 1
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            token_tids  = batch["token_type_ids"].to(device)
            labels      = batch["labels"].to(device)
            pair_types  = batch["pair_types"]

            one_hot = get_smoothed_one_hot(labels, NUM_CLASSES, LABEL_SMOOTH_EPS)

            # Forward — in-distribution sequences only (FIX-4)
            outputs = model.bert(input_ids=input_ids, attention_mask=attn_mask,
                                 token_type_ids=token_tids)
            cls = outputs.last_hidden_state[:, 0, :]
            _, alpha = model.head(cls)

            # FIX-5: per-sample weights
            sample_weights = torch.tensor(
                [PAIR_WEIGHTS.get(pt, 1.0) for pt in pair_types],
                dtype=torch.float32, device=device,
            )

            brier_loss = brier_score_loss(alpha, one_hot, NUM_CLASSES, sample_weights)
            kl_loss    = binary_kl_penalty(alpha, labels)

            # FIX-6: S-regularisation caps easy_negative strength
            s_reg = s_regularisation_loss(alpha, pair_types, S_TARGET)

            loss = brier_loss + KL_WEIGHT * kl_loss + S_REG_WEIGHT * s_reg

            # FIX-3 & FIX-4: OOD vacuity — separate forward pass, no Brier/KL
            if batch_idx % OOD_EXPOSE_EVERY == 0:
                ood_batch, ood_iter = get_ood_batch(ood_iter, ood_train_loader)
                _, ood_alpha = model(
                    input_ids=ood_batch["input_ids"].to(device),
                    attention_mask=ood_batch["attention_mask"].to(device),
                    token_type_ids=ood_batch["token_type_ids"].to(device),
                )
                loss = loss + OOD_LOSS_WEIGHT * vacuity_loss_fn(ood_alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item() * labels.size(0)
            total_count += labels.size(0)

        epoch_time = time.time() - epoch_start

        # Validation on in-distribution test set
        model.eval()
        va_list, vl_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                _, alpha = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                )
                va_list.append(alpha)
                vl_list.append(batch["labels"].to(device))

        val_acc = base_accuracy(torch.cat(va_list), torch.cat(vl_list)).item()
        train_losses.append(total_loss / total_count)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

    # ==========================================
    # 9. Full Evaluation Suite
    # ==========================================
    print("\n--- Running Full Phase 2 Metric Suite ---")
    model.eval()

    # In-distribution evaluation (quad loader)
    all_alphas_indist, all_labels_indist, all_pair_types_indist = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            _, alpha = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            )
            all_alphas_indist.append(alpha)
            all_labels_indist.append(batch["labels"].to(device))
            all_pair_types_indist.extend(batch["pair_types"])

    alpha_indist  = torch.cat(all_alphas_indist)
    labels_indist = torch.cat(all_labels_indist)

    # FIX-1: OOD evaluation via flat loader — all 500 samples, not 4
    all_alphas_ood, all_pair_types_ood = [], []
    with torch.no_grad():
        for batch in ood_eval_loader:
            _, alpha = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            )
            all_alphas_ood.append(alpha)
            all_pair_types_ood.extend(batch["pair_types"])

    alpha_ood = torch.cat(all_alphas_ood)

    # Standard metrics (in-distribution only)
    clean_acc    = base_accuracy(alpha_indist, labels_indist).item()
    clean_brier  = brier_score_loss(alpha_indist, labels_indist, NUM_CLASSES).item()
    clean_ece    = _ece_from_probs(get_expected_probs(alpha_indist),
                                    labels_indist, n_bins=15).item()
    clean_eru    = eru_mutual_information(alpha_indist).mean().item()
    mean_S_clean = alpha_indist.sum(dim=1).mean().item()
    mean_S_ood   = alpha_ood.sum(dim=1).mean().item()

    print("\n1. Predictive Performance / Calibration")
    print(f"   Accuracy : {clean_acc:.4f}")
    print(f"   Brier    : {clean_brier:.4f}")
    print(f"   ECE      : {clean_ece:.4f}")

    print("\n2. Epistemic Uncertainty Proxy")
    print(f"   ERU / MI (clean mean)  : {clean_eru:.4f}")
    print(f"   Mean Strength S (clean): {mean_S_clean:.2f}")
    print(f"   Mean Strength S (OOD)  : {mean_S_ood:.2f}  (n={len(all_pair_types_ood)})")

    # Combine in-distribution + OOD for unified uncertainty profile
    all_alphas_combined    = torch.cat([alpha_indist, alpha_ood])
    all_pair_types_combined = all_pair_types_indist + all_pair_types_ood

    uncertainty_profiles = profile_uncertainty(all_alphas_combined,
                                               all_pair_types_combined)

    results = {
        "accuracy":             round(clean_acc,    4),
        "brier":                round(clean_brier,  4),
        "ece":                  round(clean_ece,    4),
        "eru_mi":               round(clean_eru,    4),
        "mean_S_clean":         round(mean_S_clean, 4),
        "mean_S_ood":           round(mean_S_ood,   4),
        "ood_eval_n":           len(all_pair_types_ood),
        "uncertainty_profiles": uncertainty_profiles,
        "train_losses":         train_losses,
        "val_accs":             val_accs,
        "epoch_times":          epoch_times,
    }
    out_path = REPORTS_DIR / "phase2_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    train_and_evaluate()