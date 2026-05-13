from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DAEDLConfig


class EpicRankerQuadDataset(Dataset):
    TYPE_ORDER = ["positive", "hard_negative", "easy_negative", "easy_negative"]

    def __init__(self, parquet_path: Path, tokenizer, max_length: int = 128):
        df = pd.read_parquet(parquet_path)
        self.df = df[df["pair_type"] != "ood"].copy().reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.grouped = self.df.groupby("query_id")
        self.query_ids = list(self.grouped.groups.keys())
        counts = self.df["pair_type"].value_counts().to_dict()
        print(f"[QuadDataset] {parquet_path.name}: {counts}")

    def __len__(self) -> int:
        return len(self.query_ids)

    def _get_row_for_type(self, group_df, pair_type):
        subset = group_df[group_df["pair_type"] == pair_type]
        return subset.iloc[0] if len(subset) > 0 else group_df.iloc[-1]

    def __getitem__(self, idx: int) -> dict:
        qid = self.query_ids[idx]
        group_df = self.grouped.get_group(qid)
        seen_easy, rows = 0, []
        for pair_type in self.TYPE_ORDER:
            if pair_type == "easy_negative":
                easy_rows = group_df[group_df["pair_type"] == "easy_negative"]
                rows.append(
                    easy_rows.iloc[seen_easy]
                    if len(easy_rows) > seen_easy
                    else group_df.iloc[-1]
                )
                seen_easy += 1
            else:
                rows.append(self._get_row_for_type(group_df, pair_type))

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
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
            "labels": torch.tensor(quad_df["label"].astype(int).values, dtype=torch.long),
            "pair_types": quad_df["pair_type"].tolist(),
        }


class OodFlatDataset(Dataset):
    def __init__(self, parquet_path: Path, tokenizer, max_length: int = 128):
        df = pd.read_parquet(parquet_path)
        if "pair_type" in df.columns:
            df = df[df["pair_type"] == "ood"].copy().reset_index(drop=True)
        self.passages = df["passage_text"].fillna("").astype(str).tolist()
        self.queries = (
            df["query_text"].fillna("").astype(str).tolist()
            if "query_text" in df.columns
            else [""] * len(self.passages)
        )
        self.pair_types = (
            df["pair_type"].tolist() if "pair_type" in df.columns else ["ood"] * len(self.passages)
        )
        self.tokenizer = tokenizer
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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "pair_types": self.pair_types[idx],
        }


def collate_quads(batch):
    return {
        "input_ids": torch.cat([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in batch], dim=0),
        "token_type_ids": torch.cat([b["token_type_ids"] for b in batch], dim=0),
        "labels": torch.cat([b["labels"] for b in batch], dim=0),
        "pair_types": [pt for b in batch for pt in b["pair_types"]],
    }


def collate_flat(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "pair_types": [b["pair_types"] for b in batch],
    }


def materialize_local_processed(config: DAEDLConfig) -> Path:
    config.local_processed_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("train_pairs.parquet", "ood_slice.parquet", "test_4bucket.parquet"):
        src = config.processed_dir / fname
        dst = config.local_processed_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return config.local_processed_dir


def ensure_phase2_inputs(config: DAEDLConfig) -> tuple[Path, Path, Path]:
    if config.use_local_processed:
        materialize_local_processed(config)

    processed_dir = config.active_processed_dir
    train_path = processed_dir / "train_pairs.parquet"
    test_path = processed_dir / "test_4bucket.parquet"
    ood_path = processed_dir / "ood_slice.parquet"

    missing = [p for p in (train_path, ood_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {[str(p) for p in missing]}. Run mine.py first.")

    if not test_path.exists():
        print(f"Rebuilding {test_path.name}...")
        df_pairs = pd.read_parquet(train_path)
        df_ood = pd.read_parquet(ood_path)
        for frame in (df_pairs, df_ood):
            for col in ("query_text", "passage_text"):
                if col in frame.columns:
                    frame[col] = frame[col].fillna("").astype(str)
            if "passage_id" in frame.columns:
                frame["passage_id"] = frame["passage_id"].astype(str)
            if "pair_type" in frame.columns:
                frame["pair_type"] = frame["pair_type"].astype(str)
        indist_sample = (
            df_pairs[df_pairs["pair_type"] != "ood"]
            .groupby("pair_type", group_keys=False)
            .head(500)
        )
        ood_sample = df_ood.head(500).copy()
        ood_sample["query_id"] = range(-1, -len(ood_sample) - 1, -1)
        test_4bucket = pd.concat([indist_sample, ood_sample], ignore_index=True)
        test_4bucket.to_parquet(test_path, index=False, compression="snappy")
        print(f"  Created: {test_4bucket['pair_type'].value_counts().to_dict()}")

    return train_path, test_path, ood_path

