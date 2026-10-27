"""
CLINC150 data pipeline for BERT evidential / DAEDL experiments.

Loads from HuggingFace `clinc_oos` (config="plus") when available.
Note: `clinc_oos` is a script-based HF dataset; datasets>=3 drops script
support. The local `data_full.json` fallback covers that case.

Return signature:
    (train, val, test, oos_val, oos_test, meta)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset

from bert_daedl.utils.seed import seed_everything

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

OOS_LABEL = "oos"

# Expected sizes for the standard "plus" config (Larson et al. 2019).
PLUS_EXPECTED = {
    "id_train": 15000,
    "oos_train": 250,
    "id_val": 3000,
    "oos_val": 100,
    "id_test": 4500,
    "oos_test": 1000,
}


@dataclass
class CLINC150Config:
    model_name: str = "bert-base-uncased"
    max_length: int = 64
    batch_size: int = 32
    eval_batch_size: int = 128
    num_workers: int = 4
    seed: int = 0
    # "small" | "imbalanced" | "plus" (standard 150-intent + OOS benchmark)
    hf_config: str = "plus"
    hf_cache_dir: Optional[str] = None
    # Fallback: directory containing data_full.json from clinc/oos-eval.
    local_data_dir: Optional[str] = "data/CLINC150"
    return_texts: bool = False


def _load_raw_splits(cfg: CLINC150Config) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    """Returns {"train"/"validation"/"test": [(text, label_str), ...]} + ID label vocab."""
    try:
        from datasets import load_dataset

        logger.info("Loading CLINC150 ('clinc_oos', config=%s) from the HF Hub", cfg.hf_config)
        raw = load_dataset("clinc_oos", cfg.hf_config, cache_dir=cfg.hf_cache_dir)
        label_names = raw["train"].features["intent"].names

        splits = {}
        for split in ("train", "validation", "test"):
            texts = raw[split]["text"]
            labels = [label_names[i] for i in raw[split]["intent"]]
            splits[split] = list(zip(texts, labels))
        return splits, [l for l in label_names if l != OOS_LABEL]

    except Exception as exc:
        # datasets>=3 drops script-based datasets like clinc_oos; local JSON covers this.
        logger.warning("HF Hub load failed (%s). Falling back to local JSON.", exc)
        return _load_raw_splits_local(cfg)


def _resolve_local_json(data_root: Path, hf_config: str = "plus") -> Path:
    # Prefer the file that matches the requested HF config when present.
    preferred = {
        "plus": ("data_oos_plus.json", "data_full.json", "data_small.json"),
        "small": ("data_small.json", "data_full.json", "data_oos_plus.json"),
        "imbalanced": ("data_imbalanced.json", "data_full.json", "data_oos_plus.json"),
    }.get(hf_config, ("data_oos_plus.json", "data_full.json", "data_small.json"))
    for candidate in preferred:
        path = data_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No CLINC150 json file (data_full.json/...) found under {data_root}"
    )


def _load_raw_splits_local(cfg: CLINC150Config) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    if cfg.local_data_dir is None:
        raise FileNotFoundError(
            "No internet access and `local_data_dir` was not set. "
            "Either allow HF Hub access, or set CLINC150Config.local_data_dir to the "
            "directory containing data_full.json from github.com/clinc/oos-eval."
        )
    json_path = _resolve_local_json(Path(cfg.local_data_dir), cfg.hf_config)
    logger.info("Loading CLINC150 from local file %s", json_path)
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    splits = {
        "train": payload["train"] + payload.get("oos_train", []),
        "validation": payload["val"] + payload.get("oos_val", []),
        "test": payload["test"] + payload.get("oos_test", []),
    }
    label_names = sorted({label for _, label in splits["train"] if label != OOS_LABEL})
    return splits, label_names


def _split_id_oos(
    entries: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    id_entries = [(t, l) for t, l in entries if l != OOS_LABEL]
    oos_entries = [(t, l) for t, l in entries if l == OOS_LABEL]
    return id_entries, oos_entries


def build_label_maps(label_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(sorted(label_names))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _validate_plus_counts(id_train, oos_train, id_val, oos_val, id_test, oos_test) -> None:
    got = {
        "id_train": len(id_train),
        "oos_train": len(oos_train),
        "id_val": len(id_val),
        "oos_val": len(oos_val),
        "id_test": len(id_test),
        "oos_test": len(oos_test),
    }
    for k, v in PLUS_EXPECTED.items():
        if got[k] != v:
            logger.warning("Split size mismatch %s: expected %d got %d", k, v, got[k])


def truncated_share(texts: List[str], tokenizer, max_length: int) -> float:
    """Share of utterances whose token ids, including specials, exceed ``max_length``."""
    if not texts:
        return 0.0
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    n_over = sum(len(ids) > max_length for ids in encoded["input_ids"])
    return n_over / len(texts)


class IntentDataset(Dataset):
    """Pre-tokenized intent examples. label_ids == -1 marks OOS (never used in loss)."""

    def __init__(
        self,
        texts: List[str],
        label_ids: List[int],
        tokenizer,
        max_length: int,
        return_texts: bool = False,
        oos_types: Optional[List[int]] = None,
    ):
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.token_type_ids = encodings.get("token_type_ids")
        self.labels = torch.tensor(label_ids, dtype=torch.long)
        self.texts = texts if return_texts else None
        self.oos_types = (
            None if oos_types is None else torch.tensor(oos_types, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.oos_types is not None:
            item["oos_type"] = self.oos_types[idx]
        if self.texts is not None:
            item["text"] = self.texts[idx]
        return item


def build_loaders_from_entries(
    *,
    model_name: str,
    max_length: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    id_train: List[Tuple[str, str]],
    id_val: List[Tuple[str, str]],
    id_test: List[Tuple[str, str]],
    oos_val: List[Tuple[str, str]],
    oos_test: List[Tuple[str, str]],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    oos_val_types: Optional[List[int]] = None,
    oos_test_types: Optional[List[int]] = None,
    n_oos_train_unused: int = 0,
    return_texts: bool = False,
    tokenizer=None,
    extra_meta: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, dict]:
    """Tokenize five splits and build the loaders both datasets share.

    ``oos_val_types`` / ``oos_test_types`` are optional per-utterance codes
    (1 = ID-OOS, 2 = OOD-OOS). CLINC150 leaves them unset.
    """
    seed_everything(seed)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    text_splits = {
        "train": [t for t, _ in id_train],
        "val": [t for t, _ in id_val],
        "test": [t for t, _ in id_test],
        "oos_val": [t for t, _ in oos_val],
        "oos_test": [t for t, _ in oos_test],
    }
    truncation = {
        name: truncated_share(texts, tokenizer, max_length)
        for name, texts in text_splits.items()
    }
    worst = max(truncation.values()) if truncation else 0.0
    for name, share in truncation.items():
        logger.info(
            "truncation %s: %.4f (%d utterances, max_length=%d)",
            name,
            share,
            len(text_splits[name]),
            max_length,
        )
    logger.info("worst truncation %.4f at max_length=%d", worst, max_length)

    def to_id_dataset(entries: List[Tuple[str, str]]) -> IntentDataset:
        texts = [t for t, _ in entries]
        labels = [label2id[label] for _, label in entries]
        return IntentDataset(
            texts, labels, tokenizer, max_length, return_texts=return_texts
        )

    def to_oos_dataset(entries: List[Tuple[str, str]], types: Optional[List[int]]) -> IntentDataset:
        texts = [t for t, _ in entries]
        labels = [-1] * len(entries)
        return IntentDataset(
            texts,
            labels,
            tokenizer,
            max_length,
            return_texts=return_texts,
            oos_types=types,
        )

    pin_memory = torch.cuda.is_available()
    gen = Generator().manual_seed(seed)

    def _eval_loader(ds: Dataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = DataLoader(
        to_id_dataset(id_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=gen,
        pin_memory=pin_memory,
        drop_last=True,
    )
    meta = {
        "num_classes": len(label2id),
        "label2id": label2id,
        "id2label": id2label,
        "n_oos_train_unused": n_oos_train_unused,
        "model_name": model_name,
        "max_length": max_length,
        "truncation": truncation,
        "truncation_worst": worst,
    }
    if extra_meta:
        meta.update(extra_meta)
    return (
        train_loader,
        _eval_loader(to_id_dataset(id_val)),
        _eval_loader(to_id_dataset(id_test)),
        _eval_loader(to_oos_dataset(oos_val, oos_val_types)),
        _eval_loader(to_oos_dataset(oos_test, oos_test_types)),
        meta,
    )


def describe_clinc150(data_dir: str, hf_config: str = "plus") -> dict:
    """Split texts and counts from the local CLINC150 JSON, without tokenizing."""
    cfg = CLINC150Config(local_data_dir=data_dir, hf_config=hf_config)
    raw_splits, label_names = _load_raw_splits_local(cfg)
    id_train, oos_train = _split_id_oos(raw_splits["train"])
    id_val, oos_val = _split_id_oos(raw_splits["validation"])
    id_test, oos_test = _split_id_oos(raw_splits["test"])
    counts = {
        "id_train": len(id_train),
        "oos_train": len(oos_train),
        "id_val": len(id_val),
        "oos_val": len(oos_val),
        "id_test": len(id_test),
        "oos_test": len(oos_test),
    }
    return {
        "texts": {
            "train": [t for t, _ in id_train],
            "val": [t for t, _ in id_val],
            "test": [t for t, _ in id_test],
            "oos_val": [t for t, _ in oos_val],
            "oos_test": [t for t, _ in oos_test],
        },
        "counts": counts,
        "n_intents": len(label_names),
    }


def build_clinc150_dataloaders(
    cfg: CLINC150Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, dict]:
    """Build (train, val, test, oos_val, oos_test, meta).

    Classifier trains/evals on ID only. OOS loaders are for detection metrics.
    `oos_train` is intentionally unused (no outlier exposure).
    """
    raw_splits, label_names = _load_raw_splits(cfg)
    label2id, id2label = build_label_maps(label_names)

    id_train, oos_train = _split_id_oos(raw_splits["train"])
    id_val, oos_val = _split_id_oos(raw_splits["validation"])
    id_test, oos_test = _split_id_oos(raw_splits["test"])

    if cfg.hf_config == "plus":
        _validate_plus_counts(id_train, oos_train, id_val, oos_val, id_test, oos_test)

    logger.info(
        "ID classes=%d | train=%d val=%d test=%d | oos(val=%d test=%d)",
        len(label2id),
        len(id_train),
        len(id_val),
        len(id_test),
        len(oos_val),
        len(oos_test),
    )
    logger.info(
        "oos_train (n=%d) intentionally unused (no outlier exposure).",
        len(oos_train),
    )
    return build_loaders_from_entries(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        id_train=id_train,
        id_val=id_val,
        id_test=id_test,
        oos_val=oos_val,
        oos_test=oos_test,
        label2id=label2id,
        id2label=id2label,
        n_oos_train_unused=len(oos_train),
        return_texts=cfg.return_texts,
        extra_meta={"dataset": "clinc150"},
    )


def labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
