"""Banking77-OOS loader.

Reads the seven published folders from jianguoz/Few-Shot-Intent-Detection
(commit fa95533). ``id-oos/train`` is counted and checked, then left out of
every loader (no outlier exposure).

``oos_val`` is id-oos/valid followed by ood-oos/valid. ``oos_test`` is
id-oos/test followed by ood-oos/test. A parallel ``oos_type`` code marks
each OOS utterance: 1 = ID-OOS, 2 = OOD-OOS.

Set ``BERT_DAEDL_DATA_STRICT=0`` to skip the published-count checks. That
switch is for synthetic tests only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from bert_daedl.data.clinc150 import build_label_maps, build_loaders_from_entries

# Published sizes (Zhang et al. 2022; files at fa95533). Empty lines are kept.
EXPECTED = {
    "train": 5905,
    "valid": 1506,
    "test": 2000,
    "id_oos_train": 2062,
    "id_oos_valid": 530,
    "id_oos_test": 1080,
    "ood_oos_valid": 200,
    "ood_oos_test": 1000,
}
N_INTENTS = 50
N_HELD_OUT = 27
ID_OOS = 1
OOD_OOS = 2

SPLIT_FILES = {
    "train": ("train", False),
    "valid": ("valid", False),
    "test": ("test", False),
    "id_oos_train": ("id-oos/train", True),
    "id_oos_valid": ("id-oos/valid", True),
    "id_oos_test": ("id-oos/test", True),
    "ood_oos_valid": ("ood-oos/valid", False),
    "ood_oos_test": ("ood-oos/test", False),
}


@dataclass
class Banking77OOSConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 64
    batch_size: int = 32
    eval_batch_size: int = 128
    num_workers: int = 4
    seed: int = 0
    data_dir: str = "data/BANKING77-OOS"
    return_texts: bool = False


def strict_counts() -> bool:
    return os.environ.get("BERT_DAEDL_DATA_STRICT", "1") != "0"


def _read_lines(path: Path) -> List[str]:
    # splitlines keeps internal empty lines and drops a trailing newline.
    return path.read_text(encoding="utf-8").splitlines()


def _read_split(root: Path, folder: str, *, original: bool) -> dict:
    base = root / folder
    seq_path = base / "seq.in"
    label_path = base / "label"
    if not seq_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"missing seq.in or label under {base}")
    texts = _read_lines(seq_path)
    labels = _read_lines(label_path)
    if len(texts) != len(labels):
        raise RuntimeError(
            f"{base}: seq.in has {len(texts)} lines, label has {len(labels)}"
        )
    originals = None
    if original:
        orig_path = base / "label_original"
        if not orig_path.is_file():
            raise FileNotFoundError(f"missing label_original under {base}")
        originals = _read_lines(orig_path)
        if len(originals) != len(texts):
            raise RuntimeError(
                f"{base}: label_original has {len(originals)} lines, seq.in has {len(texts)}"
            )
    return {"texts": texts, "labels": labels, "originals": originals}


def load_banking77(data_dir: str | Path) -> dict:
    """Read and check the Banking77-OOS folders. Does not tokenize."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Banking77-OOS directory not found: {root}")
    splits = {}
    for name, (folder, original) in SPLIT_FILES.items():
        path = root / folder
        if not path.is_dir():
            if name == "id_oos_train" and not strict_counts():
                splits[name] = {"texts": [], "labels": [], "originals": []}
                continue
            raise FileNotFoundError(f"missing Banking77-OOS split {folder}")
        splits[name] = _read_split(root, folder, original=original)
    _validate(splits)
    in_scope_labels = sorted({
        label
        for name in ("train", "valid", "test")
        for label in splits[name]["labels"]
    })
    held_out = sorted(set(splits["id_oos_test"]["originals"] or []))
    oos_val_texts = splits["id_oos_valid"]["texts"] + splits["ood_oos_valid"]["texts"]
    oos_test_texts = splits["id_oos_test"]["texts"] + splits["ood_oos_test"]["texts"]
    oos_val_types = [ID_OOS] * len(splits["id_oos_valid"]["texts"]) + [
        OOD_OOS
    ] * len(splits["ood_oos_valid"]["texts"])
    oos_test_types = [ID_OOS] * len(splits["id_oos_test"]["texts"]) + [
        OOD_OOS
    ] * len(splits["ood_oos_test"]["texts"])
    return {
        "splits": splits,
        "in_scope_labels": in_scope_labels,
        "held_out": held_out,
        "texts": {
            "train": splits["train"]["texts"],
            "val": splits["valid"]["texts"],
            "test": splits["test"]["texts"],
            "oos_val": oos_val_texts,
            "oos_test": oos_test_texts,
        },
        "oos_val_types": oos_val_types,
        "oos_test_types": oos_test_types,
        "counts": {name: len(item["texts"]) for name, item in splits.items()},
        "n_empty": {
            name: sum(text == "" for text in item["texts"])
            for name, item in splits.items()
        },
    }


def _validate(splits: dict) -> None:
    in_scope = set()
    for name in ("train", "valid", "test"):
        labels = set(splits[name]["labels"])
        if "oos" in labels:
            raise RuntimeError(f"in-scope split {name} contains an 'oos' label")
        in_scope |= labels

    held_sets = []
    for name in ("id_oos_train", "id_oos_valid", "id_oos_test"):
        originals = splits[name]["originals"] or []
        if not originals and name == "id_oos_train" and not strict_counts():
            continue
        held = set(originals)
        overlap = held & in_scope
        if overlap:
            raise RuntimeError(
                f"held-out intents also appear in-scope ({name}): {sorted(overlap)[:5]}"
            )
        held_sets.append(held)
    if held_sets and any(held != held_sets[0] for held in held_sets[1:]):
        raise RuntimeError("held-out intents differ across id-oos splits")

    def _overlap(a: Sequence[str], b: Sequence[str], what: str) -> None:
        shared = set(a) & set(b)
        shared.discard("")
        if shared:
            raise RuntimeError(f"{what} share {len(shared)} utterances")

    _overlap(splits["valid"]["texts"], splits["test"]["texts"], "valid and test")
    _overlap(
        splits["id_oos_test"]["texts"],
        splits["ood_oos_test"]["texts"],
        "id-oos and ood-oos test",
    )

    if not strict_counts():
        return
    for name, expected in EXPECTED.items():
        got = len(splits[name]["texts"])
        if got != expected:
            raise RuntimeError(
                f"Banking77-OOS split {name}: expected {expected} utterances, got {got}"
            )
    if len(in_scope) != N_INTENTS:
        raise RuntimeError(f"expected {N_INTENTS} in-scope intents, got {len(in_scope)}")
    if not held_sets or len(held_sets[0]) != N_HELD_OUT:
        got = 0 if not held_sets else len(held_sets[0])
        raise RuntimeError(f"expected {N_HELD_OUT} held-out intents, got {got}")


def build_banking77_dataloaders(cfg: Banking77OOSConfig):
    """Build (train, val, test, oos_val, oos_test, meta).

    ``id-oos/train`` is never passed to a loader.
    """
    bundle = load_banking77(cfg.data_dir)
    splits = bundle["splits"]
    label2id, id2label = build_label_maps(bundle["in_scope_labels"])

    def _pairs(name: str) -> List[Tuple[str, str]]:
        item = splits[name]
        return list(zip(item["texts"], item["labels"]))

    # OOS labels are unused: the dataset stores -1. Order is ID-OOS, then OOD-OOS.
    oos_val = [(text, "oos") for text in bundle["texts"]["oos_val"]]
    oos_test = [(text, "oos") for text in bundle["texts"]["oos_test"]]
    return build_loaders_from_entries(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        id_train=_pairs("train"),
        id_val=_pairs("valid"),
        id_test=_pairs("test"),
        oos_val=oos_val,
        oos_test=oos_test,
        label2id=label2id,
        id2label=id2label,
        oos_val_types=bundle["oos_val_types"],
        oos_test_types=bundle["oos_test_types"],
        n_oos_train_unused=bundle["counts"].get("id_oos_train", 0),
        return_texts=cfg.return_texts,
        extra_meta={
            "dataset": "banking77_oos",
            "n_intents": len(label2id),
            "n_held_out": len(bundle["held_out"]),
            "held_out": bundle["held_out"],
            "oos_type_codes": {"id": ID_OOS, "ood": OOD_OOS},
        },
    )
