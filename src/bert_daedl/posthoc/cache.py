"""Extract, save, load, and find cached evaluation arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SPLITS = ("train", "val", "test", "oos_val", "oos_test")


def eval_order_loader(loader, batch_size: int | None = None, num_workers: int = 0) -> DataLoader:
    """File order, nothing dropped. Replaces the shuffled train loader for GDA fits."""
    bs = batch_size or loader.batch_size or 128
    return DataLoader(
        loader.dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )


@torch.no_grad()
def extract_split(model, loader, device, with_pooled: bool = False) -> dict:
    model.eval()
    logits, feats, labels, pooled, oos_types = [], [], [], [], []
    got_type = None
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        tt = batch.get("token_type_ids")
        tt = tt.to(device) if tt is not None else None
        if with_pooled:
            z, f, p = model(ids, am, tt, return_pooled=True)
            pooled.append(p.detach().cpu().numpy().astype(np.float32))
        else:
            z, f = model(ids, am, tt, return_features=True)
        logits.append(z.detach().cpu().numpy().astype(np.float32))
        feats.append(f.detach().cpu().numpy().astype(np.float32))
        labels.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
        if got_type is None:
            got_type = "oos_type" in batch
        if got_type:
            oos_types.append(batch["oos_type"].detach().cpu().numpy().astype(np.int64))
    out = {
        "logits": np.concatenate(logits),
        "feats": np.concatenate(feats),
        "labels": np.concatenate(labels),
    }
    if with_pooled:
        out["pooled"] = np.concatenate(pooled)
    if got_type:
        out["oos_type"] = np.concatenate(oos_types)
    return out


def save_cache(output_dir, arrays: dict, meta: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {}
    pooled = {}
    for split, data in arrays.items():
        for key in ("logits", "feats", "labels", "oos_type"):
            if key in data:
                payload[f"{split}__{key}"] = data[key]
        if "pooled" in data:
            pooled[f"{split}__pooled"] = data["pooled"]
    np.savez_compressed(out / "cache.npz", **payload)
    if pooled:
        np.savez_compressed(out / "pooled.npz", **pooled)
    meta = dict(meta)
    meta["split_sizes"] = {s: int(arrays[s]["labels"].shape[0]) for s in arrays}
    meta["has_pooled"] = bool(pooled)
    meta["has_oos_type"] = any("oos_type" in data for data in arrays.values())
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _apply_oos_subset(splits: dict, oos_subset: str) -> dict:
    """Keep one OOS block. In-scope splits are not touched.

    Banking77-OOS caches store ``oos_type`` 1 (ID-OOS) and 2 (OOD-OOS).
    CLINC150 has no codes; only ``all`` is valid there.
    """
    if oos_subset in (None, "all"):
        return splits
    code = {"id": 1, "ood": 2}[oos_subset]
    for split in ("oos_val", "oos_test"):
        if split not in splits:
            continue
        types = splits[split].get("oos_type")
        if types is None:
            raise ValueError(
                f"oos_subset={oos_subset!r} needs {split}__oos_type in the cache. "
                "CLINC150 has one undivided OOS set; use oos_subset='all'."
            )
        mask = np.asarray(types) == code
        if not bool(mask.any()):
            raise ValueError(f"no utterances with oos_type={code} in {split}")
        splits[split] = {key: np.asarray(value)[mask] for key, value in splits[split].items()}
    return splits


def load_cache(run_dir, oos_subset: str = "all") -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "cache_meta.json").read_text(encoding="utf-8"))
    blob = np.load(run_dir / "cache.npz")
    pooled_path = run_dir / "pooled.npz"
    pooled = np.load(pooled_path) if pooled_path.exists() else None
    splits = {}
    names = sorted({k.split("__", 1)[0] for k in blob.files})
    for split in names:
        item = {
            "logits": blob[f"{split}__logits"],
            "feats": blob[f"{split}__feats"],
            "labels": blob[f"{split}__labels"],
        }
        type_key = f"{split}__oos_type"
        if type_key in blob.files:
            item["oos_type"] = blob[type_key]
        key = f"{split}__pooled"
        if pooled is not None and key in pooled.files:
            item["pooled"] = pooled[key]
        splits[split] = item
    _apply_oos_subset(splits, oos_subset)
    return {"path": run_dir, "meta": meta, "splits": splits}


def find_runs(root, arm: str | None = None, tag: str | None = None) -> list[Path]:
    root = Path(root)
    found: list[Path] = []
    seen: dict[tuple, Path] = {}
    for meta_path in sorted(root.glob("**/cache_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if arm is not None and meta.get("arm") != arm:
            continue
        if tag is not None and meta.get("tag", "") != tag:
            continue
        key = (meta.get("arm"), meta.get("tag", ""), meta.get("seed"))
        if key in seen:
            raise RuntimeError(
                f"Two cached runs for arm={key[0]} tag={key[1]!r} seed={key[2]}: "
                f"{seen[key]} and {meta_path.parent}"
            )
        seen[key] = meta_path.parent
        found.append(meta_path.parent)
    return found


def load_runs(root, arm: str | None = None, tag: str | None = None, oos_subset: str = "all") -> list[dict]:
    runs = [load_cache(p, oos_subset=oos_subset) for p in find_runs(root, arm=arm, tag=tag)]
    runs.sort(key=lambda r: (str(r["meta"].get("arm")), str(r["meta"].get("tag", "")), int(r["meta"].get("seed", 0))))
    return runs
