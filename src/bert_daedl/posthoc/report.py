"""CSV, JSON, and markdown writers shared by the post-hoc scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj) -> Path:
    path = ensure_parent(Path(path))
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def write_md(path, text: str) -> Path:
    path = ensure_parent(Path(path))
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def write_csv(path, rows: list[dict]) -> Path:
    path = ensure_parent(Path(path))
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def sample_std(vals, ddof: int = 1) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size <= ddof:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def format_mean_std(vals, ddof: int = 1, digits: int = 4) -> str:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0:
        return ""
    return f"{float(arr.mean()):.{digits}f} ± {sample_std(arr, ddof):.{digits}f}"
