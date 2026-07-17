"""JSON / results I/O with config hashing and git SHA stamping."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_RESULT_KEYS = (
    "run_id",
    "arm",
    "seed",
    "git_sha",
    "config_hash",
    "config",
    "in_scope_acc",
    "ece",
    "per_score",
    "thresholded",
)


def git_sha(short: bool = True) -> str:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def save_json(output_dir: str | Path, name: str, obj: Any) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(obj: Any):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_results(output_dir: str | Path, results: Dict[str, Any],
                 filename: str = "results.json") -> Path:
    """Stamp bookkeeping fields and write schema-valid results.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = results.get("config") or {}
    arm = results.get("arm", cfg.get("arm", "unknown"))
    seed = results.get("seed", cfg.get("seed", 0))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    results.setdefault("run_id", f"{arm}_seed{seed}_{ts}")
    results.setdefault("git_sha", git_sha())
    results.setdefault("config_hash", config_hash(cfg) if cfg else "none")
    results.setdefault("diagnostics_file", "diagnostics.json")

    missing = [k for k in REQUIRED_RESULT_KEYS if k not in results]
    if missing:
        raise ValueError(f"results.json missing required keys: {missing}")

    path = out / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
