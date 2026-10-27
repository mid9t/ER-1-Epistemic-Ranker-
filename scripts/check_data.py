#!/usr/bin/env python
"""Split sizes and truncation, before any training run.

Exits non-zero when a published count is wrong or more than 2% of any split
is truncated at the requested length. A truncation failure means both arms
should use 96 tokens; that choice is made here, before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.data.banking77_oos import EXPECTED as B77_EXPECTED
from bert_daedl.data.banking77_oos import N_HELD_OUT, N_INTENTS, load_banking77
from bert_daedl.data.clinc150 import PLUS_EXPECTED, describe_clinc150, truncated_share

TRUNC_LIMIT = 0.02


def _print_sizes(sizes, n_intents, n_held_out):
    text = (
        f"sizes {sizes[0]:,} / {sizes[1]:,} / {sizes[2]:,} / "
        f"{sizes[3]:,} / {sizes[4]:,} with {n_intents} intents"
    )
    if n_held_out is not None:
        text += f" and {n_held_out} held out"
    print(text)


def _truncation(texts, tokenizer, max_length):
    shares = {name: truncated_share(rows, tokenizer, max_length) for name, rows in texts.items()}
    worst_name = max(shares, key=shares.get)
    return shares, worst_name, shares[worst_name]


def _finish(payload, out, stop, message=None):
    payload["stop"] = bool(stop)
    if message:
        payload["message"] = message
        print(message)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    if stop:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["banking77_oos", "clinc150"])
    p.add_argument("--data_dir", default=None)
    p.add_argument("--model_name", required=True)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None, help="Accepted so the Makefile can pass DEVICE. Unused.")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    payload = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "max_length": args.max_length,
    }
    if args.dataset == "banking77_oos":
        data_dir = args.data_dir or "data/BANKING77-OOS"
        try:
            bundle = load_banking77(data_dir)
        except (RuntimeError, FileNotFoundError) as exc:
            payload["data_dir"] = data_dir
            _finish(payload, args.out, True, str(exc))
        sizes = (
            bundle["counts"]["train"],
            bundle["counts"]["valid"],
            bundle["counts"]["test"],
            bundle["counts"]["id_oos_valid"] + bundle["counts"]["ood_oos_valid"],
            bundle["counts"]["id_oos_test"] + bundle["counts"]["ood_oos_test"],
        )
        _print_sizes(sizes, len(bundle["in_scope_labels"]), len(bundle["held_out"]))
        shares, worst_name, worst = _truncation(bundle["texts"], tokenizer, args.max_length)
        payload.update({
            "data_dir": data_dir,
            "sizes": {
                "train": sizes[0],
                "val": sizes[1],
                "test": sizes[2],
                "oos_val": sizes[3],
                "oos_test": sizes[4],
            },
            "split_counts": bundle["counts"],
            "expected": dict(B77_EXPECTED),
            "n_intents": len(bundle["in_scope_labels"]),
            "n_held_out": len(bundle["held_out"]),
            "n_empty": bundle["n_empty"],
            "truncation": shares,
            "worst_split": worst_name,
            "worst_truncation": worst,
        })
        count_ok = (
            sizes == (5905, 1506, 2000, 730, 2080)
            and len(bundle["in_scope_labels"]) == N_INTENTS
            and len(bundle["held_out"]) == N_HELD_OUT
        )
    else:
        data_dir = args.data_dir or "data/CLINC150"
        try:
            described = describe_clinc150(data_dir)
        except (RuntimeError, FileNotFoundError, json.JSONDecodeError) as exc:
            payload["data_dir"] = data_dir
            _finish(payload, args.out, True, str(exc))
        counts = described["counts"]
        sizes = (
            counts["id_train"],
            counts["id_val"],
            counts["id_test"],
            counts["oos_val"],
            counts["oos_test"],
        )
        _print_sizes(sizes, described["n_intents"], None)
        shares, worst_name, worst = _truncation(described["texts"], tokenizer, args.max_length)
        payload.update({
            "data_dir": data_dir,
            "sizes": {
                "train": sizes[0],
                "val": sizes[1],
                "test": sizes[2],
                "oos_val": sizes[3],
                "oos_test": sizes[4],
            },
            "split_counts": counts,
            "expected": dict(PLUS_EXPECTED),
            "n_intents": described["n_intents"],
            "truncation": shares,
            "worst_split": worst_name,
            "worst_truncation": worst,
        })
        count_ok = counts == dict(PLUS_EXPECTED) and described["n_intents"] == 150

    print(f"worst truncation {worst:.2%} ({worst_name}) at {args.max_length} tokens")
    if not count_ok:
        _finish(payload, args.out, True, "split counts differ from the published sizes; fix the data before training")
    if worst > TRUNC_LIMIT:
        _finish(
            payload,
            args.out,
            True,
            f"more than 2% of {worst_name} is truncated at {args.max_length} tokens; "
            "use 96 tokens for both arms, decided before training",
        )
    _finish(payload, args.out, False, "data check passed")


if __name__ == "__main__":
    main()
