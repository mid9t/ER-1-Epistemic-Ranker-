"""Banking77-OOS loader, OOS subsets, and the pre-registered decision statistics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.data.banking77_oos import load_banking77
from bert_daedl.data.clinc150 import build_loaders_from_entries, build_label_maps
from bert_daedl.posthoc.cache import load_cache, save_cache
from bert_daedl.posthoc.stats import combined_interval, holm, paired_bootstrap_did


def _script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"replication_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_split(root, folder, texts, labels, originals=None):
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / "seq.in").write_text("\n".join(texts) + ("\n" if texts else ""), encoding="utf-8")
    (path / "label").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    if originals is not None:
        (path / "label_original").write_text("\n".join(originals) + "\n", encoding="utf-8")


def _tiny_banking(root):
    _write_split(root, "train", ["move my money", "check the balance"], ["transfer", "balance"])
    _write_split(root, "valid", ["send a transfer", ""], ["transfer", "balance"])
    _write_split(root, "test", ["what is my balance", "wire the funds"], ["balance", "transfer"])
    _write_split(
        root, "id-oos/train", ["held train utt"], ["oos"], ["card_arrival"],
    )
    _write_split(
        root, "id-oos/valid", ["", "held valid utt"], ["oos", "oos"], ["card_arrival", "card_arrival"],
    )
    _write_split(
        root, "id-oos/test", ["", "held test utt"], ["oos", "oos"], ["card_arrival", "card_arrival"],
    )
    _write_split(root, "ood-oos/valid", ["ood valid utt"], ["oos"])
    _write_split(root, "ood-oos/test", ["ood test utt"], ["oos"])


def test_banking77_keeps_order_types_and_drops_oos_train(tmp_path, monkeypatch):
    monkeypatch.setenv("BERT_DAEDL_DATA_STRICT", "0")
    root = tmp_path / "BANKING77-OOS"
    _tiny_banking(root)
    bundle = load_banking77(root)
    assert bundle["oos_val_types"] == [1, 1, 2]
    assert bundle["texts"]["oos_val"] == ["", "held valid utt", "ood valid utt"]
    assert bundle["oos_test_types"] == [1, 1, 2]
    assert "held train utt" not in bundle["texts"]["train"]
    assert bundle["counts"]["id_oos_train"] == 1
    assert bundle["held_out"] == ["card_arrival"]
    assert set(bundle["in_scope_labels"]) == {"balance", "transfer"}


def test_banking77_strict_counts_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("BERT_DAEDL_DATA_STRICT", "1")
    root = tmp_path / "BANKING77-OOS"
    _tiny_banking(root)
    with pytest.raises(RuntimeError, match="train"):
        load_banking77(root)


def test_banking77_overlap_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("BERT_DAEDL_DATA_STRICT", "0")
    root = tmp_path / "BANKING77-OOS"
    _tiny_banking(root)
    _write_split(root, "test", ["send a transfer", "wire the funds"], ["transfer", "transfer"])
    with pytest.raises(RuntimeError, match="valid and test"):
        load_banking77(root)


def test_loaders_record_truncation_and_oos_type():
    label2id, id2label = build_label_maps(["balance", "transfer"])

    class _Tok:
        def __call__(self, texts, padding="max_length", truncation=True, max_length=8,
                     return_tensors=None, add_special_tokens=True):
            rows = []
            for text in texts:
                n = len(str(text).split()) + 2
                if truncation:
                    n = min(n, max_length)
                rows.append(list(range(n)))
            if return_tensors == "pt":
                width = max_length
                ids = torch.zeros(len(texts), width, dtype=torch.long)
                mask = torch.zeros(len(texts), width, dtype=torch.long)
                for i, seq in enumerate(rows):
                    ids[i, : len(seq)] = torch.tensor(seq)
                    mask[i, : len(seq)] = 1
                return {"input_ids": ids, "attention_mask": mask}
            return {"input_ids": rows}

    long_text = "one two three four five six seven eight nine ten"
    train, val, test, oos_val, oos_test, meta = build_loaders_from_entries(
        model_name="unused",
        max_length=8,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=0,
        id_train=[("move money", "transfer"), ("check balance", "balance")],
        id_val=[("send funds", "transfer")],
        id_test=[(long_text, "balance")],
        oos_val=[("held val", "oos"), ("ood val", "oos")],
        oos_test=[("held test", "oos"), ("ood test", "oos")],
        label2id=label2id,
        id2label=id2label,
        oos_val_types=[1, 2],
        oos_test_types=[1, 2],
        tokenizer=_Tok(),
    )
    assert meta["truncation"]["test"] == 1.0
    assert meta["truncation"]["train"] == 0.0
    assert train.dataset.oos_types is None
    types = torch.cat([batch["oos_type"] for batch in oos_test]).tolist()
    assert types == [1, 2]
    assert len(val.dataset) == 1


def test_oos_subset_masks_only_oos_arrays(tmp_path):
    rng = np.random.default_rng(0)

    def block(n, labels):
        return {
            "logits": rng.normal(size=(n, 3)).astype(np.float32),
            "feats": rng.normal(size=(n, 4)).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
        }

    arrays = {
        "train": block(6, [0, 1, 0, 1, 0, 1]),
        "val": block(4, [0, 1, 0, 1]),
        "test": block(4, [0, 1, 1, 0]),
        "oos_val": block(4, [-1, -1, -1, -1]),
        "oos_test": block(4, [-1, -1, -1, -1]),
    }
    arrays["oos_val"]["oos_type"] = np.array([1, 1, 2, 2])
    arrays["oos_test"]["oos_type"] = np.array([1, 2, 1, 2])
    arrays["oos_test"]["pooled"] = rng.normal(size=(4, 5)).astype(np.float32)
    save_cache(tmp_path, arrays, {"arm": "edl", "seed": 0, "tag": "", "config": {}})
    full = load_cache(tmp_path, oos_subset="all")
    ident = load_cache(tmp_path, oos_subset="id")
    ood = load_cache(tmp_path, oos_subset="ood")
    assert len(ident["splits"]["test"]["labels"]) == 4
    np.testing.assert_array_equal(ident["splits"]["test"]["logits"], full["splits"]["test"]["logits"])
    assert ident["splits"]["oos_test"]["logits"].shape[0] == 2
    assert set(ident["splits"]["oos_test"]["oos_type"].tolist()) == {1}
    assert set(ood["splits"]["oos_val"]["oos_type"].tolist()) == {2}
    assert ident["splits"]["oos_test"]["pooled"].shape == (2, 5)
    bare = tmp_path / "bare"
    bare_arrays = {k: {kk: vv for kk, vv in block.items() if kk != "oos_type"} for k, block in arrays.items()}
    for split in ("oos_val", "oos_test"):
        bare_arrays[split].pop("oos_type", None)
    save_cache(bare, bare_arrays, {"arm": "edl", "seed": 1, "tag": "", "config": {}})
    with pytest.raises(ValueError, match="oos_type"):
        load_cache(bare, oos_subset="id")


def test_combined_interval_adds_seed_variance():
    delta = 0.01
    se_boot = 0.002
    seeds = [0.008, 0.012, 0.009, 0.011, 0.010]
    out = combined_interval(delta, se_boot, seeds)
    # s^2 = 0.00001 / 4, SE = sqrt(0.002^2 + s^2 / 5)
    assert out["n_seeds"] == 5
    assert out["se_combined"] == pytest.approx(0.0021213203435596424)
    assert out["se_combined"] > se_boot
    half = out["ci95_high"] - delta
    assert out["delta"] - half == pytest.approx(out["ci95_low"])
    assert (out["ci90_high"] - out["ci90_low"]) < (out["ci95_high"] - out["ci95_low"])


def test_holm_stepdown():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06)
    assert out["b"]["p_holm"] == pytest.approx(0.06)
    assert out["a"]["reject"]
    assert not out["b"]["reject"]
    assert not out["c"]["reject"]


def test_did_bootstrap_is_positive_when_near_oos_gains_more():
    id_scores = [np.zeros(20)]
    near_a = [np.ones(12)]
    near_b = [np.zeros(12)]
    far = [np.zeros(8)]
    out = paired_bootstrap_did(
        id_scores, near_a, id_scores, near_b,
        id_scores, far, id_scores, far,
        B=30, seed=0,
    )
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.0)
    assert out["delta"] == pytest.approx(0.5)
    assert out["ci_low"] > 0


def test_lambda_grid_extension_and_secondary_ece_filter():
    grid = _script("coupling_grid")
    assert grid.REPLICATION_LAMS == (1, 2, 5, 10, 20, 40, 80, 160)
    assert set(grid.extend_lams(grid.REPLICATION_LAMS, 160)) == {320, 640}
    assert set(grid.extend_lams(grid.REPLICATION_LAMS, 1)) == {0.25, 0.5}
    assert grid.extend_lams(grid.REPLICATION_LAMS, 5) == []
    configs = grid._configs(grid.REPLICATION_LAMS)
    assert sum(cfg["mode"] == "mul" for cfg in configs) == 3
    assert sum(cfg["mode"] == "add_log" for cfg in configs) == 24

    rows = []
    for seed in range(5):
        rows.append({
            "spec": "daedl:qsigmoid:mul/vacuity",
            "normalizer": "qsigmoid",
            "mode": "mul",
            "lam": None,
            "val_materiality": "ACTIVE",
            "val_auroc": 0.90,
            "test_auroc": 0.80,
            "test_delta_auroc": 0.05,
            "val_ece_scaled": 0.20,
            "val_ece_plain": 0.05,
        })
        rows.append({
            "spec": "daedl:qsigmoid:add_log:5/vacuity",
            "normalizer": "qsigmoid",
            "mode": "add_log",
            "lam": 5.0,
            "val_materiality": "ACTIVE",
            "val_auroc": 0.70,
            "test_auroc": 0.75,
            "test_delta_auroc": 0.02,
            "val_ece_scaled": 0.05,
            "val_ece_plain": 0.05,
        })
    primary = grid._select(rows, [1, 5, 10])
    secondary = grid._select_secondary(rows, [1, 5, 10])
    assert primary["combine_mode"] == "mul"
    assert secondary["combine_mode"] == "add_log"
    assert secondary["lam"] == 5.0
    assert secondary["ece_filter_emptied"] is False


def test_decision_rules_follow_the_preregistration():
    rules = _script("decide_hypotheses")
    assert rules.verdict_h1(0.01, 0.02, 0.011, 0.019) == "supported"
    assert rules.verdict_h1(-0.001, 0.001, -0.0005, 0.0005) == "tie"
    assert rules.verdict_h1(-0.02, -0.01, -0.019, -0.011) == "reversed"
    assert rules.verdict_h1(-0.01, 0.01, -0.008, 0.008) == "inconclusive"
    assert rules.verdict_h2(-0.01, 0.001, 5, 5) == "supported"
    assert rules.verdict_h2(0.01, 0.02, 5, 5) == "refuted"
    assert rules.verdict_h2(-0.01, 0.01, 3, 5) == "inconclusive"
    assert rules.verdict_h3(0.01, 0.02) == "supported"
    assert rules.verdict_h3(-0.02, -0.01) == "refuted"
    assert rules.verdict_h3(-0.01, 0.01) == "inconclusive"
    assert rules.verdict_positive(0.01, 0.02) == "supported"
    assert rules.verdict_positive(0.01, -0.001) == "kill"


@pytest.mark.skipif(
    not Path("data/BANKING77-OOS/train/seq.in").exists(),
    reason="Banking77-OOS files are not in data/",
)
def test_published_banking77_files_match_the_preregistration():
    bundle = load_banking77("data/BANKING77-OOS")
    assert bundle["counts"]["train"] == 5905
    assert bundle["counts"]["valid"] == 1506
    assert bundle["counts"]["test"] == 2000
    assert bundle["counts"]["id_oos_train"] == 2062
    assert bundle["oos_val_types"].count(1) == 530
    assert bundle["oos_val_types"].count(2) == 200
    assert bundle["oos_test_types"][:1080] == [1] * 1080
    assert bundle["oos_test_types"][1080:] == [2] * 1000
    assert len(bundle["in_scope_labels"]) == 50
    assert len(bundle["held_out"]) == 27
    assert bundle["n_empty"]["train"] == 2
    assert bundle["n_empty"]["id_oos_valid"] == 2
    assert bundle["n_empty"]["id_oos_test"] == 4
