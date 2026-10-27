"""Identities the post-hoc analyses rely on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.density.gda import scale_alpha
from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.posthoc.cache import load_cache, save_cache
from bert_daedl.posthoc.gate import benefit_verdict, materiality, pair_decomposition, pair_flip_gate
from bert_daedl.posthoc.gda_variants import partial_mahalanobis, relative_mahalanobis
from bert_daedl.posthoc.scoring import cross_fit_phat, format_spec, logit_scores, parse_spec, score_array
from bert_daedl.posthoc.stats import WeightedAUROC, auroc_scores, paired_bootstrap
from bert_daedl.uncertainty.scores import scores_from_alpha


def test_auroc_matches_sklearn_without_ties():
    rng = np.random.default_rng(0)
    id_s = rng.normal(0, 1, size=40)
    oos_s = rng.normal(1, 1, size=25)
    y = np.r_[np.zeros(40), np.ones(25)]
    s = np.r_[id_s, oos_s]
    assert auroc_scores(id_s, oos_s) == roc_auc_score(y, s)


def test_auroc_matches_sklearn_with_ties():
    id_s = np.array([0.0, 1.0, 1.0, 2.0])
    oos_s = np.array([1.0, 1.0, 3.0])
    y = np.r_[np.zeros(len(id_s)), np.ones(len(oos_s))]
    s = np.r_[id_s, oos_s]
    assert auroc_scores(id_s, oos_s) == roc_auc_score(y, s)


def test_weighted_auroc_matches_resampled_arrays():
    rng = np.random.default_rng(1)
    id_s = rng.normal(size=30)
    oos_s = rng.normal(loc=0.5, size=20)
    w_id = rng.integers(0, 4, size=30)
    w_oos = rng.integers(1, 4, size=20)
    got = WeightedAUROC(id_s, oos_s).auroc(w_id, w_oos)
    expected = auroc_scores(np.repeat(id_s, w_id), np.repeat(oos_s, w_oos))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_pair_flip_decomposition_is_exact():
    rng = np.random.default_rng(2)
    s0_id, s0_oos = rng.normal(size=12), rng.normal(size=8)
    s1_id, s1_oos = rng.normal(size=12), rng.normal(size=8)
    parts = pair_decomposition(s0_id, s0_oos, s1_id, s1_oos)
    assert parts["delta_auroc"] == parts["fixed"] - parts["broken"]
    assert parts["flip_rate"] == parts["fixed"] + parts["broken"]
    np.testing.assert_allclose(parts["delta_auroc"], parts["auroc_gap"], atol=1e-12)


def test_gate_verdicts():
    assert materiality(0.0001) == "INERT"
    assert materiality(0.01) == "ACTIVE"
    assert benefit_verdict(-0.0002, 0.0002) == "NEUTRAL"
    assert benefit_verdict(0.01, 0.02) == "USEFUL"
    assert benefit_verdict(-0.03, -0.01) == "HARMFUL"
    assert benefit_verdict(-0.01, 0.02) == "INCONCLUSIVE"
    id_s = np.linspace(0, 1, 20)
    oos_s = np.linspace(0, 1, 15)
    same = pair_flip_gate(id_s, oos_s, id_s, oos_s, B=50)
    assert same["materiality"] == "INERT"
    assert same["benefit"] == "NEUTRAL"
    strong = pair_flip_gate(np.ones(20), np.zeros(15), np.zeros(20), np.ones(15), B=40)
    assert strong["materiality"] == "ACTIVE"
    assert strong["benefit"] == "USEFUL"


def test_bootstrap_identical_scores_are_zero():
    rng = np.random.default_rng(3)
    id_s = rng.normal(size=25)
    oos_s = rng.normal(size=18)
    out = paired_bootstrap([id_s], [oos_s], [id_s], [oos_s], B=30, seed=0)
    assert out["ci_low"] == 0.0
    assert out["ci_high"] == 0.0
    assert out["delta"] == 0.0


def test_bootstrap_excludes_zero_for_a_real_difference():
    id_a = np.zeros(30)
    oos_a = np.ones(20)
    id_b = np.ones(30)
    oos_b = np.zeros(20)
    out = paired_bootstrap([id_a], [oos_a], [id_b], [oos_b], B=40, seed=0)
    assert out["ci_low"] > 0


def test_vacuity_ranks_like_clipped_energy():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(80, 12))
    scores = logit_scores(logits, arm="edl")
    rho = np.corrcoef(np.argsort(np.argsort(scores["vacuity"])), np.argsort(np.argsort(scores["energy_clip"])))[0, 1]
    assert rho == 1.0


def test_add_log_matches_energy_and_keeps_mean():
    rng = np.random.default_rng(5)
    logits = rng.normal(scale=1.5, size=(40, 6))
    phat = rng.uniform(0.2, 1.0, size=40)
    lam = 4.0
    z = np.clip(logits, -15, 15)
    energy = -logsumexp(z, axis=-1)
    alpha = scale_alpha(torch.from_numpy(logits), torch.from_numpy(phat), "add_log", lam, 15.0)
    vac = scores_from_alpha(alpha)["vacuity"].numpy()
    np.testing.assert_allclose(np.log(vac), np.log(6) + energy + lam * (1 - phat), atol=1e-8)
    plain = torch.softmax(torch.from_numpy(z), dim=-1).numpy()
    scaled = (alpha / alpha.sum(-1, keepdim=True)).numpy()
    np.testing.assert_allclose(scaled, plain, atol=1e-8)


def test_mul_does_not_change_argmax_when_phat_positive():
    rng = np.random.default_rng(6)
    logits = rng.normal(scale=2.0, size=(30, 5))
    phat = rng.uniform(0.05, 1.0, size=30)
    alpha = scale_alpha(torch.from_numpy(logits), torch.from_numpy(phat), "mul", 5.0, 15.0)
    plain = np.clip(logits, -15, 15).argmax(-1)
    assert np.array_equal(alpha.numpy().argmax(-1), plain)


def test_score_specs_round_trip():
    for text in ("msp", "energy_clip", "daedl:qsigmoid:mul", "daedl:ecdf:add_log:5/vacuity"):
        spec = parse_spec(text)
        assert parse_spec(format_spec(spec)) == spec


def test_cross_fit_normalizers_stay_in_unit_interval():
    rng = np.random.default_rng(7)
    lp = rng.normal(size=50)
    for name in ("minmax", "qsigmoid", "ecdf"):
        out = cross_fit_phat(lp, name)
        assert out.shape == lp.shape
        assert np.all(out >= 0) and np.all(out <= 1)


def test_relative_and_partial_mahalanobis_rank_far_points_higher():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 8))
    y = np.array([0] * 30 + [1] * 30)
    X[30:] += 2.0
    far = np.ones((8, 8)) * 15
    rel_id = relative_mahalanobis(X, y, X, pca_dim=4)
    rel_far = relative_mahalanobis(X, y, far, pca_dim=4)
    assert rel_far.mean() > rel_id.mean()
    part_id = partial_mahalanobis(X, y, X, top_k=3)
    part_far = partial_mahalanobis(X, y, far, top_k=3)
    assert part_far["top"].mean() > part_id["top"].mean()
    assert part_far["bottom"].mean() > part_id["bottom"].mean()


def test_cache_round_trip_scores_every_spec(tmp_path):
    rng = np.random.default_rng(9)
    def block(n, k=4, d=6):
        return {
            "logits": rng.normal(size=(n, k)).astype(np.float32),
            "feats": rng.normal(size=(n, d)).astype(np.float32),
            "labels": rng.integers(0, 2, size=n).astype(np.int64),
        }
    arrays = {
        "train": block(40),
        "val": block(16),
        "test": block(16),
        "oos_val": block(8),
        "oos_test": block(8),
    }
    for split in ("oos_val", "oos_test"):
        arrays[split]["labels"][:] = -1
    save_cache(tmp_path, arrays, {"arm": "edl", "seed": 0, "tag": "", "config": {}})
    run = load_cache(tmp_path)
    from bert_daedl.posthoc.scoring import density_maps, phats_for

    _gda, logp = density_maps(run)
    phat = phats_for(logp, "minmax", "train")
    for text in ("msp", "vacuity", "neg_logp", "daedl:minmax:mul/vacuity"):
        spec = parse_spec(text)
        scored = score_array(run, spec, logp=logp, phat=phat if spec.kind == "daedl" else None)
        direct = logit_scores(arrays["test"]["logits"], arm="edl")
        if spec.kind == "plain" and spec.name != "neg_logp":
            np.testing.assert_allclose(scored["test"], direct[spec.name], atol=1e-5)
        else:
            assert scored["test"].shape == (16,)
    # silence unused import if logits_to_alpha stays available for readers
    assert logits_to_alpha(torch.zeros(1, 2)).shape == (1, 2)
