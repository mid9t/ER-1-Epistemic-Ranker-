# Epistemic Ranker (public)

Public snapshot of **Milestone 1**: evidential BERT on CLINC150 (CE / EDL / DAEDL arms). Private experiment notebooks, CIFAR `DAEDL/` reference code, and training artifacts are not included.

Milestone 1: leak-free CLINC150 pipeline and three experiment arms
(**CE**, **vanilla EDL**, **DAEDL-style density-scaled EDL**) with diagnostics
that catch an inert density term.

See [`bert_edl_clinc150_implementation_plan.md`](bert_edl_clinc150_implementation_plan.md)
for the full design. Early-exit heads are deferred to Milestone 2.

## The bug story, and what we found

Density-aware evidential models (DAEDL) scale Dirichlet evidence by a
feature-space density \(\hat{p}(x)\). On our CIFAR replication, that term
could look healthy in the code path while doing almost nothing: min-max
normalization squashed ID log-densities into a tiny band near 1, so
\(\exp(z \odot \hat{p}) \approx \exp(z)\). AUROC still looked fine because
it was vanilla EDL wearing a costume. We call that failure mode **INERT**.

This repo treats that as a first-class bug class. Every DAEDL run writes
`diagnostics.json` with a **materiality gate** (rank correlation of
\(\alpha_0\), relative \(\Delta\alpha_0\), prediction flip-rate). Configs
marked `density_effect: INERT` are disqualified even if AUROC looks good.
We also redesign normalization for NLP (`qsigmoid` / `ecdf`, not only
min-max), default to tied-covariance GDA with PCA+whitening on BERT
features, and keep a single score convention: **higher = more OOS**.

### Milestone 1 results (CLINC150-plus, 5 seeds)

BERT-base, SN bottleneck \(768\to256\), 5 epochs. Primary scores: CE=`msp`,
EDL=`vacuity`, DAEDL=`daedl_vacuity` (`qsigmoid` × `mul`).

| Arm | In-scope acc | Primary OOS AUROC | FPR@95 |
|-----|-------------:|------------------:|-------:|
| CE | 0.9662 ± 0.0011 | 0.9655 ± 0.0009 | 0.129 ± 0.005 |
| EDL | 0.9653 ± 0.0009 | **0.9763 ± 0.0005** | **0.101 ± 0.005** |
| DAEDL | 0.9654 ± 0.0008 | 0.9713 ± 0.0005 | 0.115 ± 0.003 |

House CE baselines (same runs): energy **0.979**, Mahalanobis (`neg_logp`)
**0.974**, MSP 0.966. Raw GDA density alone separates OOS strongly
(val AUROC ~0.97–0.98). Materiality on DAEDL seeds is **ACTIVE** (not the
CIFAR inert trap): density moves \(\alpha_0\) (median relative change ~0.80),
but `daedl_vacuity` still underperforms plain EDL vacuity by ~0.5 pts AUROC
and fails the inclusion gate (lift must exceed seed noise).

**Takeaway:** on near-OOD intent detection, EDL vacuity is a clean win over
CE MSP with almost no accuracy cost; density scaling is measurable but does
not improve the headline score under the default combine rule. The writeup
keeps CE+energy / CE+Mahalanobis / EDL as the table, and DAEDL as a
diagnosed negative rather than a silent no-op.

Numbers above are from `results/summary.csv` (seeds 0–4).

## Layout


```
src/bert_daedl/          # package (data, model, losses, density, eval, train)
scripts/run_experiment.py
scripts/aggregate_results.py
configs/default.yaml
tests/
DAEDL/                   # CIFAR reference — do not modify
```

## Setup

```bash
# reuse the DAEDL venv, or create a fresh one
uv venv .venv && uv pip install -r requirements.txt
export PYTHONPATH=src
```

Local CLINC150 JSON is expected at `data/CLINC150/data_full.json`
(fallback when HF `clinc_oos` is unavailable; `datasets>=3` drops script support).

## Quick commands

```bash
make test                          # unit tests (excludes slow smoke)
make ce SEED=0                     # CE baseline
make edl SEED=0
make daedl SEED=0 NORM=qsigmoid MODE=mul
make seeds ARM=ce                  # seeds 0..4
make aggregate
```

Or:

```bash
PYTHONPATH=src python scripts/run_experiment.py --arm ce --seed 0 --device mps
```

## Arms

| Arm | Train loss | Scores |
|-----|------------|--------|
| `ce` | cross-entropy | msp, entropy, energy, neg_logp |
| `edl` | EDL (EXP + KL anneal) | + vacuity, pred_entropy, mutual_info, … |
| `daedl` | same weights as `edl` (post-hoc) | + `daedl_*` density-scaled scores |

Selection uses `id_val` + `oos_val` only; test is touched once per chosen config.
Every run writes `results/<arm>_seed<N>/results.json` and `diagnostics.json`
(`density_effect: ACTIVE|INERT`).
