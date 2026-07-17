# Epistemic Ranker (public)

Public snapshot of **Milestone 1**: evidential BERT on CLINC150 (CE / EDL / DAEDL arms). Private experiment notebooks, CIFAR `DAEDL/` reference code, and training artifacts are not included.


Milestone 1: leak-free CLINC150 pipeline and three experiment arms
(**CE**, **vanilla EDL**, **DAEDL-style density-scaled EDL**) with diagnostics
that catch an inert density term.

See [`bert_edl_clinc150_implementation_plan.md`](bert_edl_clinc150_implementation_plan.md)
for the full design. Early-exit heads are deferred to Milestone 2.

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
