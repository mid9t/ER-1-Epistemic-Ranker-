# How to check the published numbers

This folder is a public snapshot of the replication branch (`dd09637`). It contains the README, the frozen analysis rules, the code that produced the tables, and the tables themselves. Raw datasets and the float caches (`cache.npz`, `pooled.npz`, about 465 MB for Banking77 and 1.6 GB for CLINC150) are not included. Every number in the README can be read off the files below without those caches.

Commit `eb040dd`, tag `prereg-replication`, is the pre-registration. `PREREGISTRATION.md` is that text. Banking77 test numbers were computed after that commit.

## CLINC150 milestone table

`results/summary.csv` is the five-seed aggregate cited in the README.

| README cell | Column |
|---|---|
| CE in-scope acc 0.9662 ± 0.0011 | `ce` / `in_scope_acc_mean`, `in_scope_acc_std` |
| EDL vacuity 0.9763 ± 0.0005 | `edl` / `vacuity_auroc_mean`, `vacuity_auroc_std` |
| DAEDL vacuity 0.9713 ± 0.0005 | `daedl` / `daedl_vacuity_auroc_mean`, `daedl_vacuity_auroc_std` |
| CE energy 0.979 | `ce` / `energy_auroc_mean` |
| CE Mahalanobis 0.974 | `ce` / `neg_logp_auroc_mean` |

Score-matched detail is in `results/posthoc/scores/score_table.md`. Coupling, complementarity, and bootstrap tables are the other folders under `results/posthoc/`.

## Banking77-OOS

Data check, before training: `results_b77/data_check.json` (sizes 5905 / 1506 / 2000 / 730 / 2080, worst truncation 0.46% at 64 tokens).

Per seed, `results_b77/<arm>_seed<N>/`:

| File | What it checks |
|---|---|
| `results.json` | in-scope accuracy and ECE for that seed |
| `cache_meta.json` | dataset `banking77_oos`, model `bert-base-uncased`, truncation, `has_oos_type` |
| `diagnostics.json` | whether the training-time density term was active |

Pooled in-scope accuracy and ECE in the README are the mean and sample standard deviation of `results_b77/posthoc/all/scores_port/runs.csv` (`in_scope_acc`, `ece`). `clamp_share` is 0 on every seed.

| README block | File |
|---|---|
| Pooled score table, bottleneck density row | `results_b77/posthoc/all/scores_port/score_table.md` |
| Full `[CLS]` density row | `results_b77/posthoc/all/scores_full/score_table.md` (`neg_logp`) |
| ID-OOS and OOD-OOS vacuity | `results_b77/posthoc/id/scores_port/score_table.md` and `.../ood/...` |
| Selected D_full coupling | `results_b77/posthoc/all/grid_full/selected.json` (`ecdf` add_log λ=5) |
| Selected D_port coupling | `results_b77/posthoc/all/grid_port/selected.json` (`ecdf` × mul) |
| H1, H2, positive result | `results_b77/posthoc/all/bootstrap_full.csv` and `bootstrap_port.csv` |
| H2 ECE, 5/5 seeds | `test_ece_scaled` vs `test_ece_plain` in `grid_full/grid_runs.csv` and `grid_port/grid_runs.csv` for `daedl:qsigmoid:mul/vacuity` |
| H3 | `results_b77/posthoc/h3_did.csv` |
| Verdicts, including Holm | `results_b77/posthoc/verdicts/verdicts.md` |

Quote the combined interval (`combined_ci95_low`, `combined_ci95_high`), not the utterance-only interval. Selection files exist only under `results_b77/posthoc/all/`.

`results_b77/posthoc/all/scores_port/checks.json` records that cached scores reproduce the training-time AUROCs (gap below 10⁻⁴) and that vacuity ranks identically with clipped energy on every EDL run.

## Code

`src/bert_daedl/` is the training and scoring package. `scripts/check_data.py`, `scripts/run_experiment.py`, `scripts/replication_posthoc.sh`, and `scripts/decide_hypotheses.py` are the Banking77 entry points. `tests/test_replication.py` locks the loader, the OOS-subset mask, the λ-grid edge rule, and the H1–H3 decision rules. Those tests do not need the result caches.
