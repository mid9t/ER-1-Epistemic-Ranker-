# Pre-registration — Banking77-OOS and RoBERTa replication

Copied from the implementation plan, section 2, before any Banking77-OOS or RoBERTa test-set number is computed. Nothing in this file may change after a test-set number has been seen. The commit that adds this file is the pre-registration commit.

## 2.1 Fixed settings

| Setting | Value |
| --- | --- |
| Backbone | bert-base-uncased (RoBERTa study: roberta-base) |
| Input length | 64 tokens (Section 8 checks truncation before any run) |
| Head | dropout 0.1 → Linear 768→256 with spectral norm, no activation → Linear 256→K (K = 50 for Banking77-OOS, 150 for CLINC150) |
| Optimization | AdamW, lr 2×10⁻⁵ encoder / 10⁻³ head, weight decay 0.01, 10% warmup, clip 1.0, batch 32, 5 epochs, epoch with best validation in-scope accuracy |
| EDL arm | exp evidence, clamp ±15, Bayes-risk digamma loss, KL weight annealed to 10⁻³ over 5 epochs |
| Seeds | 0–4 for both arms |
| Outlier exposure | None. The 2,062 ID-OOS training utterances are never loaded. |
| Density D_port | 256-d bottleneck, PCA to 128, whitened, tied covariance |
| Density D_full | 768-d [CLS] vector, no PCA, tied covariance |
| Density fit | All training utterances, file order |
| Normalizers | minmax, qsigmoid, ecdf, fit on training log-densities |
| Couplings | mul; add_log with λ ∈ {1, 2, 5, 10, 20, 40, 80, 160} |
| Validation OOS for selection | Pooled ID-OOS + OOD-OOS validation (730 utterances) |
| Test OOS subsets | ID-OOS (1,080), OOD-OOS (1,000) and pooled (2,080), each against the same 2,000 in-scope test utterances |
| Statistics | Paired stratified bootstrap, 5,000 replicates; SE combined with seed variance; Holm over the three directional claims in 2.3 (H1, H3, positive result) |

The low-variance-only density is not pre-specified. If it is run, it is reported as exploratory.

## 2.2 Selection rules

- **Primary** (unchanged from CLINC150), run once per density: drop configurations INERT on validation in at least half the seeds; pick the highest mean validation AUROC of the coupled vacuity; if the pick is an add_log configuration on an edge of the λ grid, extend the grid once by two values beyond that edge (0.25 and 0.5, or 320 and 640) and repeat; if the pick is still on the edge, report its gain as a lower bound.
- **Secondary** (new, declared here): the same, after first dropping configurations whose coupled validation ECE exceeds the plain validation ECE by more than 0.01. Because add_log leaves the predictive mean unchanged, this rule can only pick an add_log configuration.

## 2.3 Hypotheses and decision rules

All comparisons are on test, pooled OOS, unless stated. "Combined CI" uses SE = √(SE²_boot + s²_seed / 5).

| Claim | Comparison | Supported if | Otherwise |
| --- | --- | --- | --- |
| H1: score matching replicates | CE energy − EDL vacuity | 95% combined CI > 0 | Tie if the 90% CI lies within ±0.002; else reversed or inconclusive |
| H2: multiplicative coupling gives no benefit | qsigmoid×mul − vacuity, separately on D_full and D_port | Upper 95% bound < +0.002, and coupled ECE > plain ECE on 5/5 seeds | Refuted if the lower 95% bound > 0 |
| H3: the density helps more on near-OOS | DiD = (A − vacuity on ID-OOS) − (A − vacuity on OOD-OOS), with A the secondary rule's pick on D_full | 95% combined CI > 0 | Refuted if the 95% CI < 0; else inconclusive |
| Positive coupling result | Primary rule's pick on D_full vs vacuity, and vs D_full alone | Both 95% combined CIs > 0 | Kill branch: written up as a diagnosed negative |

H2 is a bound check, not a test, so it is left out of the Holm correction. Descriptive only, with no test: the per-subset score tables, the agreement in sign between validation and test effects, and the test regret of each selection rule.

## 2.4 Stop and branch conditions

- Any split count differs from the published Banking77-OOS sizes (train 5,905, valid 1,506, test 2,000, validation OOS 730, test OOS 2,080; 50 in-scope intents and 27 held out): stop and fix the data before training.
- More than 2% of any split truncated at 64 tokens: use 96 tokens for both arms, decided before training.
- The full cross (RoBERTa on Banking77-OOS) runs only if the Banking77-OOS and RoBERTa studies disagree on H1 or H2.
