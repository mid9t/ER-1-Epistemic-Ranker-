# Epistemic-Ranker
This is a research project to investigate the usage of Evidential Deep Learning (EDL) in Early-Exit with Uncertainty Quantification.

## Phase 1 — UQ sanity on controlled text classification
### Goal: prove your EDL head + metrics behave sensibly.
dataset: any stable text classification benchmark
tests: 
- inject label noise ⇒ U(alea)U should rise
- domain shift (topic shift) ⇒ vacuity/density should rise
- calibration curves vs accuracy

## Phase 2 — Pointwise relevance classification (IR-adjacent, simpler than ranking)
### Dataset: MS MARCO passage relevance (or similar).
- Train cross-encoder with EDL head at final layer only.
- Validate:
  - OOD slices (spam/gibberish) don’t look confident (this often fails without density/conflict features)

## Phase 3 — Add early exits + teacher distillation
- Add exit heads at 3/6/9 with the same EDL parameterization.
- Train with:
  - standard relevance loss at each exit
  - distillation from layer-12 logits/scores (for monotonic consistency)
- Collect offline Δℓ labels to train gℓ.
  
## Phase 4 — Robustness hardening (the “Truth Gap” phase)
- Add density-aware feature/scaling
- Add conflict-aware/stability features (light perturbations; optional metamorphic transforms if feasible)
- Evaluate against adversarial lexical overlap and spam corpora.

## Phase 5 — Ranking integration + compute policy eval
- Move to pairwise/listwise training objective.
- Evaluate:
  - NDCG/MRR vs compute budget
  - selective risk curves (quality vs “coverage”/exits)
  - worst-case slices (gibberish / OOD / spam)
