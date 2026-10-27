# Verdicts

Decided from the combined interval, SE = √(SE²_boot + s²_seed / n). The verdict column is the rule in the pre-registration. Holm-adjusted one-sided p-values are reported for the three directional claims and do not replace that rule. H2 is a bound check and is not in the Holm correction.

| Claim | Verdict | Estimate | Combined 95% CI |
|---|---|---:|---|
| H1: score matching replicates | inconclusive | -0.0040 | [-0.0086, +0.0006] |
| H2: multiplicative coupling gives no benefit (D_full) | inconclusive | -0.0026 | [-0.0072, +0.0021] |
| H2: multiplicative coupling gives no benefit (D_port) | refuted | +0.0105 | [+0.0064, +0.0146] |
| H3: the density helps more on near-OOS | inconclusive | +0.0053 | [-0.0005, +0.0111]; a +0.0061, b +0.0008 |
| Positive coupling result | supported | +0.0035 | [+0.0005, +0.0066]; density [+0.0060, +0.0132] |

## Holm

| Claim | one-sided p | Holm p | reject at 0.05 |
|---|---:|---:|---|
| H1 | 0.9559 | 0.9559 | no |
| positive | 0.01116 | 0.03349 | yes |
| H3 | 0.03764 | 0.07528 | no |
