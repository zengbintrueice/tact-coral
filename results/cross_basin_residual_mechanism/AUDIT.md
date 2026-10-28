# AUDIT — Cross-basin Residual Mechanism Consistency Test

Generated: 2026-09-30 09:03:13  
Frozen records: `rec_structural_ablation.npz` md5 `4035a19fd30fa4d14151f4c8055dfe59` (verified).

## Protocol adherence

- R = TSAR(seed-42) − PersA; frozen record set; no retraining, no target fitting/tuning, no threshold optimisation.
- Candidate variables read verbatim from the causal v2 datasets at the record issue time; unified complete-case mask (identical records across all parts).
- Partial Spearman controls current_DHW (= PersA×10, the clipped coordinate); raw and partial families FDR-corrected (BH) per basin.
- Paired week-block bootstrap (1,000, seed 42) per basin; the GBR−CARIB difference CI is formed from independent per-basin bootstrap streams (element-wise difference of replicates; the two basins are independent record sets, so no cross-basin pairing is assumed).
- Fixed rank-OLS (8 candidates, fold-internal ECDF maps, GroupKFold 5 by issue week, pooled OOS R²); rank-scale coefficients reported with week-block bootstrap CIs.
- SCS is reported as the source/reference only; the consistency claim is evaluated exclusively on GBR vs CARIB.
- Associative / representation-level interpretation only; no causal claim.

## Pre-specified verdict rule

A variable is **consistent** iff GBR and CARIB partial associations share sign AND both are FDR-significant (q<0.05) AND the GBR−CARIB difference 95% CI includes 0 (homogeneous effect size). Same-sign + both-significant with the difference CI excluding 0 is reported as *same-sign but heterogeneous effect size*; anything else is **not consistent**.

## Verdicts (GBR vs CARIB, partial | current DHW)

| Variable | GBR ρ (q) | CARIB ρ (q) | diff [95% CI] | verdict |
|---|---|---|---|---|
| dDHW_4w | -0.000 (9.6e-01) | +0.208 (0.0e+00) | -0.209 [-0.311, -0.103] | not consistent |
| cumsum_4w | +0.186 (5.4e-148) | +0.542 (0.0e+00) | -0.355 [-0.441, -0.262] | same-sign but heterogeneous effect size |
| cumsum_12w | +0.342 (0.0e+00) | +0.397 (0.0e+00) | -0.055 [-0.161, +0.044] | consistent |
| ssta | +0.138 (2.0e-81) | +0.341 (0.0e+00) | -0.203 [-0.302, -0.105] | same-sign but heterogeneous effect size |
| hotspot | +0.142 (6.2e-86) | +0.517 (0.0e+00) | -0.375 [-0.461, -0.285] | same-sign but heterogeneous effect size |
| chl | +0.201 (1.8e-173) | +0.039 (3.7e-29) | +0.163 [+0.127, +0.200] | same-sign but heterogeneous effect size |
| par | +0.257 (1.1e-283) | +0.387 (0.0e+00) | -0.130 [-0.222, -0.033] | same-sign but heterogeneous effect size |
| wind | -0.202 (1.2e-173) | -0.126 (4.4e-293) | -0.075 [-0.170, +0.017] | consistent |

## Interpretation guard

Statistical association is not mechanism: no claim is made that the residual is produced by, or causally depends on, these variables. Consistency of association across two basins is representation-level evidence only and does not establish a shared causal pathway.

