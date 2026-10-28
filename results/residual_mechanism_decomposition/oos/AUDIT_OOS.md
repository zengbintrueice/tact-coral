# AUDIT_OOS — Strict out-of-sample ΔR² upgrade

Generated: 2026-09-30 06:56:43  
Frozen records: `rec_structural_ablation.npz` md5 `4035a19fd30fa4d14151f4c8055dfe59` (verified).
Environment: piecf_env; **GPU unused by design** — pure numpy/sklearn statistics, no neural-network inference.

## Protocol adherence

- R = TSAR(seed-42 C_anchored, scale 0.4) − PersA_DHW; per-basin analyses on the same complete-case record set as the in-sample decomposition (SCS dropped 87 PAR-NaN records; GBR/CARIB none).
- 5-fold GroupKFold grouped by issue week; fold-internal assertion that train/test weeks are disjoint (no week leakage).
- Standardization (means/SDs) fitted **within each training fold** and applied to that fold's test weeks; no global preprocessing.
- Rank-OLS variant: training-fold empirical-CDF maps for predictors and response, also fold-internal; pooled on the rank scale.
- Primary statistic: **pooled OOS R² = 1 − SSE/SST** over all OOS predictions; per-fold R² reported only as a stability check; the fold average is not used.
- ΔR²(M1−M0), ΔR²(M2−M1) computed on the pooled OOS records, which are identical across M0/M1/M2 by construction (same folds, same records).
- Bootstrap: paired week-block (1,000, seed 42) on the pooled OOS predictions — the SAME resampled weeks are applied to M0/M1/M2, so the ΔR² CIs are paired.
- Negative OOS R² is allowed and reported without truncation.
- No future target or test-week information enters any model choice (model form fixed a priori; no tuning anywhere).

## Results (pooled OOS)

| Basin | Model | OLS OOS R² [95% CI] | rank-OLS OOS R² [95% CI] |
|---|---|---|---|
| SCS | M0 | +0.3111 [+0.2504, +0.3589] | +0.5640 [+0.5092, +0.6094] |
| SCS | M1 | +0.4647 [+0.4086, +0.5092] | +0.6785 [+0.6407, +0.7098] |
| SCS | M2 | +0.4647 [+0.4089, +0.5092] | +0.7368 [+0.7101, +0.7597] |
| GBR | M0 | +0.0003 [-0.0093, +0.0014] | +0.1281 [+0.0641, +0.1827] |
| GBR | M1 | +0.0067 [-0.0030, +0.0143] | +0.2718 [+0.1934, +0.3485] |
| GBR | M2 | +0.0197 [+0.0084, +0.0315] | +0.3213 [+0.2461, +0.3937] |
| CARIB | M0 | +0.0015 [-0.0051, +0.0043] | +0.0892 [+0.0619, +0.1129] |
| CARIB | M1 | +0.2109 [+0.1558, +0.2541] | +0.3943 [+0.3432, +0.4426] |
| CARIB | M2 | +0.2273 [+0.1756, +0.2671] | +0.4514 [+0.4047, +0.4950] |

| Basin | ΔR² contrast | OLS point [95% CI] | rank-OLS point [95% CI] |
|---|---|---|---|
| SCS | M1_minus_M0 | +0.1536 [+0.1036, +0.2052] | +0.1145 [+0.0920, +0.1403] |
| SCS | M2_minus_M1 | +0.0001 [-0.0003, +0.0004] | +0.0583 [+0.0409, +0.0759] |
| GBR | M1_minus_M0 | +0.0065 [+0.0024, +0.0167] | +0.1437 [+0.1007, +0.1899] |
| GBR | M2_minus_M1 | +0.0129 [+0.0058, +0.0198] | +0.0495 [+0.0394, +0.0604] |
| CARIB | M1_minus_M0 | +0.2094 [+0.1564, +0.2525] | +0.3051 [+0.2629, +0.3466] |
| CARIB | M2_minus_M1 | +0.0164 [+0.0102, +0.0221] | +0.0571 [+0.0423, +0.0722] |

## Interpretation guard

OOS ΔR² quantifies out-of-sample incremental explanatory/predictive information about the residual R. It is NOT a causal mechanism claim: no intervention, no structural model, and no claim that R 'is produced by' these variables.

