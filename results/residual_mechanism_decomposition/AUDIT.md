# AUDIT — Residual Mechanism Decomposition

Generated: 2026-09-29 18:44:40  
Frozen record set: `physics_anchored/structural_ablation/rec_structural_ablation.npz` (md5 `4035a19fd30fa4d14151f4c8055dfe59`, verified before run)

## Fixed protocol (adherence)

- R = TSAR (C_anchored, scale 0.4, seed-42 checkpoint) − PersA_DHW, per record, from the frozen record set; no model was retrained.
- No target fitting / target tuning / threshold optimisation anywhere: Part B models are explanatory regressions of R on mechanism variables (fixed OLS form, 5-fold CV grouped by issue week, no hyperparameter search); Part D scores are pre-specified training-free rank means.
- All uncertainty from the paired week-block bootstrap (1,000 replicates, seed 42); identical resampled weeks across compared models/scores.
- FDR: Benjamini–Hochberg within family (raw / partial) per basin.
- Interpretation restricted to associative / representation level; no causal claims.

## Mechanism variables (verbatim project definitions)

| Variable | Definition | Source |
|---|---|---|
| current_DHW | PersA×10 = clip(DHW/10,0,1)×10 (clipped coordinate) | frozen record set (A_PersA_pred) |
| dDHW_4w | y(t) − y(t−4), 4-week change of the clipped DHW coordinate (causal: t−4 ≤ t) | causal v2 y_regression |
| d_hotspot_dt | existing weekly HotSpot-derivative channel (alternate derivative, reported alongside) | causal v2 channel |
| cumsum_4w / cumsum_12w | existing trailing 4/12-week HotSpot exposure channels (persistence) | causal v2 channels |
| ssta, hotspot | existing thermal state channels | causal v2 channels |
| chl, par, wind | existing nonthermal context channels (chlorophyll, PAR, wind_speed) | causal v2 channels |

No window or label definition was changed.

## Complete-case handling

A single complete-case mask over ALL candidate variables is applied so that every part (A–D) uses identical records per basin.

| Basin | records | complete-case | dropped | severe (kept) | dominant NaN source |
|---|---|---|---|---|---|
| SCS | 49322 | 49235 | 87 | 419 | par (87) |
| GBR | 19065 | 19065 | 0 | 676 | current_DHW (0) |
| CARIB | 83025 | 83025 | 0 | 1356 | current_DHW (0) |

## Pre-specified analysis choices

- Primary explanatory model: OLS on standardized predictors. Because R = gate×tanh(·)×0.4 is bounded and zero-inflated, a **pre-fixed rank-OLS** (rank-transformed response and predictors, same folds) is reported alongside as the robust fallback; conclusions are read jointly from both.
- CV: 5-fold GroupKFold grouped by issue week (prevents week-level temporal leakage in CV R²); model form fixed a priori, no tuning.
- ΔR² increments are in-sample, with week-block bootstrap CIs (documented; CV R² reported as the out-of-sample reference).
- Part D combination rule (pre-specified, training-free): equal-weight mean of within-basin percentile ranks of {PersA, dDHW_4w, cumsum_4w, cumsum_12w} (thermal block), optionally + {chl, par, wind} (context block). No coefficients are fitted, so no target information enters the score construction.
- dDHW_4w is the first difference of the clipped coordinate; DHW > 10 degree-weeks is clipped to 1.0 in the coordinate, which slightly compresses both current_DHW and dDHW_4w at extreme heat. This is the coordinate definition the models themselves use.

## Deviations

- None from the requested protocol. `d_hotspot_dt` is reported in addition to dDHW_4w because the code base defines it (per the instruction to report existing alternate derivative definitions).

## Interpretation guard

All associations are representation-level: they describe what the frozen residual covaries with, not a causal mechanism. No claim is made that the residual 'uses' these variables, and no retraining or threshold selection was performed.

