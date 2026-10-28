# AUDIT — Residual Hierarchy / Incremental Information Test

Generated: 2026-09-29 20:38:26  
Frozen record set: `physics_anchored/structural_ablation/rec_structural_ablation.npz` (md5 `4035a19fd30fa4d14151f4c8055dfe59`, verified before run)

## Fixed protocol (adherence)

- PersA = clip(DHW/10,0,1) from the frozen record set; R = TSAR − PersA (TSAR = seed-42 C_anchored, scale 0.4). No model was retrained; no target fitting, target tuning or threshold optimisation was performed anywhere in this test.
- Identical frozen records per basin (single complete-case mask over all hierarchy variables); identical issue-week grouped 5-fold CV for the residual-explanation models.
- All uncertainty: paired week-block bootstrap (1,000, seed 42); the SAME resampled weeks are used across all hierarchy levels, so adjacent level deltas are paired.
- Interpretation restricted to predictive / representation level; no causal claims.

## Pre-specified hierarchy construction

The hierarchy scores are TRAINING-FREE equal-weight means of within-basin percentile ranks (no coefficient is fitted to the target, so no target information enters score construction):

| Level | Components |
|---|---|
| H0 | PersA |
| H1 | PersA + {dDHW_4w, cumsum_4w, cumsum_12w} (thermal trajectory/persistence) |
| H2 | H1 + {chl, par, wind} (nonthermal context) |
| H3 | H2 + R (the frozen TSAR residual) |

The frozen TSAR score itself is reported alongside for the consistency check H3 vs TSAR. Because H1–H3 are rank means, Brier and log-loss are **diagnostic within-family** comparisons, not calibrated probabilities; log-loss uses the pre-fixed clipping 1e-06.

## Complete-case handling

| Basin | records | complete-case | dropped | severe |
|---|---|---|---|---|
| SCS | 49322 | 49235 | 87 | 419 |
| GBR | 19065 | 19065 | 0 | 676 |
| CARIB | 83025 | 83025 | 0 | 1356 |

## Pre-specified analysis choices

- Residual-explanation models: fixed OLS on standardized predictors; out-of-sample R² via deterministic week-grouped 5-fold CV; the paired bootstrap of the out-of-sample ΔR² recomputes the same grouped-fold split (weeks enumerated in sorted order, folded i mod 5) within each replicate — fixed rule, no tuning.
- Verdict rule (pre-specified): the residual's additional information is declared *established* for a basin iff the H3−H2 95% CI excludes 0 in the positive direction for ROC-AUC or PR-AUC; otherwise reported as **not established**.

## Deviations

- None from the requested protocol.

## Verdict summary (pre-specified rule)

| Basin | H3−H2 ΔROC [CI] | H3−H2 ΔPR [CI] | residual extra info |
|---|---|---|---|
| SCS | +0.0006 [-0.0002, +0.0017] | -0.0081 [-0.0161, +0.0032] | not established |
| GBR | +0.0049 [+0.0028, +0.0074] | +0.0263 [+0.0156, +0.0417] | **established** |
| CARIB | +0.0025 [-0.0005, +0.0051] | +0.0138 [+0.0041, +0.0207] | **established** |

## Interpretation guard

Hierarchy levels share all lower-level components by construction; the equal-weight rank mean is one pre-specified combination rule. A failure of H3 to beat H2 does not prove the residual is a re-combination of the listed variables under other combination rules; a success shows only that, under this fixed rule, the residual adds discrimination beyond the explicit candidate set. All statements are predictive/representation-level.

