# AUDIT — Final Mechanism Discrimination Test

Generated: 2026-09-30 21:15:55  
Frozen records md5 `4035a19fd30fa4d14151f4c8055dfe59` (verified).

## Protocol adherence

- R = TSAR(seed-42) − PersA; frozen record set; single complete-case mask; no retraining / target fitting / target tuning / threshold optimisation.
- DHW-matched surface: fixed 0.5-DHW bins on the clipped coordinate; pre-specified merge rule (keep bins with n≥200 and severe≥10, merge with the adjacent more-populated bin, drop the tail if unsatisfiable); R tertiles within bin (quartile sensitivity for bins n≥600); Q1→Q3 severe-rate difference/risk ratio + in-bin Spearman; week-block bootstrap CI (1,000, seed 42) within each bin.
- Pooled gradient = record-weighted mean of per-bin Q3−Q1 differences; bootstrap basin-wide over issue weeks.
- Mechanism ladder: training-free percentile-rank means (no fitted coefficient); paired week-block bootstrap with the same resampled weeks across M0/M1/M2/M2+R/TSAR.
- E4: fixed rank-OLS (8 candidates, fold-internal ECDF maps, GroupKFold 5 by issue week, pooled OOS R²).
- Associative / representation-level interpretation only.

## Evidence matrix (pre-specified)

| Basin | E1 partial assoc excl 0 | E4 OOS R² of R | E5 M2+R gain |
|---|---|---|---|
| SCS | True | +0.746 | False |
| GBR | True | +0.315 | True |
| CARIB | True | +0.461 | True |

GBR−CARIB pooled-gradient difference 95% CI: [-0.0222, +0.4306] (excludes 0 = basin-specific effect size).

## Interpretation guard

All quantities are associative. The evidence matrix tests whether the residual behaves like a basin-specific susceptibility signal under the pre-specified criteria; it does not establish a causal mechanism, and a failure of any criterion is reported as NOT ESTABLISHED rather than resolved in favour of H1.

