# AUDIT — DHW-Conditioned Basin Susceptibility Test

Generated: 2026-09-30 19:58:25  
Frozen records md5 `4035a19fd30fa4d14151f4c8055dfe59` (verified); seed-42 TSAR / PersA from the frozen record set. Environment: piecf_env (pure statistics; no GPU inference — no model is run).

## Protocol adherence

- current DHW = PersA×10 (clipped coordinate), fixed 0.5 °C-week bins; only bins with ≥40 records in EVERY basin are used (20 common bins).
- Future severe = y(t+4) ≥ 0.8 from the frozen record targets; no threshold was optimised, no model retrained, no target fitting.
- All CIs: week-block bootstrap (1,000, seed 42) within basin; basin-pair differences use independent per-basin streams (no cross-basin pairing).
- Within-bin R tertiles use boundaries fixed from the full record set (outcome-free); the pooled conditional effect is the bin-count-weighted mean of (Q3−Q1) severe-rate changes.
- A vs B comparison uses a pre-specified training-free aggregation (mean of within-basin percentile ranks of DHW and R); no coefficients fitted; paired week-block bootstrap on shared resampled weeks.
- Language: associative / conditional risk. NOT causal mechanism.

## Pre-specified verdicts

1. Basin-specific susceptibility at matched DHW: 0/20 common bins have CARIB−GBR risk-difference CIs excluding 0 → **not established**.

2a. R separates within matched DHW (SCS): +0.0019 [-0.0020, +0.0050] → **not established**.
2b. R separates within matched DHW (GBR): +0.0466 [+0.0266, +0.0642] → **established**.
2c. R separates within matched DHW (CARIB): +0.0232 [+0.0151, +0.0304] → **established**.

3. GBR/CARIB residual-pattern sign consistency: GBR +0.0466 vs CARIB +0.0232 → sign-consistent.

## Interpretation guard

Rates are conditional-risk descriptions of the frozen record set. 'Susceptibility' means higher observed future severe-thermal-event frequency at matched current DHW; it is not a causal or biological claim.

