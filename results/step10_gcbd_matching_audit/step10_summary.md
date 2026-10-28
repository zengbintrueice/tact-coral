# Step 10 — GCBD Spatiotemporal Matching Sensitivity Audit

Generated: 2026-09-17 19:55:24  
Robustness audit of Step 9 only. No retraining; frozen checkpoint MD5 7cdbca26cae03fa53be53210940e8ce1. Never claims biological prediction.

## Temporal matching sensitivity (issue = obs - 4wk + Delta, window +/-4wk)

| Delta (wk) | n | severe | ROC(sev vs bg) | MW p | Cliff's d | RR Q5/Q1 | cluster 95% CI |
|---|---|---|---|---|---|---|---|
| -2 | 1910 | 78 | 0.8145 | 4.05e-21 | 0.629 | 34.73 | [8.21, 43.21] |
| -1 | 1906 | 78 | 0.8121 | 8.12e-21 | 0.624 | 33.17 | [7.70, 40.43] |
| +0 | 1898 | 78 | 0.8763 | 2.15e-29 | 0.753 | 35.19 | [8.95, 48.31] |
| +1 | 1874 | 78 | 0.8756 | 2.88e-29 | 0.751 | 37.12 | [8.65, 47.42] |
| +2 | 1868 | 78 | 0.8881 | 4.2e-31 | 0.776 | 37.19 | [8.80, 47.74] |

All predefined windows reported; none selected post hoc.

## Site-level aggregation (primary cohort)

- sites: 384 (with >=1 severe: 37)
- AUROC site-has-severe vs mean risk: 0.8623; vs max risk: 0.8532
- Cliff's delta (severe-sites vs non-severe sites, mean risk): 0.725 (MW p 2.11e-13)
- site-mean quintile RR Q5/Q1: nan (Fisher p 1.44e-11)

| Site-mean-risk quintile | n sites | severe sites | rate |
|---|---|---|---|
| Q1 | 80 | 0 | 0.0000 |
| Q2 | 74 | 1 | 0.0135 |
| Q3 | 78 | 3 | 0.0385 |
| Q4 | 75 | 3 | 0.0400 |
| Q5 | 77 | 30 | 0.3896 |

## Site-preserving permutation negative control

- D_obs = 0.1799
- Perm B (site-blocks among same-size sites, primary): p = 0.0005 (null mean 0.0683, 97.5th 0.1006)
- Perm A (plain label permutation, comparison): p = 0.0005
- RR_obs = 35.19; null finite median 4.52; exceedance p = 0.0005

## VERDICT: ROBUST

Independent severe-bleaching observations remain enriched among higher model-derived thermal-risk states after sensitivity to temporal and spatial matching.

This is a robustness audit only; it does not establish biological prediction.
