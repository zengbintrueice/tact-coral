# Step 9 — Independent GCBD Ecological Validation

Generated: 2026-09-17 19:42:06  
Steps 4-5 remain the formal ML headline results. This step tests whether independent bleaching observations are enriched at higher model thermal risk. NOT biological prediction; no calibration; GCBD fully external.

## Cohort (pinned)

n=1898, any=530, severe=78, background=1368; 384 sites (max cluster 24). Checkpoint MD5 7cdbca26cae03fa53be53210940e8ce1.

## A. Risk distribution by group

| Group | n | mean | median | IQR |
|---|---|---|---|---|
| no_bleaching | 1368 | 0.3339 | 0.3358 | [0.2542, 0.3945] |
| any_bleaching | 530 | 0.4075 | 0.3358 | [0.2485, 0.4122] |
| severe_bleaching | 78 | 0.5000 | 0.4111 | [0.4079, 0.4572] |

## B. Independent ranking (ecological risk enrichment)

| Label | ROC AUC (naive 95% CI) | ROC AUC (site-cluster CI) | PR AUC (naive CI) | PR AUC (cluster CI) |
|---|---|---|---|---|
| any bleaching | 0.5304 (0.531 [0.498, 0.563]) | 0.530 [0.473, 0.583] | 0.4502 (0.451 [0.409, 0.494]) | 0.451 [0.384, 0.514] |
| severe (vs rest) | 0.8477 (0.848 [0.804, 0.890]) | 0.847 [0.785, 0.902] | 0.1719 (0.177 [0.129, 0.233]) | 0.181 [0.122, 0.250] |
| severe (vs background) | 0.8763 (0.876 [0.830, 0.922]) | 0.876 [0.808, 0.936] | 0.3749 (0.382 [0.281, 0.481]) | 0.389 [0.247, 0.529] |

AUROC interpreted as risk ENRICHMENT/ranking, not biological accuracy.

## C. Non-parametric group comparison

| Comparison | MW U | p (one-sided) | Cliff's delta | delta CI (naive) |
|---|---|---|---|---|
| background vs any | 384594 | 0.0197 | 0.061 | [-0.004, 0.126] |
| background vs severe | 93508 | 2.15e-29 | 0.753 | [0.659, 0.844] |

## D. Quintile enrichment (Q1=lowest risk, Q5=highest)

| Event | Bin | n | n event | rate |
|---|---|---|---|---|
| severe | Q1 | 380 | 2 | 0.0053 |
| severe | Q2 | 382 | 4 | 0.0105 |
| severe | Q3 | 378 | 2 | 0.0053 |
| severe | Q4 | 380 | 0 | 0.0000 |
| severe | Q5 | 378 | 70 | 0.1852 |

Primary statistic — RR = P(severe|Q5)/P(severe|Q1):

- severe RR (naive CI): 40.478 [12.685, 82.480]; Fisher p = 1.64e-20
- severe RR (site-cluster CI): 26.623 [8.845, 44.400]
- any RR (site-cluster CI): 1.460 [1.091, 1.914]

## E. Sensitivity (predefined bins)

| Bins | Event | RR | cluster 95% CI | Fisher p |
|---|---|---|---|---|
| quintile | any | 1.440 | [1.091, 1.914] | 1.05e-05 |
| quintile | severe | 35.185 | [8.845, 44.400] | 1.64e-20 |
| tertile | any | 1.087 | [0.834, 1.418] | 0.174 |
| tertile | severe | 17.723 | [6.287, 41.919] | 6.79e-18 |
| quartile | any | 1.267 | [0.965, 1.665] | 0.00327 |
| quartile | severe | 35.295 | [8.699, 43.946] | 3.06e-20 |

## F. Cluster-aware vs naive

Site-level bootstrap reported alongside naive observation-level CIs in step9_bootstrap.csv; cluster-aware CIs are the primary inference.

## OUTCOME A

Independent ecological enrichment CONFIRMED: severe-bleaching observations are significantly enriched in the top model-risk quintile (cluster-aware RR CI excludes 1) and rank above background in predicted thermal risk (MW p<0.05, Cliff's delta>0).

**Interpretation**: independent coral bleaching observations are enriched among locations/times assigned higher model-derived thermal risk. This STRENGTHENS the reading that the frozen TSCF representation estimates a transferable thermal-risk state — it does not directly predict biological bleaching, and no biological probability calibration was fitted.
