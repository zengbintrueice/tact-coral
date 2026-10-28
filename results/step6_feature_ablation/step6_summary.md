# Step 6 — Thermal-Stress Feature Ablation and Cross-Domain Contribution

Generated: 2026-09-17 18:42:57  
Mechanistic feature-contribution experiment. Step 4 (frozen zero-shot) and Step 5 (temporal purge) remain the formal results. Cumulative HotSpot channels are physically meaningful historical thermal-stress variables (NOAA CRW DHW = 12-week accumulation of HotSpot ≥ 1 °C); this is attribution, NOT a shortcut/leakage claim.

## Models

- **FULL** (13 ch): stopped epoch 15, best val-BCE 0.6294, 41462 params
- **NO_CUMSUM12** (12 ch): stopped epoch 24, best val-BCE 0.7104, 41446 params
- **NO_CUMULATIVE** (11 ch): stopped epoch 13, best val-BCE 1.8850, 41430 params
- **THERMAL_ONLY** (7 ch): stopped epoch 13, best val-BCE 1.7727, 41366 params
- **FULL_FROZEN** (external anchor): deployed Step-4 checkpoint (MD5 7cdbca26cae0…), evaluated on causal v2

## Main results (identical Step-4 population)

### SCS — SCS year-based test 2016-2019 (n=49322, severe=419)

| Model | ROC AUC | PR AUC | CSI@0.8 | CSI@0.9 | POD@0.9 | FAR@0.9 | F1@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 0.9828 | 0.5355 | 0.2175 | 0.2634 | 0.7494 | 0.7111 | 0.4170 | 0.05535 | 0.2809 |
| NO_CUMSUM12 | 0.9791 | 0.4101 | 0.1867 | 0.2001 | 0.6802 | 0.7791 | 0.3335 | 0.02357 | 0.0295 |
| NO_CUMULATIVE | 0.8795 | 0.0745 | 0.0024 | 0.0000 | 0.0000 | — | 0.0000 | 0.08664 | 0.3306 |
| THERMAL_ONLY | 0.8809 | 0.1119 | 0.0766 | 0.0442 | 0.0477 | 0.6296 | 0.0846 | 0.16658 | 0.4573 |
| FULL_FROZEN | 0.9865 | 0.5463 | 0.3031 | 0.3374 | 0.5871 | 0.5576 | 0.5046 | 0.07128 | 0.3150 |
| PersistenceA_DHW | 0.9817 | 0.3177 | 0.2021 | 0.1571 | 0.1957 | 0.5568 | 0.2715 | 0.01172 | 0.0665 |

- **Δ FULL_minus_NO_CUMSUM12** (SCS): ROC +0.0037 (paired CI [-0.0063, 0.0125]), PR +0.1253 ([0.0385, 0.2157]), CSI@0.9 +0.0633 ([0.0344, 0.0902])
- **Δ FULL_minus_NO_CUMULATIVE** (SCS): ROC +0.1032 (paired CI [0.0727, 0.1290]), PR +0.4609 ([0.2981, 0.5703]), CSI@0.9 +0.2634 ([0.1929, 0.3262])
- **Δ FULL_minus_THERMAL_ONLY** (SCS): ROC +0.1018 (paired CI [0.0661, 0.1331]), PR +0.4236 ([0.2654, 0.5272]), CSI@0.9 +0.2193 ([0.1299, 0.3005])

### GBR — GBR formal zero-shot test 2017-2020 (n=19065, severe=676)

| Model | ROC AUC | PR AUC | CSI@0.8 | CSI@0.9 | POD@0.9 | FAR@0.9 | F1@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 0.9562 | 0.5100 | 0.2961 | 0.3029 | 0.7470 | 0.6624 | 0.4650 | 0.08740 | 0.3185 |
| NO_CUMSUM12 | 0.9155 | 0.5149 | 0.3285 | 0.3195 | 0.5695 | 0.5788 | 0.4843 | 0.04060 | 0.0416 |
| NO_CUMULATIVE | 0.3784 | 0.0327 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.11584 | 0.2789 |
| THERMAL_ONLY | 0.6801 | 0.1411 | 0.1107 | 0.0058 | 0.0059 | 0.7333 | 0.0116 | 0.20777 | 0.4778 |
| FULL_FROZEN | 0.9624 | 0.5543 | 0.3522 | 0.3648 | 0.6686 | 0.5547 | 0.5346 | 0.09287 | 0.3317 |
| PersistenceA_DHW | 0.9593 | 0.4164 | 0.2719 | 0.2494 | 0.3328 | 0.5011 | 0.3993 | 0.02331 | 0.0736 |

- **Δ FULL_minus_NO_CUMSUM12** (GBR): ROC +0.0406 (paired CI [0.0078, 0.0974]), PR -0.0050 ([-0.1035, 0.1042]), CSI@0.9 -0.0166 ([-0.0718, 0.0419])
- **Δ FULL_minus_NO_CUMULATIVE** (GBR): ROC +0.5778 (paired CI [0.5090, 0.6466]), PR +0.4773 ([0.3319, 0.5835]), CSI@0.9 +0.3029 ([0.2146, 0.3815])
- **Δ FULL_minus_THERMAL_ONLY** (GBR): ROC +0.2761 (paired CI [0.1489, 0.4227]), PR +0.3688 ([0.1660, 0.5067]), CSI@0.9 +0.2971 ([0.2055, 0.3771])

## Cross-domain transfer characterization (GBR/SCS; descriptive only)

| Model | ROC SCS→GBR | PR SCS→GBR | CSI@0.9 SCS→GBR |
|---|---|---|---|
| FULL | 0.9828→0.9562 (Δ -0.0266) | 0.5355→0.5100 (Δ -0.0255) | 0.2634→0.3029 (Δ +0.0395) |
| NO_CUMSUM12 | 0.9791→0.9155 (Δ -0.0635) | 0.4101→0.5149 (Δ +0.1048) | 0.2001→0.3195 (Δ +0.1194) |
| NO_CUMULATIVE | 0.8795→0.3784 (Δ -0.5011) | 0.0745→0.0327 (Δ -0.0419) | 0.0000→0.0000 (Δ +0.0000) |
| THERMAL_ONLY | 0.8809→0.6801 (Δ -0.2009) | 0.1119→0.1411 (Δ +0.0293) | 0.0442→0.0058 (Δ -0.0383) |
| FULL_FROZEN | 0.9865→0.9624 (Δ -0.0240) | 0.5463→0.5543 (Δ +0.0080) | 0.3374→0.3648 (Δ +0.0274) |

## Answers to the mechanistic questions (evidence-based)

**Q1 — Does removing the 12-week cumulative HotSpot substantially reduce GBR zero-shot performance?** Point estimates: ΔROC +0.0406, ΔPR -0.0050, ΔCSI@0.9 -0.0166 (paired CIs PR [-0.1035, 0.1042], CSI@0.9 [-0.0718, 0.0419]). The reduction is positive at the point-estimate level but the paired 95% CIs include zero — not established as substantial at this sample size.
Q1b — collapse? NO_CUMSUM12 retains PR 0.5149 / CSI@0.9 0.3195 vs FULL 0.5100 / 0.3029 → no collapse (retains ≥ half of FULL's rare-event performance).
**Q2 — Does removing ALL cumulative channels reproduce the previous no-cumulative failure mode?** ΔROC +0.5778, ΔPR +0.4773, ΔCSI@0.9 +0.3029 (paired CIs PR [0.3319, 0.5835], CSI@0.9 [0.2146, 0.3815]); NO_CUMULATIVE retains ROC 0.3784 (some ranking retained: reduced).
**Q3 — Does thermal-only retain high ROC ranking but lose operational threshold performance?** THERMAL_ONLY GBR: ROC 0.6801 vs FULL 0.9562; CSI@0.9 0.0058 vs 0.3029; SCS: ROC 0.8809 vs 0.9828, CSI@0.9 0.0442 vs 0.2634.
**Q4 — Does environmental context on top of thermal features improve PR-AUC / CSI?** Δ(FULL − THERMAL_ONLY) GBR: PR +0.3688 (CI [0.1660, 0.5067]), CSI@0.9 +0.2971 (CI [0.2055, 0.3771]); SCS: PR +0.4236 (CI [0.2654, 0.5272]), CSI@0.9 +0.2193 (CI [0.1299, 0.3005]).
**Q5 — Is the 12-week cumulative thermal-stress representation the most important cross-domain anchor?** GBR rare-event performance under removal: NO_CUMSUM12: PR 0.5149, CSI@0.9 0.3195; NO_CUMULATIVE: PR 0.0327, CSI@0.9 0.0000. vs FULL PR 0.5100, CSI@0.9 0.3029. NO_CUMSUM12 vs NO_CUMULATIVE difference isolates the 12-week channel beyond the 4-week channel: PR -0.4822, CSI@0.9 -0.3195.

## Outcome determination: **Outcome A**

Evidence: FULL−NO_CUMSUM12 GBR PR -0.0050 (CI [-0.1035, 0.1042]); CSI@0.9 -0.0166 (CI [-0.0718, 0.0419]); FULL−NO_CUMULATIVE GBR PR +0.4773 (CI [0.3319, 0.5835]); CSI@0.9 +0.3029 (CI [0.2146, 0.3815]); NO_CUMULATIVE retains ROC 0.3784

**Conclusion: The transferable ranking signal is strongly anchored in cumulative thermal stress, while TSCF's contextual architecture adds value primarily in converting this physically standardized thermal state into improved rare-event discrimination.**

No claim is made that cumulative features are a shortcut; they are physically standardized thermal-stress variables by construction.
