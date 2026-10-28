# Step 7 — Temporal-Scale Thermal-Stress Decomposition Summary

Generated: 2026-09-17 18:58:32  
Mechanistic temporal-scale decomposition. Steps 4/5 remain the formal headline results; Step 6 the feature-group attribution. Cumulative HotSpot channels are physically meaningful historical thermal-stress variables (NOAA CRW DHW = 12-week accumulation of HotSpot ≥ 1 °C).

## Horizon alignment (descriptive only)

| window (weeks) | forecast horizon | window/horizon ratio |
|---|---|---|
| 1 | 4 | 0.25 |
| 2 | 4 | 0.50 |
| 4 | 4 | 1.00 |
| 8 | 4 | 2.00 |
| 12 | 4 | 3.00 |

## Main results (identical Step-4 population)

### SCS — SCS year-based test 2016-2019 (n=49322, severe=419)

| Model | window | ROC AUC | PR AUC | CSI@0.9 | POD@0.9 | FAR@0.9 | F1@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M_C1 | 1 | 0.9876 | 0.3729 | 0.2235 | 0.8496 | 0.7673 | 0.3653 | 0.02809 | 0.1887 |
| M_C2 | 2 | 0.9878 | 0.4244 | 0.2459 | 0.7518 | 0.7324 | 0.3947 | 0.02133 | 0.1454 |
| M_C4 | 4 | 0.9902 | 0.4839 | 0.3115 | 0.7709 | 0.6567 | 0.4750 | 0.01965 | 0.1394 |
| M_C8 | 8 | 0.9898 | 0.5989 | 0.3669 | 0.6611 | 0.5481 | 0.5368 | 0.01687 | 0.1350 |
| M_C12 | 12 | 0.9837 | 0.5119 | 0.3111 | 0.7017 | 0.6415 | 0.4746 | 0.01943 | 0.1346 |
| M_C4_C12 | 4+12 | 0.9597 | 0.2806 | 0.1418 | 0.1790 | 0.5946 | 0.2483 | 0.12268 | 0.4032 |
| M_C2_C4_C12 | 2+4+12 | 0.9890 | 0.5402 | 0.2384 | 0.9093 | 0.7558 | 0.3850 | 0.01992 | 0.0742 |
| M_C1_C4_C12 | 1+4+12 | 0.9902 | 0.5493 | 0.1847 | 0.9666 | 0.8141 | 0.3118 | 0.02289 | 0.0772 |
| FULL | 4+12 (orig) | 0.9828 | 0.5355 | 0.2634 | 0.7494 | 0.7111 | 0.4170 | 0.05535 | 0.2809 |
| FULL_FROZEN | 4+12 (orig, frozen) | 0.9865 | 0.5463 | 0.3374 | 0.5871 | 0.5576 | 0.5046 | 0.07128 | 0.3150 |
| CURRENT_DHW | current (0) | 0.9817 | 0.3177 | 0.1571 | 0.1957 | 0.5568 | 0.2715 | 0.01172 | 0.0665 |

### GBR — GBR formal zero-shot test 2017-2020 (n=19065, severe=676)

| Model | window | ROC AUC | PR AUC | CSI@0.9 | POD@0.9 | FAR@0.9 | F1@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M_C1 | 1 | 0.9213 | 0.4738 | 0.3518 | 0.6938 | 0.5835 | 0.5205 | 0.05816 | 0.2058 |
| M_C2 | 2 | 0.8973 | 0.4298 | 0.2935 | 0.6183 | 0.6415 | 0.4539 | 0.07648 | 0.1946 |
| M_C4 | 4 | 0.9000 | 0.4614 | 0.3323 | 0.6583 | 0.5984 | 0.4989 | 0.06014 | 0.1779 |
| M_C8 | 8 | 0.8146 | 0.4134 | 0.2874 | 0.4068 | 0.5054 | 0.4464 | 0.05008 | 0.1404 |
| M_C12 | 12 | 0.8161 | 0.3337 | 0.2309 | 0.4290 | 0.6667 | 0.3752 | 0.05600 | 0.1487 |
| M_C4_C12 | 4+12 | 0.9119 | 0.4588 | 0.1450 | 0.1538 | 0.2828 | 0.2533 | 0.13366 | 0.3873 |
| M_C2_C4_C12 | 2+4+12 | 0.8234 | 0.4410 | 0.2065 | 0.6864 | 0.7720 | 0.3423 | 0.12659 | 0.2144 |
| M_C1_C4_C12 | 1+4+12 | 0.8380 | 0.4188 | 0.1647 | 0.7441 | 0.8254 | 0.2828 | 0.14680 | 0.2337 |
| FULL | 4+12 (orig) | 0.9562 | 0.5100 | 0.3029 | 0.7470 | 0.6624 | 0.4650 | 0.08740 | 0.3185 |
| FULL_FROZEN | 4+12 (orig, frozen) | 0.9624 | 0.5543 | 0.3648 | 0.6686 | 0.5547 | 0.5346 | 0.09287 | 0.3317 |
| CURRENT_DHW | current (0) | 0.9593 | 0.4164 | 0.2494 | 0.3328 | 0.5011 | 0.3993 | 0.02331 | 0.0736 |

## Transfer summary (GBR − SCS; descriptive, not a formal test)

| Model | window | SCS PR | GBR PR | ΔPR | retention PR | SCS CSI@0.9 | GBR CSI@0.9 | ΔCSI@0.9 |
|---|---|---|---|---|---|---|---|---|
| M_C1 | 1 | 0.3729 | 0.4738 | +0.1009 | 1.2706 | 0.2235 | 0.3518 | +0.1284 |
| M_C2 | 2 | 0.4244 | 0.4298 | +0.0054 | 1.0128 | 0.2459 | 0.2935 | +0.0476 |
| M_C4 | 4 | 0.4839 | 0.4614 | -0.0225 | 0.9536 | 0.3115 | 0.3323 | +0.0209 |
| M_C8 | 8 | 0.5989 | 0.4134 | -0.1855 | 0.6903 | 0.3669 | 0.2874 | -0.0795 |
| M_C12 | 12 | 0.5119 | 0.3337 | -0.1783 | 0.6517 | 0.3111 | 0.2309 | -0.0802 |
| M_C4_C12 | 4+12 | 0.2806 | 0.4588 | +0.1782 | 1.6348 | 0.1418 | 0.1450 | +0.0033 |
| M_C2_C4_C12 | 2+4+12 | 0.5402 | 0.4410 | -0.0992 | 0.8163 | 0.2384 | 0.2065 | -0.0319 |
| M_C1_C4_C12 | 1+4+12 | 0.5493 | 0.4188 | -0.1305 | 0.7624 | 0.1847 | 0.1647 | -0.0200 |
| FULL | 4+12 (orig) | 0.5355 | 0.5100 | -0.0255 | 0.9524 | 0.2634 | 0.3029 | +0.0395 |
| FULL_FROZEN | 4+12 (orig, frozen) | 0.5463 | 0.5543 | +0.0080 | 1.0147 | 0.3374 | 0.3648 | +0.0274 |
| CURRENT_DHW | current (0) | 0.3177 | 0.4164 | +0.0987 | 1.3106 | 0.1571 | 0.2494 | +0.0924 |

## Key paired comparisons (GBR zero-shot; 1000 week-block replicates, seed 42)

- M_C1 − M_C12: ΔPR +0.1401 [+0.0429, +0.2439], ΔCSI@0.9 +0.1209 [+0.0566, +0.1830]
- M_C2 − M_C12: ΔPR +0.0962 [-0.0169, +0.2074], ΔCSI@0.9 +0.0626 [-0.0141, +0.1306]
- M_C4 − M_C12: ΔPR +0.1277 [+0.0531, +0.2191], ΔCSI@0.9 +0.1014 [+0.0507, +0.1575]
- M_C8 − M_C12: ΔPR +0.0798 [+0.0240, +0.1416], ΔCSI@0.9 +0.0565 [+0.0162, +0.1028]
- M_C1 − FULL: ΔPR -0.0362 [-0.1022, +0.0391], ΔCSI@0.9 +0.0489 [+0.0023, +0.0922]
- M_C2 − FULL: ΔPR -0.0801 [-0.1689, +0.0065], ΔCSI@0.9 -0.0094 [-0.0735, +0.0433]
- M_C4 − FULL: ΔPR -0.0486 [-0.1043, +0.0172], ΔCSI@0.9 +0.0294 [-0.0042, +0.0661]
- M_C8 − FULL: ΔPR -0.0965 [-0.1505, -0.0393], ΔCSI@0.9 -0.0156 [-0.0943, +0.0543]
- M_C12 − FULL: ΔPR -0.1763 [-0.2449, -0.1113], ΔCSI@0.9 -0.0720 [-0.1321, -0.0254]

## Answers to the required questions

1. **Highest SCS performance (single-scale, PR AUC primary)**: M_C8 (PR 0.5989, ROC 0.9898, CSI@0.9 0.3669). Multi-scale best on SCS: M_C1_C4_C12.
2. **Highest GBR performance (single-scale)**: M_C1 (PR 0.4738, ROC 0.9213, CSI@0.9 0.3518). Multi-scale best on GBR: M_C4_C12 (PR 0.4588).
3. **Best SCS→GBR retention (PR AUC ratio, single-scale)**: M_C1 (1.271); full ranking: M_C1 1.271, M_C2 1.013, M_C4 0.954, M_C8 0.690, M_C12 0.652
4. **Does C4 outperform C12 for the 4-week horizon?** GBR PR 0.4614 vs 0.3337 (paired ΔCI PR [0.0531, 0.2191]); CSI@0.9 0.3323 vs 0.2309. Yes — CI-supported.
5. **Does C12 remain the most physically canonical representation?** Yes by definition — NOAA CRW DHW is the 12-week accumulation of HotSpot ≥ 1 °C; C12 is the model-input counterpart of that physical quantity. This is a statement about physical canonicality, independent of statistical forecasting utility, and the two must not be conflated.
6. **Do multi-scale representations outperform single-scale?** Best multi (M_C4_C12) GBR PR 0.4588 vs best single (M_C1) 0.4738; paired ΔCI [-0.0629, 0.0280] (see step7_paired_bootstrap.csv for all pairs).
7. **Does current-DHW persistence remain competitive?** CURRENT_DHW GBR: ROC 0.9593, PR 0.4164, CSI@0.9 0.2494 — vs best temporal-scale model and FULL (see tables).
8. **Is the temporal-scale effect statistically supported?** See the paired CIs above and `step7_paired_bootstrap.csv`; no superiority claim is made where the CI includes zero.
9. **Strengthen or weaken the Step-6 conclusion?** Step 6 established that cumulative thermal stress (as a class) is the transferable anchor. Step 7 decomposes that anchor by temporal scale; the outcome below states whether the 12-week scale specifically carries the signal.
10. **Exact claim justified for the manuscript**: see the outcome paragraph below; only CI-supported statements are included.

## OUTCOME A

**Near-term operational discrimination may be driven primarily by recent cumulative exposure, consistent with the four-week forecasting horizon. (Statistical forecasting utility only — this does NOT conclude that 4-week accumulation is physically superior to NOAA's 12-week DHW.)**

Decision logic (transparent): multi-scale-dominance checked first (best multi significantly beats all singles on GBR PR and CSI@0.9: False); then C12-dominance (C12 significantly beats all four shorter windows on both: False); then A (C4/C8 ≥ C12 on GBR PR or CSI@0.9: True); else D. No positive conclusion is forced where confidence intervals include zero.
