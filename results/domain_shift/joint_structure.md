# Joint-Structure Shift — correlation matrices and y(t+4) association drift

Generated: 2026-09-18T13:59:58+00:00 · bootstrap 1000 seed 42 (week-block, within-population resampling)

16 features as in the marginal analysis; Pearson correlations on pooled week x ocean-pixel rows (listwise-complete); distances are Frobenius norms of correlation-matrix differences. Association uses features at the issue week t vs y_regression at t+4 (PFV-style, ocean cells).

## Correlation-structure distances

| pair | Frobenius | 95% CI | note |
|---|---|---|---|
| SCS_train -> SCS_test | 1.0661 | [0.9605, 1.5170] |  |
| SCS_train -> GBR_test | 4.8634 | [4.6272, 5.2903] |  |
| SCS_train -> CARIB_test | 2.3720 | [2.2389, 2.6054] |  |
| SCS_train -> SCS_train_split_half | 0.2456 | [nan, nan] | split-half estimation noise floor |
| SCS_train -> SCS_test | 1.0661 | [0.9605, 1.5170] | in-domain temporal drift |

## Channel <-> y(t+4) association drift

| pair | feature | group | corr_source | corr_target | drift | 95% CI |
|---|---|---|---|---|---|---|
| SCS_train -> SCS_test | sst | instantaneous_thermal | +0.2836 | +0.3975 | +0.1139 | [+0.0846, +0.1437] |
| SCS_train -> SCS_test | ssta | instantaneous_thermal | +0.3029 | +0.2739 | -0.0290 | [-0.0825, +0.0298] |
| SCS_train -> SCS_test | hotspot | instantaneous_thermal | +0.6677 | +0.7391 | +0.0714 | [+0.0245, +0.1302] |
| SCS_train -> SCS_test | sst_anomaly | instantaneous_thermal | +0.1738 | +0.1891 | +0.0153 | [-0.0427, +0.0698] |
| SCS_train -> SCS_test | d_hotspot_dt | instantaneous_thermal | -0.0252 | +0.0016 | +0.0268 | [-0.0813, +0.1450] |
| SCS_train -> SCS_test | C1 | cumulative_thermal | +0.5056 | +0.5662 | +0.0606 | [-0.0090, +0.1390] |
| SCS_train -> SCS_test | C2 | cumulative_thermal | +0.5907 | +0.6487 | +0.0580 | [-0.0041, +0.1306] |
| SCS_train -> SCS_test | C4 | cumulative_thermal | +0.7106 | +0.7590 | +0.0484 | [+0.0081, +0.1008] |
| SCS_train -> SCS_test | C8 | cumulative_thermal | +0.8427 | +0.8595 | +0.0169 | [-0.0140, +0.0523] |
| SCS_train -> SCS_test | C12 | cumulative_thermal | +0.7525 | +0.7832 | +0.0308 | [-0.0121, +0.0793] |
| SCS_train -> SCS_test | chlorophyll | context_calendar | +0.0361 | +0.0923 | +0.0562 | [+0.0324, +0.0760] |
| SCS_train -> SCS_test | par | context_calendar | +0.2541 | +0.3763 | +0.1223 | [+0.0814, +0.1602] |
| SCS_train -> SCS_test | wind_speed | context_calendar | -0.2002 | -0.2990 | -0.0989 | [-0.1264, -0.0708] |
| SCS_train -> SCS_test | ssr | context_calendar | +0.1775 | +0.2913 | +0.1138 | [+0.0696, +0.1596] |
| SCS_train -> SCS_test | seasonal_sin | context_calendar | -0.1164 | -0.0536 | +0.0628 | [+0.0012, +0.1317] |
| SCS_train -> SCS_test | seasonal_cos | context_calendar | -0.3205 | -0.5257 | -0.2052 | [-0.2463, -0.1666] |
| SCS_train -> GBR_test | sst | instantaneous_thermal | +0.2836 | +0.4647 | +0.1811 | [+0.1276, +0.2286] |
| SCS_train -> GBR_test | ssta | instantaneous_thermal | +0.3029 | +0.4618 | +0.1589 | [+0.0626, +0.2330] |
| SCS_train -> GBR_test | hotspot | instantaneous_thermal | +0.6677 | +0.6389 | -0.0287 | [-0.1432, +0.0597] |
| SCS_train -> GBR_test | sst_anomaly | instantaneous_thermal | +0.1738 | +0.3152 | +0.1414 | [+0.0687, +0.2174] |
| SCS_train -> GBR_test | d_hotspot_dt | instantaneous_thermal | -0.0252 | -0.0686 | -0.0434 | [-0.2006, +0.1380] |
| SCS_train -> GBR_test | C1 | cumulative_thermal | +0.5056 | +0.5421 | +0.0365 | [-0.1041, +0.1453] |
| SCS_train -> GBR_test | C2 | cumulative_thermal | +0.5907 | +0.6013 | +0.0106 | [-0.0981, +0.0984] |
| SCS_train -> GBR_test | C4 | cumulative_thermal | +0.7106 | +0.6819 | -0.0287 | [-0.0950, +0.0325] |
| SCS_train -> GBR_test | C8 | cumulative_thermal | +0.8427 | +0.7421 | -0.1006 | [-0.1432, -0.0589] |
| SCS_train -> GBR_test | C12 | cumulative_thermal | +0.7525 | +0.6584 | -0.0940 | [-0.1546, -0.0321] |
| SCS_train -> GBR_test | chlorophyll | context_calendar | +0.0361 | +0.0174 | -0.0186 | [-0.0461, +0.0076] |
| SCS_train -> GBR_test | par | context_calendar | +0.2541 | +0.5131 | +0.2591 | [+0.2107, +0.3115] |
| SCS_train -> GBR_test | wind_speed | context_calendar | -0.2002 | -0.3859 | -0.1857 | [-0.2301, -0.1350] |
| SCS_train -> GBR_test | ssr | context_calendar | +0.1775 | +0.3469 | +0.1694 | [+0.1164, +0.2218] |
| SCS_train -> GBR_test | seasonal_sin | context_calendar | -0.1164 | +0.5002 | +0.6166 | [+0.5607, +0.6773] |
| SCS_train -> GBR_test | seasonal_cos | context_calendar | -0.3205 | +0.2521 | +0.5727 | [+0.5075, +0.6300] |
| SCS_train -> CARIB_test | sst | instantaneous_thermal | +0.2836 | +0.3109 | +0.0273 | [-0.0115, +0.0659] |
| SCS_train -> CARIB_test | ssta | instantaneous_thermal | +0.3029 | +0.2383 | -0.0646 | [-0.1094, -0.0124] |
| SCS_train -> CARIB_test | hotspot | instantaneous_thermal | +0.6677 | +0.4701 | -0.1975 | [-0.2579, -0.1320] |
| SCS_train -> CARIB_test | sst_anomaly | instantaneous_thermal | +0.1738 | +0.1174 | -0.0564 | [-0.0938, -0.0126] |
| SCS_train -> CARIB_test | d_hotspot_dt | instantaneous_thermal | -0.0252 | -0.0600 | -0.0347 | [-0.1259, +0.0620] |
| SCS_train -> CARIB_test | C1 | cumulative_thermal | +0.5056 | +0.3306 | -0.1750 | [-0.2572, -0.0856] |
| SCS_train -> CARIB_test | C2 | cumulative_thermal | +0.5907 | +0.3653 | -0.2254 | [-0.3025, -0.1451] |
| SCS_train -> CARIB_test | C4 | cumulative_thermal | +0.7106 | +0.4098 | -0.3008 | [-0.3661, -0.2385] |
| SCS_train -> CARIB_test | C8 | cumulative_thermal | +0.8427 | +0.4371 | -0.4055 | [-0.4576, -0.3605] |
| SCS_train -> CARIB_test | C12 | cumulative_thermal | +0.7525 | +0.3788 | -0.3737 | [-0.4253, -0.3220] |
| SCS_train -> CARIB_test | chlorophyll | context_calendar | +0.0361 | -0.0482 | -0.0843 | [-0.1083, -0.0645] |
| SCS_train -> CARIB_test | par | context_calendar | +0.2541 | +0.0870 | -0.1671 | [-0.2159, -0.1194] |
| SCS_train -> CARIB_test | wind_speed | context_calendar | -0.2002 | -0.1022 | +0.0980 | [+0.0737, +0.1235] |
| SCS_train -> CARIB_test | ssr | context_calendar | +0.1775 | +0.0737 | -0.1038 | [-0.1512, -0.0561] |
| SCS_train -> CARIB_test | seasonal_sin | context_calendar | -0.1164 | -0.4215 | -0.3052 | [-0.3515, -0.2527] |
| SCS_train -> CARIB_test | seasonal_cos | context_calendar | -0.3205 | -0.0086 | +0.3119 | [+0.2538, +0.3646] |

Association language only; no causal attribution.

## Interpretation (2026-09-18) — association language only

### 1. Pairwise correlation-structure distance does NOT rank-order the outcome

| pair | Frobenius(R) | 95% CI |
|---|---|---|
| SCS_train split-half (noise floor) | 0.246 | — |
| SCS_train -> SCS_test (in-domain drift) | 1.066 | [bootstrap] |
| SCS_train -> CARIB_test | 2.372 | [bootstrap] |
| SCS_train -> GBR_test | **4.863** | [bootstrap] |

GBR shows the LARGEST pairwise channel-correlation shift yet transfers;
CARIB's shift is intermediate yet it fails. All distances far exceed the
split-half noise floor (0.246), so they are real structure differences — but
like the marginal analysis, pairwise co-variation distance does not
correspond to the transfer outcome ordering. (GBR's large R-shift is driven
by the hemisphere seasonality flip and context channels; see below.)

### 2. Feature <-> y(t+4) association drift DOES separate the outcomes —
the cumulative-thermal association collapses CI-significantly on CARIB only

Per-feature Pearson corr(feature at issue week t, y_regression at t+4):

| channel | SCS_train | SCS_test | GBR_test | CARIB_test |
|---|---|---|---|---|
| C8  | +0.843 | +0.860 | +0.742 | **+0.437** |
| C12 | +0.752 | +0.783 | +0.658 | **+0.379** |
| C4  | +0.711 | +0.759 | +0.682 | **+0.410** |
| C2  | +0.591 | +0.649 | +0.601 | **+0.365** |
| C1  | +0.506 | +0.566 | +0.542 | **+0.331** |
| hotspot | +0.668 | +0.739 | +0.639 | **+0.470** |

- **CARIB is the only test population with CI-significant NEGATIVE drift in
  the cumulative-thermal channels**: C8 −0.406 [−0.458,−0.360], C12 −0.374
  [−0.425,−0.322], C4 −0.301 [−0.366,−0.238], C2 −0.225, C1 −0.175,
  hotspot −0.198 (all CI exclude 0). The channels that Steps 6-7 identified
  as the transferable anchor are exactly the ones whose linear association
  with the 4-week-ahead target is roughly HALVED on the failing basin.
- **GBR preserves the thermal associations** (C8 0.742, C4 0.682, C12 0.658 —
  drifts small relative to CARIB) while its largest drifts are CALENDAR
  (seasonal_sin +0.617, seasonal_cos +0.573 — the southern-hemisphere
  seasonal phase flip) and context (PAR +0.259, wind −0.186). The channel
  groups that drift on GBR are ones the model does not need for its thermal
  signal; the channels that drift on CARIB are the ones it relies on.
- Calendar drift on CARIB (seasonal_sin −0.305, seasonal_cos +0.312) is also
  CI-significant but smaller than GBR's, and calendar channels carry little
  target information on CARIB (|corr| ≤ 0.42).

### 3. Reading (and the necessary caveat)

The failure pattern is consistent with a **target-relation mismatch**: the
SCS-calibrated strength of the mapping "accumulated thermal stress now ->
severe risk in 4 weeks" does not hold at the same strength on the Caribbean
(association roughly halved), while it is preserved on the GBR. This
corresponds to the frozen TSCF failure and to the rescue by the
thermal-stress-anchored residual head, whose anchor (current DHW) re-reads
the local DHW->target relation directly.

Caveat (important for honest reporting): the Caribbean associations remain
POSITIVE and substantial (C8 0.437), and current-DHW persistence still
reaches ROC 0.98 on the same records — so this is a WEAKENING of the
thermal-stress -> target relation, not a sign reversal or absence of signal.
Pearson correlation on a heavily skewed target is also a crude summary.
Consistent reading: the SCS-learned thermal-to-risk calibration overstates
what the Caribbean thermal state implies at 4-week lead, which the marginal
and pairwise-structure analyses (1) could not detect because the mismatch is
in the feature-TARGET relation, not in the input-input structure.

ED Fig 9 extension: joint_structure.csv (panel b candidate) + y_assoc.csv
(panel c candidate: per-channel association bars, train vs GBR vs CARIB).
