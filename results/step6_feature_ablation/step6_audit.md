# Step 6 — Leakage / Protocol Audit (feature ablation)

Generated: 2026-09-17 18:42:57

Step 6 is a mechanistic feature-contribution experiment. Cumulative HotSpot channels are legitimate thermal-stress predictors (NOAA CRW DHW definition); nothing here implies a shortcut or leakage.

| ID | Requirement | Verdict | Evidence |
|---|---|---|---|
| A1 | Every ablation uses causal-v2 data | **PASS** | train/eval on scs_causal_v2.nc / gbr_causal_v2.nc only |
| A2 | No future target enters any input | **PASS** | PFV t+4 labels only; inputs are trailing channels from causal v2 (30/30 builder checks) |
| A3 | All cumulative features are trailing | **PASS** | hotspot_cumsum_4w/12w recomputed trailing at load time (aligned_dataset), verified in causal v2 audit |
| A4 | Monthly variables remain causal | **PASS** | month-end hold + past-only ffill (causal v2 builder) |
| A5 | Climatology remains cutoff at 2010 | **PASS** | causal v2 ERA5 climatology year<=2010 both domains |
| A6 | Normalization for every ablation uses SCS train 2003-2010 only | **PASS** | SCS_CHANNEL_MEAN/STD (frozen, SCS-train-only) subset per model; subset vectors stored in each training_log.json |
| A7 | GBR is completely zero-shot | **PASS** | no GBR fitting; GBR used only for sliding-window inference of the SCS-trained checkpoints |
| A8 | GBR does not affect weights | **PASS** | checkpoints trained on SCS only; GBR inference is no-grad eval |
| A9 | GBR does not affect normalization | **PASS** | normalization stats are SCS-train buffers baked in checkpoints |
| A10 | GBR does not determine tau | **PASS** | tau ∈ {0.8, 0.9} pinned a priori; no per-ablation threshold selection on any test set |
| A11 | GCBD is not used | **PASS** | no GCBD file in the Step-6 chain |
| A12 | No feature silently retained after removal designation | **PASS** | verified: {"NO_CUMSUM12": true, "NO_CUMULATIVE": true, "THERMAL_ONLY": true} |
| A13 | Feature ordering identical to the declared manifest order | **PASS** | channel order asserted against dataset dyn_feature_names and manifest at runtime |
| A14 | Identical evaluation population across models | **PASS** | all models evaluated on the Step-4 common (t,i,j) set; per-model finiteness verified |
| A15 | Training protocol identical across ablations (except input dim) | **PASS** | seed/optimizer/lr/scheduler/epochs/batch/patience/pos_weight/loss/selection identical; only input_proj dim differs |
| A16 | Step-4/5 files untouched | **PASS** | 14 files hash-verified before/after |

**OVERALL: PASS**

| Check | Verdict | Detail |
|---|---|---|
| Cfinite_SCS_FULL | [SCS] FULL predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_SCS_NO_CUMSUM12 | [SCS] NO_CUMSUM12 predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_SCS_NO_CUMULATIVE | [SCS] NO_CUMULATIVE predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_SCS_THERMAL_ONLY | [SCS] THERMAL_ONLY predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_SCS_FULL_FROZEN | [SCS] FULL_FROZEN predictions finite on shared population | **PASS** | n_nan=0 |
| Canchor_SCS | [SCS] FULL_FROZEN reproduces stored Step-2/4 TSCF predictions | **PASS** | max|diff|=0.00e+00 |
| Cfinite_GBR_FULL | [GBR] FULL predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_GBR_NO_CUMSUM12 | [GBR] NO_CUMSUM12 predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_GBR_NO_CUMULATIVE | [GBR] NO_CUMULATIVE predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_GBR_THERMAL_ONLY | [GBR] THERMAL_ONLY predictions finite on shared population | **PASS** | n_nan=0 |
| Cfinite_GBR_FULL_FROZEN | [GBR] FULL_FROZEN predictions finite on shared population | **PASS** | n_nan=0 |
| Canchor_GBR | [GBR] FULL_FROZEN reproduces stored Step-2/4 TSCF predictions | **PASS** | max|diff|=0.00e+00 |
| C99 | Step-4/5 files not overwritten (hashes identical) | **PASS** | 14 files |
