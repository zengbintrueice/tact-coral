# Step 4 — Definitive Causal Zero-Shot Evaluation of TSCF (Frozen v2 Protocol)

Generated: 2026-09-17 18:26:01

## Frozen protocol

| Item | Value |
|---|---|
| Checkpoint | `results/model_final_Btail3/model.pt` (MD5 `7cdbca26cae0…`, SCS-trained, frozen; directory name is a stale label — content is the D1 selected formal model) |
| Normalization | SCS training years 2003–2010 only, baked inside checkpoint |
| Climatology / features | causal v2 (ERA5 year≤2010 cutoff; 30/30 causality checks PASS) |
| SCS splits | train 2003–2010 / val 2011–2015 / **test 2016–2019** |
| GBR slices | 2003–2010 & 2014–2016 = protocol/diagnostic only (no fitting); 2011–2013 unused; **test 2017–2020** |
| Severe threshold | y ≥ 0.8 (fixed) |
| Decision threshold | **τ = 0.90, pinned a priori** (sweep diagnostic only) |
| Lead / history | 4 weeks / 24 weeks |
| Verification | PFV t+4, prediction-year filtering, shared common valid masks |
| GBR refitting | none (no weight update, no normalization recalculation, no threshold tuning) |

## Main results (identical samples & masks for all methods)

### SCS — SCS year-based test 2016-2019

| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.8 | CSI@0.9 | POD@0.8 | POD@0.9 | FAR@0.8 | FAR@0.9 | F1@0.8 | F1@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | 49322 | 419 | 0.0085 | 0.9865 | 0.5463 | 0.3031 | 0.3374 | 0.7661 | 0.5871 | 0.6660 | 0.5576 | 0.4652 | 0.5046 | 0.07128 | 0.3150 |
| current-DHW persistence | 49322 | 419 | 0.0085 | 0.9817 | 0.3177 | 0.2021 | 0.1571 | 0.3222 | 0.1957 | 0.6484 | 0.5568 | 0.3362 | 0.2715 | 0.01172 | 0.0665 |
| cumulative-only (12wk HotSpot) | 49322 | 419 | 0.0085 | 0.9720 | 0.1957 | 0.1419 | 0.1746 | 0.7446 | 0.6229 | 0.8509 | 0.8048 | 0.2485 | 0.2973 | 0.04583 | 0.1778 |

**Δ (TSCF − current-DHW persistence)**: ROC +0.0047, PR AUC +0.2286, CSI@0.9 +0.1804, POD@0.9 +0.3914, FAR@0.9 +0.0008, F1@0.9 +0.2331

**Δ (TSCF − cumulative-only)**: ROC +0.0145, PR AUC +0.3506, CSI@0.9 +0.1629, POD@0.9 -0.0358, FAR@0.9 -0.2472, F1@0.9 +0.2073

### GBR — GBR formal zero-shot test 2017-2020

| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.8 | CSI@0.9 | POD@0.8 | POD@0.9 | FAR@0.8 | FAR@0.9 | F1@0.8 | F1@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | 19065 | 676 | 0.0355 | 0.9624 | 0.5543 | 0.3522 | 0.3648 | 0.7751 | 0.6686 | 0.6078 | 0.5547 | 0.5209 | 0.5346 | 0.09287 | 0.3317 |
| current-DHW persistence | 19065 | 676 | 0.0355 | 0.9593 | 0.4164 | 0.2719 | 0.2494 | 0.4275 | 0.3328 | 0.5725 | 0.5011 | 0.4275 | 0.3993 | 0.02331 | 0.0736 |
| cumulative-only (12wk HotSpot) | 19065 | 676 | 0.0355 | 0.9318 | 0.2490 | 0.2141 | 0.2141 | 0.5962 | 0.4808 | 0.7495 | 0.7215 | 0.3527 | 0.3527 | 0.05657 | 0.1550 |

**Δ (TSCF − current-DHW persistence)**: ROC +0.0031, PR AUC +0.1379, CSI@0.9 +0.1154, POD@0.9 +0.3358, FAR@0.9 +0.0536, F1@0.9 +0.1353

**Δ (TSCF − cumulative-only)**: ROC +0.0306, PR AUC +0.3053, CSI@0.9 +0.1507, POD@0.9 +0.1879, FAR@0.9 -0.1668, F1@0.9 +0.1819

## Threshold sweep (diagnostic only — τ = 0.90 remains the pinned primary)

Full sweep (τ = 0.10 … 0.95, step 0.05) for all methods and domains: `causal_zero_shot_threshold_sweep_diagnostic.csv`. The sweep was NOT used to select or adjust the reported τ = 0.90.

## Interpretation (primary scientific reading)

The thermal-stress-aware model is evaluated as zero-shot transfer of **thermal-risk estimation**, not as transferable biological bleaching prediction. (1) The physically standardized cumulative thermal-stress state carries strong transferable ranking information (cumulative-only ROC AUC 0.9318 under GBR zero-shot). (2) Current DHW provides a strong persistence reference. (3) TSCF adds value mainly in rare-event precision/recall and fixed-threshold severe-risk detection (PR AUC and CSI@0.9 deltas), not in global ranking (ROC deltas ≈ 0.001–0.03). (4) The model does not establish transferable biological bleaching prediction. (5) GBR results represent zero-shot transfer of thermal-risk estimation. This model is thermal-stress-guided / thermal-stress-aware; it is NOT physics-informed.

## Leakage / causality audit — final **PASS**

| ID | Requirement | Verdict | Evidence |
|---|---|---|---|
| A1 | No future target enters X (target y(t+4) used only as label) | **PASS** | build_causal_dataset_v2.py 30/30 checks; target at t+4 never an input; PFV verification at t+4 |
| A2 | All dynamic derived features are trailing | **PASS** | d_hotspot_dt / cumsum_4w / cumsum_12w recomputed trailing in aligned_dataset._compute_derived_channels; GBR D-checks max_err <= 6.7e-05 |
| A3 | 4wk/12wk cumulative HotSpot contain no future observations | **PASS** | trailing (t-3..t] / (t-11..t] windows verified bit-exact on causal v2 (causal_availability_audit.md) |
| A4 | Monthly variables use only causally available values | **PASS** | month-end hold + past-only ffill (M_wind_speed/M_ssr/M_sst_anomaly/M_hold_avail checks PASS) |
| A5 | No linear interpolation introduces future monthly information | **PASS** | legacy 'linear' path removed from v2 builder; causal hold verified vs raw OC-CCI recompute (max diff 1.9e-8) |
| A6 | Climatology uses only data available by end of SCS training | **PASS** | ERA5 climatology year<=2010 both domains (A_clim_cutoff/A2_not_fullrecord PASS; SCS basis 96 months, GBR 312 months) |
| A7 | Normalization uses SCS training period only | **PASS** | ChannelNormalizer stats baked in checkpoint, computed from SCS train years 2003-2010 only (frozen 2026-07-31). Note: stats predate the causal v2 rebuild; sst_anomaly mean/std differ from normalization_v2.json by the causal-shift only; model NOT refit, so baked stats ARE the frozen source-domain state |
| A8 | GBR data do not alter model weights | **PASS** | checkpoint loaded once, eval mode, no optimizer/gradient step on GBR; sliding-window inference only |
| A9 | GBR data do not alter normalization | **PASS** | SCS statistics fixed for the entire GBR zero-shot protocol (paths.py / gbr_zeroshot_evaluation.py:179-181) |
| A10 | GBR data do not determine tau | **PASS** | tau=0.90 pinned a priori from source-domain (SCS) optimization; GBR val (2014-2016) used only for labelled diagnostic/adaptation analyses, excluded from this evaluation |
| A11 | GCBD observations are not used | **PASS** | evaluation targets are CRW 5km Jensen-aware DHW only; no GCBD file read in the evaluation chain |
| A12 | Static features are not falsely claimed to be used | **PASS** | FinalSingleHeadModel forward does not consume x_static (forward_features ignores static channels); paper must not claim static-feature usage |

**OVERALL: PASS** — exact frozen source-domain checkpoint; exact SCS normalization; causal v2 preprocessing; GBR 2017–2020 formal test; τ=0.90 fixed a priori; no GBR fitting; no future information; reproducible predictions (source arrays SHA256-locked in `causal_zero_shot_protocol.json`); identical masks for TSCF and baselines.

## Files

- `causal_zero_shot_results.json`
- `causal_zero_shot_results.csv`
- `causal_zero_shot_threshold_sweep_diagnostic.csv`
- `causal_zero_shot_predictions.npz`
- `causal_zero_shot_protocol.json`

Previous experiment results were not modified or overwritten.
