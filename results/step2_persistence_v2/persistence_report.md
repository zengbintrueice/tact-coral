# Persistence Baseline v2 — Strong Physics-Based Persistence vs TSCF

Generated: 2026-09-17 18:25:43  
Data: causal v2 datasets (`scs_causal_v2.nc`, `gbr_causal_v2.nc`)  
Protocol: Pure Forecasting Verification — prediction issued at week t using information available at t only, verified against the target at t+4 weeks; prediction-year filtering; severe threshold y ≥ 0.8; detection thresholds **pinned** at τ ∈ {0.8, 0.9} (never selected on the test set).

## Baseline definitions

**These are strong physics-based persistence baselines — not naive baselines.** Carrying current thermal stress forward is the operational industry standard (NOAA CRW DHW outlooks are persistence-anchored), so beating them is the relevant bar for added value.

- **Baseline A — DHW persistence**: `yhat_persistence = clip(DHW_t / 10, 0, 1)` (current Jensen-aware DHW carried forward; identical target definition).
- **Baseline B — 12-week cumulative HotSpot**: `yhat_cumsum12 = clip(hotspot_cumsum_12w_t / 10, 0, 1)` (the model's own trailing 12-week thermal-accumulation channel, carried forward).
- **TSCF**: deployed single-head model, sliding-window inference (`model_final_Btail3/model.pt`).

## Evaluation sets

- **own**: each method evaluated where its own prediction is finite (persistence is undefined where the current DHW/HotSpot is missing).
- **common**: all methods evaluated on the shared valid mask (finite TSCF & Baseline A & Baseline B & finite target) — the like-for-like comparison.

## Publication-ready comparison table (common set, τ = 0.9)

ROC AUC / PR AUC / CSI shown as value [95% week-block bootstrap CI]. Brier = mean (ŷ − y)²; ECE = 10-bin expected calibration error (diagnostic only).

### SCS — SCS year-based test (2016-2019) (grid 21x23)

| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | 49322 | 419 | 0.0085 | 0.9865 [0.9784, 0.9921] | 0.5463 [0.4080, 0.6378] | 0.3374 [0.2585, 0.3994] | 0.5871 | 0.5576 | 0.5046 | 0.07128 | 0.3150 |
| PersistenceA_DHW | 49322 | 419 | 0.0085 | 0.9817 [0.9752, 0.9879] | 0.3177 [0.2166, 0.4403] | 0.1571 [0.0817, 0.2410] | 0.1957 | 0.5568 | 0.2715 | 0.01172 | 0.0665 |
| PersistenceB_cumsum12 | 49322 | 419 | 0.0085 | 0.9720 [0.9621, 0.9811] | 0.1957 [0.1297, 0.2632] | 0.1746 [0.1239, 0.2249] | 0.6229 | 0.8048 | 0.2973 | 0.04583 | 0.1778 |

Δ (TSCF − baseline), SCS common set: 
- TSCF_minus_PersistenceA_DHW: ROC 0.0047, PR 0.2286, CSI@0.9 0.1804, Brier 0.05956
- TSCF_minus_PersistenceB_cumsum12: ROC 0.0145, PR 0.3506, CSI@0.9 0.1629, Brier 0.02545

### SCS — own valid sets (per-method availability)

| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | 49322 | 419 | 0.0085 | 0.9865 [0.9784, 0.9921] | 0.5463 [0.4080, 0.6378] | 0.3374 [0.2585, 0.3994] | 0.5871 | 0.5576 | 0.5046 | 0.07128 | 0.3150 |
| PersistenceA_DHW | 49322 | 419 | 0.0085 | 0.9817 [0.9752, 0.9879] | 0.3177 [0.2166, 0.4403] | 0.1571 [0.0817, 0.2410] | 0.1957 | 0.5568 | 0.2715 | 0.01172 | 0.0665 |
| PersistenceB_cumsum12 | 49322 | 419 | 0.0085 | 0.9720 [0.9621, 0.9811] | 0.1957 [0.1297, 0.2632] | 0.1746 [0.1239, 0.2249] | 0.6229 | 0.8048 | 0.2973 | 0.04583 | 0.1778 |

### GBR — GBR formal zero-shot test (2017-2020) (grid 15x12)

| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | 19065 | 676 | 0.0355 | 0.9624 [0.9490, 0.9749] | 0.5543 [0.4159, 0.6423] | 0.3648 [0.2753, 0.4347] | 0.6686 | 0.5547 | 0.5346 | 0.09287 | 0.3317 |
| PersistenceA_DHW | 19065 | 676 | 0.0355 | 0.9593 [0.9350, 0.9805] | 0.4164 [0.2204, 0.6235] | 0.2494 [0.0968, 0.3970] | 0.3328 | 0.5011 | 0.3993 | 0.02331 | 0.0736 |
| PersistenceB_cumsum12 | 19065 | 676 | 0.0355 | 0.9318 [0.9026, 0.9600] | 0.2490 [0.1453, 0.3614] | 0.2141 [0.1228, 0.3006] | 0.4808 | 0.7215 | 0.3527 | 0.05657 | 0.1550 |

Δ (TSCF − baseline), GBR common set: 
- TSCF_minus_PersistenceA_DHW: ROC 0.0031, PR 0.1379, CSI@0.9 0.1154, Brier 0.06955
- TSCF_minus_PersistenceB_cumsum12: ROC 0.0306, PR 0.3053, CSI@0.9 0.1507, Brier 0.03630

### GBR — own valid sets (per-method availability)

| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | 19065 | 676 | 0.0355 | 0.9624 [0.9490, 0.9749] | 0.5543 [0.4159, 0.6423] | 0.3648 [0.2753, 0.4347] | 0.6686 | 0.5547 | 0.5346 | 0.09287 | 0.3317 |
| PersistenceA_DHW | 19065 | 676 | 0.0355 | 0.9593 [0.9350, 0.9805] | 0.4164 [0.2204, 0.6235] | 0.2494 [0.0968, 0.3970] | 0.3328 | 0.5011 | 0.3993 | 0.02331 | 0.0736 |
| PersistenceB_cumsum12 | 19065 | 676 | 0.0355 | 0.9318 [0.9026, 0.9600] | 0.2490 [0.1453, 0.3614] | 0.2141 [0.1228, 0.3006] | 0.4808 | 0.7215 | 0.3527 | 0.05657 | 0.1550 |

## Notes

- The persistence baselines use only issue-time thermal state; they are physically grounded: 4-week DHW persistence is strongly autocorrelated during events, which is why these baselines are strong.
- TSCF numbers here are computed on the **causal v2** datasets; they may differ slightly from legacy JSONs computed on the pre-causal files (the sst_anomaly channel differs).
- Week-level (temporal block) bootstrap, 1000 replicates, seed 42: whole weeks are resampled, preserving within-week spatial correlation and event clustering. Raw per-pixel arrays with (t, i, j) indices are saved in `persistence_raw_predictions.npz` for later spatial/event block variants.
- ECE is reported as a diagnostic only; persistence scores are not calibrated probabilities.
- Thresholds τ = 0.8 / 0.9 are pinned a priori; no threshold was selected on the test set.
