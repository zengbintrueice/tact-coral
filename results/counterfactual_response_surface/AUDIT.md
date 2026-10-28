# AUDIT — Counterfactual Thermal-Response Surface

Generated: 2026-09-30 18:03:17  
Frozen records md5 `4035a19fd30fa4d14151f4c8055dfe59`; frozen TSAR (seed-42) md5 `ae86e017bc5ba634b0c68e5d13b4f9f0` (both verified). Device: cuda.

## Protocol adherence

- Frozen-model counterfactual inference ONLY: no retraining, no target fitting/tuning, no threshold optimisation. Real data are never modified; counterfactual thermal states are built on copies of model input windows.
- **These are model-response surfaces, i.e. associative evidence about the frozen model's input-output map — NOT causal effects.**
- Intervention is identical in all basins and pre-specified (see summary.json / script header): anchor + level-scaled HotSpot window + δ/28 per week on the last 4 weeks + consistent cumsum shift + recomputed d_hotspot_dt. sst/ssta and nonthermal channels are held at the observed values (documented limitation: the counterfactual is defined on the thermal-trajectory channels only).
- Anti-extrapolation: cells require ≥100 real records within |10a−d| ≤ 0.5 and |10δ_obs−δ| ≤ 1.0 °C-weeks; ≤256 records per cell (seed-42 subsample); grid restricted to the common supported DHW range [0, 8.64] °C-weeks.
- Uncertainty: per-cell week-block bootstrap (1,000, seed 42) over saved per-record counterfactual responses (no re-inference). Cross-basin differences use independent per-basin bootstrap streams (no cross-basin pairing is assumed).
- δ/28 weekly-HotSpot approximation to the CRW 84-day DHW accumulation is an approximation; it is applied identically everywhere and does not use target information.

## Trajectory contrast at fixed DHW (R at δ=+4 minus R at δ=−4)

- CARIB: median over supported d = +0.0202 (coordinate units); positive at 3/3 levels.

## Interpretation guard

The surface describes what the FROZEN MODEL would output under hypothetical thermal trajectories; it does not describe how real reefs respond to heat stress. Any 'susceptibility' language is shorthand for basin differences in the model's response map.

