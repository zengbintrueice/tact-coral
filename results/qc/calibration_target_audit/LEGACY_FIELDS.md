# Legacy metric fields — mapping and deprecation

Canonical definitions: `experiments/08_infrastructure/metric_semantics.py`.
Canonical values: `results/qc/calibration_target_audit/metrics_v2.json` /
`.csv` (produced by `recompute_calibration_metrics.py`).

## Field mapping

| Legacy field (still present in old result files) | Actual meaning | Canonical field |
|---|---|---|
| `brier` (in `structural_ablation.json`, `structural_ablation_corrected.json`, `xgb_transfer_control/metrics.json`, `physics_anchored_eval.json`, persistence outputs) | `mean((clip(pred,0,1) − y(t+4))²)` — **continuous MSE**, not the binary Brier score | `mse_cont` (clipped) / `mse_cont_raw` (unclipped) |
| `brier_cont_target` (in `probability_metrics_audit/structural_ablation_corrected.json`) | same continuous MSE | `mse_cont` |
| `ece` (all legacy files) | `Σ_k (n_k/N)·|observed severe-event rate − mean score|` over 10 bins | `ece_probability` **only for TSCF**; `score_reliability` for PersA / ungated / TSAR / XGBoost |
| `brier` (in `calibration_curves_4group.py`, `scs_probability_calibration.py`) | `mean((p − 1[y≥0.8])²)` — **standard binary Brier** (correct probability metric) | `brier_binary` |

## Files that still emit a legacy `brier` field

Each now carries a `LEGACY METRIC NAMING` header:

- `experiments/08_infrastructure/compute_structural_ablation_analysis.py`
- `experiments/05_thermal_attribution/xgb_transfer_control.py`
- `experiments/08_infrastructure/compute_physics_anchored_analysis.py`
- `experiments/08_infrastructure/compute_persistence_baseline.py`
- `experiments/04_structural_ablation/pfv_utils.py`
- `scripts/figures/nc_style.py::reliability_curve` (returned `brier`)

They are retained so that the original evidence-locked result files remain
reproducible. They must not be used for new tables or figures.

## Files on the correct probability path

Each carries a `PROBABILITY PATH` header:

- `experiments/06_calibration/calibration_curves_4group.py`
- `experiments/02_scs_benchmark/scs_probability_calibration.py`

## What the manuscript now uses

- Tables 3(a) and 3(b): `MSE (continuous)`, `Brier (severe event)`,
  `ECE (probability)`, `Score reliability` — generated from `metrics_v2.json`.
- Figures 3/4/5/6 source data: `mse_cont`, `score_reliability`,
  `ece_probability`, `brier_binary` (+ `brier_binary_descriptive` only for
  non-probability outputs in the XGB source data, explicitly flagged).
- No manuscript text calls a continuous MSE "Brier", and no non-probability
  score's ECE is called probability calibration.
