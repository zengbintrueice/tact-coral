#!/usr/bin/env bash
# Reproduce the manuscript SUPPLEMENTARY results (SI figures S1-S13 + R-figures).
set -e

PY=${PY:-python}

echo "== Sup. Fig. 1: threshold sensitivity =="
$PY experiments/03_gbr_transfer/threshold_sensitivity_analysis.py

# Structural ablation (paper A/B/C/D) is produced from the shipped frozen
# checkpoints by:
#   experiments/08_infrastructure/compute_structural_ablation_infer.py
#   experiments/08_infrastructure/compute_structural_ablation_analysis.py
# (not invoked here: it requires a full test-set inference pass).

echo "== Sup. Fig. 3: cross-domain isotonic calibration =="
$PY experiments/06_calibration/calibration_curves_4group.py

echo "== Sup. Fig. 4: representative event trajectories =="

echo "== Sup. Fig. 5: attribution consistency =="
$PY experiments/05_thermal_attribution/feature_importance_consistency.py

echo "== Sup. Fig. 6: conceptual synthesis =="


$PY scripts/figures/fig_supplementary.py
$PY scripts/figures/fig_s8_seed_sensitivity.py
$PY scripts/figures/fig_s_x_xgb_transfer.py

echo "SUPPLEMENTARY DONE"

