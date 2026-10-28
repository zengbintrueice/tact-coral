#!/usr/bin/env bash
# Reproduce the manuscript results (all main-text figures/tables).
# All commands run from the repository root. Frozen weights ship in results/.
set -e

PY=${PY:-python}

echo "== 0. smoke test (no datasets needed) =="
$PY scripts/smoke_test.py

echo "== 1. SCS source-domain benchmark (Fig. 2 / Sup. Table S4) =="
$PY experiments/02_scs_benchmark/scs_threshold_calibration.py
$PY experiments/02_scs_benchmark/scs_severe_event_evaluation.py
$PY experiments/02_scs_benchmark/scs_reliability_diagram.py
$PY experiments/02_scs_benchmark/scs_probability_calibration.py

echo "== 2. GBR zero-shot + threshold adaptation (Fig. 3 / Sup. Table S5) =="
$PY experiments/03_gbr_transfer/gbr_zeroshot_evaluation.py
$PY experiments/03_gbr_transfer/gbr_threshold_adaptation.py
$PY experiments/03_gbr_transfer/gbr_score_recalibration.py
$PY experiments/03_gbr_transfer/gbr_domain_shift_diagnosis.py
$PY experiments/03_gbr_transfer/gbr_gate_activation_analysis.py
$PY experiments/03_gbr_transfer/gbr_xgboost_zeroshot_baseline.py

echo "== 3. thermal ablations + attribution (Fig. 4) =="
$PY experiments/05_thermal_attribution/thermal_only_ablation.py
$PY experiments/05_thermal_attribution/thermal_removal_ablation.py
$PY experiments/05_thermal_attribution/permutation_importance_cross_domain.py
$PY experiments/05_thermal_attribution/xgboost_thermal_ablation.py
$PY experiments/05_thermal_attribution/xgboost_permutation_importance.py

echo "== 4. context-magnitude analysis (Fig. 5) =="
$PY scripts/figures/fig6_gated_residual.py

echo "== 5. GCBD independent validation (Fig. 6 / Sup. Table S7) =="
$PY experiments/07_gcbd_validation/gcbd_independent_validation.py

$PY scripts/figures/fig_supplementary.py   # SI figures (S1-S13, R-figures)

echo "MAIN DONE - compare outputs against the verification targets in README.md section 5."
