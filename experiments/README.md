# experiments/ — entry points grouped by paper theme

Each script carries a provenance header (Experiment / Paper / Purpose /
Output / Corresponding Figure-Table / Random seed). Result files produced by
newer analyses live in `results/<dir>/` together with `AUDIT.md` and
`provenance_tier1.json`, which record the producing script, input hashes and
QC verdicts.

| Directory | Paper scope | Scripts |
|---|---|---|
| `01_training/` | optional retraining; frozen checkpoints shipped in `results/` | `train_severe_classifier.py` (TSCF / D1), `train_physics_anchored.py` (TSAR), `train_structural_ablation_B.py` (ungated control), `train_xgboost_scs.py`; reverse-source seed models under `results/transfer_coordinate/weights/` |
| `02_scs_benchmark/` | SCS in-source benchmark (Figs. 2–3 diagonal; SI Tables) | `scs_threshold_calibration.py`, `scs_severe_event_evaluation.py`, `scs_reliability_diagram.py`, `scs_probability_calibration.py` |
| `03_gbr_transfer/` | zero-shot transfer, threshold adaptation, domain-shift diagnosis (Figs. 2–3) | `gbr_zeroshot_evaluation.py`, `gbr_threshold_adaptation.py`, `gbr_score_recalibration.py`, `gbr_domain_shift_diagnosis.py`, `gbr_gate_activation_analysis.py`, `threshold_sensitivity_analysis.py`, `gbr_xgboost_zeroshot_baseline.py` |
| `04_structural_ablation/` | model definitions used by the A/B/C/D ablation (Fig. 5) | `physics_anchored_model.py` (TSAR), `structural_ablation_models.py` (ungated control), `pfv_utils.py` (shared PFV helpers) |
| `05_thermal_attribution/` | thermal-only / thermal-removal / permutation / XGBoost controls (Fig. 6; SI) | `thermal_only_ablation.py`, `thermal_removal_ablation.py`, `permutation_importance_cross_domain.py`, `xgboost_thermal_ablation.py`, `xgboost_permutation_importance.py` |
| `06_calibration/` | calibration curves (SI) | `calibration_curves_4group.py` |
| `07_gcbd_validation/` | GCBD independent validation (Fig. 10; SI R5) | `gcbd_independent_validation.py` |
| `08_infrastructure/` | PFV 3-level evaluation, normalization, dual-output maps (Methods) | `pfv_evaluation_3level.py`, `export_dual_output_maps.py`, `compute_train_normalization.py` |

Six-direction transfer matrix, seed scan and regime controls:
`results/transfer_coordinate/` (AUDIT.md documents the producing scripts).
Residual hierarchy, conditional residual information, matched-DHW
susceptibility, mechanism decomposition and counterfactual probe:
respective `results/` directories, each with its own `AUDIT.md` +
`provenance_tier1.json`.

Run order recommendation: `scripts/reproduce_main.sh` →
`scripts/reproduce_supplementary.sh` (or `scripts/reproduce_all.sh`).
