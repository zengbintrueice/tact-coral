# paths.py - Shared path configuration for the LTTS release
#
#   Every experiment script imports this module for consistent paths.
#   LTTS is self-contained: it carries its own src/ package, frozen model
#   weights (results/) and result files (results/<category>/). The only
#   external dependency is the raw dataset directory (see data/README.md).

import os
import sys
from pathlib import Path

# LTTS repository root (directory containing this file)
A7_DIR = Path(__file__).resolve().parent
LTTS_ROOT = A7_DIR  # alias; kept for readability

# PIECF_ROOT: kept for backward compatibility with imports. The LTTS root
# itself carries the authoritative src/ package, so it is the project root.
PIECF_ROOT = A7_DIR

# Raw data directory (LTTS_CRW: CRW-standard regenerated datasets).
# Resolution order: (1) PIECF_DATA_DIR env var, (2) ./data shipped with this
# repository (self-contained default), (3) D:\piecf_crw_data (author machine).
# `./data` is checked BEFORE the author path so a fresh clone never silently
# picks up an external directory.
_env = os.environ.get("PIECF_DATA_DIR")
if _env:
    DATA_DIR = Path(_env)
elif (A7_DIR / "data").is_dir():
    DATA_DIR = A7_DIR / "data"
else:
    DATA_DIR = Path(r"D:\piecf_crw_data")

# Frozen final model weights (shipped). LTTS_CRW: the formal model is the D1
# severe-event binary classifier — a single sigmoid head trained with
# class-weighted BCE (pos_weight=1000) on the severe label y(t+4) >= 0.8,
# producing P(severe) in [0, 1]. See experiments/01_training/train_severe_classifier.py.
# Load with ablate_a_dh_stopgrad.FinalSingleHeadModel (single head, same
# state_dict keys). The directory name "model_final_Btail3" is a stale label:
# the shipped file is byte-identical (MD5 7cdbca26cae03fa53be53210940e8ce1) to
# results/model_severe_classifier/model.pt, i.e. the D1 classifier, NOT the
# tail-weighted regression that the directory name suggests.
A6_WEIGHT = A7_DIR / "results" / "model_final_Btail3" / "model.pt"

# Frozen XGBoost weights (shipped)
XGB_WEIGHT_DIR = A7_DIR / "results" / "xgb_retrained_20260911"

# Results directory (experiment outputs land in category subdirectories)
RESULTS_DIR = A7_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for _sub in ["scs_benchmark", "gbr_transfer", "structural_ablation",
             "thermal_attribution", "calibration", "gcbd", "infrastructure", "figures"]:
    (RESULTS_DIR / _sub).mkdir(parents=True, exist_ok=True)

# Put the LTTS root FIRST on sys.path so the local src/ package is
# authoritative (no dependency on any parent project).
if str(A7_DIR) not in sys.path:
    sys.path.insert(0, str(A7_DIR))
