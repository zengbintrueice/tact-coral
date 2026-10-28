# ==========================================================================
# Source: paper/a7_dual_output/scs_probability_calibration.py (verbatim except path fix)
# Experiment: Isotonic probability calibration + ECE
# Paper: Results 1 (Methods: Operational evaluation and calibration)
# Purpose: isotonic mapping fitted on validation, ECE on test
# Output: results/scs_probability_calibration.json
# Corresponding Figure/Table: Fig. 2b
# ==========================================================================

# scs_probability_calibration.py — Platt scaling + Isotonic regression for a6
#
#   Goal: Fix a6's over-confidence (predicted prob systematically higher than
#     actual severe frequency) using post-hoc calibration.
#
#   Method:
#     1. Sliding-window inference on val (2013-2015) and test (2016-2019)
#     2. Fit calibration model on val:
#        - Platt scaling: logistic regression on raw output
#        - Isotonic regression: non-parametric monotonic fit
#     3. Apply to test, compare Brier before/after
#
#   Saves: results/scs_probability_calibration.json + scs_probability_calibration.png

import os, sys, io, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# a7 shared paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, PIECF_ROOT, A6_WEIGHT, XGB_WEIGHT_DIR, DATA_DIR, RESULTS_DIR

from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import build_pfv_arrays, get_year_range_for_mode

A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8


def brier_calc(p, t):
    if len(p) < 1:
        return float("nan")
    return float(np.mean((p - t) ** 2))


def ece_calc(pred, actual, n_bins=10):
    """Expected Calibration Error: weighted average of |acc - conf| per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(pred)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred < bins[i + 1])
        if i == n_bins - 1:
            mask = (pred >= bins[i]) & (pred <= bins[i + 1])
        n = mask.sum()
        if n > 0:
            acc = float(np.mean(actual[mask]))
            conf = float(np.mean(pred[mask]))
            weight = n / total
            ece += weight * abs(acc - conf)
            bin_data.append({"bin_center": float((bins[i] + bins[i + 1]) / 2),
                             "accuracy": acc, "confidence": conf, "n": int(n), "weight": float(weight)})
        else:
            bin_data.append({"bin_center": float((bins[i] + bins[i + 1]) / 2),
                             "accuracy": None, "confidence": None, "n": 0, "weight": 0.0})
    return float(ece), bin_data


def restore_a5():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_sliding_temporal(model, ds, device):
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X_full = ds.X_dynamic_full
    bs = 16
    n_valid = n_dates - T + 1
    t_start_valid = T - 1
    with torch.no_grad():
        for start_batch in range(0, n_valid, bs):
            batch_end = min(start_batch + bs, n_valid)
            bs_actual = batch_end - start_batch
            xd_batch = np.zeros((bs_actual, T, H, W, C_dyn), dtype=np.float32)
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                start_idx = t_end - T + 1
                xd_batch[k] = X_full[start_idx:t_end + 1]
            xd = torch.FloatTensor(xd_batch).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (bs_actual, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, scale=A5_CFG["scale"],
                        tau=A5_CFG["tau"], temp=A5_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
    return forecast_map


def build_pixel_arrays(forecast_map, y_full, T, n_dates, year_range=None, date_strings=None):
    """Build pixel-level prediction and binary severe label arrays (year-filtered)."""
    preds, targets, _ = build_pfv_arrays(forecast_map, y_full, T, n_dates,
                                         year_range=year_range, date_strings=date_strings)
    severe_labels = (targets >= SEVERE_THRESHOLD).astype(np.float32)
    return preds, targets, severe_labels


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Probability Calibration — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nRestoring a6...")
a5_model = restore_a5().to(device)

# Load validation set (2013-2015) for fitting calibration
print("\nLoading validation dataset (2013-2015)...")
ds_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                         lead_weeks=4, year_based_split=True,
                         train_years=(2003, 2010), val_years=(2011, 2015),
                         test_years=(2016, 2019))
T, H, W = ds_val.history_weeks, ds_val.n_lat, ds_val.n_lon
n_dates_val = ds_val.n_dates
print(f"  T={T} H={H} W={W}  n_dates={n_dates_val}")

print("\nSliding-window inference (val)...")
forecast_map_val = predict_sliding_temporal(a5_model, ds_val, device)
val_preds, val_targets, val_labels = build_pixel_arrays(
    forecast_map_val, ds_val.y_regression_full, T, n_dates_val,
    year_range=get_year_range_for_mode(ds_val), date_strings=ds_val.date_strings)
print(f"  val n_pixels={len(val_preds)}")

# Load test set (2016-2019) for evaluation
print("\nLoading test dataset (2016-2019)...")
ds_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                          lead_weeks=4, year_based_split=True,
                          train_years=(2003, 2010), val_years=(2011, 2015),
                          test_years=(2016, 2019))
n_dates_test = ds_test.n_dates
print(f"  n_dates={n_dates_test}")

print("\nSliding-window inference (test)...")
forecast_map_test = predict_sliding_temporal(a5_model, ds_test, device)
test_preds, test_targets, test_labels = build_pixel_arrays(
    forecast_map_test, ds_test.y_regression_full, T, n_dates_test,
    year_range=get_year_range_for_mode(ds_test), date_strings=ds_test.date_strings)
print(f"  test n_pixels={len(test_preds)}")

# ═══════════════════════════════════════════════════════════════
# Fit calibration models on validation set
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Fitting calibration models on validation set...")
print("=" * 60)

# Isotonic regression
print("Fitting Isotonic regression...")
iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso.fit(val_preds, val_labels)
print(f"  Isotonic fit on {len(val_preds)} samples")

# ═══════════════════════════════════════════════════════════════
# Apply to test set, compute Brier
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test set Brier comparison")
print("=" * 60)

# Before calibration
brier_before = brier_calc(test_preds, test_labels)
brier_before_severe = brier_calc(test_preds[test_labels == 1], test_labels[test_labels == 1])
brier_before_mild = brier_calc(test_preds[test_labels == 0], test_labels[test_labels == 0])

# Isotonic regression
test_iso = iso.predict(test_preds).astype(np.float32)
brier_iso = brier_calc(test_iso, test_labels)
brier_iso_severe = brier_calc(test_iso[test_labels == 1], test_labels[test_labels == 1])
brier_iso_mild = brier_calc(test_iso[test_labels == 0], test_labels[test_labels == 0])

print(f"\n  Before calibration:")
print(f"    Brier (all)     = {brier_before:.6f}")
print(f"    Brier (severe)  = {brier_before_severe:.6f}  n={int(test_labels.sum())}")
print(f"    Brier (mild)    = {brier_before_mild:.6f}  n={int((1-test_labels).sum())}")

print(f"\n  Isotonic regression:")
print(f"    Brier (all)     = {brier_iso:.6f}  (delta={brier_iso - brier_before:+.6f})")
print(f"    Brier (severe)  = {brier_iso_severe:.6f}  (delta={brier_iso_severe - brier_before_severe:+.6f})")
print(f"    Brier (mild)    = {brier_iso_mild:.6f}  (delta={brier_iso_mild - brier_before_mild:+.6f})")

# ECE calculation
print("\n" + "=" * 60)
print("Expected Calibration Error (ECE)")
print("=" * 60)
ece_before, ece_before_bins = ece_calc(test_preds, test_labels)
ece_iso, ece_iso_bins = ece_calc(test_iso, test_labels)
print(f"\n  ECE before calibration: {ece_before:.6f}")
print(f"  ECE after Isotonic:     {ece_iso:.6f}  (delta={ece_iso - ece_before:+.6f})")

# ═══════════════════════════════════════════════════════════════
# Plot: calibration curve (before + after on same axes) + ECE text
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
fig.suptitle(f"Calibration Curve: a6 test set (t+{LEAD_WEEKS}, severe >= {SEVERE_THRESHOLD})",
             fontsize=12, fontweight="bold")

NBINS = 10
bins = np.linspace(0, 1, NBINS + 1)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")

# Before calibration
valid_before = [(b["confidence"], b["accuracy"]) for b in ece_before_bins if b["accuracy"] is not None]
if valid_before:
    conf_b, acc_b = zip(*valid_before)
    ax.plot(conf_b, acc_b, "o-", color="#FF9800", linewidth=2, markersize=7,
            label=f"Before (ECE={ece_before:.4f})")

# After isotonic
valid_iso = [(b["confidence"], b["accuracy"]) for b in ece_iso_bins if b["accuracy"] is not None]
if valid_iso:
    conf_i, acc_i = zip(*valid_iso)
    ax.plot(conf_i, acc_i, "s-", color="#2196F3", linewidth=2, markersize=7,
            label=f"Isotonic (ECE={ece_iso:.4f})")

ax.set_xlabel("Mean predicted probability (confidence)", fontsize=11)
ax.set_ylabel("Actual severe frequency (accuracy)", fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_aspect("equal")

plt.tight_layout()
png_path = RESULTS_DIR / "scs_benchmark/scs_probability_calibration.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: {png_path}")

# Save JSON
out = {
    "method": "scs_probability_calibration",
    "severe_threshold": SEVERE_THRESHOLD,
    "lead_weeks": LEAD_WEEKS,
    "val_years": [2013, 2015],
    "test_years": [2016, 2019],
    "ece": {
        "before": ece_before,
        "isotonic": ece_iso,
        "delta": ece_iso - ece_before,
    },
    "brier": {
        "before": {"all": brier_before, "severe": brier_before_severe, "mild": brier_before_mild},
        "isotonic": {"all": brier_iso, "severe": brier_iso_severe, "mild": brier_iso_mild},
    },
    "ece_bins": {
        "before": ece_before_bins,
        "isotonic": ece_iso_bins,
    },
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "scs_benchmark/scs_probability_calibration.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

print("\nDone.")
