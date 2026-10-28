# ==========================================================================
# Source: paper/a7_dual_output/scs_threshold_calibration.py (verbatim except path fix)
# Experiment: SCS source-domain threshold calibration (PIECF + XGBoost)
# Paper: Results 1, Fig. 2, Supplementary Table S4
# Purpose: validation-selected operating point, threshold sweep, CSI/POD/FAR/F1 on SCS test
# Input: D:/piecf_data SCS dataset; frozen a6 + XGB weights
# Output: results/scs_threshold_calibration.json
# Corresponding Figure/Table: Fig. 2a; Supplementary Table S4
# Random seed: 42
# ==========================================================================

# scs_threshold_calibration.py — Threshold calibration for a6 AND XGBoost lead-4
#
#   Goal: Find optimal operating point (pred_threshold) for severe classification,
#     comparing a6 vs XGBoost on equal footing (Pure Forecasting Verification,
#     t+4 pixel-level, year-based split).
#
#   Method:
#     - Sliding-window inference for a6 (only position T-1, no artifact)
#     - XGBoost lead-4 per-window inference
#     - Sweep pred_threshold in [0.5, 0.6, 0.7, 0.8, 0.9]
#     - Select best CSI/F1 operating point on VALIDATION set (2011-2015)
#     - Compute CSI, F1, POD, FAR at each threshold on TEST set (2016-2019)
#     - Report test metrics at the val-selected operating point
#     - Plot both models on same axes + mark val-selected operating points
#
#   Saves: results/scs_threshold_calibration.json + scs_threshold_calibration.png

import os, sys, io, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# a7 shared paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, PIECF_ROOT, A6_WEIGHT, XGB_WEIGHT_DIR, DATA_DIR, RESULTS_DIR

WEIGHT_DIR = XGB_WEIGHT_DIR

from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import build_pfv_arrays, get_year_range_for_mode

A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]  # fine low-τ grid (LTTS_CRW)


def classification_metrics(pred, actual, severe_thresh, pred_thresh):
    pred_bin = pred >= pred_thresh
    actual_bin = actual >= severe_thresh
    hits = int(np.sum(pred_bin & actual_bin))
    misses = int(np.sum(~pred_bin & actual_bin))
    false_alarms = int(np.sum(pred_bin & ~actual_bin))
    correct_negatives = int(np.sum(~pred_bin & ~actual_bin))
    denom_csi = hits + misses + false_alarms
    csi = float(hits / denom_csi) if denom_csi > 0 else float("nan")
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else float("nan")
    denom_f1 = 2 * hits + false_alarms + misses
    f1 = float(2 * hits / denom_f1) if denom_f1 > 0 else float("nan")
    return {"csi": csi, "pod": pod, "far": far, "f1": f1,
            "hits": hits, "misses": misses, "false_alarms": false_alarms,
            "correct_negatives": correct_negatives, "n_total": int(len(pred))}


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


def build_xgb_pixel_array(ds, xgb_model, xgb_scaler, T, H, W, C_dyn, C_stat,
                           year_range, date_strings):
    """XGBoost lead-4 pixel-level predictions with PFV t+4 verification.

    Uses forecasting target y_full[t+LEAD_WEEKS] (window-external), NOT
    autoencoding y_regression[:,L-1]. Filters to year_range (issue year)."""
    n_dates = ds.n_dates
    y_full = ds.y_regression_full
    X_stat = ds.X_static[0]
    pfv_pred = []
    for t in range(T - 1, n_dates):
        if year_range is not None and date_strings is not None:
            year = int(date_strings[t][:4])
            if not (year_range[0] <= year <= year_range[1]):
                continue
        t_target = t + LEAD_WEEKS
        if t_target >= n_dates:
            continue
        s = t - T + 1
        dyn_win = ds.X_dynamic_full[s:s + T]
        dyn_flat = dyn_win.transpose(1, 2, 0, 3).reshape(H * W, T * C_dyn)
        stat_flat = X_stat.reshape(H * W, C_stat)
        X = np.concatenate([dyn_flat, stat_flat], axis=1).astype(np.float32)
        X_s = xgb_scaler.transform(X)
        pred_t = np.clip(xgb_model.predict(X_s), 0, 1).astype(np.float32).reshape(H, W)
        target_t = y_full[t_target]
        valid = ~np.isnan(pred_t) & ~np.isnan(target_t)
        if valid.sum() < 1:
            continue
        pfv_pred.extend(pred_t[valid].tolist())
    return np.array(pfv_pred, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Threshold Calibration (a6 vs XGBoost) — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))

print("\nRestoring a6...")
a5_model = restore_a5().to(device)

print("Loading XGBoost lead-4 model...")
xgb_model = joblib.load(WEIGHT_DIR / "xgb_lead4.joblib")
xgb_scaler = joblib.load(WEIGHT_DIR / "xgb_lead4_scaler.joblib")

# ═══════════════════════════════════════════════════════════════
# Load VALIDATION set (2011-2015) — used to SELECT the operating point
# ═══════════════════════════════════════════════════════════════
print("\nLoading validation dataset (2011-2015) for threshold selection...")
ds_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                         lead_weeks=4, **split_kw)
T, H, W = ds_val.history_weeks, ds_val.n_lat, ds_val.n_lon
n_dates_val = ds_val.n_dates
y_full_val = ds_val.y_regression_full
C_dyn = ds_val.C_dyn
C_stat = ds_val.C_stat
print(f"  T={T} H={H} W={W}  n_dates={n_dates_val}")

print("\nSliding-window inference (val, a6)...")
t0 = time.time()
forecast_map_val = predict_sliding_temporal(a5_model, ds_val, device)
print(f"  Done in {time.time()-t0:.1f}s")

year_range_val = get_year_range_for_mode(ds_val)
date_strings_val = ds_val.date_strings
val_pred_a6, val_target, _ = build_pfv_arrays(
    forecast_map_val, y_full_val, T, n_dates_val,
    year_range=year_range_val, date_strings=date_strings_val)
print(f"  val n_pixels={len(val_pred_a6)}, n_severe={int((val_target>=SEVERE_THRESHOLD).sum())}")

print("XGBoost inference (val)...")
val_pred_xgb = build_xgb_pixel_array(
    ds_val, xgb_model, xgb_scaler, T, H, W, C_dyn, C_stat,
    year_range_val, date_strings_val)
print(f"  XGB val n_pixels={len(val_pred_xgb)}")

# ═══════════════════════════════════════════════════════════════
# Select operating point on VALIDATION set (NOT test)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Threshold selection on VALIDATION (severe >= {SEVERE_THRESHOLD})")
print("=" * 60)

val_a6_sweep, val_xgb_sweep = [], []
for thr in THRESHOLDS:
    m_a6 = classification_metrics(val_pred_a6, val_target, SEVERE_THRESHOLD, thr)
    m_xgb = classification_metrics(val_pred_xgb, val_target, SEVERE_THRESHOLD, thr)
    val_a6_sweep.append({"threshold": thr, **m_a6})
    val_xgb_sweep.append({"threshold": thr, **m_xgb})
    print(f"  thr={thr:.1f} | a6  CSI={m_a6['csi']:.4f} F1={m_a6['f1']:.4f} POD={m_a6['pod']:.4f} FAR={m_a6['far']:.4f}")
    print(f"         | xgb CSI={m_xgb['csi']:.4f} F1={m_xgb['f1']:.4f} POD={m_xgb['pod']:.4f} FAR={m_xgb['far']:.4f}")

best_a6_csi = max(val_a6_sweep, key=lambda r: r["csi"])
best_a6_f1 = max(val_a6_sweep, key=lambda r: r["f1"])
best_xgb_csi = max(val_xgb_sweep, key=lambda r: r["csi"])
best_xgb_f1 = max(val_xgb_sweep, key=lambda r: r["f1"])
print(f"\n  [val-selected] a6  best CSI: thr={best_a6_csi['threshold']:.1f} (val CSI={best_a6_csi['csi']:.4f})")
print(f"  [val-selected] a6  best F1:  thr={best_a6_f1['threshold']:.1f} (val F1={best_a6_f1['f1']:.4f})")
print(f"  [val-selected] xgb best CSI: thr={best_xgb_csi['threshold']:.1f} (val CSI={best_xgb_csi['csi']:.4f})")
print(f"  [val-selected] xgb best F1:  thr={best_xgb_f1['threshold']:.1f} (val F1={best_xgb_f1['f1']:.4f})")

# ═══════════════════════════════════════════════════════════════
# Load TEST set (2016-2019) — evaluate at every threshold + at val-selected point
# ═══════════════════════════════════════════════════════════════
print("\nLoading test dataset (2016-2019) for evaluation...")
ds_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                          lead_weeks=4, **split_kw)
n_dates = ds_test.n_dates
y_full = ds_test.y_regression_full
print(f"  n_dates={n_dates}")

print("\nSliding-window inference (test, a6)...")
t0 = time.time()
forecast_map = predict_sliding_temporal(a5_model, ds_test, device)
print(f"  Done in {time.time()-t0:.1f}s")

year_range = get_year_range_for_mode(ds_test)
date_strings = ds_test.date_strings
pfv_pred_a6, pfv_target, _ = build_pfv_arrays(
    forecast_map, y_full, T, n_dates,
    year_range=year_range, date_strings=date_strings)
print(f"  test n_pixels={len(pfv_pred_a6)}, n_severe={int((pfv_target>=SEVERE_THRESHOLD).sum())}")

print("XGBoost inference (test)...")
pfv_pred_xgb = build_xgb_pixel_array(
    ds_test, xgb_model, xgb_scaler, T, H, W, C_dyn, C_stat,
    year_range, date_strings)
print(f"  XGB test n_pixels={len(pfv_pred_xgb)}")

# ═══════════════════════════════════════════════════════════════
# Threshold sweep on TEST (report metrics at each threshold)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Threshold sweep on TEST (severe >= {SEVERE_THRESHOLD})")
print("=" * 60)

a6_results, xgb_results = [], []
for thr in THRESHOLDS:
    m_a6 = classification_metrics(pfv_pred_a6, pfv_target, SEVERE_THRESHOLD, thr)
    m_xgb = classification_metrics(pfv_pred_xgb, pfv_target, SEVERE_THRESHOLD, thr)
    a6_results.append({"threshold": thr, **m_a6})
    xgb_results.append({"threshold": thr, **m_xgb})
    print(f"  thr={thr:.1f} | a6  CSI={m_a6['csi']:.4f} F1={m_a6['f1']:.4f} POD={m_a6['pod']:.4f} FAR={m_a6['far']:.4f}")
    print(f"         | xgb CSI={m_xgb['csi']:.4f} F1={m_xgb['f1']:.4f} POD={m_xgb['pod']:.4f} FAR={m_xgb['far']:.4f}")

# Test metrics AT the val-selected operating point
test_a6_at_best_csi = next(r for r in a6_results if r["threshold"] == best_a6_csi["threshold"])
test_a6_at_best_f1 = next(r for r in a6_results if r["threshold"] == best_a6_f1["threshold"])
test_xgb_at_best_csi = next(r for r in xgb_results if r["threshold"] == best_xgb_csi["threshold"])
test_xgb_at_best_f1 = next(r for r in xgb_results if r["threshold"] == best_xgb_f1["threshold"])
print(f"\n  [test @ val-selected] a6  CSI: thr={test_a6_at_best_csi['threshold']:.1f} test CSI={test_a6_at_best_csi['csi']:.4f}")
print(f"  [test @ val-selected] a6  F1:  thr={test_a6_at_best_f1['threshold']:.1f} test F1={test_a6_at_best_f1['f1']:.4f}")
print(f"  [test @ val-selected] xgb CSI: thr={test_xgb_at_best_csi['threshold']:.1f} test CSI={test_xgb_at_best_csi['csi']:.4f}")
print(f"  [test @ val-selected] xgb F1:  thr={test_xgb_at_best_f1['threshold']:.1f} test F1={test_xgb_at_best_f1['f1']:.4f}")

# ═══════════════════════════════════════════════════════════════
# Plot: a6 vs XGB on same axes
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"Threshold Calibration: a6 vs XGBoost lead-4\n(severe >= {SEVERE_THRESHOLD}, t+{LEAD_WEEKS} pixel-level; "
             f"operating point selected on val 2011-2015, metrics on test 2016-2019)",
             fontsize=12, fontweight="bold")

thr_vals = THRESHOLDS
metrics = ["csi", "f1", "pod", "far"]
titles = ["CSI (Critical Success Index)", "F1 Score", "POD (Probability of Detection)",
          "FAR (False Alarm Rate)"]

for ax, met, title in zip(axes.flat, metrics, titles):
    a6_vals = [r[met] for r in a6_results]
    xgb_vals = [r[met] for r in xgb_results]
    ax.plot(thr_vals, a6_vals, "o-", color="#2196F3", linewidth=2, markersize=7, label="a6 (test)")
    ax.plot(thr_vals, xgb_vals, "s-", color="#FF9800", linewidth=2, markersize=7, label="XGBoost (test)")

    # Mark val-selected operating points (showing their test-metric value)
    best_a6 = best_a6_csi if met == "csi" else (best_a6_f1 if met == "f1" else None)
    best_xgb = best_xgb_csi if met == "csi" else (best_xgb_f1 if met == "f1" else None)
    if best_a6:
        test_val = test_a6_at_best_csi[met] if met == "csi" else test_a6_at_best_f1[met]
        ax.scatter([best_a6["threshold"]], [test_val], s=150, c="#2196F3",
                   marker="*", zorder=5)
    if best_xgb:
        test_val = test_xgb_at_best_csi[met] if met == "csi" else test_xgb_at_best_f1[met]
        ax.scatter([best_xgb["threshold"]], [test_val], s=150, c="#FF9800",
                   marker="*", zorder=5)

    ax.set_xlabel("Prediction threshold", fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.45, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    for x, ya, yx in zip(thr_vals, a6_vals, xgb_vals):
        ax.annotate(f"{ya:.3f}", (x, ya), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="#2196F3")
        ax.annotate(f"{yx:.3f}", (x, yx), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="#FF9800")

plt.tight_layout()
png_path = RESULTS_DIR / "scs_benchmark/scs_threshold_calibration.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: {png_path}")

# Save JSON
out = {
    "method": "scs_threshold_calibration",
    "selection_protocol": "operating point (argmax CSI/F1) selected on VALIDATION 2011-2015; metrics reported on TEST 2016-2019",
    "severe_threshold": SEVERE_THRESHOLD,
    "lead_weeks": LEAD_WEEKS,
    "thresholds": THRESHOLDS,
    "a6": {"val_sweep": val_a6_sweep,
           "test_sweep": a6_results,
           "best_csi": {"threshold": best_a6_csi["threshold"], "val_csi": best_a6_csi["csi"],
                        "test": {k: test_a6_at_best_csi[k] for k in ("csi", "f1", "pod", "far")}},
           "best_f1": {"threshold": best_a6_f1["threshold"], "val_f1": best_a6_f1["f1"],
                       "test": {k: test_a6_at_best_f1[k] for k in ("csi", "f1", "pod", "far")}}},
    "xgb_lead4": {"val_sweep": val_xgb_sweep,
                  "test_sweep": xgb_results,
                  "best_csi": {"threshold": best_xgb_csi["threshold"], "val_csi": best_xgb_csi["csi"],
                               "test": {k: test_xgb_at_best_csi[k] for k in ("csi", "f1", "pod", "far")}},
                  "best_f1": {"threshold": best_xgb_f1["threshold"], "val_f1": best_xgb_f1["f1"],
                              "test": {k: test_xgb_at_best_f1[k] for k in ("csi", "f1", "pod", "far")}}},
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "scs_benchmark/scs_threshold_calibration.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

print("\nDone.")
