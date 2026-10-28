# ==========================================================================
# Source: paper/a7_dual_output/scs_reliability_diagram.py (verbatim except path fix)
# Experiment: Reliability diagram (PIECF raw vs isotonic)
# Paper: Results 1, Fig. 2b, Supplementary Fig. 3
# Purpose: predicted vs observed severe frequency before/after isotonic mapping
# Output: results/scs_reliability_diagram.json
# Corresponding Figure/Table: Fig. 2b
# ==========================================================================

# scs_reliability_diagram.py — Calibration curve: predicted prob vs actual severe freq
#
#   Goal: Show whether a6's predicted probabilities are calibrated — i.e., does
#     a prediction of 0.8 correspond to ~80% actual severe frequency?
#
#   Method:
#     - Bin predictions into [0-0.1), [0.1-0.2), ..., [0.9-1.0]
#     - For each bin, compute actual severe frequency (y >= 0.8)
#     - Plot predicted mean vs actual frequency (reliability diagram)
#     - Perfect calibration lies on diagonal
#     - Also plot prediction distribution (histogram) to show where mass lies
#
#   Saves: results/scs_reliability_diagram.json + scs_reliability_diagram.png

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
NBINS = 10


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


def compute_reliability(pred, actual, severe_thresh, n_bins):
    """Compute reliability diagram data: bin centers, actual freq, counts."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    actual_freq = []
    pred_mean = []
    counts = []
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred < bins[i + 1])
        if i == n_bins - 1:
            mask = (pred >= bins[i]) & (pred <= bins[i + 1])
        n = mask.sum()
        counts.append(int(n))
        if n > 0:
            actual_freq.append(float(np.mean(actual[mask] >= severe_thresh)))
            pred_mean.append(float(np.mean(pred[mask])))
        else:
            actual_freq.append(None)
            pred_mean.append(None)
    return {
        "bin_centers": bin_centers.tolist(),
        "bin_edges": bins.tolist(),
        "actual_severe_freq": actual_freq,
        "pred_mean": pred_mean,
        "counts": counts,
    }


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Reliability Diagram — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))

print("\nRestoring a6...")
a5_model = restore_a5().to(device)

print("Loading test dataset...")
ds_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                          lead_weeks=4, **split_kw)
T, H, W = ds_test.history_weeks, ds_test.n_lat, ds_test.n_lon
n_dates = ds_test.n_dates
y_full = ds_test.y_regression_full
C_dyn = ds_test.C_dyn
C_stat = ds_test.C_stat
X_stat = ds_test.X_static[0]
print(f"  T={T} H={H} W={W}  n_dates={n_dates}")

# a6 predictions
print("\nSliding-window inference (a6)...")
t0 = time.time()
forecast_map = predict_sliding_temporal(a5_model, ds_test, device)
print(f"  Done in {time.time()-t0:.1f}s")

year_range = get_year_range_for_mode(ds_test)
date_strings = ds_test.date_strings
pfv_pred_a6, pfv_target, _ = build_pfv_arrays(
    forecast_map, y_full, T, n_dates,
    year_range=year_range, date_strings=date_strings)
print(f"  n_pixels={len(pfv_pred_a6)}")

# XGBoost lead-4 predictions
print("\nLoading XGBoost lead-4 model...")
xgb_model = joblib.load(WEIGHT_DIR / "xgb_lead4.joblib")
xgb_scaler = joblib.load(WEIGHT_DIR / "xgb_lead4_scaler.joblib")

pfv_pred_xgb = []
for t in range(T - 1, n_dates):
    # Year filtering: only include test years
    if year_range is not None and date_strings is not None:
        year = int(date_strings[t][:4])
        if not (year_range[0] <= year <= year_range[1]):
            continue
    t_target = t + LEAD_WEEKS
    if t_target >= n_dates:
        continue
    s = t - T + 1
    dyn_win = ds_test.X_dynamic_full[s:s + T]
    dyn_flat = dyn_win.transpose(1, 2, 0, 3).reshape(H * W, T * C_dyn)
    stat_flat = X_stat.reshape(H * W, C_stat)
    X = np.concatenate([dyn_flat, stat_flat], axis=1).astype(np.float32)
    X_s = xgb_scaler.transform(X)
    pred_t = np.clip(xgb_model.predict(X_s), 0, 1).astype(np.float32).reshape(H, W)
    target_t = y_full[t_target]
    valid = ~np.isnan(pred_t) & ~np.isnan(target_t)
    if valid.sum() < 1:
        continue
    pfv_pred_xgb.extend(pred_t[valid].tolist())

pfv_pred_xgb = np.array(pfv_pred_xgb, dtype=np.float32)
print(f"  XGB n_pixels={len(pfv_pred_xgb)}")

# Compute reliability
print("\nComputing reliability diagrams...")
a6_rel = compute_reliability(pfv_pred_a6, pfv_target, SEVERE_THRESHOLD, NBINS)
xgb_rel = compute_reliability(pfv_pred_xgb, pfv_target, SEVERE_THRESHOLD, NBINS)

print("\n  a6 reliability (bin_center -> actual_severe_freq):")
for i, (c, f, n) in enumerate(zip(a6_rel["bin_centers"], a6_rel["actual_severe_freq"], a6_rel["counts"])):
    fs = f"{f:.4f}" if f is not None else "N/A"
    print(f"    [{i/10:.1f}-{(i+1)/10:.1f})  pred_mean={a6_rel['pred_mean'][i]}  actual_freq={fs}  n={n}")

# ═══════════════════════════════════════════════════════════════
# Plot: reliability diagram + prediction distribution
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Reliability Diagram: a6 vs XGBoost lead-4\n(severe >= {SEVERE_THRESHOLD}, t+{LEAD_WEEKS} pixel-level)",
             fontsize=12, fontweight="bold")

# Left: reliability curve
ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
a6_valid = [(c, f) for c, f in zip(a6_rel["pred_mean"], a6_rel["actual_severe_freq"]) if f is not None]
xgb_valid = [(c, f) for c, f in zip(xgb_rel["pred_mean"], xgb_rel["actual_severe_freq"]) if f is not None]

if a6_valid:
    ax1.plot(*zip(*a6_valid), "o-", color="#2196F3", linewidth=2, markersize=8, label="a6")
if xgb_valid:
    ax1.plot(*zip(*xgb_valid), "s-", color="#FF9800", linewidth=2, markersize=8, label="XGBoost")

ax1.set_xlabel("Mean predicted value", fontsize=11)
ax1.set_ylabel(f"Actual severe frequency (y >= {SEVERE_THRESHOLD})", fontsize=11)
ax1.set_title("Reliability Curve", fontsize=11)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_aspect("equal")

# Right: prediction distribution (histogram)
bin_edges = np.linspace(0, 1, NBINS + 1)
ax2.hist(pfv_pred_a6, bins=bin_edges, alpha=0.6, color="#2196F3", label="a6", density=True)
ax2.hist(pfv_pred_xgb, bins=bin_edges, alpha=0.6, color="#FF9800", label="XGBoost", density=True)
ax2.set_xlabel("Predicted value", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Prediction Distribution", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
png_path = RESULTS_DIR / "scs_benchmark/scs_reliability_diagram.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: {png_path}")

# Save JSON
out = {
    "method": "scs_reliability_diagram",
    "severe_threshold": SEVERE_THRESHOLD,
    "lead_weeks": LEAD_WEEKS,
    "n_bins": NBINS,
    "a6": a6_rel,
    "xgb_lead4": xgb_rel,
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "scs_benchmark/scs_reliability_diagram.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

print("\nDone.")
