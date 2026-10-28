# ==========================================================================
# Source: paper/a7_dual_output/scs_severe_event_evaluation.py (verbatim except path fix)
# Experiment: Final severe-event classification evaluation
# Paper: Results 1, Fig. 2
# Purpose: severe-event (y>=0.8) operating characteristics at val-selected thresholds
# Output: results/scs_severe_event_evaluation.json
# Corresponding Figure/Table: Fig. 2d; Supplementary Table S4
# ==========================================================================

# scs_severe_event_evaluation.py — Final Severe Classification Comparison (Main Result)
#
#   Goal: Produce the definitive severe classification table for the paper,
#     comparing a7 (raw risk_score) vs XGBoost (raw lead-4) at their
#     respective optimal operating points.
#
#   Method:
#     - Pure Forecasting Verification (sliding-window, t+4 pixel-level)
#     - a7:  raw risk_score, threshold=0.9 (optimized via threshold_calibration)
#     - XGB: raw lead-4 output, threshold=0.7 (optimized via threshold_calibration)
#     - Metrics: CSI, F1, POD, FAR, Hits, Misses, FA, n_total
#
#   Saves: results/scs_severe_event_evaluation.json + scs_severe_event_evaluation_table.png
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' scs_severe_event_evaluation.py

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

# Operating points: loaded from threshold_calibration.json, where they are
# SELECTED ON VALIDATION (2011-2015) — not on this test set. Falls back to the
# val-derived defaults below if the JSON is absent (e.g. first run).
_THRESH_JSON = RESULTS_DIR / "scs_benchmark/scs_threshold_calibration.json"
_A7_DEFAULT = 0.9
_XGB_DEFAULT = 0.7
try:
    with open(_THRESH_JSON, encoding="utf-8") as _f:
        _tc = json.load(_f)
    A7_THRESHOLD = _tc["a6"]["best_csi"]["threshold"]
    XGB_THRESHOLD = _tc["xgb_lead4"]["best_csi"]["threshold"]
    _thresh_source = f"threshold_calibration.json (val-selected)"
except Exception as _e:
    A7_THRESHOLD = _A7_DEFAULT
    XGB_THRESHOLD = _XGB_DEFAULT
    _thresh_source = f"hardcoded defaults (JSON unavailable: {_e})"
print(f"[final_severe] thresholds: a6={A7_THRESHOLD}, xgb={XGB_THRESHOLD}  (source: {_thresh_source})")


def classification_metrics(pred, actual, severe_thresh, pred_thresh):
    """Compute severe classification metrics at given prediction threshold."""
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
    return {
        "csi": round(csi, 4),
        "f1": round(f1, 4),
        "pod": round(pod, 4),
        "far": round(far, 4),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "n_total": int(len(pred)),
    }


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


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Final Severe Evaluation — {datetime.now():%Y-%m-%d %H:%M:%S}")
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

# ═══════════════════════════════════════════════════════════════
# a7: Sliding-window inference → pixel-level t+4 predictions
# ═══════════════════════════════════════════════════════════════
print("\nSliding-window inference (a7)...")
t0 = time.time()
forecast_map = predict_sliding_temporal(a5_model, ds_test, device)
print(f"  Done in {time.time()-t0:.1f}s")

year_range = get_year_range_for_mode(ds_test)
date_strings = ds_test.date_strings
pfv_pred_a7, pfv_target, _ = build_pfv_arrays(
    forecast_map, y_full, T, n_dates,
    year_range=year_range, date_strings=date_strings)
print(f"  n_pixels={len(pfv_pred_a7)}, n_severe={int((pfv_target>=SEVERE_THRESHOLD).sum())}")

# ═══════════════════════════════════════════════════════════════
# XGBoost: lead-4 per-window inference → pixel-level t+4 predictions
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# Compute metrics at optimal operating points
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Final Severe Classification (optimal operating points)")
print("=" * 60)

a7_metrics = classification_metrics(pfv_pred_a7, pfv_target, SEVERE_THRESHOLD, A7_THRESHOLD)
xgb_metrics = classification_metrics(pfv_pred_xgb, pfv_target, SEVERE_THRESHOLD, XGB_THRESHOLD)

print(f"\n  a7  (raw, thr={A7_THRESHOLD}):")
print(f"    CSI={a7_metrics['csi']:.4f}  F1={a7_metrics['f1']:.4f}  "
      f"POD={a7_metrics['pod']:.4f}  FAR={a7_metrics['far']:.4f}")
print(f"    Hits={a7_metrics['hits']}  Misses={a7_metrics['misses']}  "
      f"FA={a7_metrics['false_alarms']}  n={a7_metrics['n_total']}")

print(f"\n  XGB (raw, thr={XGB_THRESHOLD}):")
print(f"    CSI={xgb_metrics['csi']:.4f}  F1={xgb_metrics['f1']:.4f}  "
      f"POD={xgb_metrics['pod']:.4f}  FAR={xgb_metrics['far']:.4f}")
print(f"    Hits={xgb_metrics['hits']}  Misses={xgb_metrics['misses']}  "
      f"FA={xgb_metrics['false_alarms']}  n={xgb_metrics['n_total']}")

# ═══════════════════════════════════════════════════════════════
# Plot: Final comparison table
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis("off")
ax.set_title(f"Final Severe Classification Comparison\n(severe >= {SEVERE_THRESHOLD}, t+{LEAD_WEEKS} pixel-level, year-based test 2016-2019)",
             fontsize=12, fontweight="bold", pad=15)

# Table data
columns = ["Model", "Threshold", "CSI", "F1", "POD", "FAR"]
cell_data = [
    ["a7 (raw)", f"{A7_THRESHOLD}", f"{a7_metrics['csi']:.4f}",
     f"{a7_metrics['f1']:.4f}", f"{a7_metrics['pod']:.4f}", f"{a7_metrics['far']:.4f}"],
    ["XGB (raw)", f"{XGB_THRESHOLD}", f"{xgb_metrics['csi']:.4f}",
     f"{xgb_metrics['f1']:.4f}", f"{xgb_metrics['pod']:.4f}", f"{xgb_metrics['far']:.4f}"],
]

table = ax.table(cellText=cell_data, colLabels=columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.6)

# Style header
for j in range(len(columns)):
    table[0, j].set_facecolor("#E3F2FD")
    table[0, j].set_text_props(fontweight="bold")

# Style rows
colors = ["#F3E5F5", "#FFF3E0"]
for i, color in enumerate(colors):
    for j in range(len(columns)):
        table[i + 1, j].set_facecolor(color)

# Bold the winning values
if a7_metrics["csi"] > xgb_metrics["csi"]:
    table[1, 2].set_text_props(fontweight="bold")
else:
    table[2, 2].set_text_props(fontweight="bold")
if a7_metrics["f1"] > xgb_metrics["f1"]:
    table[1, 3].set_text_props(fontweight="bold")
else:
    table[2, 3].set_text_props(fontweight="bold")

plt.tight_layout()
png_path = RESULTS_DIR / "scs_benchmark/scs_severe_event_evaluation_table.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Table saved: {png_path}")

# ═══════════════════════════════════════════════════════════════
# Save JSON
# ═══════════════════════════════════════════════════════════════
out = {
    "method": "final_severe_classification",
    "description": "Definitive severe classification comparison at optimal operating points",
    "severe_threshold": SEVERE_THRESHOLD,
    "lead_weeks": LEAD_WEEKS,
    "evaluation": "Pure Forecasting Verification (sliding-window, t+4 pixel-level)",
    "test_period": "2016-2019",
    "models": {
        "a7_raw": {
            "threshold": A7_THRESHOLD,
            "output": "raw risk_score (S_raw)",
            **a7_metrics,
        },
        "xgb_raw": {
            "threshold": XGB_THRESHOLD,
            "output": "raw XGBoost lead-4",
            **xgb_metrics,
        },
    },
    "conclusion": {
        "better_csi": "a7" if a7_metrics["csi"] > xgb_metrics["csi"] else "xgb",
        "better_f1": "a7" if a7_metrics["f1"] > xgb_metrics["f1"] else "xgb",
        "delta_csi": round(a7_metrics["csi"] - xgb_metrics["csi"], 4),
        "delta_f1": round(a7_metrics["f1"] - xgb_metrics["f1"], 4),
    },
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "scs_benchmark/scs_severe_event_evaluation.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

# ═══════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SEVERE CLASSIFICATION TABLE")
print("=" * 60)
print(f"{'Model':<12} {'Threshold':<10} {'CSI':<8} {'F1':<8} {'POD':<8} {'FAR':<8}")
print("-" * 54)
print(f"{'a7 (raw)':<12} {A7_THRESHOLD:<10} {a7_metrics['csi']:<8.4f} "
      f"{a7_metrics['f1']:<8.4f} {a7_metrics['pod']:<8.4f} {a7_metrics['far']:<8.4f}")
print(f"{'XGB (raw)':<12} {XGB_THRESHOLD:<10} {xgb_metrics['csi']:<8.4f} "
      f"{xgb_metrics['f1']:<8.4f} {xgb_metrics['pod']:<8.4f} {xgb_metrics['far']:<8.4f}")
print("=" * 60)
print(f"\n  Winner CSI: {'a7' if a7_metrics['csi'] > xgb_metrics['csi'] else 'xgb'} "
      f"(delta={a7_metrics['csi'] - xgb_metrics['csi']:+.4f})")
print(f"  Winner F1:  {'a7' if a7_metrics['f1'] > xgb_metrics['f1'] else 'xgb'} "
      f"(delta={a7_metrics['f1'] - xgb_metrics['f1']:+.4f})")
print("\nDone.")
