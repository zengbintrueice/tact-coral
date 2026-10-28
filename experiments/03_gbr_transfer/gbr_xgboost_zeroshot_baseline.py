# ==========================================================================
# Source: paper/a7_dual_output/gbr_xgboost_zeroshot_baseline.py (verbatim except path fix)
# Experiment: XGBoost GBR zero-shot baseline
# Paper: Results 2 (XGBoost control)
# Output: results/zeroshot_gbr_xgb_comparison.json
# ==========================================================================

# gbr_xgboost_zeroshot_baseline.py — XGB Zero-shot Baseline + GBR Case Studies
#
#   Purpose:
#     1. Run SCS-trained XGBoost on GBR (zero-shot) for comparison
#     2. Generate GBR case study visualizations (2016 bleaching, etc.)
#
#   Configuration:
#     XGB: SCS-trained (12 leads), lead-4 for t+4 verification
#     a7: reuses forecast_map from zeroshot_gbr_evaluation.py
#     Threshold: fixed at 0.9 for a7, 0.7 for XGB (from SCS optimization)
#
#   Saves:
#     results/zeroshot_gbr_xgb_comparison.json
#     results/zeroshot_gbr_xgb_comparison.png
#     results/zeroshot_gbr_case_study_{N}.png
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' gbr_xgboost_zeroshot_baseline.py

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
from scipy.signal import find_peaks

# a7 shared paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, PIECF_ROOT, A6_WEIGHT, XGB_WEIGHT_DIR, DATA_DIR, RESULTS_DIR

from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import get_year_range_for_mode

A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
A7_THRESHOLD = 0.9
XGB_THRESHOLD = 0.8
PEAK_PROMINENCE = 0.02
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# GBR case study events (top by prominence, will be selected dynamically)
N_CASE_STUDIES = 5


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


def classification_metrics(pred, actual, severe_thresh, pred_thresh):
    pred_bin = pred >= pred_thresh
    actual_bin = actual >= severe_thresh
    hits = int(np.sum(pred_bin & actual_bin))
    misses = int(np.sum(~pred_bin & actual_bin))
    false_alarms = int(np.sum(pred_bin & ~actual_bin))
    denom_csi = hits + misses + false_alarms
    csi = float(hits / denom_csi) if denom_csi > 0 else float("nan")
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else float("nan")
    denom_f1 = 2 * hits + false_alarms + misses
    f1 = float(2 * hits / denom_f1) if denom_f1 > 0 else float("nan")
    return {"csi": round(csi, 4), "f1": round(f1, 4), "pod": round(pod, 4), "far": round(far, 4),
            "hits": hits, "misses": misses, "false_alarms": false_alarms, "n_total": int(len(pred))}


def extract_event_instances(y_full, n_dates, H, W, severe_thresh, prominence):
    instances = []
    for i in range(H):
        for j in range(W):
            ts = y_full[:, i, j]
            valid = ~np.isnan(ts)
            if valid.sum() < 3:
                continue
            peaks, props = find_peaks(ts, prominence=prominence)
            prom_vals = props["prominences"]
            for p, prom in zip(peaks, prom_vals):
                if ts[p] >= severe_thresh and p < n_dates:
                    instances.append({"t": int(p), "i": int(i), "j": int(j),
                                      "y_peak": float(ts[p]), "prominence": float(prom)})
    return instances


def compute_instance_pod(event_instances, forecast_map, lead_weeks, threshold):
    n_dates, H, W = forecast_map.shape
    hits = 0
    missed = 0
    for inst in event_instances:
        t_pred = inst["t"] - lead_weeks
        i, j = inst["i"], inst["j"]
        if t_pred < 0 or t_pred >= n_dates:
            missed += 1
            continue
        rs = forecast_map[t_pred, i, j]
        if np.isnan(rs):
            missed += 1
            continue
        if rs >= threshold:
            hits += 1
        else:
            missed += 1
    denom = hits + missed
    pod = hits / denom if denom > 0 else float("nan")
    return {"hits": hits, "missed": missed, "pod": round(pod, 4)}


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"XGB Zero-shot Baseline + GBR Case Studies — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))
GBR_SPLIT = dict(year_based_split=True, train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))  # LTTS_CRW GBR re-slice (val has severe events)

print("\n[Step 1] Loading GBR test dataset...")
ds_gbr = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                         **GBR_SPLIT)
T, H, W = ds_gbr.history_weeks, ds_gbr.n_lat, ds_gbr.n_lon
n_dates = ds_gbr.n_dates
y_full = ds_gbr.y_regression_full
C_dyn = ds_gbr.C_dyn
C_stat = ds_gbr.C_stat
X_stat = ds_gbr.X_static[0]
print(f"  T={T} H={H} W={W}  n_dates={n_dates}")

# ═══════════════════════════════════════════════════════════════
# Step 2: a7 inference (zero-shot)
# ═══════════════════════════════════════════════════════════════
print("\n[Step 2] a7 sliding-window inference on GBR...")
a5_model = restore_a5().to(device)
t0 = time.time()
forecast_map_a7 = predict_sliding_temporal(a5_model, ds_gbr, device)
print(f"  Done in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# Step 3: XGB lead-4 inference (zero-shot)
# ═══════════════════════════════════════════════════════════════
print("\n[Step 3] XGBoost lead-4 inference on GBR...")
xgb_model = joblib.load(XGB_WEIGHT_DIR / "xgb_lead4.joblib")
xgb_scaler = joblib.load(XGB_WEIGHT_DIR / "xgb_lead4_scaler.joblib")

forecast_map_xgb = np.full((n_dates, H, W), np.nan, dtype=np.float32)
for t in range(T - 1, n_dates):
    s = t - T + 1
    dyn_win = ds_gbr.X_dynamic_full[s:s + T]
    dyn_flat = dyn_win.transpose(1, 2, 0, 3).reshape(H * W, T * C_dyn)
    stat_flat = X_stat.reshape(H * W, C_stat)
    X = np.concatenate([dyn_flat, stat_flat], axis=1).astype(np.float32)
    X_s = xgb_scaler.transform(X)
    pred_t = np.clip(xgb_model.predict(X_s), 0, 1).astype(np.float32).reshape(H, W)
    forecast_map_xgb[t] = pred_t
print(f"  Done. XGB forecast_map shape: {forecast_map_xgb.shape}")

# ═══════════════════════════════════════════════════════════════
# Step 4: Pixel-level comparison (t+4 verification)
# ═══════════════════════════════════════════════════════════════
print("\n[Step 4] Pixel-level comparison (t+4)...")
# Year-filtered evaluation: only include time steps in the mode's year range
year_range = get_year_range_for_mode(ds_gbr)
pfv_pred_a7, pfv_pred_xgb, pfv_target = [], [], []
for t in range(T - 1, n_dates):
    # Year filtering
    if year_range is not None:
        year = int(ds_gbr.date_strings[t][:4])
        if not (year_range[0] <= year <= year_range[1]):
            continue
    t_target = t + LEAD_WEEKS
    if t_target >= n_dates:
        continue
    pred_a7_t = forecast_map_a7[t]
    pred_xgb_t = forecast_map_xgb[t]
    target_t = y_full[t_target]
    valid = ~np.isnan(pred_a7_t) & ~np.isnan(pred_xgb_t) & ~np.isnan(target_t)
    if valid.sum() < 1:
        continue
    pfv_pred_a7.extend(pred_a7_t[valid].tolist())
    pfv_pred_xgb.extend(pred_xgb_t[valid].tolist())
    pfv_target.extend(target_t[valid].tolist())

pfv_pred_a7 = np.array(pfv_pred_a7, dtype=np.float32)
pfv_pred_xgb = np.array(pfv_pred_xgb, dtype=np.float32)
pfv_target = np.array(pfv_target, dtype=np.float32)
print(f"  n_pixels={len(pfv_pred_a7)}, n_severe={int((pfv_target>=SEVERE_THRESHOLD).sum())}")

# Fixed threshold comparison
a7_metrics = classification_metrics(pfv_pred_a7, pfv_target, SEVERE_THRESHOLD, A7_THRESHOLD)
xgb_metrics = classification_metrics(pfv_pred_xgb, pfv_target, SEVERE_THRESHOLD, XGB_THRESHOLD)

print(f"\n  Fixed threshold comparison:")
print(f"    a7  (thr={A7_THRESHOLD}): CSI={a7_metrics['csi']:.4f}  F1={a7_metrics['f1']:.4f}  "
      f"POD={a7_metrics['pod']:.4f}  FAR={a7_metrics['far']:.4f}")
print(f"    XGB (thr={XGB_THRESHOLD}): CSI={xgb_metrics['csi']:.4f}  F1={xgb_metrics['f1']:.4f}  "
      f"POD={xgb_metrics['pod']:.4f}  FAR={xgb_metrics['far']:.4f}")

# Threshold sweep for both
print(f"\n  Threshold sweep:")
print(f"  {'Thr':<6} {'a7 CSI':<10} {'a7 F1':<10} {'XGB CSI':<10} {'XGB F1':<10}")
print("  " + "-" * 46)
a7_sweep, xgb_sweep = [], []
for thr in THRESHOLDS:
    ma = classification_metrics(pfv_pred_a7, pfv_target, SEVERE_THRESHOLD, thr)
    mx = classification_metrics(pfv_pred_xgb, pfv_target, SEVERE_THRESHOLD, thr)
    a7_sweep.append({"threshold": thr, **ma})
    xgb_sweep.append({"threshold": thr, **mx})
    print(f"  {thr:<6.1f} {ma['csi']:<10.4f} {ma['f1']:<10.4f} {mx['csi']:<10.4f} {mx['f1']:<10.4f}")

# ═══════════════════════════════════════════════════════════════
# Step 5: Event-level comparison
# ═══════════════════════════════════════════════════════════════
print("\n[Step 5] Event-instance comparison...")
event_instances = extract_event_instances(y_full, n_dates, H, W, SEVERE_THRESHOLD, PEAK_PROMINENCE)
print(f"  Found {len(event_instances)} event instances")

a7_event_pod = compute_instance_pod(event_instances, forecast_map_a7, LEAD_WEEKS, A7_THRESHOLD)
xgb_event_pod = compute_instance_pod(event_instances, forecast_map_xgb, LEAD_WEEKS, XGB_THRESHOLD)

print(f"  a7  Event POD (thr={A7_THRESHOLD}): {a7_event_pod['pod']:.4f}  "
      f"(Hits={a7_event_pod['hits']}/{len(event_instances)})")
print(f"  XGB Event POD (thr={XGB_THRESHOLD}): {xgb_event_pod['pod']:.4f}  "
      f"(Hits={xgb_event_pod['hits']}/{len(event_instances)})")

# ═══════════════════════════════════════════════════════════════
# Step 6: Plot comparison
# ═══════════════════════════════════════════════════════════════
print("\n[Step 6] Plotting comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Zero-shot SCS→GBR: a7 vs XGBoost (raw output, fixed thresholds)\n"
             f"(weights frozen, normalizer frozen, GBR test 2016-2019)",
             fontsize=11, fontweight="bold")

thr_vals = THRESHOLDS
a7_csi = [r["csi"] for r in a7_sweep]
a7_f1 = [r["f1"] for r in a7_sweep]
xgb_csi = [r["csi"] for r in xgb_sweep]
xgb_f1 = [r["f1"] for r in xgb_sweep]

# Left: CSI
ax1.plot(thr_vals, a7_csi, "o-", color="#2196F3", linewidth=2, markersize=7, label="a7 CSI")
ax1.plot(thr_vals, xgb_csi, "s-", color="#FF9800", linewidth=2, markersize=7, label="XGB CSI")
ax1.axvline(x=A7_THRESHOLD, color="#2196F3", linestyle=":", alpha=0.4)
ax1.axvline(x=XGB_THRESHOLD, color="#FF9800", linestyle=":", alpha=0.4)
ax1.set_xlabel("Threshold", fontsize=11)
ax1.set_ylabel("CSI", fontsize=11)
ax1.set_title("Pixel-level CSI vs Threshold", fontsize=11)
ax1.set_ylim(-0.05, 0.65)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Right: F1
ax2.plot(thr_vals, a7_f1, "o-", color="#2196F3", linewidth=2, markersize=7, label="a7 F1")
ax2.plot(thr_vals, xgb_f1, "s-", color="#FF9800", linewidth=2, markersize=7, label="XGB F1")
ax2.axvline(x=A7_THRESHOLD, color="#2196F3", linestyle=":", alpha=0.4)
ax2.axvline(x=XGB_THRESHOLD, color="#FF9800", linestyle=":", alpha=0.4)
ax2.set_xlabel("Threshold", fontsize=11)
ax2.set_ylabel("F1", fontsize=11)
ax2.set_title("Pixel-level F1 vs Threshold", fontsize=11)
ax2.set_ylim(-0.05, 0.65)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
png_path = RESULTS_DIR / "gbr_transfer/gbr_xgboost_zeroshot_baseline.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {png_path}")

# ═══════════════════════════════════════════════════════════════
# Step 7: GBR Case Studies
# ═══════════════════════════════════════════════════════════════
print(f"\n[Step 7] Generating {N_CASE_STUDIES} GBR case studies...")

# Select top events by prominence
sorted_events = sorted(event_instances, key=lambda x: x["prominence"], reverse=True)
case_events = sorted_events[:N_CASE_STUDIES]

for idx, event in enumerate(case_events, 1):
    t_ev = event["t"]
    i, j = event["i"], event["j"]
    y_peak = event["y_peak"]
    prom = event["prominence"]

    # Time window: [t_ev-12, t_ev+4]
    t_start = max(0, t_ev - 12)
    t_end = min(n_dates, t_ev + 5)
    t_window = np.arange(t_start, t_end)

    # Extract time series
    y_ts = y_full[t_window, i, j]
    a7_ts = np.full_like(y_ts, np.nan)
    xgb_ts = np.full_like(y_ts, np.nan)
    for k, t in enumerate(t_window):
        t_pred = t - LEAD_WEEKS
        if 0 <= t_pred < n_dates:
            a7_ts[k] = forecast_map_a7[t_pred, i, j]
            xgb_ts[k] = forecast_map_xgb[t_pred, i, j]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[1, 1])
    fig.suptitle(f"GBR Case Study #{idx} — Regional Bleaching Event\n"
                 f"Location ({i},{j}), t_peak={t_ev}, y_peak={y_peak:.3f}, prominence={prom:.3f}",
                 fontsize=11, fontweight="bold")

    # Top: time series
    ax1.plot(t_window, y_ts, "b-", linewidth=2, label="Observed y(t)", marker="o", markersize=4)
    ax1.plot(t_window, a7_ts, "r--", linewidth=2, label=f"a7 risk_score (t-{LEAD_WEEKS})",
             marker="s", markersize=4)
    ax1.plot(t_window, xgb_ts, "g--", linewidth=2, label=f"XGB score (t-{LEAD_WEEKS})",
             marker="^", markersize=4)
    ax1.axvline(x=t_ev, color="blue", linestyle=":", alpha=0.5, label="Peak time")
    ax1.axvline(x=t_ev - LEAD_WEEKS, color="red", linestyle=":", alpha=0.5,
                label=f"Prediction time (t-{LEAD_WEEKS})")
    ax1.axhline(y=SEVERE_THRESHOLD, color="gray", linestyle="--", alpha=0.4)
    ax1.axhline(y=A7_THRESHOLD, color="red", linestyle="--", alpha=0.3)
    ax1.axhline(y=XGB_THRESHOLD, color="green", linestyle="--", alpha=0.3)
    ax1.set_ylabel("y / risk_score", fontsize=10)
    ax1.set_title("Time Series at Event Location", fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_xlim(t_window[0], t_window[-1])
    ax1.set_ylim(-0.05, 1.2)
    ax1.grid(True, alpha=0.3)

    # Bottom: spatial map of a7 risk_score at t_peak-4
    t_pred_map = t_ev - LEAD_WEEKS
    if 0 <= t_pred_map < n_dates:
        rs_map = forecast_map_a7[t_pred_map].copy()
        im = ax2.imshow(rs_map, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        ax2.plot(j, i, "r*", markersize=15, label="Event location")
        ax2.set_xlabel("lon index", fontsize=10)
        ax2.set_ylabel("lat index", fontsize=10)
        ax2.set_title(f"a7 Risk Score Map at t={t_pred_map} (peak - {LEAD_WEEKS} weeks)", fontsize=10)
        ax2.legend(fontsize=8)
        plt.colorbar(im, ax=ax2, shrink=0.8, label="risk_score")

    plt.tight_layout()
    case_path = RESULTS_DIR / f"zeroshot_gbr_case_study_{idx}.png"
    plt.savefig(case_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Case #{idx}: t={t_ev} loc=({i},{j}) y_peak={y_peak:.3f} prom={prom:.3f}")

# ═══════════════════════════════════════════════════════════════
# Step 8: Save JSON
# ═══════════════════════════════════════════════════════════════
print("\n[Step 8] Saving JSON...")
out = {
    "experiment": "zeroshot_scs_to_gbr_xgb_comparison",
    "description": "Zero-shot transfer comparison: a7 vs XGBoost (SCS-trained → GBR-test)",
    "configuration": {
        "train_dataset": "SCS",
        "test_dataset": "GBR",
        "weights": "SCS-trained (fixed)",
        "normalizer": "SCS statistics (fixed)",
        "a7_threshold": A7_THRESHOLD,
        "xgb_threshold": XGB_THRESHOLD,
        "lead_weeks": LEAD_WEEKS,
        "severe_threshold": SEVERE_THRESHOLD,
    },
    "pixel_level": {
        "n_pixels": len(pfv_pred_a7),
        "n_severe": int((pfv_target >= SEVERE_THRESHOLD).sum()),
        "a7_fixed": a7_metrics,
        "xgb_fixed": xgb_metrics,
        "a7_sweep": a7_sweep,
        "xgb_sweep": xgb_sweep,
    },
    "event_level": {
        "n_event_instances": len(event_instances),
        "a7_event_pod": a7_event_pod,
        "xgb_event_pod": xgb_event_pod,
    },
    "main_comparison": {
        "note": "Primary result: a7 vs XGB zero-shot on GBR",
        "a7": {"CSI": a7_metrics["csi"], "F1": a7_metrics["f1"], "Event_POD": a7_event_pod["pod"]},
        "xgb": {"CSI": xgb_metrics["csi"], "F1": xgb_metrics["f1"], "Event_POD": xgb_event_pod["pod"]},
        "winner": "a7" if a7_metrics["csi"] > xgb_metrics["csi"] else "xgb",
    },
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "gbr_transfer/gbr_xgboost_zeroshot_baseline.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

# ═══════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ZERO-SHOT SCS→GBR: a7 vs XGB — COMPARISON")
print("=" * 60)
print(f"  Pixel-level (fixed thresholds):")
print(f"    a7  (thr={A7_THRESHOLD}): CSI={a7_metrics['csi']:.4f}  F1={a7_metrics['f1']:.4f}")
print(f"    XGB (thr={XGB_THRESHOLD}): CSI={xgb_metrics['csi']:.4f}  F1={xgb_metrics['f1']:.4f}")
print(f"")
print(f"  Event-level:")
print(f"    a7  Event POD: {a7_event_pod['pod']:.4f}")
print(f"    XGB Event POD: {xgb_event_pod['pod']:.4f}")
print(f"")
print(f"  Winner (CSI): {'a7' if a7_metrics['csi'] > xgb_metrics['csi'] else 'XGB'}")
print("\nDone.")
