# ==========================================================================
# Source: paper/a7_dual_output/gbr_zeroshot_evaluation.py (verbatim except path fix)
# Experiment: GBR zero-shot transfer (SCS-trained PIECF, no parameter update)
# Paper: Results 2, Fig. 3
# Purpose: GBR test ranking (ROC AUC, AP) and detection at transferred tau=0.90
# Output: results/gbr_zeroshot_evaluation.json
# Corresponding Figure/Table: Fig. 3a; Supplementary Table S5
# ==========================================================================

# gbr_zeroshot_evaluation.py — Experiment 1: Direct Zero-shot Transfer
#
#   Goal: Evaluate SCS-trained a6 model directly on GBR (no retraining,
#     no recalibration) to measure cross-domain generalization.
#
#   Configuration:
#     Train: SCS (weights fixed)
#     Test: GBR
#     Normalizer: SCS statistics (fixed)
#     Threshold: 0.9 (fixed, from SCS optimization)
#
#   Outputs:
#     Pixel-level: CSI, F1, POD, FAR (severe >= 0.8)
#     Event-level: Instance POD (all real event instances in GBR)
#
#   Saves:
#     results/gbr_zeroshot_evaluation.json
#     results/zeroshot_gbr_pod_far.png
#     results/zeroshot_gbr_event_table.csv
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' gbr_zeroshot_evaluation.py

import os, sys, io, json, time, warnings, importlib, csv
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# a7 shared paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, PIECF_ROOT, A6_WEIGHT, XGB_WEIGHT_DIR, DATA_DIR, RESULTS_DIR

from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import build_pfv_arrays, get_year_range_for_mode

A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
FIXED_THRESHOLD = 0.9
PEAK_PROMINENCE = 0.02
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def restore_a5():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_sliding_temporal(model, ds, device):
    """Sliding-window inference → forecast_map [n_dates, H, W]."""
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
    """Compute pixel-level severe classification metrics."""
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
    """Extract temporal bleaching event instances from observed y field."""
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
    """Compute instance-level POD at given threshold."""
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
print(f"Experiment 1: Direct Zero-shot (SCS→GBR) — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))
GBR_SPLIT = dict(year_based_split=True, train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))  # LTTS_CRW GBR re-slice (val has severe events)

print("\n[Step 1] Loading SCS-trained a6 model (weights fixed)...")
a5_model = restore_a5().to(device)
print(f"  Weight: {A5_MODEL}")

print("\n[Step 2] Loading GBR test dataset...")
ds_gbr = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                         **GBR_SPLIT)
T, H, W = ds_gbr.history_weeks, ds_gbr.n_lat, ds_gbr.n_lon
n_dates = ds_gbr.n_dates
y_full = ds_gbr.y_regression_full
print(f"  T={T} H={H} W={W}  n_dates={n_dates}")
print(f"  GBR ocean pixels: {(~np.isnan(y_full[0])).sum()}")

# ═══════════════════════════════════════════════════════════════
# Step 3: Sliding-window inference on GBR
# ═══════════════════════════════════════════════════════════════
print("\n[Step 3] Sliding-window inference on GBR (zero-shot)...")
t0 = time.time()
forecast_map = predict_sliding_temporal(a5_model, ds_gbr, device)
print(f"  Done in {time.time()-t0:.1f}s  shape={forecast_map.shape}")

# ═══════════════════════════════════════════════════════════════
# Step 4: Pixel-level metrics (t+4 verification)
# ═══════════════════════════════════════════════════════════════
print("\n[Step 4] Pixel-level severe classification (t+4)...")
pfv_pred, pfv_target, _ = build_pfv_arrays(
    forecast_map, y_full, T, n_dates,
    year_range=get_year_range_for_mode(ds_gbr),
    date_strings=ds_gbr.date_strings)
n_severe = int((pfv_target >= SEVERE_THRESHOLD).sum())
print(f"  n_pixels={len(pfv_pred)}, n_severe={n_severe} ({n_severe/len(pfv_pred)*100:.2f}%)")

# Compute metrics at all thresholds for the sweep
print(f"\n  Pixel-level threshold sweep:")
print(f"  {'Thr':<6} {'CSI':<8} {'F1':<8} {'POD':<8} {'FAR':<8}")
print("  " + "-" * 38)
sweep_results = []
for thr in THRESHOLDS:
    m = classification_metrics(pfv_pred, pfv_target, SEVERE_THRESHOLD, thr)
    sweep_results.append({"threshold": thr, **m})
    print(f"  {thr:<6.1f} {m['csi']:<8.4f} {m['f1']:<8.4f} {m['pod']:<8.4f} {m['far']:<8.4f}")

# Fixed threshold result
fixed_metrics = classification_metrics(pfv_pred, pfv_target, SEVERE_THRESHOLD, FIXED_THRESHOLD)
print(f"\n  Fixed threshold={FIXED_THRESHOLD} (from SCS optimization):")
print(f"    CSI={fixed_metrics['csi']:.4f}  F1={fixed_metrics['f1']:.4f}  "
      f"POD={fixed_metrics['pod']:.4f}  FAR={fixed_metrics['far']:.4f}")

# ═══════════════════════════════════════════════════════════════
# Step 5: Event-level POD (instance-level)
# ═══════════════════════════════════════════════════════════════
print("\n[Step 5] Event-instance evaluation on GBR...")
event_instances_gbr = extract_event_instances(y_full, n_dates, H, W,
                                              SEVERE_THRESHOLD, PEAK_PROMINENCE)
print(f"  Found {len(event_instances_gbr)} event instances in GBR")

event_pod = compute_instance_pod(event_instances_gbr, forecast_map, LEAD_WEEKS, FIXED_THRESHOLD)
print(f"  Event POD (thr={FIXED_THRESHOLD}): {event_pod['pod']:.4f}  "
      f"(Hits={event_pod['hits']}, Missed={event_pod['missed']})")

# Also sweep event POD across thresholds
print(f"\n  Event POD sweep:")
print(f"  {'Thr':<6} {'POD':<8} {'Hits':<8} {'Missed':<8}")
print("  " + "-" * 30)
event_sweep = []
for thr in THRESHOLDS:
    ep = compute_instance_pod(event_instances_gbr, forecast_map, LEAD_WEEKS, thr)
    event_sweep.append({"threshold": thr, **ep})
    print(f"  {thr:<6.1f} {ep['pod']:<8.4f} {ep['hits']:<8} {ep['missed']:<8}")

# ═══════════════════════════════════════════════════════════════
# Step 6: Plot POD/FAR vs threshold
# ═══════════════════════════════════════════════════════════════
print("\n[Step 6] Plotting...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Experiment 1: Zero-shot SCS→GBR (a6, raw risk_score)\n"
             f"(weights fixed, normalizer fixed, GBR test {THRESHOLDS})",
             fontsize=11, fontweight="bold")

thr_vals = THRESHOLDS
csi_vals = [r["csi"] for r in sweep_results]
f1_vals = [r["f1"] for r in sweep_results]
pod_vals = [r["pod"] for r in sweep_results]
far_vals = [r["far"] for r in sweep_results]
event_pod_vals = [r["pod"] for r in event_sweep]

# Left: Pixel-level CSI/F1/POD/FAR
ax1.plot(thr_vals, csi_vals, "o-", color="#2196F3", linewidth=2, markersize=7, label="CSI")
ax1.plot(thr_vals, f1_vals, "s-", color="#4CAF50", linewidth=2, markersize=7, label="F1")
ax1.plot(thr_vals, pod_vals, "^-", color="#FF9800", linewidth=2, markersize=7, label="POD (pixel)")
ax1.plot(thr_vals, far_vals, "v-", color="#F44336", linewidth=2, markersize=7, label="FAR")
ax1.axvline(x=FIXED_THRESHOLD, color="gray", linestyle="--", alpha=0.5, label=f"thr={FIXED_THRESHOLD}")
ax1.set_xlabel("Threshold", fontsize=11)
ax1.set_ylabel("Metric value", fontsize=11)
ax1.set_title("Pixel-level Metrics vs Threshold", fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Right: Event POD vs threshold
ax2.plot(thr_vals, event_pod_vals, "o-", color="#9C27B0", linewidth=2, markersize=8, label="Event POD")
ax2.axvline(x=FIXED_THRESHOLD, color="gray", linestyle="--", alpha=0.5, label=f"thr={FIXED_THRESHOLD}")
ax2.set_xlabel("Threshold", fontsize=11)
ax2.set_ylabel("Event POD", fontsize=11)
ax2.set_title("Event-Instance POD vs Threshold", fontsize=11)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
for x, y in zip(thr_vals, event_pod_vals):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8, color="#9C27B0")

plt.tight_layout()
png_path = RESULTS_DIR / "gbr_transfer/gbr_zeroshot_pod_far.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {png_path}")

# ═══════════════════════════════════════════════════════════════
# Step 7: Save event instance table
# ═══════════════════════════════════════════════════════════════
print("\n[Step 7] Saving event instance table...")
csv_path = RESULTS_DIR / "gbr_transfer" / "gbr_zeroshot_event_table.csv"
details = []
for inst in event_instances_gbr:
    t_pred = inst["t"] - LEAD_WEEKS
    i, j = inst["i"], inst["j"]
    rs = forecast_map[t_pred, i, j] if 0 <= t_pred < n_dates else None
    status = "hit" if (rs is not None and not np.isnan(rs) and rs >= FIXED_THRESHOLD) else "missed"
    details.append({**inst, "risk_score": float(rs) if rs is not None and not np.isnan(rs) else None,
                    "status": status})

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["t", "i", "j", "y_peak", "prominence",
                                            "risk_score", "status"])
    writer.writeheader()
    writer.writerows(details)
print(f"  Saved: {csv_path} ({len(details)} rows)")

# ═══════════════════════════════════════════════════════════════
# Step 8: Save JSON
# ═══════════════════════════════════════════════════════════════
print("\n[Step 8] Saving JSON...")
out = {
    "experiment": "zeroshot_scs_to_gbr",
    "description": "Direct zero-shot transfer: SCS-trained a6 on GBR test set",
    "configuration": {
        "train_dataset": "SCS",
        "test_dataset": "GBR",
        "weights": "SCS-trained (fixed)",
        "normalizer": "SCS statistics (fixed)",
        "fixed_threshold": FIXED_THRESHOLD,
        "lead_weeks": LEAD_WEEKS,
        "severe_threshold": SEVERE_THRESHOLD,
    },
    "dataset_info": {
        "gbr_grid": f"{H}x{W}",
        "gbr_n_dates": n_dates,
        "gbr_ocean_pixels": int((~np.isnan(y_full[0])).sum()),
    },
    "pixel_level": {
        "n_pixels": len(pfv_pred),
        "n_severe": n_severe,
        "severe_ratio": round(n_severe / len(pfv_pred), 4),
        "threshold_sweep": sweep_results,
        "fixed_threshold_result": fixed_metrics,
    },
    "event_level": {
        "n_event_instances": len(event_instances_gbr),
        "event_pod_sweep": event_sweep,
        "fixed_threshold_event_pod": event_pod,
    },
    "main_result": {
        "note": "Primary result for Experiment 1",
        "CSI": fixed_metrics["csi"],
        "F1": fixed_metrics["f1"],
        "Event_POD": event_pod["pod"],
    },
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "gbr_transfer/gbr_zeroshot_evaluation.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

# ═══════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 1: ZERO-SHOT SCS→GBR — MAIN RESULT")
print("=" * 60)
print(f"  Configuration: weights=SCS-fixed, normalizer=SCS-fixed, thr={FIXED_THRESHOLD}")
print(f"")
print(f"  Pixel-level (t+4, severe>={SEVERE_THRESHOLD}):")
print(f"    CSI = {fixed_metrics['csi']:.4f}")
print(f"    F1  = {fixed_metrics['f1']:.4f}")
print(f"    POD = {fixed_metrics['pod']:.4f}")
print(f"    FAR = {fixed_metrics['far']:.4f}")
print(f"")
print(f"  Event-level (instance POD):")
print(f"    Event POD = {event_pod['pod']:.4f}  (Hits={event_pod['hits']}/{len(event_instances_gbr)})")
print(f"")
print(f"  n_pixels={len(pfv_pred)}, n_severe={n_severe}, n_events={len(event_instances_gbr)}")
print("\nDone.")
