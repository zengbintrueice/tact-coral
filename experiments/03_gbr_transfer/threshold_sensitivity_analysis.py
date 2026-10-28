# ==========================================================================
# Source: paper/a7_dual_output/threshold_sensitivity_analysis.py (verbatim except path fix)
# Experiment: Decision-threshold sensitivity (fine grid, SCS + GBR)
# Paper: Supplementary Fig. 1
# Output: results/threshold_sensitivity_analysis.json
# ==========================================================================

# threshold_sensitivity_analysis.py — Severe-head threshold sensitivity analysis
#
#   Goal: Empirically validate the severe-event decision threshold by
#     characterizing the full precision–recall / detection trade-off,
#     rather than reporting only a single argmax point.
#
#   Method (val-select / test-report, consistent with threshold_calibration.py):
#     - a6 sliding-window PFV inference on SCS val/test AND GBR val/test
#     - Sweep threshold in [0.10, 0.15, ..., 0.95] (0.05 step)
#     - Per threshold: CSI, POD, FAR, F1, Precision, Recall
#       (Recall == POD for binary severe classification; reported for PR curve)
#     - Plot: (a) detection-metric trade-off vs threshold, (b) PR curve
#       with operational points marked
#     - SCS operational point τ=0.9 (val argmax CSI, from threshold_calibration)
#       characterized as precision-oriented
#     - GBR: zero-shot at τ=0.9 (POD collapses) vs τ=0.54 (val-adapted, recovers CSI)
#       → score-calibration-shift story
#
#   NOTE: This is severe-head SPATIAL classification (pixel-level t+4),
#     NOT temporal event-peak detection (kept separate from event_instance).
#
#   Saves:
#     results/threshold_sensitivity_analysis.json
#     results/threshold_sensitivity_tradeoff.png
#     results/threshold_sensitivity_pr_curve.png

import os, sys, io, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# a7 shared paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR

from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import build_pfv_arrays, get_year_range_for_mode

A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8

# Fine grid for trade-off / PR curve resolution
THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(18)]  # 0.10 .. 0.95
# Ensure the operational points are present exactly (0.9 is already in grid; 0.54 is not)
for _extra in (0.54,):
    if _extra not in THRESHOLDS:
        THRESHOLDS.append(_extra)
THRESHOLDS = sorted(set(THRESHOLDS))

# Operational points (val-selected elsewhere):
#   SCS:  τ=0.9  (threshold_calibration.py, val argmax CSI; confirmed vs 0.95 grid)
#   GBR:  τ=0.54 (experiment1p2, GBR val argmax CSI)
SCS_OPERATIONAL = 0.9
GBR_OPERATIONAL = 0.54


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
    """CSI, POD, FAR, F1, Precision, Recall at a given prediction threshold.

    Recall == POD for binary severe classification (TP/(TP+FN)); reported
    separately so the PR curve can be plotted in standard precision–recall space.
    """
    pred_bin = pred >= pred_thresh
    actual_bin = actual >= severe_thresh
    hits = int(np.sum(pred_bin & actual_bin))                      # TP
    misses = int(np.sum(~pred_bin & actual_bin))                   # FN
    false_alarms = int(np.sum(pred_bin & ~actual_bin))             # FP
    correct_negatives = int(np.sum(~pred_bin & ~actual_bin))       # TN
    denom_csi = hits + misses + false_alarms
    csi = float(hits / denom_csi) if denom_csi > 0 else float("nan")
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")          # recall
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else float("nan")
    denom_f1 = 2 * hits + false_alarms + misses
    f1 = float(2 * hits / denom_f1) if denom_f1 > 0 else float("nan")
    precision = float(hits / (hits + false_alarms)) if (hits + false_alarms) > 0 else float("nan")
    return {
        "threshold": pred_thresh,
        "csi": round(csi, 4),
        "pod": round(pod, 4),            # == recall
        "far": round(far, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(pod, 4),         # alias for PR curve
        "hits": hits, "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "n_total": int(len(pred)),
    }


def sweep(pred, target, thresholds):
    return [classification_metrics(pred, target, SEVERE_THRESHOLD, t) for t in thresholds]


def get_pfv(model, ds, device):
    """Sliding-window a6 inference → PFV pixel arrays (pred, target)."""
    T = ds.history_weeks
    forecast = predict_sliding_temporal(model, ds, device)
    pred, target, _ = build_pfv_arrays(
        forecast, ds.y_regression_full, T, ds.n_dates,
        year_range=get_year_range_for_mode(ds), date_strings=ds.date_strings)
    return pred, target


GBR_SPLIT = dict(year_based_split=True, train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))  # LTTS_CRW GBR re-slice (val has severe events)


def load_dataset(mode, dataset_file=None):
    split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                    val_years=(2011, 2015), test_years=(2016, 2019))
    kw = dict(data_dir=str(DATA_DIR), mode=mode, random_split=False,
              lead_weeks=LEAD_WEEKS, **split_kw)
    if dataset_file is not None:
        kw["dataset_file"] = dataset_file
        kw.update(GBR_SPLIT)   # GBR uses its own re-sliced val/test years
    return SCSPIECFDataset(**kw)


# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print(f"Threshold Sensitivity Analysis — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nRestoring a6...")
a5_model = restore_a5().to(device)

# ═══════════════════════════════════════════════════════════════
# SCS: val (select) + test (report)
# ═══════════════════════════════════════════════════════════════
print("\n[SCS] Loading val + test...")
ds_scs_val = load_dataset("val")
scs_val_pred, scs_val_target = get_pfv(a5_model, ds_scs_val, device)
print(f"  SCS val: n={len(scs_val_pred)}, severe={int((scs_val_target>=SEVERE_THRESHOLD).sum())} "
      f"({(scs_val_target>=SEVERE_THRESHOLD).mean():.2%})")

ds_scs_test = load_dataset("test")
scs_test_pred, scs_test_target = get_pfv(a5_model, ds_scs_test, device)
print(f"  SCS test: n={len(scs_test_pred)}, severe={int((scs_test_target>=SEVERE_THRESHOLD).sum())} "
      f"({(scs_test_target>=SEVERE_THRESHOLD).mean():.2%})")

scs_val_sweep = sweep(scs_val_pred, scs_val_target, THRESHOLDS)
scs_test_sweep = sweep(scs_test_pred, scs_test_target, THRESHOLDS)

# ═══════════════════════════════════════════════════════════════
# GBR: val (select) + test (report)  — zero-shot SCS-trained model
# ═══════════════════════════════════════════════════════════════
print("\n[GBR] Loading val + test (zero-shot SCS-trained model)...")
GBR_FILE = "gbr_coral_bleaching_dataset_v9_1deg_raw.nc"
ds_gbr_val = load_dataset("val", GBR_FILE)
gbr_val_pred, gbr_val_target = get_pfv(a5_model, ds_gbr_val, device)
print(f"  GBR val: n={len(gbr_val_pred)}, severe={int((gbr_val_target>=SEVERE_THRESHOLD).sum())} "
      f"({(gbr_val_target>=SEVERE_THRESHOLD).mean():.2%})")

ds_gbr_test = load_dataset("test", GBR_FILE)
gbr_test_pred, gbr_test_target = get_pfv(a5_model, ds_gbr_test, device)
print(f"  GBR test: n={len(gbr_test_pred)}, severe={int((gbr_test_target>=SEVERE_THRESHOLD).sum())} "
      f"({(gbr_test_target>=SEVERE_THRESHOLD).mean():.2%})")

gbr_val_sweep = sweep(gbr_val_pred, gbr_val_target, THRESHOLDS)
gbr_test_sweep = sweep(gbr_test_pred, gbr_test_target, THRESHOLDS)

# ═══════════════════════════════════════════════════════════════
# Operational-point characterization
# ═══════════════════════════════════════════════════════════════
def at(swp, thr):
    return next((r for r in swp if abs(r["threshold"] - thr) < 1e-6), None)

scs_at_09 = at(scs_test_sweep, SCS_OPERATIONAL)
gbr_at_09 = at(gbr_test_sweep, SCS_OPERATIONAL)        # zero-shot SCS threshold
gbr_at_054 = at(gbr_test_sweep, GBR_OPERATIONAL)        # GBR-adapted

print("\n" + "=" * 64)
print("OPERATIONAL-POINT CHARACTERIZATION (test)")
print("=" * 64)
print(f"\n  SCS @ τ={SCS_OPERATIONAL} (val-selected): "
      f"CSI={scs_at_09['csi']} POD={scs_at_09['pod']} FAR={scs_at_09['far']} "
      F"Prec={scs_at_09['precision']}")
print(f"\n  GBR @ τ={SCS_OPERATIONAL} (zero-shot SCS threshold): "
      f"CSI={gbr_at_09['csi']} POD={gbr_at_09['pod']} FAR={gbr_at_09['far']} "
      f"Prec={gbr_at_09['precision']}")
print(f"  GBR @ τ={GBR_OPERATIONAL} (GBR val-adapted):           "
      f"CSI={gbr_at_054['csi']} POD={gbr_at_054['pod']} FAR={gbr_at_054['far']} "
      f"Prec={gbr_at_054['precision']}")
gbr_csi_gain = gbr_at_054['csi'] - gbr_at_09['csi']
print(f"\n  GBR threshold adaptation τ {SCS_OPERATIONAL}→{GBR_OPERATIONAL}: "
      f"CSI {gbr_at_09['csi']}→{gbr_at_054['csi']} ({gbr_csi_gain:+.4f})")

# ═══════════════════════════════════════════════════════════════
# Plot 1: Detection-metric trade-off vs threshold
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Severe-head threshold sensitivity (a6, t+4 pixel-level PFV)\n"
             f"severe ≥ {SEVERE_THRESHOLD}, year-based val(2011–2015)/test(2016–2019)",
             fontsize=12, fontweight="bold")


def plot_tradeoff(ax, swp, title, mark_thr, color):
    thrs = [r["threshold"] for r in swp]
    ax.plot(thrs, [r["csi"] for r in swp], "o-", color=color, lw=2, ms=5, label="CSI")
    ax.plot(thrs, [r["pod"] for r in swp], "s--", color="#2E7D32", lw=1.5, ms=4, label="POD (recall)")
    ax.plot(thrs, [r["far"] for r in swp], "^--", color="#C62828", lw=1.5, ms=4, label="FAR")
    ax.plot(thrs, [r["precision"] for r in swp], "D--", color="#6A1B9A", lw=1.5, ms=4, label="Precision")
    ax.axvline(mark_thr, color="k", ls=":", lw=1.5, alpha=0.7)
    ax.text(mark_thr, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0, f" τ={mark_thr}",
            fontsize=8, va="top")
    ax.set_xlabel("Decision threshold", fontsize=10)
    ax.set_ylabel("Metric value", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.08, 0.97)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


plot_tradeoff(axes[0], scs_test_sweep, f"SCS test (operational τ={SCS_OPERATIONAL})",
              SCS_OPERATIONAL, "#1565C0")
plot_tradeoff(axes[1], gbr_test_sweep, f"GBR test (zero-shot; adapted τ={GBR_OPERATIONAL})",
              GBR_OPERATIONAL, "#E65100")
# also mark the SCS threshold on GBR panel (shows zero-shot failure)
axes[1].axvline(SCS_OPERATIONAL, color="#1565C0", ls=":", lw=1.5, alpha=0.7)
axes[1].text(SCS_OPERATIONAL, 0.5, f" τ={SCS_OPERATIONAL}\n (zero-shot)", fontsize=7,
             va="center", color="#1565C0")

plt.tight_layout()
p1 = RESULTS_DIR / "gbr_transfer/threshold_sensitivity_tradeoff.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {p1}")

# ═══════════════════════════════════════════════════════════════
# Plot 2: Precision–Recall curve
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.set_title("Precision–Recall: severe-event detection (a6)\n"
             "rare-event regime — PR is more informative than ROC", fontsize=11, fontweight="bold")

for swp, label, color, marks in [
    (scs_test_sweep, f"SCS test (severe {(scs_test_target>=SEVERE_THRESHOLD).mean():.1%})",
     "#1565C0", [(SCS_OPERATIONAL, "τ=0.9\n(precision-oriented)")]),
    (gbr_test_sweep, f"GBR test (severe {(gbr_test_target>=SEVERE_THRESHOLD).mean():.1%})",
     "#E65100", [(SCS_OPERATIONAL, "τ=0.9\n(zero-shot)"),
                 (GBR_OPERATIONAL, "τ=0.54\n(adapted)")]),
]:
    recalls = [r["recall"] for r in swp]
    precisions = [r["precision"] for r in swp]
    ax.plot(recalls, precisions, "o-", color=color, lw=2, ms=5, label=label)
    for thr, lbl in marks:
        r = at(swp, thr)
        if r is None:
            continue
        ax.scatter([r["recall"]], [r["precision"]], s=120, color=color,
                   edgecolor="k", zorder=5)
        ax.annotate(lbl, (r["recall"], r["precision"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=7.5)

ax.set_xlabel("Recall (= POD)", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc="best")
plt.tight_layout()
p2 = RESULTS_DIR / "gbr_transfer/threshold_sensitivity_pr_curve.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p2}")

# ═══════════════════════════════════════════════════════════════
# Save JSON
# ═══════════════════════════════════════════════════════════════
out = {
    "experiment": "severe_head_threshold_sensitivity",
    "description": "Empirical validation of the severe-event decision threshold via "
                   "full precision–recall / detection trade-off (val-select / test-report).",
    "method": "a6 sliding-window PFV, t+4 pixel-level severe classification (NOT event-peak detection)",
    "severe_threshold": SEVERE_THRESHOLD,
    "lead_weeks": LEAD_WEEKS,
    "thresholds": THRESHOLDS,
    "selection_protocol": "operating point selected on val (2011-2015); trade-off curves reported on test (2016-2019)",
    "note": "Recall == POD for binary severe classification; reported as Recall for PR-curve compatibility.",
    "scs": {
        "val_sweep": scs_val_sweep,
        "test_sweep": scs_test_sweep,
        "operational_threshold": SCS_OPERATIONAL,
        "operational_source": "threshold_calibration.py (val argmax CSI, confirmed vs 0.95 grid)",
        "test_at_operational": scs_at_09,
    },
    "gbr": {
        "val_sweep": gbr_val_sweep,
        "test_sweep": gbr_test_sweep,
        "zero_shot_threshold": SCS_OPERATIONAL,
        "adapted_threshold": GBR_OPERATIONAL,
        "adapted_source": "experiment1p2 (GBR val argmax CSI)",
        "test_at_zero_shot": gbr_at_09,
        "test_at_adapted": gbr_at_054,
        "csi_gain_from_adaptation": round(gbr_csi_gain, 4),
    },
    "interpretation": {
        "scs": f"τ={SCS_OPERATIONAL} is precision-oriented: lowest FAR among useful-CSI thresholds; "
               "conservative early-warning operating point under class imbalance.",
        "gbr": f"Zero-shot τ={SCS_OPERATIONAL} compresses POD; val-adapted τ={GBR_OPERATIONAL} recovers CSI, "
               "consistent with score-calibration shift rather than representation failure.",
    },
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "gbr_transfer/threshold_sensitivity_analysis.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

print("\nDone.")
