# ==========================================================================
# Source: paper/a7_dual_output/gbr_score_recalibration.py (verbatim except path fix)
# Experiment: Score recalibration comparison (threshold vs isotonic)
# Paper: Results 2, Supplementary Table S5
# Purpose: baseline tau=0.90 vs threshold adaptation vs isotonic on GBR test
# Output: results/gbr_score_recalibration.json
# ==========================================================================

# gbr_score_recalibration.py — Experiment 1.2: Target-domain Score Recalibration
#
#   Goal: Compare three recalibration strategies on GBR:
#     Baseline  — Raw zero-shot (SCS optimal threshold τ=0.9)
#     Method A  — GBR threshold calibration (τ_GBR=0.44 from GBR val)
#     Method B  — GBR isotonic calibration (fit on GBR val)
#
#   Metrics: CSI, Event POD, FAR
#
#   Saves:
#     results/gbr_score_recalibration.json
#     results/gbr_score_recalibration.png
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' gbr_score_recalibration.py

import os, sys, io, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

# IMPORTANT: sklearn MUST be imported before torch (Windows runtime conflict)
from sklearn.isotonic import IsotonicRegression

import numpy as np
import torch
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
THRESHOLDS = np.arange(0.01, 1.00, 0.01)


def restore_a5():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_sliding_temporal(model, ds, device):
    """Sliding-window inference returning forecast_map [n_dates, H, W]."""
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X_full = ds.X_dynamic_full
    bs = 8
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
            del xd, xs, out, preds
    return forecast_map


# build_pfv_arrays is imported from pfv_utils (with year filtering)


def classification_metrics(pred, actual, pred_thresh):
    """Compute CSI, POD, FAR, F1."""
    pred_bin = pred >= pred_thresh
    actual_bin = actual >= SEVERE_THRESHOLD
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


def sweep_best(pred, target, thresholds):
    """Sweep thresholds, return best CSI and corresponding metrics."""
    best_csi, best_f1 = -1, -1
    best_csi_metrics, best_f1_metrics = None, None
    for thr in thresholds:
        m = classification_metrics(pred, target, thr)
        if not np.isnan(m["csi"]) and m["csi"] > best_csi:
            best_csi = m["csi"]
            best_csi_metrics = {"threshold": round(float(thr), 4), **m}
        if not np.isnan(m["f1"]) and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_f1_metrics = {"threshold": round(float(thr), 4), **m}
    return best_csi_metrics, best_f1_metrics


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(f"Experiment 1.2: Target-domain Score Recalibration — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                    val_years=(2011, 2015), test_years=(2016, 2019))
    GBR_SPLIT = dict(year_based_split=True, train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))  # LTTS_CRW GBR re-slice (val has severe events)

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Load model and run inference
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 1] Loading a7 model...")
    a5_model = restore_a5().to(device)

    print("\n[Step 2] Inference on GBR val...")
    ds_gbr_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                                  lead_weeks=LEAD_WEEKS,
                                  dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                                  **GBR_SPLIT)
    T = ds_gbr_val.history_weeks
    forecast_gbr_val = predict_sliding_temporal(a5_model, ds_gbr_val, device)
    gbr_val_pred, gbr_val_target, _ = build_pfv_arrays(
        forecast_gbr_val, ds_gbr_val.y_regression_full, T, ds_gbr_val.n_dates,
        year_range=get_year_range_for_mode(ds_gbr_val), date_strings=ds_gbr_val.date_strings)
    print(f"  GBR val: n_pixels={len(gbr_val_pred)}")

    print("\n[Step 3] Inference on GBR test...")
    ds_gbr_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                                   lead_weeks=LEAD_WEEKS,
                                   dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                                   **GBR_SPLIT)
    forecast_gbr_test = predict_sliding_temporal(a5_model, ds_gbr_test, device)
    gbr_test_pred, gbr_test_target, _ = build_pfv_arrays(
        forecast_gbr_test, ds_gbr_test.y_regression_full, T, ds_gbr_test.n_dates,
        year_range=get_year_range_for_mode(ds_gbr_test), date_strings=ds_gbr_test.date_strings)
    print(f"  GBR test: n_pixels={len(gbr_test_pred)}")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Determine optimal thresholds on GBR val
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 4] Threshold sweep on GBR val...")
    best_csi_val, best_f1_val = sweep_best(gbr_val_pred, gbr_val_target, THRESHOLDS)
    tau_gbr_csi = best_csi_val["threshold"]
    tau_gbr_f1 = best_f1_val["threshold"]
    print(f"  GBR val CSI-optimal: τ={tau_gbr_csi:.3f}, CSI={best_csi_val['csi']:.4f}")
    print(f"  GBR val F1-optimal:  τ={tau_gbr_f1:.3f}, F1={best_f1_val['f1']:.4f}")

    # SCS optimal threshold (from Exp 1.5: SCS_raw best at τ=0.9)
    tau_scs = 0.90

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Fit GBR isotonic regression (Method B)
    # ═══════════════════════════════════════════════════════════════
    MIN_SEVERE_FOR_ISOTONIC = 50
    gbr_val_severe_count = int((gbr_val_target >= SEVERE_THRESHOLD).sum())
    gbr_val_total = int(len(gbr_val_pred))
    gbr_val_severe_rate = gbr_val_severe_count / gbr_val_total if gbr_val_total > 0 else 0
    gbr_iso_reliable = gbr_val_severe_count >= MIN_SEVERE_FOR_ISOTONIC
    print(f"\n[Step 5] Fitting GBR isotonic regression...")
    print(f"  GBR val severe events: {gbr_val_severe_count}/{gbr_val_total} "
          f"({gbr_val_severe_rate:.4%}) — {'RELIABLE' if gbr_iso_reliable else 'UNRELIABLE'} "
          f"(min={MIN_SEVERE_FOR_ISOTONIC})")

    iso_gbr = None
    gbr_test_cal = None
    tau_cal_csi = None
    tau_cal_f1 = None
    if gbr_iso_reliable:
        iso_gbr = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso_gbr.fit(gbr_val_pred, (gbr_val_target >= SEVERE_THRESHOLD).astype(float))

        # Calibrate GBR test predictions
        valid_mask = ~np.isnan(gbr_test_pred)
        gbr_test_cal = np.full_like(gbr_test_pred, np.nan)
        gbr_test_cal[valid_mask] = iso_gbr.predict(gbr_test_pred[valid_mask])

        # Find optimal threshold on calibrated GBR val
        gbr_val_cal = np.full_like(gbr_val_pred, np.nan)
        gbr_val_cal[~np.isnan(gbr_val_pred)] = iso_gbr.predict(gbr_val_pred[~np.isnan(gbr_val_pred)])
        best_csi_cal_val, best_f1_cal_val = sweep_best(gbr_val_cal, gbr_val_target, THRESHOLDS)
        tau_cal_csi = best_csi_cal_val["threshold"]
        tau_cal_f1 = best_f1_cal_val["threshold"]
        print(f"  GBR-Iso val CSI-optimal: τ={tau_cal_csi:.3f}, CSI={best_csi_cal_val['csi']:.4f}")
        print(f"  GBR-Iso val F1-optimal:  τ={tau_cal_f1:.3f}, F1={best_csi_cal_val['f1']:.4f}")
    else:
        print(f"  ⚠ WARNING: GBR val has only {gbr_val_severe_count} severe events — "
              f"isotonic fit is degenerate. Skipping Method B (isotonic calibrated).")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Apply to GBR test — methods
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 6] Applying to GBR test...")

    methods = {
        "Baseline (raw, τ=0.9)": {
            "pred": gbr_test_pred,
            "threshold": tau_scs,
            "description": "Raw zero-shot with SCS optimal threshold",
        },
        "Method A (threshold adapted)": {
            "pred": gbr_test_pred,
            "threshold": tau_gbr_csi,
            "description": f"GBR threshold calibration (τ_GBR={tau_gbr_csi:.3f})",
        },
    }
    if gbr_iso_reliable:
        methods["Method B (isotonic calibrated)"] = {
            "pred": gbr_test_cal,
            "threshold": tau_cal_csi,
            "description": f"GBR isotonic calibration (τ_cal={tau_cal_csi:.3f})",
        }

    results = {}
    for name, cfg in methods.items():
        m = classification_metrics(cfg["pred"], gbr_test_target, cfg["threshold"])
        results[name] = {
            "threshold_used": cfg["threshold"],
            "description": cfg["description"],
            **m,
        }
        print(f"  {name}:")
        print(f"    τ={cfg['threshold']:.3f} → CSI={m['csi']:.4f}, POD={m['pod']:.4f}, FAR={m['far']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Plot comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 7] Plotting...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Experiment 1.2: Target-domain Score Recalibration\n(GBR test set)",
                 fontsize=12, fontweight="bold")

    method_names = list(results.keys())
    all_colors = ["#FF9800", "#2196F3", "#4CAF50"]
    all_short_labels = {
        "Baseline (raw, τ=0.9)": "Baseline\n(raw, τ=0.9)",
        "Method A (threshold adapted)": "Method A\n(threshold adapted)",
        "Method B (isotonic calibrated)": "Method B\n(isotonic calibrated)",
    }
    colors = [all_colors[i] for i in range(len(method_names))]
    short_labels = [all_short_labels[n] for n in method_names]

    # (a) CSI
    ax = axes[0]
    csi_vals = [results[n]["csi"] for n in method_names]
    bars = ax.bar(short_labels, csi_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("CSI", fontsize=11)
    ax.set_title("(a) CSI (Critical Success Index)", fontsize=10)
    ax.set_ylim(0, max(csi_vals) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=9, fontweight="bold")

    # (b) POD
    ax = axes[1]
    pod_vals = [results[n]["pod"] for n in method_names]
    bars = ax.bar(short_labels, pod_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("POD (Probability of Detection)", fontsize=11)
    ax.set_title("(b) Event POD", fontsize=10)
    ax.set_ylim(0, max(pod_vals) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=9, fontweight="bold")

    # (c) FAR
    ax = axes[2]
    far_vals = [results[n]["far"] for n in method_names]
    bars = ax.bar(short_labels, far_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("FAR (False Alarm Ratio)", fontsize=11)
    ax.set_title("(c) FAR", fontsize=10)
    ax.set_ylim(0, max(far_vals) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_path = RESULTS_DIR / "gbr_transfer/gbr_score_recalibration.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Save JSON
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 8] Saving JSON...")

    json_results = {
        "experiment": "target_domain_score_recalibration",
        "description": "Compare baseline vs threshold adaptation vs isotonic calibration on GBR",
        "configuration": {
            "model": "a7 (SCS-trained, frozen)",
            "lead_weeks": LEAD_WEEKS,
            "severe_threshold": SEVERE_THRESHOLD,
            "tau_scs": tau_scs,
            "tau_gbr": tau_gbr_csi,
            "tau_cal": tau_cal_csi,
        },
        "data_stats": {
            "gbr_val": {"n_pixels": int(len(gbr_val_pred)),
                        "severe_count": gbr_val_severe_count,
                        "severe_rate": round(gbr_val_severe_rate, 6),
                        "isotonic_reliable": gbr_iso_reliable},
            "gbr_test": {"n_pixels": int(len(gbr_test_pred))},
        },
        "data_quality_note": {
            "gbr_val_severe_events": gbr_val_severe_count,
            "min_required_for_isotonic": MIN_SEVERE_FOR_ISOTONIC,
            "method_b_included": gbr_iso_reliable,
            "warning": None if gbr_iso_reliable else
                f"GBR val has only {gbr_val_severe_count} severe events "
                f"(<{MIN_SEVERE_FOR_ISOTONIC}), isotonic fit degenerate. Method B excluded.",
        },
        "results": results,
        "comparison_table": {
            "columns": ["Method", "Threshold", "CSI", "POD", "FAR"],
            "rows": [
                [name, f"{r['threshold_used']:.3f}",
                 f"{r['csi']:.4f}", f"{r['pod']:.4f}", f"{r['far']:.4f}"]
                for name, r in results.items()
            ],
        },
        "timestamp": datetime.now().isoformat(),
    }

    json_path = RESULTS_DIR / "gbr_transfer/gbr_score_recalibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY: Experiment 1.2 — Score Recalibration")
    print("=" * 60)
    print(f"\n  {'Method':<35} {'Threshold':>9} {'CSI':>8} {'POD':>8} {'FAR':>8}")
    print(f"  {'-'*35} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")
    for name in method_names:
        r = results[name]
        print(f"  {name:<35} {r['threshold_used']:>9.3f} {r['csi']:>8.4f} {r['pod']:>8.4f} {r['far']:>8.4f}")
    print()
    baseline_csi = results["Baseline (raw, τ=0.9)"]["csi"]
    method_a_csi = results["Method A (threshold adapted)"]["csi"]
    print(f"  Δ CSI (Method A vs Baseline): {method_a_csi - baseline_csi:+.4f}")
    if gbr_iso_reliable:
        method_b_csi = results["Method B (isotonic calibrated)"]["csi"]
        print(f"  Δ CSI (Method B vs Baseline): {method_b_csi - baseline_csi:+.4f}")
    else:
        print(f"  Δ CSI (Method B vs Baseline): N/A (isotonic unreliable, skipped)")
    print("=" * 60)


if __name__ == "__main__":
    main()
