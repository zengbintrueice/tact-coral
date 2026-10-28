# ==========================================================================
# Source: paper/a7_dual_output/gbr_threshold_adaptation.py (verbatim except path fix)
# Experiment: Target-domain threshold adaptation (GBR)
# Paper: Results 2, Fig. 3, Supplementary Fig. 1, Supplementary Table S5
# Purpose: GBR validation argmax-CSI threshold (tau=0.54) evaluated on GBR test
# Output: results/gbr_threshold_adaptation.json
# ==========================================================================

# gbr_threshold_adaptation.py — Experiment 1.5: Threshold Calibration Diagnostic
#
#   Goal: Distinguish two failure hypotheses:
#     H_A ("cannot predict"): y_main has no discriminative power on GBR
#     H_B ("wrong scale"): y_main preserves ranking but absolute values are shifted
#
#   Method:
#     1. Freeze a7, run inference on GBR val + GBR test
#     2. Sweep threshold on GBR val → find optimal operating point (CSI/F1)
#     3. Apply optimal threshold to GBR test → measure generalization
#     4. Compare with:
#        - Raw zero-shot (threshold from SCS)
#        - SCS-Iso calibrated (threshold from SCS)
#        - GBR-Iso oracle (threshold from GBR val)
#     5. ROC + Precision-Recall curves to diagnose ranking quality
#
#   Key diagnostic:
#     - If H_A: ROC ≈ diagonal, PR ≈ baseline, threshold tuning fails
#     - If H_B: ROC/PR show separation, threshold tuning restores detection
#
#   Saves:
#     results/gbr_threshold_adaptation.json
#     results/gbr_threshold_adaptation.png
#     results/experiment1p5_roc_pr_curves.png
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' gbr_threshold_adaptation.py

import os, sys, io, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

# IMPORTANT: sklearn MUST be imported before torch (Windows runtime conflict)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

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
# Fine-grained threshold sweep
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


def ece_calc(pred, actual, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(pred)
    ece = 0.0
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred < bins[i + 1])
        if i == n_bins - 1:
            mask = (pred >= bins[i]) & (pred <= bins[i + 1])
        n = mask.sum()
        if n > 0:
            acc = float(np.mean(actual[mask] >= SEVERE_THRESHOLD))
            conf = float(np.mean(pred[mask]))
            weight = n / total
            ece += weight * abs(acc - conf)
    return float(ece)


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(f"Experiment 1.5: Threshold Calibration Diagnostic — {datetime.now():%Y-%m-%d %H:%M:%S}")
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

    print("\n[Step 2] Inference on SCS val (reference)...")
    ds_scs_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                                  lead_weeks=LEAD_WEEKS, **split_kw)
    T = ds_scs_val.history_weeks
    forecast_scs_val = predict_sliding_temporal(a5_model, ds_scs_val, device)
    scs_val_pred, scs_val_target, _ = build_pfv_arrays(
        forecast_scs_val, ds_scs_val.y_regression_full, T, ds_scs_val.n_dates,
        year_range=get_year_range_for_mode(ds_scs_val),
        date_strings=ds_scs_val.date_strings)
    print(f"  SCS val: n_pixels={len(scs_val_pred)}")

    print("\n[Step 3] Inference on GBR val...")
    ds_gbr_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                                  lead_weeks=LEAD_WEEKS,
                                  dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                                  **GBR_SPLIT)
    forecast_gbr_val = predict_sliding_temporal(a5_model, ds_gbr_val, device)
    gbr_val_pred, gbr_val_target, _ = build_pfv_arrays(
        forecast_gbr_val, ds_gbr_val.y_regression_full, T, ds_gbr_val.n_dates,
        year_range=get_year_range_for_mode(ds_gbr_val),
        date_strings=ds_gbr_val.date_strings)
    print(f"  GBR val: n_pixels={len(gbr_val_pred)}")

    print("\n[Step 4] Inference on GBR test...")
    ds_gbr_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                                   lead_weeks=LEAD_WEEKS,
                                   dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                                   **GBR_SPLIT)
    forecast_gbr_test = predict_sliding_temporal(a5_model, ds_gbr_test, device)
    gbr_test_pred, gbr_test_target, _ = build_pfv_arrays(
        forecast_gbr_test, ds_gbr_test.y_regression_full, T, ds_gbr_test.n_dates,
        year_range=get_year_range_for_mode(ds_gbr_test),
        date_strings=ds_gbr_test.date_strings)
    print(f"  GBR test: n_pixels={len(gbr_test_pred)}")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Fit isotonic regressions (for calibrated variants)
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 5] Fitting isotonic regressions...")

    # Data quality gate: GBR val must have enough severe events for reliable isotonic fit
    gbr_val_severe_count = int((gbr_val_target >= SEVERE_THRESHOLD).sum())
    gbr_val_total = int(len(gbr_val_pred))
    gbr_val_severe_rate = gbr_val_severe_count / gbr_val_total if gbr_val_total > 0 else 0
    MIN_SEVERE_FOR_ISOTONIC = 50
    gbr_iso_reliable = gbr_val_severe_count >= MIN_SEVERE_FOR_ISOTONIC
    print(f"  GBR val severe events: {gbr_val_severe_count}/{gbr_val_total} "
          f"({gbr_val_severe_rate:.4%}) — {'RELIABLE' if gbr_iso_reliable else 'UNRELIABLE'} "
          f"(min={MIN_SEVERE_FOR_ISOTONIC})")

    # SCS-Iso: fit on SCS val
    iso_scs = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_scs.fit(scs_val_pred, (scs_val_target >= SEVERE_THRESHOLD).astype(float))

    # GBR-Iso: fit on GBR val (oracle) — ONLY if enough severe events
    iso_gbr = None
    if gbr_iso_reliable:
        iso_gbr = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso_gbr.fit(gbr_val_pred, (gbr_val_target >= SEVERE_THRESHOLD).astype(float))
        print("  GBR isotonic fitted (oracle).")
    else:
        print(f"  ⚠ WARNING: GBR val has only {gbr_val_severe_count} severe events — "
              f"isotonic fit is degenerate (maps all values to ~0). "
              f"Skipping GBR-isotonic oracle variant.")

    # Calibrate predictions
    def calibrate(pred, iso):
        valid_mask = ~np.isnan(pred)
        calibrated = np.full_like(pred, np.nan)
        calibrated[valid_mask] = iso.predict(pred[valid_mask])
        return calibrated

    scs_val_cal = calibrate(scs_val_pred, iso_scs)
    gbr_val_cal_scs = calibrate(gbr_val_pred, iso_scs)
    gbr_val_cal_gbr = calibrate(gbr_val_pred, iso_gbr) if iso_gbr is not None else None
    gbr_test_cal_scs = calibrate(gbr_test_pred, iso_scs)
    gbr_test_cal_gbr = calibrate(gbr_test_pred, iso_gbr) if iso_gbr is not None else None

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Threshold sweep on GBR val → find optimal
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 6] Threshold sweep on GBR val...")

    variants = {
        "GBR_raw": gbr_val_pred,
        "GBR_scs_iso": gbr_val_cal_scs,
    }
    if gbr_iso_reliable:
        variants["GBR_gbr_iso"] = gbr_val_cal_gbr

    # Also sweep SCS val for reference
    scs_variants = {
        "SCS_raw": scs_val_pred,
        "SCS_scs_iso": scs_val_cal,
    }

    def sweep_thresholds(pred, target, thresholds):
        """Sweep thresholds, return best CSI and F1."""
        results = []
        for thr in thresholds:
            m = classification_metrics(pred, target, thr)
            results.append({"threshold": round(float(thr), 4), **m})
        best_csi = max(results, key=lambda r: r["csi"] if not np.isnan(r["csi"]) else -1)
        best_f1 = max(results, key=lambda r: r["f1"] if not np.isnan(r["f1"]) else -1)
        return results, best_csi, best_f1

    # Sweep on GBR val
    gbr_val_sweep = {}
    for name, pred in variants.items():
        results, best_csi, best_f1 = sweep_thresholds(pred, gbr_val_target, THRESHOLDS)
        gbr_val_sweep[name] = {
            "sweep": results,
            "best_csi": best_csi,
            "best_f1": best_f1,
        }
        print(f"  {name}: best_CSI={best_csi['csi']:.4f} @ thr={best_csi['threshold']:.3f}, "
              f"best_F1={best_f1['f1']:.4f} @ thr={best_f1['threshold']:.3f}")

    # Sweep on SCS val (reference)
    scs_val_sweep = {}
    for name, pred in scs_variants.items():
        results, best_csi, best_f1 = sweep_thresholds(pred, scs_val_target, THRESHOLDS)
        scs_val_sweep[name] = {
            "sweep": results,
            "best_csi": best_csi,
            "best_f1": best_f1,
        }
        print(f"  {name}: best_CSI={best_csi['csi']:.4f} @ thr={best_csi['threshold']:.3f}, "
              f"best_F1={best_f1['f1']:.4f} @ thr={best_f1['threshold']:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Apply optimal thresholds to GBR test
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 7] Applying optimal thresholds to GBR test...")

    test_variants = {
        "GBR_raw": gbr_test_pred,
        "GBR_scs_iso": gbr_test_cal_scs,
    }
    if gbr_iso_reliable:
        test_variants["GBR_gbr_iso"] = gbr_test_cal_gbr

    test_results = {}
    for variant_name, pred_test in test_variants.items():
        # Get optimal thresholds from GBR val sweep
        best_csi_thr = gbr_val_sweep[variant_name]["best_csi"]["threshold"]
        best_f1_thr = gbr_val_sweep[variant_name]["best_f1"]["threshold"]

        # Apply to test
        m_csi = classification_metrics(pred_test, gbr_test_target, best_csi_thr)
        m_f1 = classification_metrics(pred_test, gbr_test_target, best_f1_thr)

        test_results[variant_name] = {
            "optimal_csi_threshold": best_csi_thr,
            "test_at_csi_threshold": m_csi,
            "optimal_f1_threshold": best_f1_thr,
            "test_at_f1_threshold": m_f1,
        }
        print(f"  {variant_name}:")
        print(f"    CSI-optimal thr={best_csi_thr:.3f} → test CSI={m_csi['csi']:.4f}, F1={m_csi['f1']:.4f}")
        print(f"    F1-optimal  thr={best_f1_thr:.3f} → test CSI={m_f1['csi']:.4f}, F1={m_f1['f1']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: ROC + PR curves (diagnostic for H_A vs H_B)
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 8] Computing ROC/PR curves...")

    def compute_roc_pr(pred, target):
        """Compute ROC AUC and Average Precision."""
        actual_bin = (target >= SEVERE_THRESHOLD).astype(int)
        valid = ~np.isnan(pred)
        pred_valid = pred[valid]
        actual_valid = actual_bin[valid]

        try:
            roc_auc = float(roc_auc_score(actual_valid, pred_valid))
        except Exception:
            roc_auc = float("nan")
        try:
            avg_prec = float(average_precision_score(actual_valid, pred_valid))
        except Exception:
            avg_prec = float("nan")

        # Compute ROC curve points
        fpr_list, tpr_list = [], []
        for thr in THRESHOLDS:
            pred_bin = (pred_valid >= thr).astype(int)
            tp = np.sum((pred_bin == 1) & (actual_valid == 1))
            fp = np.sum((pred_bin == 1) & (actual_valid == 0))
            tn = np.sum((pred_bin == 0) & (actual_valid == 0))
            fn = np.sum((pred_bin == 0) & (actual_valid == 1))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        # Compute PR curve points
        prec_list, rec_list = [], []
        for thr in THRESHOLDS:
            pred_bin = (pred_valid >= thr).astype(int)
            tp = np.sum((pred_bin == 1) & (actual_valid == 1))
            fp = np.sum((pred_bin == 1) & (actual_valid == 0))
            fn = np.sum((pred_bin == 0) & (actual_valid == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec_list.append(prec)
            rec_list.append(rec)

        return {
            "roc_auc": roc_auc,
            "avg_precision": avg_prec,
            "baseline": float(actual_valid.mean()),
            "roc_curve": {"fpr": fpr_list, "tpr": tpr_list},
            "pr_curve": {"precision": prec_list, "recall": rec_list},
        }

    # Compute for GBR test
    roc_pr_results = {}
    for name, pred in test_variants.items():
        roc_pr = compute_roc_pr(pred, gbr_test_target)
        roc_pr_results[name] = roc_pr
        print(f"  {name}: ROC_AUC={roc_pr['roc_auc']:.4f}, "
              f"AP={roc_pr['avg_precision']:.4f}, "
              f"baseline={roc_pr['baseline']:.4f}")

    # Also compute for SCS val (reference)
    scs_roc_pr = compute_roc_pr(scs_val_pred, scs_val_target)
    print(f"  SCS_val_raw: ROC_AUC={scs_roc_pr['roc_auc']:.4f}, "
          f"AP={scs_roc_pr['avg_precision']:.4f}, "
          f"baseline={scs_roc_pr['baseline']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Plot
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 9] Plotting...")

    # Figure 1: Threshold sweep + test performance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Experiment 1.5: Threshold Calibration Diagnostic\n"
                 f"(GBR val → optimal threshold → GBR test)",
                 fontsize=12, fontweight="bold")

    colors = {"GBR_raw": "#FF9800", "GBR_scs_iso": "#2196F3", "GBR_gbr_iso": "#4CAF50"}

    # (a) CSI vs threshold on GBR val
    ax = axes[0, 0]
    for name, sweep_data in gbr_val_sweep.items():
        thr_vals = [r["threshold"] for r in sweep_data["sweep"]]
        csi_vals = [r["csi"] for r in sweep_data["sweep"]]
        ax.plot(thr_vals, csi_vals, color=colors[name], linewidth=1.5, label=name)
        # Mark best
        best = sweep_data["best_csi"]
        ax.scatter([best["threshold"]], [best["csi"]], s=100, c=colors[name], marker="*", zorder=5)
    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("CSI", fontsize=10)
    ax.set_title("(a) CSI vs Threshold (GBR val)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) F1 vs threshold on GBR val
    ax = axes[0, 1]
    for name, sweep_data in gbr_val_sweep.items():
        thr_vals = [r["threshold"] for r in sweep_data["sweep"]]
        f1_vals = [r["f1"] for r in sweep_data["sweep"]]
        ax.plot(thr_vals, f1_vals, color=colors[name], linewidth=1.5, label=name)
        best = sweep_data["best_f1"]
        ax.scatter([best["threshold"]], [best["f1"]], s=100, c=colors[name], marker="*", zorder=5)
    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("F1", fontsize=10)
    ax.set_title("(b) F1 vs Threshold (GBR val)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Test performance comparison (bar chart)
    ax = axes[1, 0]
    config_names = list(test_results.keys())
    x = np.arange(len(config_names))
    width = 0.35
    csi_vals = [test_results[n]["test_at_csi_threshold"]["csi"] for n in config_names]
    f1_vals = [test_results[n]["test_at_f1_threshold"]["f1"] for n in config_names]
    bars1 = ax.bar(x - width/2, csi_vals, width, label="CSI (at CSI-optimal thr)", color="#FF9800")
    bars2 = ax.bar(x + width/2, f1_vals, width, label="F1 (at F1-optimal thr)", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("(c) Test Performance (threshold from GBR val)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    # (d) Threshold transfer visualization
    ax = axes[1, 1]
    # Show optimal thresholds for each variant
    variant_names = list(gbr_val_sweep.keys())
    csi_thrs = [gbr_val_sweep[n]["best_csi"]["threshold"] for n in variant_names]
    f1_thrs = [gbr_val_sweep[n]["best_f1"]["threshold"] for n in variant_names]
    x = np.arange(len(variant_names))
    bars1 = ax.bar(x - width/2, csi_thrs, width, label="CSI-optimal threshold", color="#FF9800")
    bars2 = ax.bar(x + width/2, f1_thrs, width, label="F1-optimal threshold", color="#2196F3")
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='SCS default (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Optimal Threshold", fontsize=10)
    ax.set_title("(d) Optimal Thresholds (from GBR val)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig_path1 = RESULTS_DIR / "gbr_transfer/gbr_threshold_adaptation.png"
    fig.savefig(fig_path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path1}")

    # Figure 2: ROC and PR curves
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Experiment 1.5: ROC & PR Curves (GBR test)\n"
                  "Diagnostic: H_A (no discriminative power) vs H_B (wrong scale)",
                  fontsize=11, fontweight="bold")

    # (a) ROC curves
    ax = axes2[0]
    for name, roc_pr in roc_pr_results.items():
        fpr = roc_pr["roc_curve"]["fpr"]
        tpr = roc_pr["roc_curve"]["tpr"]
        ax.plot(fpr, tpr, color=colors[name], linewidth=1.5,
                label=f"{name} (AUC={roc_pr['roc_auc']:.3f})")
    # SCS reference
    ax.plot(scs_roc_pr["roc_curve"]["fpr"], scs_roc_pr["roc_curve"]["tpr"],
            color="gray", linewidth=1.5, linestyle="--",
            label=f"SCS val ref (AUC={scs_roc_pr['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("(a) ROC Curves", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # (b) PR curves
    ax = axes2[1]
    for name, roc_pr in roc_pr_results.items():
        rec = roc_pr["pr_curve"]["recall"]
        prec = roc_pr["pr_curve"]["precision"]
        ax.plot(rec, prec, color=colors[name], linewidth=1.5,
                label=f"{name} (AP={roc_pr['avg_precision']:.3f})")
    # SCS reference
    ax.plot(scs_roc_pr["pr_curve"]["recall"], scs_roc_pr["pr_curve"]["precision"],
            color="gray", linewidth=1.5, linestyle="--",
            label=f"SCS val ref (AP={scs_roc_pr['avg_precision']:.3f})")
    # Baseline (random classifier)
    baseline = scs_roc_pr["baseline"]
    ax.axhline(y=baseline, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label=f"Baseline (prev={baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("(b) Precision-Recall Curves", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig_path2 = RESULTS_DIR / "gbr_transfer/gbr_threshold_adaptation_roc_pr.png"
    fig2.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fig_path2}")

    # ═══════════════════════════════════════════════════════════════
    # Step 7: Save JSON results
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 10] Saving JSON results...")

    json_results = {
        "experiment": "threshold_calibration_diagnostic",
        "description": "Distinguish H_A (cannot predict) vs H_B (wrong scale) on GBR",
        "configuration": {
            "model": "a7 (SCS-trained, frozen)",
            "lead_weeks": LEAD_WEEKS,
            "severe_threshold": SEVERE_THRESHOLD,
            "threshold_sweep": [float(THRESHOLDS[0]), float(THRESHOLDS[-1]),
                                float(THRESHOLDS[1] - THRESHOLDS[0])],
            "n_thresholds": len(THRESHOLDS),
        },
        "data_stats": {
            "scs_val": {"n_pixels": int(len(scs_val_pred))},
            "gbr_val": {"n_pixels": int(len(gbr_val_pred)),
                        "severe_count": gbr_val_severe_count,
                        "severe_rate": round(gbr_val_severe_rate, 6),
                        "isotonic_reliable": gbr_iso_reliable},
            "gbr_test": {"n_pixels": int(len(gbr_test_pred))},
        },
        "data_quality_note": {
            "gbr_val_severe_events": gbr_val_severe_count,
            "min_required_for_isotonic": MIN_SEVERE_FOR_ISOTONIC,
            "gbr_isotonic_included": gbr_iso_reliable,
            "warning": None if gbr_iso_reliable else
                f"GBR val has only {gbr_val_severe_count} severe events "
                f"(<{MIN_SEVERE_FOR_ISOTONIC}), isotonic fit is degenerate. "
                f"GBR-Iso oracle variant excluded.",
        },
        "gbr_val_sweep": {
            name: {
                "best_csi_threshold": d["best_csi"]["threshold"],
                "best_csi_value": d["best_csi"]["csi"],
                "best_f1_threshold": d["best_f1"]["threshold"],
                "best_f1_value": d["best_f1"]["f1"],
            }
            for name, d in gbr_val_sweep.items()
        },
        "scs_val_sweep": {
            name: {
                "best_csi_threshold": d["best_csi"]["threshold"],
                "best_csi_value": d["best_csi"]["csi"],
                "best_f1_threshold": d["best_f1"]["threshold"],
                "best_f1_value": d["best_f1"]["f1"],
            }
            for name, d in scs_val_sweep.items()
        },
        "test_results": {
            name: {
                "optimal_csi_threshold": d["optimal_csi_threshold"],
                "test_at_csi_threshold": d["test_at_csi_threshold"],
                "optimal_f1_threshold": d["optimal_f1_threshold"],
                "test_at_f1_threshold": d["test_at_f1_threshold"],
            }
            for name, d in test_results.items()
        },
        "roc_pr_diagnostic": {
            name: {
                "roc_auc": d["roc_auc"],
                "avg_precision": d["avg_precision"],
                "baseline_prevalence": d["baseline"],
            }
            for name, d in roc_pr_results.items()
        },
        "scs_reference": {
            "roc_auc": scs_roc_pr["roc_auc"],
            "avg_precision": scs_roc_pr["avg_precision"],
            "baseline_prevalence": scs_roc_pr["baseline"],
        },
        "diagnostic_interpretation": {
            "H_A_no_discriminative_power": "If ROC_AUC ≈ 0.5 and AP ≈ baseline, y_main has no ranking ability on GBR",
            "H_B_wrong_scale": "If ROC_AUC >> 0.5 and AP >> baseline, ranking is preserved but absolute values shifted",
            "threshold_tuning_benefit": "Compare test CSI/F1 with GBR-val-tuned threshold vs SCS default threshold",
        },
        "timestamp": datetime.now().isoformat(),
    }

    json_path = RESULTS_DIR / "gbr_transfer/gbr_threshold_adaptation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY: H_A vs H_B Diagnostic")
    print("=" * 60)
    print(f"\n  GBR test ROC AUC:")
    for name, roc_pr in roc_pr_results.items():
        print(f"    {name}: {roc_pr['roc_auc']:.4f}")
    print(f"    SCS val ref: {scs_roc_pr['roc_auc']:.4f}")
    print(f"\n  GBR test Average Precision:")
    for name, roc_pr in roc_pr_results.items():
        print(f"    {name}: {roc_pr['avg_precision']:.4f}")
    print(f"    SCS val ref: {scs_roc_pr['avg_precision']:.4f}")
    print(f"\n  Optimal thresholds (from GBR val):")
    for name in gbr_val_sweep:
        csi_thr = gbr_val_sweep[name]["best_csi"]["threshold"]
        f1_thr = gbr_val_sweep[name]["best_f1"]["threshold"]
        print(f"    {name}: CSI-thr={csi_thr:.3f}, F1-thr={f1_thr:.3f}")
    print(f"\n  Test performance (with GBR-val-tuned threshold):")
    for name, d in test_results.items():
        csi = d["test_at_csi_threshold"]["csi"]
        f1 = d["test_at_f1_threshold"]["f1"]
        print(f"    {name}: CSI={csi:.4f}, F1={f1:.4f}")
    print(f"\n  Diagnosis:")
    mean_roc = np.mean([roc_pr_results[n]["roc_auc"] for n in roc_pr_results])
    if mean_roc > 0.7:
        print(f"    → H_B supported: mean ROC AUC = {mean_roc:.4f} >> 0.5")
        print(f"      Ranking is preserved; threshold tuning can restore detection.")
    elif mean_roc > 0.55:
        print(f"    → Partial H_B: mean ROC AUC = {mean_roc:.4f} (weak ranking)")
        print(f"      Some discriminative power but domain shift is significant.")
    else:
        print(f"    → H_A supported: mean ROC AUC = {mean_roc:.4f} ≈ 0.5")
        print(f"      y_main has no discriminative power on GBR.")
    print("=" * 60)


if __name__ == "__main__":
    main()