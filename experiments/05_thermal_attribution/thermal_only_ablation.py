# ==========================================================================
# Source: paper/a7_dual_output/thermal_only_ablation.py (verbatim except path fix)
# Experiment: Thermal-only inference ablation (GBR)
# Paper: Results 3/4, Fig. 4a, Fig. 5
# Purpose: retain thermal channels only; ranking vs score-magnitude consequences
# Output: results/thermal_only_ablation.json
# ==========================================================================

# thermal_only_ablation.py — Thermal Dependency + Static Ablation + Permutation Importance
#
#   Three sub-experiments:
#     7.1 Thermal core dependency: Full vs Thermal-only
#     7.2 Static spatial prior ablation: Full vs No-static
#     7.3 Feature permutation importance: shuffle each channel, measure AUC drop
#
#   Metrics: AUC, CSI, Event POD (GBR test, threshold=0.54)
#
#   Saves:
#     results/thermal_only_ablation.json
#     results/thermal_only_ablation.png
#     results/experiment7_1_permutation_importance.png

import os, sys, io, json, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

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
GBR_THRESHOLD = 0.54  # val-selected GBR operating point (experiment1p2)

# Channel indices (from dataset output):
# Dynamic (13): sst(0), ssta(1), hotspot(2), chlorophyll(3), par(4),
#               seasonal_sin(5), wind_speed(6), ssr(7), sst_anomaly(8),
#               d_hotspot_dt(9), hotspot_cumsum_4w(10), hotspot_cumsum_12w(11), seasonal_cos(12)
# Static (5): bathymetry(0), dist_to_shore(1), cyclone_frequency(2), salinity(3), ocean_mask(4)

THERMAL_DYN_INDICES = [0, 1, 2, 8, 9, 10, 11]  # sst, ssta, hotspot, sst_anomaly, d_hs, cumsum_4w, cumsum_12w
NON_THERMAL_DYN_INDICES = [3, 4, 5, 6, 7, 12]  # chlorophyll, par, seasonal_sin, wind, ssr, seasonal_cos
# ocean_mask (idx=4) is ESSENTIAL for land/ocean separation — always keep it
# Static spatial priors = bathymetry(0), dist_to_shore(1), cyclone_frequency(2), salinity(3)
STATIC_SPATIAL_PRIOR_INDICES = [0, 1, 2, 3]  # spatial priors (test for domain bias)
STATIC_ALL_INDICES = [0, 1, 2, 3, 4]  # all static (including ocean_mask)
OCEAN_MASK_IDX = 4  # must always keep

DYN_CHANNEL_NAMES = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par',
                     'seasonal_sin', 'wind_speed', 'ssr', 'sst_anomaly',
                     'd_hotspot_dt', 'hotspot_cumsum_4w', 'hotspot_cumsum_12w', 'seasonal_cos']
STATIC_CHANNEL_NAMES = ['bathymetry', 'dist_to_shore', 'cyclone_frequency', 'salinity', 'ocean_mask']


def restore_a5():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_with_ablation(model, ds, device, dyn_zero_indices=None, stat_zero_indices=None,
                           dyn_shuffle_idx=None):
    """Sliding-window inference with channel ablation.

    Args:
        dyn_zero_indices: list of dynamic channel indices to zero out
        stat_zero_indices: list of static channel indices to zero out
        dyn_shuffle_idx: single dynamic channel index to shuffle (permutation importance)
    """
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X_full = ds.X_dynamic_full  # [n_dates, H, W, C_dyn]
    bs = 8
    n_valid = n_dates - T + 1
    t_start_valid = T - 1

    # Prepare shuffled version if needed
    if dyn_shuffle_idx is not None:
        # Shuffle the channel across ALL dates (break temporal/spatial correlation)
        flat = X_full[:, :, :, dyn_shuffle_idx].flatten()
        np.random.shuffle(flat)
        X_full_shuffled = X_full.copy()
        X_full_shuffled[:, :, :, dyn_shuffle_idx] = flat.reshape(n_dates, H, W)
    else:
        X_full_shuffled = X_full

    with torch.no_grad():
        for start_batch in range(0, n_valid, bs):
            batch_end = min(start_batch + bs, n_valid)
            bs_actual = batch_end - start_batch
            xd_batch = np.zeros((bs_actual, T, H, W, C_dyn), dtype=np.float32)
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                start_idx = t_end - T + 1
                xd_batch[k] = X_full_shuffled[start_idx:t_end + 1]

            # Apply zeroing ablation
            if dyn_zero_indices is not None:
                for idx in dyn_zero_indices:
                    xd_batch[:, :, :, :, idx] = 0.0

            xd = torch.FloatTensor(xd_batch).to(device)

            # Static features
            xs_raw = ds.X_static[0]  # [H, W, C_stat]
            xs = xs_raw.copy()
            if stat_zero_indices is not None:
                for idx in stat_zero_indices:
                    xs[:, :, idx] = 0.0
            xs_tiled = np.tile(xs, (bs_actual, 1, 1, 1))
            xs_t = torch.FloatTensor(xs_tiled).to(device)

            out = model(xd, xs_t, lm, scale=A5_CFG["scale"],
                        tau=A5_CFG["tau"], temp=A5_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
            del xd, xs_t, out, preds
    return forecast_map


# build_pfv_arrays is imported from pfv_utils (with year filtering)


def compute_auc(pred, target):
    actual_bin = (target >= SEVERE_THRESHOLD).astype(int)
    valid = ~np.isnan(pred)
    try:
        return float(roc_auc_score(actual_bin[valid], pred[valid]))
    except Exception:
        return float("nan")


def compute_csi(pred, target, threshold):
    pred_bin = pred >= threshold
    actual_bin = target >= SEVERE_THRESHOLD
    hits = np.sum(pred_bin & actual_bin)
    misses = np.sum(~pred_bin & actual_bin)
    false_alarms = np.sum(pred_bin & ~actual_bin)
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else float("nan")


def compute_pod(pred, target, threshold):
    pred_bin = pred >= threshold
    actual_bin = target >= SEVERE_THRESHOLD
    hits = np.sum(pred_bin & actual_bin)
    misses = np.sum(~pred_bin & actual_bin)
    return float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print(f"Experiment 7.1: Ablation Study — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                    val_years=(2011, 2015), test_years=(2016, 2019))

    # ═══════════════════════════════════════════════════════════════
    # Load model and data
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 1] Loading a7 model...")
    a5_model = restore_a5().to(device)

    print("\n[Step 2] Loading GBR test dataset...")
    ds_gbr_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                                   lead_weeks=LEAD_WEEKS,
                                   dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                                   **split_kw)
    T = ds_gbr_test.history_weeks
    print(f"  GBR test: n_dates={ds_gbr_test.n_dates}, grid=({ds_gbr_test.n_lat},{ds_gbr_test.n_lon})")

    # ═══════════════════════════════════════════════════════════════
    # Experiment 7.1 & 7.2: Ablation (thermal-only, no-static)
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 3] Running ablation experiments...")

    configs = {
        "Full (all channels)": {
            "dyn_zero": None,
            "stat_zero": None,
        },
        "Thermal-only": {
            "dyn_zero": NON_THERMAL_DYN_INDICES,  # zero out non-thermal dynamic
            "stat_zero": STATIC_SPATIAL_PRIOR_INDICES,  # remove spatial priors (keep ocean_mask)
        },
        "No-static-priors": {
            "dyn_zero": None,
            "stat_zero": STATIC_SPATIAL_PRIOR_INDICES,  # remove spatial priors only (keep ocean_mask)
        },
        "No-static (ocean_mask only)": {
            "dyn_zero": None,
            "stat_zero": STATIC_ALL_INDICES,  # remove ALL static (including ocean_mask)
        },
    }

    ablation_results = {}
    for name, cfg in configs.items():
        print(f"  Running: {name}...")
        forecast = predict_with_ablation(
            a5_model, ds_gbr_test, device,
            dyn_zero_indices=cfg["dyn_zero"],
            stat_zero_indices=cfg["stat_zero"],
        )
        pred, target, _ = build_pfv_arrays(forecast, ds_gbr_test.y_regression_full, T, ds_gbr_test.n_dates,
            year_range=get_year_range_for_mode(ds_gbr_test), date_strings=ds_gbr_test.date_strings)
        auc = compute_auc(pred, target)
        csi = compute_csi(pred, target, GBR_THRESHOLD)
        pod = compute_pod(pred, target, GBR_THRESHOLD)
        ablation_results[name] = {
            "auc": auc, "csi": csi, "pod": pod,
            "n_pixels": int(len(pred)),
        }
        print(f"    AUC={auc:.4f}, CSI={csi:.4f}, POD={pod:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Experiment 7.3: Permutation importance
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 4] Permutation importance (shuffling each channel)...")

    # First get baseline (no shuffling)
    print("  Baseline (no shuffle)...")
    forecast_baseline = predict_with_ablation(a5_model, ds_gbr_test, device)
    pred_baseline, target_baseline, _ = build_pfv_arrays(
        forecast_baseline, ds_gbr_test.y_regression_full, T, ds_gbr_test.n_dates,
        year_range=get_year_range_for_mode(ds_gbr_test), date_strings=ds_gbr_test.date_strings)
    baseline_auc = compute_auc(pred_baseline, target_baseline)
    print(f"  Baseline AUC: {baseline_auc:.4f}")

    # Permute each dynamic channel
    perm_importance = {}
    np.random.seed(42)  # reproducible

    for idx in range(len(DYN_CHANNEL_NAMES)):
        ch_name = DYN_CHANNEL_NAMES[idx]
        print(f"  Shuffling dynamic[{idx}]: {ch_name}...")
        forecast_perm = predict_with_ablation(
            a5_model, ds_gbr_test, device, dyn_shuffle_idx=idx)
        pred_perm, _, _ = build_pfv_arrays(
            forecast_perm, ds_gbr_test.y_regression_full, T, ds_gbr_test.n_dates,
            year_range=get_year_range_for_mode(ds_gbr_test), date_strings=ds_gbr_test.date_strings)
        perm_auc = compute_auc(pred_perm, target_baseline)
        importance = baseline_auc - perm_auc
        perm_importance[f"dyn_{ch_name}"] = {
            "channel_index": idx,
            "channel_type": "dynamic",
            "auc_after_shuffle": perm_auc,
            "auc_drop": importance,
        }
        print(f"    AUC={perm_auc:.4f}, drop={importance:+.4f}")

    # Permute each static channel
    for idx in range(len(STATIC_CHANNEL_NAMES)):
        ch_name = STATIC_CHANNEL_NAMES[idx]
        print(f"  Shuffling static[{idx}]: {ch_name}...")
        # For static, we shuffle the 2D spatial map
        forecast_perm = predict_with_ablation_static_shuffle(
            a5_model, ds_gbr_test, device, stat_shuffle_idx=idx)
        pred_perm, _, _ = build_pfv_arrays(
            forecast_perm, ds_gbr_test.y_regression_full, T, ds_gbr_test.n_dates,
            year_range=get_year_range_for_mode(ds_gbr_test), date_strings=ds_gbr_test.date_strings)
        perm_auc = compute_auc(pred_perm, target_baseline)
        importance = baseline_auc - perm_auc
        perm_importance[f"stat_{ch_name}"] = {
            "channel_index": idx,
            "channel_type": "static",
            "auc_after_shuffle": perm_auc,
            "auc_drop": importance,
        }
        print(f"    AUC={perm_auc:.4f}, drop={importance:+.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Plot
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 5] Plotting...")

    # Figure 1: Ablation comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Experiment 7.1/7.2: Ablation Study (GBR test, τ=0.54)",
                 fontsize=12, fontweight="bold")

    config_names = list(ablation_results.keys())
    n_configs = len(config_names)
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"][:n_configs]
    short_labels = ["Full", "Thermal-\nonly", "No-spatial\npriors", "No-static\n(mask only)"]

    # (a) AUC
    ax = axes[0]
    vals = [ablation_results[n]["auc"] for n in config_names]
    bars = ax.bar(range(n_configs), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("(a) AUC", fontsize=10)
    ax.set_ylim(min(vals) * 0.95, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", fontsize=8, fontweight="bold")

    # (b) CSI
    ax = axes[1]
    vals = [ablation_results[n]["csi"] for n in config_names]
    bars = ax.bar(range(n_configs), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("CSI", fontsize=11)
    ax.set_title("(b) CSI (τ=0.54)", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=8, fontweight="bold")

    # (c) POD
    ax = axes[2]
    vals = [ablation_results[n]["pod"] for n in config_names]
    bars = ax.bar(range(n_configs), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Event POD", fontsize=11)
    ax.set_title("(c) Event POD (τ=0.54)", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig_path1 = RESULTS_DIR / "thermal_attribution/thermal_only_ablation.png"
    fig.savefig(fig_path1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path1}")

    # Figure 2: Permutation importance ranking
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Sort by importance
    sorted_items = sorted(perm_importance.items(), key=lambda x: x[1]["auc_drop"], reverse=True)
    names = [k for k, _ in sorted_items]
    drops = [v["auc_drop"] for _, v in sorted_items]
    colors_bar = ["#FF5722" if "dyn_" in k else "#607D8B" for k in names]

    # Clean names for display
    display_names = [n.replace("dyn_", "").replace("stat_", "[static] ") for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, drops, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("AUC Drop (baseline - shuffled)", fontsize=11)
    ax.set_title("Experiment 7.3: Feature Permutation Importance (GBR test)\n"
                 "Higher = more important for ranking severe events",
                 fontsize=11, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, drop) in enumerate(zip(bars, drops)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{drop:.4f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#FF5722", label="Dynamic"),
                       Patch(facecolor="#607D8B", label="Static")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig_path2 = RESULTS_DIR / "thermal_attribution/thermal_only_permutation_importance.png"
    fig2.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fig_path2}")

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Save JSON
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 6] Saving JSON...")

    json_results = {
        "experiment": "ablation_study",
        "description": "Thermal dependency + Static ablation + Permutation importance",
        "configuration": {
            "model": "a7 (SCS-trained, frozen)",
            "lead_weeks": LEAD_WEEKS,
            "severe_threshold": SEVERE_THRESHOLD,
            "gbr_threshold": GBR_THRESHOLD,
            "thermal_dynamic_indices": THERMAL_DYN_INDICES,
            "thermal_dynamic_names": [DYN_CHANNEL_NAMES[i] for i in THERMAL_DYN_INDICES],
            "non_thermal_dynamic_indices": NON_THERMAL_DYN_INDICES,
            "non_thermal_dynamic_names": [DYN_CHANNEL_NAMES[i] for i in NON_THERMAL_DYN_INDICES],
        },
        "ablation_results": ablation_results,
        "permutation_importance": {
            "baseline_auc": baseline_auc,
            "ranking": [
                {"rank": i+1, "channel": name, **data}
                for i, (name, data) in enumerate(sorted_items)
            ],
        },
        "timestamp": datetime.now().isoformat(),
    }

    json_path = RESULTS_DIR / "thermal_attribution/thermal_only_ablation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: Ablation Study")
    print("=" * 70)
    print(f"\n  {'Config':<25} {'AUC':>8} {'CSI':>8} {'POD':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for name in config_names:
        r = ablation_results[name]
        print(f"  {name:<25} {r['auc']:>8.4f} {r['csi']:>8.4f} {r['pod']:>8.4f}")

    full_auc = ablation_results["Full (all channels)"]["auc"]
    thermal_auc = ablation_results["Thermal-only"]["auc"]
    full_csi = ablation_results["Full (all channels)"]["csi"]
    thermal_csi = ablation_results["Thermal-only"]["csi"]
    spatial_auc = ablation_results["No-static-priors"]["auc"]
    nostatic_auc = ablation_results["No-static (ocean_mask only)"]["auc"]

    print(f"\n  Thermal-only vs Full: ΔAUC = {thermal_auc - full_auc:+.4f}, "
          f"ΔCSI = {thermal_csi - full_csi:+.4f}")
    print(f"  No-spatial-priors vs Full: ΔAUC = {spatial_auc - full_auc:+.4f}")
    print(f"  No-static (mask only) vs Full: ΔAUC = {nostatic_auc - full_auc:+.4f}")

    print(f"\n  Permutation Importance (top-5):")
    for i, (name, data) in enumerate(sorted_items[:5]):
        print(f"    {i+1}. {name}: AUC drop = {data['auc_drop']:.4f}")

    print(f"\n  Interpretation:")
    if abs(thermal_auc - full_auc) < 0.02:
        print(f"    → Thermal-only ≈ Full: Model primarily learns thermal stress physics")
    else:
        print(f"    → Thermal-only < Full (ΔAUC={thermal_auc - full_auc:+.4f}): "
              f"Auxiliary channels provide meaningful signal")
        if thermal_csi == 0 and full_csi > 0:
            print(f"      Note: Thermal-only AUC={thermal_auc:.4f} (ranking preserved) "
                  f"but CSI=0 at τ=0.54")
            print(f"      → Auxiliary channels boost absolute score magnitude")

    if spatial_auc >= full_auc:
        print(f"    → No-spatial-priors ≥ Full: Static spatial priors NOT needed (good for transfer)")
    else:
        print(f"    → No-spatial-priors < Full: Static spatial priors contribute meaningfully")

    if nostatic_auc >= full_auc:
        print(f"    → No-static (mask only) ≥ Full: Model does not rely on static features at all")
    print("=" * 70)


def predict_with_ablation_static_shuffle(model, ds, device, stat_shuffle_idx):
    """Sliding-window inference with one static channel spatially shuffled."""
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

    # Prepare shuffled static channel (shuffle spatial pixels)
    xs_raw = ds.X_static[0].copy()  # [H, W, C_stat]
    flat = xs_raw[:, :, stat_shuffle_idx].flatten()
    np.random.shuffle(flat)
    xs_raw[:, :, stat_shuffle_idx] = flat.reshape(H, W)

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
            xs_tiled = np.tile(xs_raw, (bs_actual, 1, 1, 1))
            xs_t = torch.FloatTensor(xs_tiled).to(device)

            out = model(xd, xs_t, lm, scale=A5_CFG["scale"],
                        tau=A5_CFG["tau"], temp=A5_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
            del xd, xs_t, out, preds
    return forecast_map


if __name__ == "__main__":
    main()