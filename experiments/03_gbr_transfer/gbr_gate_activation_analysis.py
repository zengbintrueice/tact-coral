# ==========================================================================
# Source: paper/a7_dual_output/gbr_gate_activation_analysis.py (verbatim except path fix)
# Experiment: Severe-event gate diagnostics under domain shift
# Paper: Results 2, Fig. 3d
# Output: results/gbr_gate_activation_analysis.json
# ==========================================================================

# gbr_gate_activation_analysis.py — Experiment 5: Gate Threshold Analysis
#
#   Goal: Visualize and quantify the gate deactivation mechanism on GBR.
#
#   Analysis:
#     1. Gate vs y_main density heatmap (SCS vs GBR)
#     2. Gate CDF comparison (SCS vs GBR, all pixels + severe only)
#     3. "What if" counterfactual: if gate were forced active on GBR severe pixels,
#        how much would y_final improve?
#     4. Gate activation rate as function of y_main threshold
#
#   Saves:
#     results/gbr_gate_activation_analysis.json
#     results/experiment5_gate_threshold.png
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' gbr_gate_activation_analysis.py

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
from matplotlib.colors import LogNorm

# a7 shared paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, PIECF_ROOT, A6_WEIGHT, XGB_WEIGHT_DIR, DATA_DIR, RESULTS_DIR

from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import get_year_range_for_mode

A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8


def restore_a5():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_with_components(model, ds, device):
    """Sliding-window inference returning y_final, y_main, gate, residual."""
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn

    y_final_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    y_main_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    gate_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    res_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)

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
            y_final, y_main, gate, res = model(xd, xs, lm, scale=A5_CFG["scale"],
                                                tau=A5_CFG["tau"], temp=A5_CFG["temp"])
            y_final_last = y_final[:, -1, :, :].cpu().numpy()
            y_main_last = y_main[:, -1, :, :].cpu().numpy()
            gate_last = gate[:, -1, :, :].cpu().numpy()
            res_last = res[:, -1, :, :].cpu().numpy()

            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                y_final_map[t_end] = y_final_last[k]
                y_main_map[t_end] = y_main_last[k]
                gate_map[t_end] = gate_last[k]
                res_map[t_end] = res_last[k]
            del xd, xs, y_final, y_main, gate, res

    return {
        "y_final": y_final_map,
        "y_main": y_main_map,
        "gate": gate_map,
        "residual": res_map,
    }


def build_verification_arrays(components, y_full, T, n_dates,
                              year_range=None, date_strings=None):
    """Build pixel-level arrays for t+4 verification (year-filtered)."""
    y_final_pfv, y_main_pfv, gate_pfv, res_pfv, target = [], [], [], [], []
    for t in range(T - 1, n_dates):
        # Year filtering: only include time steps in the specified year range
        if year_range is not None and date_strings is not None:
            year = int(date_strings[t][:4])
            if not (year_range[0] <= year <= year_range[1]):
                continue
        t_target = t + LEAD_WEEKS
        if t_target >= n_dates:
            continue
        valid = (~np.isnan(components["y_final"][t]) &
                 ~np.isnan(y_full[t_target]))
        if valid.sum() < 1:
            continue
        y_final_pfv.extend(components["y_final"][t][valid].tolist())
        y_main_pfv.extend(components["y_main"][t][valid].tolist())
        gate_pfv.extend(components["gate"][t][valid].tolist())
        res_pfv.extend(components["residual"][t][valid].tolist())
        target.extend(y_full[t_target][valid].tolist())
    return (np.array(y_final_pfv), np.array(y_main_pfv), np.array(gate_pfv),
            np.array(res_pfv), np.array(target))


def r2_calc(p, t):
    if len(p) < 5:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(f"Experiment 5: Gate Threshold Analysis — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                    val_years=(2011, 2015), test_years=(2016, 2019))
    GBR_SPLIT = dict(year_based_split=True, train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))  # LTTS_CRW GBR re-slice (val has severe events)

    print("\n[Step 1] Loading a7 model...")
    a5_model = restore_a5().to(device)

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Inference on SCS and GBR
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 2] Inference on SCS...")
    ds_scs = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                              lead_weeks=LEAD_WEEKS, **split_kw)
    T = ds_scs.history_weeks
    scs_components = predict_with_components(a5_model, ds_scs, device)
    scs_yf, scs_ym, scs_g, scs_r, scs_y = build_verification_arrays(
        scs_components, ds_scs.y_regression_full, T, ds_scs.n_dates,
        year_range=get_year_range_for_mode(ds_scs), date_strings=ds_scs.date_strings)
    print(f"  SCS: n_pixels={len(scs_yf)}")

    print("\n[Step 3] Inference on GBR...")
    ds_gbr = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                              lead_weeks=LEAD_WEEKS,
                              dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                              **GBR_SPLIT)
    gbr_components = predict_with_components(a5_model, ds_gbr, device)
    gbr_yf, gbr_ym, gbr_g, gbr_r, gbr_y = build_verification_arrays(
        gbr_components, ds_gbr.y_regression_full, T, ds_gbr.n_dates,
        year_range=get_year_range_for_mode(ds_gbr), date_strings=ds_gbr.date_strings)
    print(f"  GBR: n_pixels={len(gbr_yf)}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Gate threshold analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 4] Gate threshold analysis...")

    # Severe masks
    scs_severe = scs_y >= SEVERE_THRESHOLD
    gbr_severe = gbr_y >= SEVERE_THRESHOLD

    # Gate activation rate as function of y_main threshold
    y_main_thresholds = np.linspace(0.1, 0.9, 17)
    scs_activation_rates = []
    gbr_activation_rates = []
    for thr in y_main_thresholds:
        scs_rate = float(np.mean(scs_g[scs_ym >= thr] > 0.5)) if (scs_ym >= thr).sum() > 0 else 0.0
        gbr_rate = float(np.mean(gbr_g[gbr_ym >= thr] > 0.5)) if (gbr_ym >= thr).sum() > 0 else 0.0
        scs_activation_rates.append(scs_rate)
        gbr_activation_rates.append(gbr_rate)

    # Counterfactual: if gate were forced to 1 on GBR severe pixels
    # y_final_counterfactual = y_main + 1.0 * residual (gate forced to 1)
    gbr_yf_counter = gbr_ym + 1.0 * gbr_r  # gate forced to 1
    gbr_severe_r2_zs = r2_calc(gbr_yf[gbr_severe], gbr_y[gbr_severe]) if gbr_severe.sum() > 0 else float("nan")
    gbr_severe_r2_counter = r2_calc(gbr_yf_counter[gbr_severe], gbr_y[gbr_severe]) if gbr_severe.sum() > 0 else float("nan")

    print(f"  GBR severe R² (zero-shot): {gbr_severe_r2_zs:.4f}")
    print(f"  GBR severe R² (gate forced=1): {gbr_severe_r2_counter:.4f}")
    print(f"  Improvement from gate activation: {gbr_severe_r2_counter - gbr_severe_r2_zs:+.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Plot
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 5] Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Experiment 5: Gate Threshold Analysis\n"
                 f"(gate = sigmoid((y_main - 0.5)/0.1), gate > 0.5 means severe correction active)",
                 fontsize=12, fontweight="bold")

    # (a) Gate vs y_main density heatmap — SCS
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(scs_ym), min(20000, len(scs_ym)), replace=False)
    h = ax.hist2d(scs_ym[sample_idx], scs_g[sample_idx], bins=80,
                  norm=LogNorm(), cmap="Blues")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="tau=0.5")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="gate=0.5")
    ax.set_xlabel("y_main", fontsize=10)
    ax.set_ylabel("gate", fontsize=10)
    ax.set_title("(a) SCS: Gate vs y_main density", fontsize=10)
    ax.legend(fontsize=8)
    plt.colorbar(h[3], ax=ax, shrink=0.8)

    # (b) Gate vs y_main density heatmap — GBR
    ax = axes[0, 1]
    sample_idx = np.random.choice(len(gbr_ym), min(20000, len(gbr_ym)), replace=False)
    h = ax.hist2d(gbr_ym[sample_idx], gbr_g[sample_idx], bins=80,
                  norm=LogNorm(), cmap="Oranges")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="tau=0.5")
    ax.axhline(y=0.5, color="blue", linestyle="--", alpha=0.7, label="gate=0.5")
    ax.set_xlabel("y_main", fontsize=10)
    ax.set_ylabel("gate", fontsize=10)
    ax.set_title("(b) GBR: Gate vs y_main density", fontsize=10)
    ax.legend(fontsize=8)
    plt.colorbar(h[3], ax=ax, shrink=0.8)

    # (c) Gate CDF comparison
    ax = axes[1, 0]
    scs_g_sorted = np.sort(scs_g)
    gbr_g_sorted = np.sort(gbr_g)
    scs_cdf = np.arange(1, len(scs_g_sorted) + 1) / len(scs_g_sorted)
    gbr_cdf = np.arange(1, len(gbr_g_sorted) + 1) / len(gbr_g_sorted)
    ax.plot(scs_g_sorted, scs_cdf, color="#2196F3", linewidth=1.5, label="SCS")
    ax.plot(gbr_g_sorted, gbr_cdf, color="#FF9800", linewidth=1.5, label="GBR")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="gate=0.5")
    ax.set_xlabel("gate", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title("(c) Gate CDF Comparison (All Pixels)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Gate activation rate vs y_main threshold
    ax = axes[1, 1]
    ax.plot(y_main_thresholds, scs_activation_rates, "o-", color="#2196F3",
            linewidth=2, markersize=5, label="SCS")
    ax.plot(y_main_thresholds, gbr_activation_rates, "s-", color="#FF9800",
            linewidth=2, markersize=5, label="GBR")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("y_main threshold (pixels with y_main >= thr)", fontsize=10)
    ax.set_ylabel("Gate activation rate (gate > 0.5)", fontsize=10)
    ax.set_title("(d) Gate Activation Rate vs y_main Threshold", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    png_path = RESULTS_DIR / "gbr_transfer/gbr_gate_activation_analysis.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {png_path}")

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Save JSON
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 6] Saving JSON...")
    out = {
        "experiment": "gate_threshold_analysis",
        "description": "Visualize and quantify gate deactivation on GBR",
        "configuration": {
            "model": "a7 (SCS-trained, frozen)",
            "gate_formula": "sigmoid((y_main - 0.5) / 0.1)",
            "severe_threshold": SEVERE_THRESHOLD,
        },
        "gate_statistics": {
            "scs": {
                "mean": round(float(scs_g.mean()), 4),
                "std": round(float(scs_g.std()), 4),
                "activation_rate_gt05": round(float((scs_g > 0.5).mean()), 4),
                "activation_rate_gt01": round(float((scs_g > 0.1).mean()), 4),
                "pct_with_gate_lt01": round(float((scs_g < 0.1).mean()), 4),
            },
            "gbr": {
                "mean": round(float(gbr_g.mean()), 4),
                "std": round(float(gbr_g.std()), 4),
                "activation_rate_gt05": round(float((gbr_g > 0.5).mean()), 4),
                "activation_rate_gt01": round(float((gbr_g > 0.1).mean()), 4),
                "pct_with_gate_lt01": round(float((gbr_g < 0.1).mean()), 4),
            },
        },
        "severe_pixel_gate": {
            "scs": {
                "n_severe": int(scs_severe.sum()),
                "gate_mean": round(float(scs_g[scs_severe].mean()), 4) if scs_severe.sum() > 0 else None,
                "gate_std": round(float(scs_g[scs_severe].std()), 4) if scs_severe.sum() > 0 else None,
                "activation_rate_gt05": round(float((scs_g[scs_severe] > 0.5).mean()), 4) if scs_severe.sum() > 0 else None,
            },
            "gbr": {
                "n_severe": int(gbr_severe.sum()),
                "gate_mean": round(float(gbr_g[gbr_severe].mean()), 4) if gbr_severe.sum() > 0 else None,
                "gate_std": round(float(gbr_g[gbr_severe].std()), 4) if gbr_severe.sum() > 0 else None,
                "activation_rate_gt05": round(float((gbr_g[gbr_severe] > 0.5).mean()), 4) if gbr_severe.sum() > 0 else None,
            },
        },
        "counterfactual": {
            "description": "If gate were forced to 1 on GBR severe pixels",
            "gbr_severe_r2_zeroshot": round(gbr_severe_r2_zs, 4),
            "gbr_severe_r2_gate_forced_1": round(gbr_severe_r2_counter, 4),
            "r2_improvement": round(gbr_severe_r2_counter - gbr_severe_r2_zs, 4),
        },
        "gate_activation_curve": {
            "y_main_thresholds": [round(float(x), 3) for x in y_main_thresholds],
            "scs_activation_rates": [round(float(x), 4) for x in scs_activation_rates],
            "gbr_activation_rates": [round(float(x), 4) for x in gbr_activation_rates],
        },
        "timestamp": datetime.now().isoformat(),
    }
    json_path = RESULTS_DIR / "gbr_transfer/gbr_gate_activation_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
