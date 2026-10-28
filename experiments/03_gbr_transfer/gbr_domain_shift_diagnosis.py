# ==========================================================================
# Source: paper/a7_dual_output/gbr_domain_shift_diagnosis.py (verbatim except path fix)
# Experiment: Domain-shift diagnosis (score distributions, gate activation)
# Paper: Results 2, Fig. 3b
# Output: results/gbr_domain_shift_diagnosis.json
# ==========================================================================

# gbr_domain_shift_diagnosis.py — Experiment 2: Domain-Shift Diagnosis
#
#   Goal: Diagnose WHY a7's risk_score is systematically lower on GBR than SCS.
#
#   Hypotheses to test:
#     H1: Backbone domain shift — y_main distribution shifts lower on GBR
#     H2: Gate deactivation — gate = sigmoid((y_main - 0.5)/0.1) stays ≈0 on GBR
#     H3: Severe head saturation — residual distribution differs between domains
#
#   Method:
#     - Run a7 on both SCS and GBR test sets
#     - Extract y_main, gate, residual at position T-1 (verified position)
#     - Compare distributions
#
#   Saves:
#     results/gbr_domain_shift_diagnosis.json
#     results/experiment2_distributions.png
#     results/experiment2_gate_analysis.png
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' gbr_domain_shift_diagnosis.py

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
    """Sliding-window inference returning y_final, y_main, gate, residual.

    Returns dict with arrays [n_dates, H, W] for each component.
    """
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
            y_final, y_main, gate, res = model(xd, xs, lm, scale=A5_CFG["scale"],
                                                tau=A5_CFG["tau"], temp=A5_CFG["temp"])
            # Take last position (position T-1) — the verified forecast
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

    return {
        "y_final": y_final_map,
        "y_main": y_main_map,
        "gate": gate_map,
        "residual": res_map,
    }


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Experiment 2: Domain-Shift Diagnosis — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))
GBR_SPLIT = dict(year_based_split=True, train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))  # LTTS_CRW GBR re-slice (val has severe events)

print("\n[Step 1] Loading a7 model...")
a5_model = restore_a5().to(device)

# ═══════════════════════════════════════════════════════════════
# Step 2: Inference on SCS (source domain)
# ═══════════════════════════════════════════════════════════════
print("\n[Step 2] Inference on SCS (source domain)...")
ds_scs = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                          lead_weeks=4, **split_kw)
T, H_scs, W_scs = ds_scs.history_weeks, ds_scs.n_lat, ds_scs.n_lon
n_dates_scs = ds_scs.n_dates
y_full_scs = ds_scs.y_regression_full
print(f"  SCS: T={T} H={H_scs} W={W_scs} n_dates={n_dates_scs}")

t0 = time.time()
scs_components = predict_with_components(a5_model, ds_scs, device)
print(f"  Done in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# Step 3: Inference on GBR (target domain)
# ═══════════════════════════════════════════════════════════════
print("\n[Step 3] Inference on GBR (target domain)...")
ds_gbr = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                          lead_weeks=4, dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                          **GBR_SPLIT)
H_gbr, W_gbr = ds_gbr.n_lat, ds_gbr.n_lon
n_dates_gbr = ds_gbr.n_dates
y_full_gbr = ds_gbr.y_regression_full
print(f"  GBR: T={T} H={H_gbr} W={W_gbr} n_dates={n_dates_gbr}")

t0 = time.time()
gbr_components = predict_with_components(a5_model, ds_gbr, device)
print(f"  Done in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# Step 4: Build pixel-level t+4 verification arrays
# ═══════════════════════════════════════════════════════════════
print("\n[Step 4] Building t+4 verification arrays...")

def build_verification_arrays(components, y_full, T, n_dates, H, W,
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

scs_yf, scs_ym, scs_g, scs_r, scs_y = build_verification_arrays(
    scs_components, y_full_scs, T, n_dates_scs, H_scs, W_scs,
    year_range=get_year_range_for_mode(ds_scs), date_strings=ds_scs.date_strings)
gbr_yf, gbr_ym, gbr_g, gbr_r, gbr_y = build_verification_arrays(
    gbr_components, y_full_gbr, T, n_dates_gbr, H_gbr, W_gbr,
    year_range=get_year_range_for_mode(ds_gbr), date_strings=ds_gbr.date_strings)

print(f"  SCS: n_pixels={len(scs_yf)}")
print(f"  GBR: n_pixels={len(gbr_yf)}")

# ═══════════════════════════════════════════════════════════════
# Step 5: Distribution comparison
# ═══════════════════════════════════════════════════════════════
print("\n[Step 5] Distribution comparison...")

def stats(arr, name):
    return {
        "name": name,
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }

comparisons = {
    "y_final": {"SCS": stats(scs_yf, "SCS"), "GBR": stats(gbr_yf, "GBR")},
    "y_main": {"SCS": stats(scs_ym, "SCS"), "GBR": stats(gbr_ym, "GBR")},
    "gate": {"SCS": stats(scs_g, "SCS"), "GBR": stats(gbr_g, "GBR")},
    "residual": {"SCS": stats(scs_r, "SCS"), "GBR": stats(gbr_r, "GBR")},
}

print(f"\n  {'Component':<12} {'Domain':<6} {'Mean':<10} {'Std':<10} {'Median':<10} {'P25':<10} {'P75':<10}")
print("  " + "-" * 66)
for comp_name, domains in comparisons.items():
    for domain_name, s in domains.items():
        print(f"  {comp_name:<12} {domain_name:<6} {s['mean']:<10.4f} {s['std']:<10.4f} "
              f"{s['median']:<10.4f} {s['p25']:<10.4f} {s['p75']:<10.4f}")
    print()

# Severe-only stats
scs_severe = scs_y >= SEVERE_THRESHOLD
gbr_severe = gbr_y >= SEVERE_THRESHOLD
print(f"  Severe pixels (y>={SEVERE_THRESHOLD}):")
print(f"    SCS: {scs_severe.sum()}/{len(scs_y)} ({scs_severe.mean()*100:.2f}%)")
print(f"    GBR: {gbr_severe.sum()}/{len(gbr_y)} ({gbr_severe.mean()*100:.2f}%)")

if scs_severe.sum() > 0 and gbr_severe.sum() > 0:
    print(f"\n  Severe-only component means:")
    print(f"    y_main:     SCS={scs_ym[scs_severe].mean():.4f}  GBR={gbr_ym[gbr_severe].mean():.4f}")
    print(f"    gate:       SCS={scs_g[scs_severe].mean():.4f}  GBR={gbr_g[gbr_severe].mean():.4f}")
    print(f"    residual:   SCS={scs_r[scs_severe].mean():.4f}  GBR={gbr_r[gbr_severe].mean():.4f}")
    print(f"    y_final:    SCS={scs_yf[scs_severe].mean():.4f}  GBR={gbr_yf[gbr_severe].mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# Step 6: Plot distributions
# ═══════════════════════════════════════════════════════════════
print("\n[Step 6] Plotting distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Experiment 2: Domain-Shift Diagnosis — Component Distributions\n"
             f"(a7, SCS-trained, t+{LEAD_WEEKS} pixel-level verification)",
             fontsize=12, fontweight="bold")

components_data = [
    ("y_final", scs_yf, gbr_yf, "Final Risk Score (y_main + gate × res)"),
    ("y_main", scs_ym, gbr_ym, "Backbone Output (y_main)"),
    ("gate", scs_g, gbr_g, "Gate (sigmoid((y_main-0.5)/0.1))"),
    ("residual", scs_r, gbr_r, "Severe Head Residual (tanh × 0.3)"),
]

for ax, (name, scs_arr, gbr_arr, title) in zip(axes.flat, components_data):
    bins = np.linspace(min(scs_arr.min(), gbr_arr.min()),
                       max(scs_arr.max(), gbr_arr.max()), 80)
    ax.hist(scs_arr, bins=bins, alpha=0.5, color="#2196F3", label="SCS", density=True)
    ax.hist(gbr_arr, bins=bins, alpha=0.5, color="#FF9800", label="GBR", density=True)
    ax.axvline(x=np.median(scs_arr), color="#2196F3", linestyle="--", alpha=0.7)
    ax.axvline(x=np.median(gbr_arr), color="#FF9800", linestyle="--", alpha=0.7)
    ax.set_xlabel(name, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
png_path = RESULTS_DIR / "gbr_transfer/gbr_domain_shift_distributions.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {png_path}")

# ═══════════════════════════════════════════════════════════════
# Step 7: Gate analysis
# ═══════════════════════════════════════════════════════════════
print("\n[Step 7] Gate activation analysis...")

# Gate activation rate (gate > 0.5 means "severe mode")
scs_gate_active = (scs_g > 0.5).mean()
gbr_gate_active = (gbr_g > 0.5).mean()
print(f"  Gate activation rate (gate > 0.5):")
print(f"    SCS: {scs_gate_active*100:.2f}%")
print(f"    GBR: {gbr_gate_active*100:.2f}%")

# Gate vs y_main scatter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Gate Analysis: sigmoid((y_main - 0.5) / 0.1)\n"
             f"(gate > 0.5 means severe correction is active)",
             fontsize=11, fontweight="bold")

# Sample for scatter plot
np.random.seed(42)
scs_idx = np.random.choice(len(scs_ym), min(5000, len(scs_ym)), replace=False)
gbr_idx = np.random.choice(len(gbr_ym), min(5000, len(gbr_ym)), replace=False)

ax1.scatter(scs_ym[scs_idx], scs_g[scs_idx], alpha=0.3, s=5, c="#2196F3", label="SCS")
ax1.scatter(gbr_ym[gbr_idx], gbr_g[gbr_idx], alpha=0.3, s=5, c="#FF9800", label="GBR")
ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="tau=0.5")
ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
ax1.set_xlabel("y_main", fontsize=11)
ax1.set_ylabel("gate", fontsize=11)
ax1.set_title("Gate vs y_main", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Gate histogram (severe pixels only)
if scs_severe.sum() > 0 and gbr_severe.sum() > 0:
    ax2.hist(scs_g[scs_severe], bins=50, alpha=0.5, color="#2196F3",
             label=f"SCS severe (n={scs_severe.sum()})", density=True)
    ax2.hist(gbr_g[gbr_severe], bins=50, alpha=0.5, color="#FF9800",
             label=f"GBR severe (n={gbr_severe.sum()})", density=True)
    ax2.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="gate=0.5")
    ax2.set_xlabel("gate (severe pixels only)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Gate Distribution for Severe Pixels", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
png_path = RESULTS_DIR / "gbr_transfer/gbr_domain_shift_gate.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {png_path}")

# ═══════════════════════════════════════════════════════════════
# Step 8: Diagnosis summary
# ═══════════════════════════════════════════════════════════════
print("\n[Step 8] Diagnosis summary...")

# Quantify shifts
y_main_shift = gbr_ym.mean() - scs_ym.mean()
gate_shift = gbr_g.mean() - scs_g.mean()
res_shift = gbr_r.mean() - scs_r.mean()
y_final_shift = gbr_yf.mean() - scs_yf.mean()

print(f"  Mean shifts (GBR - SCS):")
print(f"    y_main:     {y_main_shift:+.4f}")
print(f"    gate:       {gate_shift:+.4f}")
print(f"    residual:   {res_shift:+.4f}")
print(f"    y_final:    {y_final_shift:+.4f}")

# Determine primary cause
if abs(y_main_shift) > abs(gate_shift):
    primary_cause = "backbone domain shift (y_main lower on GBR)"
else:
    primary_cause = "gate deactivation (gate stays ≈0 on GBR)"

print(f"\n  Primary diagnosis: {primary_cause}")

# ═══════════════════════════════════════════════════════════════
# Step 9: Save JSON
# ═══════════════════════════════════════════════════════════════
print("\n[Step 9] Saving JSON...")
out = {
    "experiment": "domain_shift_diagnosis",
    "description": "Diagnose why a7 risk_score is lower on GBR than SCS",
    "configuration": {
        "model": "a7 (SCS-trained, frozen)",
        "test_datasets": ["SCS", "GBR"],
        "lead_weeks": LEAD_WEEKS,
    },
    "sample_sizes": {
        "scs_pixels": len(scs_yf),
        "gbr_pixels": len(gbr_yf),
        "scs_severe": int(scs_severe.sum()),
        "gbr_severe": int(gbr_severe.sum()),
    },
    "component_distributions": comparisons,
    "severe_only_means": {
        "y_main": {"SCS": round(float(scs_ym[scs_severe].mean()), 4) if scs_severe.sum() > 0 else None,
                   "GBR": round(float(gbr_ym[gbr_severe].mean()), 4) if gbr_severe.sum() > 0 else None},
        "gate": {"SCS": round(float(scs_g[scs_severe].mean()), 4) if scs_severe.sum() > 0 else None,
                 "GBR": round(float(gbr_g[gbr_severe].mean()), 4) if gbr_severe.sum() > 0 else None},
        "residual": {"SCS": round(float(scs_r[scs_severe].mean()), 4) if scs_severe.sum() > 0 else None,
                     "GBR": round(float(gbr_r[gbr_severe].mean()), 4) if gbr_severe.sum() > 0 else None},
    },
    "gate_activation_rate": {
        "SCS": round(float(scs_gate_active), 4),
        "GBR": round(float(gbr_gate_active), 4),
    },
    "mean_shifts_gbr_minus_scs": {
        "y_main": round(float(y_main_shift), 4),
        "gate": round(float(gate_shift), 4),
        "residual": round(float(res_shift), 4),
        "y_final": round(float(y_final_shift), 4),
    },
    "diagnosis": {
        "primary_cause": primary_cause,
        "y_main_shift": round(float(y_main_shift), 4),
        "gate_shift": round(float(gate_shift), 4),
    },
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "gbr_transfer/gbr_domain_shift_diagnosis.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

print("\nDone.")
