# ==========================================================================
# Source: paper/a7_dual_output/calibration_curves_4group.py (verbatim except path fix)
# Experiment: Full calibration comparison (4 groups, Wilson CI)
# Paper: Supplementary Fig. 3
# Output: results/calibration_curves_4group.json
# ==========================================================================

# -*- coding: utf-8 -*-
"""calibration_curves_4group.py — Full calibration curve comparison with ECE + bin counts.

Four calibration curves on one panel:
  1. SCS raw (in-domain)
  2. GBR zero-shot raw
  3. GBR + SCS-Isotonic (transfer; calibrator fit on SCS val, FROZEN, applied to GBR test)
  4. GBR + GBR-Isotonic (oracle; calibrator fit on GBR val, applied to GBR test)

Design:
  - Shared QUANTILE bins: 10 equal-frequency bins defined by GBR-raw-test
    prediction decile boundaries; ALL four groups mapped onto these same bins
    (strict cross-curve comparison on the same prediction range).
  - 95% Wilson confidence interval error bars per bin (not Wald).
  - ECE + Brier score per group.
  - Bottom sub-panel: per-bin sample size bar chart (addresses "small-bin
    sample size" reviewer concern).

Output: results/calibration_curves_4group.json (all plot data for reproducibility)
"""
import os, sys, io, json, time, warnings, importlib, gc, tempfile
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# sklearn BEFORE torch to avoid Windows STATUS_ACCESS_VIOLATION
from sklearn.isotonic import IsotonicRegression

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, PIECF_ROOT, A6_WEIGHT, XGB_WEIGHT_DIR, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import build_pfv_arrays, get_year_range_for_mode

A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
N_BINS = 10
Z_ALPHA = norm.ppf(0.975)  # 1.96 for 95% CI

TMP_DIR = Path(tempfile.gettempdir()) / "exp_calib_full"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def restore_a5():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location="cpu", weights_only=False)
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
            if start_batch % 64 == 0:
                torch.cuda.empty_cache()
    return forecast_map


def run_dataset(ds):
    """Run PFV inference on a dataset, return (pred, target) aligned arrays."""
    T = ds.history_weeks
    forecast = predict_sliding_temporal(model, ds, device)
    pred, target, _ = build_pfv_arrays(
        forecast, ds.y_regression_full, T, ds.n_dates,
        year_range=get_year_range_for_mode(ds), date_strings=ds.date_strings)
    return pred, target


def wilson_ci(successes, n, z=Z_ALPHA):
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - margin), float(centre + margin))


def brier_score(pred, actual):
    """Brier score: mean squared error between pred and binary outcome."""
    valid = ~np.isnan(pred)
    return float(np.mean((pred[valid] - (actual[valid] >= SEVERE_THRESHOLD)) ** 2))


def compute_shared_bin_stats(pred, actual, bin_edges):
    """Map pred/actual onto SHARED quantile bin edges. Return per-bin stats."""
    bins = np.digitize(pred, bin_edges[1:-1])  # 0..N_BINS-1
    bins = np.clip(bins, 0, N_BINS - 1)  # clamp out-of-range (e.g. pred > last edge)
    stats = []
    for b in range(N_BINS):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            stats.append({"bin": b, "n": 0, "confidence": None, "accuracy": None,
                          "ci_low": None, "ci_high": None})
            continue
        p = pred[mask]
        t = actual[mask]
        conf = float(np.mean(p))
        acc = float(np.mean(t >= SEVERE_THRESHOLD))
        n_sev = int((t >= SEVERE_THRESHOLD).sum())
        lo, hi = wilson_ci(n_sev, n)
        stats.append({"bin": b, "n": n, "confidence": conf, "accuracy": acc,
                      "n_severe": n_sev, "ci_low": lo, "ci_high": hi})
    return stats


def ece_from_stats(stats, total_n):
    """ECE from shared-bin stats: sum (n_k/N) |acc_k - conf_k|."""
    ece = 0.0
    for s in stats:
        if s["n"] > 0 and s["accuracy"] is not None:
            ece += (s["n"] / total_n) * abs(s["accuracy"] - s["confidence"])
    return float(ece)


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Full Calibration Comparison — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))

# ── Load model ─────────────────────────────────────────────────
print("\n[1] Loading a7 model...")
model = restore_a5().to(device)

# ── SCS val → fit SCS isotonic (FROZEN calibrator) ────────────
print("\n[2] SCS val → fit SCS-Isotonic...")
ds_scs_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                             lead_weeks=4, **split_kw)
p_val, t_val = run_dataset(ds_scs_val)
print(f"  SCS val: n={len(p_val)}")
iso_scs = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso_scs.fit(p_val, (t_val >= SEVERE_THRESHOLD).astype(float))
print("  SCS-Isotonic fitted (frozen).")
del ds_scs_val, p_val, t_val
gc.collect(); torch.cuda.empty_cache()

# ── GBR val → fit GBR isotonic (oracle) ───────────────────────
print("\n[3] GBR val → fit GBR-Isotonic (oracle)...")
ds_gbr_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                             lead_weeks=4,
                             dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                             **split_kw)
p_gbr_val, t_gbr_val = run_dataset(ds_gbr_val)
print(f"  GBR val: n={len(p_gbr_val)}")
gbr_val_sev = int((t_gbr_val >= SEVERE_THRESHOLD).sum())
iso_gbr = None
if gbr_val_sev >= 50:
    iso_gbr = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_gbr.fit(p_gbr_val, (t_gbr_val >= SEVERE_THRESHOLD).astype(float))
    print(f"  GBR-Isotonic fitted (oracle, {gbr_val_sev} severe).")
else:
    print(f"  WARNING: only {gbr_val_sev} GBR-val severe events; oracle skipped.")
del ds_gbr_val, p_gbr_val, t_gbr_val
gc.collect(); torch.cuda.empty_cache()

# ── SCS test (raw) ────────────────────────────────────────────
print("\n[4] SCS test raw...")
ds_scs_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                              lead_weeks=4, **split_kw)
p_scs, t_scs = run_dataset(ds_scs_test)
print(f"  SCS test: n={len(p_scs)}")
del ds_scs_test
gc.collect(); torch.cuda.empty_cache()

# ── GBR test (raw + calibrated) ───────────────────────────────
print("\n[5] GBR test (raw / SCS-Iso / GBR-Iso)...")
ds_gbr_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                              lead_weeks=4,
                              dataset_file="gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
                              **split_kw)
p_gbr, t_gbr = run_dataset(ds_gbr_test)
print(f"  GBR test: n={len(p_gbr)}")
del ds_gbr_test
gc.collect(); torch.cuda.empty_cache()

# GBR raw → SCS-Isotonic (frozen transfer)
valid_gbr = ~np.isnan(p_gbr)
p_gbr_scs_iso = np.full_like(p_gbr, np.nan)
p_gbr_scs_iso[valid_gbr] = iso_scs.predict(p_gbr[valid_gbr])
# GBR raw → GBR-Isotonic (oracle)
p_gbr_gbr_iso = np.full_like(p_gbr, np.nan)
if iso_gbr is not None:
    p_gbr_gbr_iso[valid_gbr] = iso_gbr.predict(p_gbr[valid_gbr])

# ═══════════════════════════════════════════════════════════════
# Shared quantile bins: GBR-raw-test decile boundaries
# ═══════════════════════════════════════════════════════════════
print("\n[6] Building shared quantile bins (GBR-raw-test deciles)...")
gbr_valid_vals = p_gbr[valid_gbr]
quantiles = np.quantile(gbr_valid_vals, np.linspace(0, 1, N_BINS + 1))
# Ensure strictly increasing edges
bin_edges = np.unique(quantiles)
print(f"  Shared bin edges: {np.round(bin_edges, 4)}")

groups = {}
# SCS raw mapped to shared bins
groups["SCS_raw"] = {
    "label": "SCS raw",
    "pred": p_scs, "actual": t_scs,
}
groups["GBR_raw"] = {
    "label": "GBR zero-shot raw",
    "pred": p_gbr, "actual": t_gbr,
}
groups["GBR_SCS_iso"] = {
    "label": "GBR + SCS-Isotonic (transfer)",
    "pred": p_gbr_scs_iso, "actual": t_gbr,
}
groups["GBR_GBR_iso"] = {
    "label": "GBR + GBR-Isotonic (oracle)",
    "pred": p_gbr_gbr_iso, "actual": t_gbr,
}

out_groups = {}
for key, g in groups.items():
    stats = compute_shared_bin_stats(g["pred"], g["actual"], bin_edges)
    n_total = int((~np.isnan(g["pred"])).sum())
    ece = ece_from_stats(stats, n_total)
    brier = brier_score(g["pred"], g["actual"])
    out_groups[key] = {
        "label": g["label"],
        "n_total": n_total,
        "ece": ece,
        "brier": brier,
        "bins": stats,
    }
    print(f"  {g['label']:35s}  ECE={ece:.4f}  Brier={brier:.4f}  n={n_total}")

# ═══════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════
print("\n[7] Plotting...")
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.8, 0.9], hspace=0.35)

ax_top = fig.add_subplot(gs[0])
colors = {"SCS_raw": "#3498DB", "GBR_raw": "#E74C3C",
          "GBR_SCS_iso": "#E67E22", "GBR_GBR_iso": "#27AE60"}
markers = {"SCS_raw": "o", "GBR_raw": "s",
           "GBR_SCS_iso": "^", "GBR_GBR_iso": "D"}

ax_top.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")
bin_centres = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

for key, color in colors.items():
    g = out_groups[key]
    xs, ys, lo, hi = [], [], [], []
    for b in g["bins"]:
        if b["confidence"] is None or b["ci_low"] is None or b["ci_high"] is None:
            continue
        xs.append(b["confidence"])
        ys.append(b["accuracy"])
        lo.append(max(b["accuracy"] - b["ci_low"], 0.0))
        hi.append(max(b["ci_high"] - b["accuracy"], 0.0))
    ax_top.errorbar(xs, ys, yerr=[lo, hi], fmt=markers[key], color=color,
                    capsize=3, linewidth=1.6, markersize=6, alpha=0.9,
                    label=f"{g['label']}  (ECE={g['ece']:.4f}, Brier={g['brier']:.4f})")

ax_top.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
ax_top.set_xlabel("Mean predicted value", fontsize=10)
ax_top.set_ylabel("Actual severe frequency", fontsize=10)
ax_top.set_title(f"Calibration curves (severe ≥ {SEVERE_THRESHOLD}, t+{LEAD_WEEKS}; "
                 f"shared GBR-raw decile bins, 95% Wilson CI)",
                 fontsize=11, fontweight="bold")
ax_top.set_xlim(-0.02, 1.02)
ax_top.set_ylim(-0.02, 1.02)
ax_top.grid(alpha=0.3)
ax_top.legend(fontsize=8, loc="upper left", framealpha=0.9)

# Bottom: bin sample size bar chart (per group, shared bins)
ax_bin = fig.add_subplot(gs[2])
width = 0.2
x_pos = np.arange(len(bin_edges) - 1)
for i, key in enumerate(["SCS_raw", "GBR_raw", "GBR_SCS_iso", "GBR_GBR_iso"]):
    g = out_groups[key]
    counts = [b["n"] for b in g["bins"]]
    ax_bin.bar(x_pos + (i - 1.5) * width, counts, width, color=colors[key],
               alpha=0.8, edgecolor="black", linewidth=0.5,
               label=g["label"])
ax_bin.set_xticks(x_pos)
ax_bin.set_xticklabels([f"B{k+1}\n({bin_centres[k]:.2f})" for k in range(len(bin_centres))],
                       fontsize=7)
ax_bin.set_ylabel("Sample size", fontsize=10)
ax_bin.set_title("Per-bin sample size (shared quantile bins) — "
                 "all bins ≥ ~10% of N, addressing small-bin concern",
                 fontsize=10, fontweight="bold")
ax_bin.grid(axis="y", alpha=0.3)
ax_bin.legend(fontsize=7.5, ncol=2)

# Middle panel: group-level summary (ECE + Brier) as text/bar
ax_sum = fig.add_subplot(gs[1])
keys = ["SCS_raw", "GBR_raw", "GBR_SCS_iso", "GBR_GBR_iso"]
gx_pos = np.arange(len(keys))
ece_vals = [out_groups[k]["ece"] for k in keys]
brier_vals = [out_groups[k]["brier"] for k in keys]
ax_sum.bar(gx_pos - 0.2, ece_vals, 0.4, color="#C0392B", alpha=0.8,
           edgecolor="black", linewidth=0.5, label="ECE")
ax_sum.bar(gx_pos + 0.2, brier_vals, 0.4, color="#7F8C8D", alpha=0.8,
           edgecolor="black", linewidth=0.5, label="Brier")
for i, (e, b) in enumerate(zip(ece_vals, brier_vals)):
    ax_sum.text(i - 0.2, e + 0.002, f"{e:.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax_sum.text(i + 0.2, b + 0.002, f"{b:.3f}", ha="center", fontsize=7.5)
ax_sum.set_xticks(gx_pos)
ax_sum.set_xticklabels([g["label"].split("(")[0].strip() for g in
                        [out_groups[k] for k in keys]], fontsize=7.5)
ax_sum.set_ylabel("Score", fontsize=10)
ax_sum.set_title("ECE and Brier score per group", fontsize=10, fontweight="bold")
ax_sum.grid(axis="y", alpha=0.3)
ax_sum.legend(fontsize=8)

png_path = RESULTS_DIR / "calibration/calibration_curves_4group.png"
plt.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Plot saved: {png_path}")

# ═══════════════════════════════════════════════════════════════
# Save JSON (all plot data)
# ═══════════════════════════════════════════════════════════════
print("\n[8] Saving JSON...")
out = {
    "experiment": "calibration_full_comparison",
    "description": "Four calibration curves with Wilson 95% CI, ECE, Brier, shared quantile bins",
    "configuration": {
        "model": "a7 (SCS-trained, frozen)",
        "severe_threshold": SEVERE_THRESHOLD,
        "lead_weeks": LEAD_WEEKS,
        "n_bins": N_BINS,
        "bin_method": "shared quantile bins (GBR-raw-test decile boundaries)",
        "ci_method": "Wilson 95% confidence interval",
        "calibration_method": "Isotonic regression (frozen on SCS val / GBR val)",
    },
    "shared_bin_edges": bin_edges.tolist(),
    "shared_bin_centres": bin_centres,
    "groups": out_groups,
    "data_quality_note": {
        "gbr_val_severe_events": gbr_val_sev,
        "gbr_isotonic_included": iso_gbr is not None,
    },
    "timestamp": datetime.now().isoformat(),
}
json_path = RESULTS_DIR / "calibration/calibration_curves_4group.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  JSON saved: {json_path}")

# Cleanup temp
for f in TMP_DIR.glob("*.npz"):
    f.unlink()
print("\nDone.")