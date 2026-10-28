# ==========================================================================
# Source: paper/a7_dual_output/xgboost_permutation_importance.py (verbatim except path fix)
# Experiment: XGBoost cross-domain feature importance (control)
# Paper: Supplementary Fig. 5 (XGB control)
# Output: results/xgboost_permutation_importance.json
# ==========================================================================

# -*- coding: utf-8 -*-
"""xgboost_permutation_importance.py — XGB Cross-Domain Feature Importance.

Tree-model counterpart of experiment7_4_cross_domain_importance.py (PIECF
permutation importance on sliding-window forecasts). Same question:

    Are the learned predictors PHYSICALLY CONSISTENT across domains
    (SCS training domain vs GBR target domain), or SCS-specific overfitting?

Method (adapted to the deployed XGBoost, results/xgb_retrained_20260911 lead-4):

  1. Permutation importance at the WINDOW-FEATURE level (protocol-equivalent
     to 7.4's channel shuffling): for each dynamic channel c, shuffle the
     T-column block [t*C .. t*C+C) of every window (same permutation across
     the block, destroying temporal structure while preserving marginals),
     re-predict, measure ΔAUC of severe-event ranking. n_shuffles=5 seeds,
     mean ± std. Baseline and shuffled predictions both restricted to the
     dataset's own test years (2016-2019) via label-masked window features —
     identical pixel population to the thermal ablation experiment.
  2. Native gain importance (XGBoost-only cross-check): per-channel sum of
     gain, computed separately on SCS and GBR test windows by re-fitting a
     same-params model on each domain's *evaluation* windows (importance
     probing only — not a performance claim). Spearman vs permutation ranks
     reported as consistency evidence.

  Channels and thermal definition identical to 7.4/7.5 (13 dynamic + 5 static).

Outputs:
  results/xgboost_permutation_importance.json
  results/xgboost_permutation_importance.png  (3 panels like 7.4)

Run:
  & 'D:\...\piecf_env\Scripts\python.exe' xgboost_permutation_importance.py
"""
import os, sys, io, json, time, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_DIR, RESULTS_DIR, XGB_WEIGHT_DIR

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

from src.data.aligned_dataset import SCSPIECFDataset

# ── constants ───────────────────────────────────────────────────
SCS_FILE = "scs_coral_bleaching_dataset_v9_1deg_raw.nc"
GBR_FILE = "gbr_coral_bleaching_dataset_v9_1deg_raw.nc"
LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
N_SHUFFLES = 5
SPLIT_KW = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))

DYN_CHANNEL_NAMES = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par',
                     'seasonal_sin', 'wind_speed', 'ssr', 'sst_anomaly',
                     'd_hotspot_dt', 'hotspot_cumsum_4w',
                     'hotspot_cumsum_12w', 'seasonal_cos']
STATIC_CHANNEL_NAMES = ['bathymetry', 'dist_to_shore', 'cyclone_frequency',
                        'salinity', 'ocean_mask']
THERMAL_INDICES = [1, 2, 9, 10, 11]          # same as 7.4/7.5
THERMAL_NAMES = {DYN_CHANNEL_NAMES[i] for i in THERMAL_INDICES}

# stability classification thresholds — identical to 7.4
STABLE_TH, WEAK_TH, NEG_TH = 0.005, 0.003, 0.005


# ── feature construction (identical to train_xgb_scs / thermal ablation) ──
def build_features(ds, dyn_idx=None):
    """[N*H*W, T*len(dyn_idx)+C_stat] window features (dynamic first, static last)."""
    X_dyn = ds.X_dynamic if dyn_idx is None else ds.X_dynamic[:, :, :, :, dyn_idx]
    X_stat = ds.X_static
    N, T, H, W, Cp = X_dyn.shape
    C_stat = X_stat.shape[-1]
    dyn_flat = X_dyn.reshape(N, T, H * W, Cp).transpose(0, 2, 1, 3)
    dyn_feat = dyn_flat.reshape(N, H * W, T * Cp)
    stat_feat = X_stat.reshape(N, H * W, C_stat)
    return np.concatenate([dyn_feat, stat_feat], axis=2).reshape(
        N * H * W, T * Cp + C_stat).astype(np.float32)


def forecast_targets(ds, T, L):
    y_full = ds.y_regression_full
    H, W = ds.n_lat, ds.n_lon
    y = np.full((len(ds.window_indices), H, W), np.nan, dtype=np.float32)
    for i, s in enumerate(ds.window_indices):
        li = s + T - 1 + L
        if 0 <= li < ds.n_dates:
            y[i] = y_full[li]
    return y


def eval_arrays(ds):
    """Masked (X, y, severe_bin) for the dataset's own test-year windows."""
    T = ds.history_weeks
    X = build_features(ds)
    y = forecast_targets(ds, T, LEAD_WEEKS).reshape(-1)
    m = ds.label_mask.reshape(-1) & ~np.isnan(y)
    return X[m], y[m]


def predict_auc(xgb, scaler, X, y):
    p = np.clip(xgb.predict(scaler.transform(X)), 0, 1)
    sev = (y >= SEVERE_THRESHOLD).astype(int)
    return float(roc_auc_score(sev, p)), p


def shuffle_channel_block(X, ch_idx, T=24, C=13, seed=0):
    """Shuffle the T-column block of dynamic channel ch_idx across samples.

    Same permutation applied to the whole block (channel values move together),
    destroying feature-target pairing and temporal structure while preserving
    the channel's marginal distribution — window-level equivalent of 7.4's
    per-pixel time-series shuffle.
    """
    Xs = X.copy()
    cols = [t * C + ch_idx for t in range(T)]
    block = Xs[:, cols]                       # [n, T]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(block.shape[0])
    Xs[:, cols] = block[perm]
    return Xs


# ── permutation importance (SCS and GBR) ────────────────────────
def permutation_importance(xgb, scaler, X, y, tag, n_shuffles=N_SHUFFLES):
    baseline_auc, _ = predict_auc(xgb, scaler, X, y)
    print(f"    Baseline AUC: {baseline_auc:.6f}")
    results = []
    for ch_idx, ch_name in enumerate(DYN_CHANNEL_NAMES):
        drops = []
        for seed in range(n_shuffles):
            Xp = shuffle_channel_block(X, ch_idx, seed=seed)
            auc_p, _ = predict_auc(xgb, scaler, Xp, y)
            drops.append(baseline_auc - auc_p)
        results.append({
            "channel": ch_name, "channel_index": ch_idx,
            "auc_drop_mean": float(np.mean(drops)),
            "auc_drop_std": float(np.std(drops)),
        })
        print(f"    [{ch_idx+1:2d}/{len(DYN_CHANNEL_NAMES)}] {ch_name:<22s} "
              f"ΔAUC={np.mean(drops):+.5f} ± {np.std(drops):.5f}")
    return baseline_auc, results


# ── native gain importance (XGB-only cross-check) ───────────────
def gain_importance(X, y, tag):
    """Fit same-params XGB on evaluation windows, return per-channel gain share."""
    xgb = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.08,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       n_jobs=-1, verbosity=0, importance_type="gain")
    xgb.fit(X, y)
    gain = xgb.feature_importances_          # normalized gain per feature column
    T, C = 24, 13
    per_ch = {DYN_CHANNEL_NAMES[c]: float(gain[[t * C + c for t in range(T)]].sum())
              for c in range(C)}
    per_ch["static (5 ch pooled)"] = float(gain[T * C:].sum())
    return per_ch


def classify_stability(scs_v, gbr_v):
    if scs_v > STABLE_TH and gbr_v > STABLE_TH:
        return "stable"
    if scs_v < -NEG_TH and gbr_v < -NEG_TH:
        return "negative_both"
    if abs(scs_v) < WEAK_TH and abs(gbr_v) < WEAK_TH:
        return "weak"
    if scs_v * gbr_v < 0:
        return "flip"
    return "moderate"


def main():
    print("=" * 70)
    print("XGB Cross-Domain Feature Importance (counterpart of exp 7.4)")
    print("=" * 70)

    xgb = joblib.load(XGB_WEIGHT_DIR / f"xgb_lead{LEAD_WEEKS}.joblib")
    scaler = joblib.load(XGB_WEIGHT_DIR / f"xgb_lead{LEAD_WEEKS}_scaler.joblib")
    print(f"Model: {XGB_WEIGHT_DIR / f'xgb_lead{LEAD_WEEKS}.joblib'} (deployed)")

    print("\n[1] SCS test windows...")
    ds_scs = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test",
                             dataset_file=SCS_FILE, **SPLIT_KW)
    X_scs, y_scs = eval_arrays(ds_scs)
    print(f"    n={len(y_scs)} severe={(y_scs >= SEVERE_THRESHOLD).sum()}")

    print("\n[2] GBR test windows...")
    ds_gbr = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test",
                             dataset_file=GBR_FILE, **SPLIT_KW)
    X_gbr, y_gbr = eval_arrays(ds_gbr)
    print(f"    n={len(y_gbr)} severe={(y_gbr >= SEVERE_THRESHOLD).sum()}")

    print("\n[3] SCS permutation importance...")
    scs_base, scs_results = permutation_importance(xgb, scaler, X_scs, y_scs, "SCS")
    print("\n[4] GBR permutation importance...")
    gbr_base, gbr_results = permutation_importance(xgb, scaler, X_gbr, y_gbr, "GBR")

    print("\n[5] Native gain importance (cross-check)...")
    t0 = time.time()
    scs_gain = gain_importance(X_scs, y_scs, "SCS")
    gbr_gain = gain_importance(X_gbr, y_gbr, "GBR")
    print(f"    done ({time.time()-t0:.0f}s)")

    # ── comparison ──
    scs_d = {r["channel"]: r["auc_drop_mean"] for r in scs_results}
    gbr_d = {r["channel"]: r["auc_drop_mean"] for r in gbr_results}
    comparison = []
    for c in DYN_CHANNEL_NAMES:
        comparison.append({
            "channel": c,
            "thermal": c in THERMAL_NAMES,
            "scs_auc_drop": scs_d[c], "gbr_auc_drop": gbr_d[c],
            "scs_gain": scs_gain[c], "gbr_gain": gbr_gain[c],
            "stability": classify_stability(scs_d[c], gbr_d[c]),
        })
    comparison.sort(key=lambda r: abs(r["scs_auc_drop"]) + abs(r["gbr_auc_drop"]),
                    reverse=True)

    from scipy.stats import spearmanr
    rho_perm, p_perm = spearmanr(
        [abs(r["scs_auc_drop"]) for r in comparison],
        [abs(r["gbr_auc_drop"]) for r in comparison])
    rho_gain, p_gain = spearmanr(
        [r["scs_gain"] for r in comparison], [r["gbr_gain"] for r in comparison])
    rho_xperm, _ = spearmanr(
        [abs(r["scs_auc_drop"]) for r in comparison], [r["scs_gain"] for r in comparison])
    print(f"\nSpearman |ΔAUC| SCS vs GBR: rho={rho_perm:.3f} (p={p_perm:.4f})")
    print(f"Spearman gain  SCS vs GBR: rho={rho_gain:.3f} (p={p_gain:.4f})")
    print(f"Spearman SCS ΔAUC vs SCS gain: rho={rho_xperm:.3f}")

    stable = [r["channel"] for r in comparison if r["stability"] == "stable"]
    flip = [r["channel"] for r in comparison if r["stability"] == "flip"]
    weak = [r["channel"] for r in comparison if r["stability"] == "weak"]

    # ── save JSON ──
    out = {
        "experiment": "xgb_cross_domain_feature_importance",
        "description": "XGB counterpart of experiment 7.4: window-feature "
                       "permutation importance + native gain, SCS vs GBR",
        "created": datetime.now().isoformat(),
        "method": {
            "model": str(XGB_WEIGHT_DIR / f"xgb_lead{LEAD_WEEKS}.joblib"),
            "permutation": "shuffle T-column block per dynamic channel across "
                           "samples (5 seeds), ΔAUC of severe ranking",
            "gain": "per-channel gain share from same-params refit on each "
                    "domain's evaluation windows (probing only)",
            "n_shuffles": N_SHUFFLES,
            "split": {"test": [2016, 2019]},
        },
        "scs_baseline_auc": scs_base, "gbr_baseline_auc": gbr_base,
        "scs_permutation": scs_results, "gbr_permutation": gbr_results,
        "scs_gain": scs_gain, "gbr_gain": gbr_gain,
        "feature_comparison": comparison,
        "stable_features": stable, "flip_features": flip, "weak_features": weak,
        "spearman": {
            "perm_scs_vs_gbr": {"rho": float(rho_perm), "p": float(p_perm)},
            "gain_scs_vs_gbr": {"rho": float(rho_gain), "p": float(p_gain)},
            "scs_perm_vs_gain": {"rho": float(rho_xperm)},
        },
        "interpretation": {
            "stable": "Channels important in BOTH domains -> physically "
                      "consistent signal",
            "flip": "Sign flips between domains -> domain-specific association",
            "weak": "Near-zero importance in both domains",
        },
    }
    out_path = RESULTS_DIR / "thermal_attribution/xgboost_permutation_importance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}")

    # ── figure (3 panels, mirrors 7.4 layout) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.97, 'XGB Cross-Domain Feature Importance (permutation)',
             ha='center', va='top', fontsize=14, fontweight='bold')
    names = [r["channel"] for r in comparison]
    colors = []
    for r in comparison:
        s = r["stability"]
        colors.append({'stable': '#27AE60', 'flip': '#E74C3C',
                       'negative_both': '#E74C3C', 'weak': '#95A5A6'}.get(s, '#F39C12'))
    y_pos = np.arange(len(names))

    for ax, key, title in [
            (axes[0], "scs_auc_drop", f'(a) SCS (training domain)\nAUC={scs_base:.4f}'),
            (axes[1], "gbr_auc_drop", f'(b) GBR (target domain)\nAUC={gbr_base:.4f}')]:
        vals = [r[key] for r in comparison]
        ax.barh(y_pos, vals, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('ΔAUC (permutation importance)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(vals):
            if abs(v) > 0.002:
                ax.text(v + (0.002 if v >= 0 else -0.002), i, f'{v:+.3f}',
                        va='center', ha='left' if v >= 0 else 'right',
                        fontsize=7, fontweight='bold')

    ax = axes[2]
    for r in comparison:
        color = '#27AE60' if r["stability"] == 'stable' else (
            '#E74C3C' if r["stability"] in ('flip', 'negative_both') else (
                '#95A5A6' if r["stability"] == 'weak' else '#F39C12'))
        marker = 'o' if r["stability"] == 'stable' else (
            'X' if r["stability"] == 'flip' else ('v' if r["stability"] == 'negative_both' else 's'))
        ax.scatter(r["scs_auc_drop"], r["gbr_auc_drop"], c=color, marker=marker,
                   s=100, alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)
        ax.annotate(r["channel"], (r["scs_auc_drop"], r["gbr_auc_drop"]),
                    fontsize=7, alpha=0.85,
                    xytext=(5, 3), textcoords='offset points')
    ax.axhline(0, color='black', lw=0.6, alpha=0.4)
    ax.axvline(0, color='black', lw=0.6, alpha=0.4)
    ax.set_xlabel('SCS ΔAUC', fontsize=10)
    ax.set_ylabel('GBR ΔAUC', fontsize=10)
    ax.set_title(f'(c) Stability scatter\nSpearman ρ={rho_perm:.3f}',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    fig_path = RESULTS_DIR / "thermal_attribution/xgboost_permutation_importance.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {fig_path}")

    print("\nDONE")
    print(f"  stable: {stable}")
    print(f"  flip:   {flip}")
    print(f"  weak:   {weak}")


if __name__ == "__main__":
    main()
