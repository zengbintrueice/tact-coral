# ==========================================================================
# Source: paper/a7_dual_output/xgboost_thermal_ablation.py (verbatim except path fix)
# Experiment: XGBoost thermal ablation (retrained feature subsets)
# Paper: Results 3, Fig. 4e, Supplementary Fig. 4, Supplementary Table S6
# Output: results/xgboost_thermal_ablation.json
# ==========================================================================

# -*- coding: utf-8 -*-
"""xgboost_thermal_ablation.py — XGBoost physics-informed feature ablation.

Tree-model counterpart of experiment7_5_remove_thermal.py (PIECF inference-time
channel zeroing). XGBoost predicts per pixel independently, so a proper feature
ablation RETRAINS the model on a feature subset (not just zeroing at inference),
using the exact same pipeline as paper/gbr_retrain/train_xgb_scs.py:

  - SCS train 2003-2010 (30k-pixel subsample, seed 42), train-only StandardScaler
  - lead-4 true forecasting target (y at window end + 4 weeks)
  - xgb_params identical: 300 trees, depth 7, lr 0.08
  - zero-shot GBR test 2016-2019 evaluation (AUC / CSI / POD / FAR)

Feature subsets (channel indices in the 13-dynamic list; static 5 always kept):
  XGB-full        : all 13 dynamic channels            (reference, matches
                    results/xgb_retrained_20260911 — retrained here for a
                    like-for-like comparison)
  XGB-thermal-only: thermal channels only [1,2,9,10,11]
                    (ssta, hotspot, d_hotspot_dt, hotspot_cumsum_4w,
                     hotspot_cumsum_12w) + static
  XGB-no-thermal  : non-thermal channels [0,3,4,5,6,7,8,12]
                    (sst, chlorophyll, par, seasonal_sin, wind_speed, ssr,
                     sst_anomaly, seasonal_cos) + static

Thermal-channel definition mirrors experiment7_5_remove_thermal.py exactly.

Outputs: results/xgboost_thermal_ablation.json (+ console table)

Run:
  & 'D:\...\piecf_env\Scripts\python.exe' xgboost_thermal_ablation.py
"""
import os, sys, io, json, time, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, DATA_DIR, RESULTS_DIR, XGB_WEIGHT_DIR

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

from src.data.aligned_dataset import SCSPIECFDataset

# ── constants (identical to train_xgb_scs.py) ──────────────────
SCS_FILE = "scs_coral_bleaching_dataset_v9_1deg_raw.nc"
GBR_FILE = "gbr_coral_bleaching_dataset_v9_1deg_raw.nc"
XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 7, "learning_rate": 0.08,
    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
    "n_jobs": -1, "verbosity": 0,
}
MAX_TR = 30000
LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
SPLIT_KW = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))

DYN_CHANNEL_NAMES = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par',
                     'seasonal_sin', 'wind_speed', 'ssr', 'sst_anomaly',
                     'd_hotspot_dt', 'hotspot_cumsum_4w',
                     'hotspot_cumsum_12w', 'seasonal_cos']
THERMAL_INDICES = [1, 2, 9, 10, 11]                       # = experiment7_5
NON_THERMAL_INDICES = [i for i in range(13) if i not in THERMAL_INDICES]

SUBSETS = {
    "XGB-full": list(range(13)),
    "XGB-thermal-only": THERMAL_INDICES,
    "XGB-no-thermal": NON_THERMAL_INDICES,
}


# ── feature builder (subset-aware port of train_xgb_scs.build_features) ──
def build_features(ds, dyn_idx):
    """[N*H*W, T*len(dyn_idx) + C_stat]; dynamic channels restricted to dyn_idx."""
    X_dyn = ds.X_dynamic[:, :, :, :, dyn_idx]              # [N,T,H,W,C']
    X_stat = ds.X_static
    N, T, H, W, Cp = X_dyn.shape
    C_stat = X_stat.shape[-1]
    dyn_flat = X_dyn.reshape(N, T, H * W, Cp).transpose(0, 2, 1, 3)
    dyn_feat = dyn_flat.reshape(N, H * W, T * Cp)
    stat_feat = X_stat.reshape(N, H * W, C_stat)
    X = np.concatenate([dyn_feat, stat_feat], axis=2).reshape(
        N * H * W, T * Cp + C_stat).astype(np.float32)
    return X, N, H, W


def forecast_targets(ds, T, L):
    """y_full[start + T - 1 + L] — same PFV target as train_xgb_scs."""
    y_full = ds.y_regression_full
    H, W = ds.n_lat, ds.n_lon
    nW = len(ds.window_indices)
    y = np.full((nW, H, W), np.nan, dtype=np.float32)
    for i, s in enumerate(ds.window_indices):
        li = s + T - 1 + L
        if 0 <= li < ds.n_dates:
            y[i] = y_full[li]
    return y


def train_subset(ds_train, ds_val, dyn_idx, tag):
    """Train one XGB on the channel subset; returns (model, scaler)."""
    T = ds_train.history_weeks
    X_tr, N_tr, H_tr, W_tr = build_features(ds_train, dyn_idx)
    X_va, N_va, H_va, W_va = build_features(ds_val, dyn_idx)

    y_tr = forecast_targets(ds_train, T, LEAD_WEEKS).reshape(N_tr * H_tr * W_tr)
    y_va = forecast_targets(ds_val, T, LEAD_WEEKS).reshape(N_va * H_va * W_va)
    m_tr = ds_train.label_mask.reshape(-1) & ~np.isnan(y_tr)
    m_va = ds_val.label_mask.reshape(-1) & ~np.isnan(y_va)

    X_tr_v = X_tr[m_tr]
    y_tr_v = np.where(m_tr, y_tr, 0.0).astype(np.float32)[m_tr]
    X_va_v = X_va[m_va]
    y_va_v = np.where(m_va, y_va, 0.0).astype(np.float32)[m_va]

    if X_tr_v.shape[0] > MAX_TR:                    # same subsample as full model
        rng = np.random.default_rng(42)
        idx = rng.choice(X_tr_v.shape[0], MAX_TR, replace=False)
        X_tr_v, y_tr_v = X_tr_v[idx], y_tr_v[idx]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_v)          # train-only fit
    xgb = XGBRegressor(**XGB_PARAMS)
    t0 = time.time()
    xgb.fit(X_tr_sc, y_tr_v)

    # SCS val sanity (R2, like train_xgb_scs)
    X_va_sc = scaler.transform(X_va_v)
    p_va = np.clip(xgb.predict(X_va_sc), 0, 1)
    ss_res = np.sum((y_va_v - p_va) ** 2)
    ss_tot = np.sum((y_va_v - y_va_v.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    print(f"  [{tag}] trained {len(y_tr_v)} px, dim={X_tr_sc.shape[1]}, "
          f"SCS val R2={r2:.4f}  ({time.time()-t0:.0f}s)")
    return xgb, scaler, r2


def predict_flat(xgb, scaler, ds, dyn_idx):
    """Predict + (pred, target, mask) flat arrays over dataset windows."""
    T = ds.history_weeks
    X, N, H, W = build_features(ds, dyn_idx)
    y = forecast_targets(ds, T, LEAD_WEEKS).reshape(-1)
    m = ds.label_mask.reshape(-1) & ~np.isnan(y)
    X_v = X[m]
    p = np.clip(xgb.predict(scaler.transform(X_v)), 0, 1).astype(np.float32)
    t = np.where(m, y, 0.0).astype(np.float32)[m]
    return p, t


def find_best_threshold(pred, target):
    """SCS val threshold sweep, argmax CSI (same grid as component ablation)."""
    severe_t = (target >= SEVERE_THRESHOLD).astype(int)
    best_csi, best_thr = -1.0, 0.5
    for thr in np.arange(0.50, 0.96, 0.05):
        severe_p = (pred >= thr).astype(int)
        hits = int(((severe_p == 1) & (severe_t == 1)).sum())
        fas = int(((severe_p == 1) & (severe_t == 0)).sum())
        misses = int(((severe_p == 0) & (severe_t == 1)).sum())
        denom = hits + fas + misses
        csi = hits / denom if denom > 0 else 0.0
        if csi > best_csi:
            best_csi, best_thr = csi, float(thr)
    return best_thr


def gbr_metrics(pred, target, threshold):
    """Zero-shot GBR test metrics at fixed threshold."""
    severe_t = (target >= SEVERE_THRESHOLD).astype(int)
    severe_p = (pred >= threshold).astype(int)
    hits = int(((severe_p == 1) & (severe_t == 1)).sum())
    misses = int(((severe_p == 0) & (severe_t == 1)).sum())
    fas = int(((severe_p == 1) & (severe_t == 0)).sum())
    denom = hits + fas + misses
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(severe_t, pred)) if severe_t.sum() >= 2 else None,
        "csi": hits / denom if denom > 0 else 0.0,
        "pod": hits / (hits + misses) if (hits + misses) > 0 else 0.0,
        "far": fas / (hits + fas) if (hits + fas) > 0 else 0.0,
        "n_valid": int(len(pred)), "n_severe": int(severe_t.sum()),
    }


# ── main ───────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("XGBoost physics-informed feature ablation (thermal / no-thermal)")
    print("=" * 70)

    print("\n[Data] SCS train/val + GBR test...")
    ds_train = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="train",
                               dataset_file=SCS_FILE, **SPLIT_KW)
    ds_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val",
                             dataset_file=SCS_FILE, **SPLIT_KW)
    ds_gbr = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test",
                             dataset_file=GBR_FILE, **SPLIT_KW)

    results = {}
    for tag, dyn_idx in SUBSETS.items():
        print(f"\n{'=' * 60}\n[{tag}] dynamic channels: "
              f"{[DYN_CHANNEL_NAMES[i] for i in dyn_idx]}")
        xgb, scaler, r2_val = train_subset(ds_train, ds_val, dyn_idx, tag)

        print("  SCS val: threshold sweep...")
        p_va, t_va = predict_flat(xgb, scaler, ds_val, dyn_idx)
        thr = find_best_threshold(p_va, t_va)

        print(f"  GBR test: zero-shot evaluation (tau={thr:.2f})...")
        p_g, t_g = predict_flat(xgb, scaler, ds_gbr, dyn_idx)
        m = gbr_metrics(p_g, t_g, thr)
        print(f"  GBR: AUC={m['auc']:.4f}  CSI={m['csi']:.4f}  "
              f"POD={m['pod']:.4f}  FAR={m['far']:.4f}  "
              f"(n={m['n_valid']}, severe={m['n_severe']})")

        results[tag] = {
            "dynamic_channels": [DYN_CHANNEL_NAMES[i] for i in dyn_idx],
            "n_dynamic_channels": len(dyn_idx),
            "feature_dim": int(len(dyn_idx) * ds_train.history_weeks
                               + ds_train.C_stat),
            "scs_val_r2": r2_val,
            "scs_val_threshold": thr,
            "gbr": m,
        }

    # ── reference row: deployed full model (xgb_retrained_20260911) ──
    ref_model = joblib.load(XGB_WEIGHT_DIR / f"xgb_lead{LEAD_WEEKS}.joblib")
    ref_scaler = joblib.load(XGB_WEIGHT_DIR / f"xgb_lead{LEAD_WEEKS}_scaler.joblib")
    p_va, t_va = predict_flat(ref_model, ref_scaler, ds_val, list(range(13)))
    thr_ref = find_best_threshold(p_va, t_va)
    p_g, t_g = predict_flat(ref_model, ref_scaler, ds_gbr, list(range(13)))
    m_ref = gbr_metrics(p_g, t_g, thr_ref)
    results["XGB-full-deployed"] = {
        "dynamic_channels": DYN_CHANNEL_NAMES,
        "n_dynamic_channels": 13,
        "note": "deployed weights results/xgb_retrained_20260911 (reference)",
        "scs_val_threshold": thr_ref,
        "gbr": m_ref,
    }
    print(f"\n[XGB-full-deployed] GBR: AUC={m_ref['auc']:.4f}  "
          f"CSI={m_ref['csi']:.4f}  (tau={thr_ref:.2f})")

    # ── save ──
    out = {
        "experiment": "xgb_physics_informed_feature_ablation",
        "description": "XGBoost feature ablation: thermal-only vs no-thermal, "
                       "retrained per subset (same pipeline as train_xgb_scs), "
                       "zero-shot SCS->GBR evaluation",
        "created": datetime.now().isoformat(),
        "configuration": {
            "lead_weeks": LEAD_WEEKS, "severe_threshold": SEVERE_THRESHOLD,
            "max_train_pixels": MAX_TR, "xgb_params": XGB_PARAMS,
            "split": {"train": [2003, 2010], "val": [2011, 2015],
                      "test": [2016, 2019]},
            "thermal_indices": THERMAL_INDICES,
            "thermal_names": [DYN_CHANNEL_NAMES[i] for i in THERMAL_INDICES],
            "thermal_definition_source": "experiment7_5_remove_thermal.py",
            "data_files": {"scs": SCS_FILE, "gbr": GBR_FILE},
        },
        "results": results,
        "summary_table": {
            "columns": ["Model", "GBR AUC", "GBR CSI"],
            "rows": [[tag, results[tag]["gbr"]["auc"], results[tag]["gbr"]["csi"]]
                     for tag in ["XGB-full-deployed", "XGB-full",
                                 "XGB-thermal-only", "XGB-no-thermal"]],
        },
    }
    out_path = RESULTS_DIR / "thermal_attribution/xgboost_thermal_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}")

    # ── console table ──
    print("\n" + "=" * 60)
    print(f"{'Model':<22}{'GBR AUC':>10}{'GBR CSI':>10}")
    print("-" * 60)
    for tag in ["XGB-full-deployed", "XGB-full",
                "XGB-thermal-only", "XGB-no-thermal"]:
        g = results[tag]["gbr"]
        print(f"{tag:<22}{g['auc']:>10.4f}{g['csi']:>10.4f}")
    print("=" * 60)
    print("\nDONE")


if __name__ == "__main__":
    main()
