# ==========================================================================
# Experiment: XGBoost SCS training (12 lead-time models)
# Paper: Methods (XGBoost baseline), Results 1
# Purpose: 30k-pixel subsample, train-only StandardScaler, lead-4 targets
# Output: results/xgb_retrained_20260911/
# Random seed: 42
# ==========================================================================

# -*- coding: utf-8 -*-
"""SCS per-lead XGBoost (12 leads) — retrained 2026-08-01 on causal SCS data.

Same logic as train_xgb_gbr.py but on SCS dataset.
- StandardScaler fit on TRAIN only (no normalization leakage)
- Output to paper/a7_dual_output/results/ (NOT backup)
- n_estimators=300, max_depth=7, lr=0.08 (matches original SCS XGB params)

Usage:
    piecf_env/Scripts/python.exe paper/gbr_retrain/train_xgboost_scs.py
"""
import os, sys, io, json, time, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

_PROJECT_FILE = Path(__file__)
PROJECT_ROOT = _PROJECT_FILE.resolve().parents[2]  # LTTS/experiments/01_training -> LTTS root
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
WEIGHT_DIR = PROJECT_ROOT / "results" / "xgb_retrained_20260911"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aligned_dataset import SCSPIECFDataset

SCS_DATASET_FILE = "scs_coral_bleaching_dataset_v9_1deg_raw.nc"
XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 7, "learning_rate": 0.08,
    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
    "n_jobs": -1, "verbosity": 0,
}
MAX_TR = 30000
LEADS = list(range(1, 13))


def build_features(ds):
    """Build [N*H*W, T*C_dyn + C_stat] feature matrix from dataset."""
    X_dyn = ds.X_dynamic
    X_stat = ds.X_static
    N, T, H, W, C_dyn = X_dyn.shape
    C_stat = X_stat.shape[-1]
    dyn_flat = X_dyn.reshape(N, T, H * W, C_dyn).transpose(0, 2, 1, 3)
    dyn_feat = dyn_flat.reshape(N, H * W, T * C_dyn)
    stat_feat = X_stat.reshape(N, H * W, C_stat)
    X = np.concatenate([dyn_feat, stat_feat], axis=2).reshape(
        N * H * W, T * C_dyn + C_stat).astype(np.float32)
    return X, N, H, W


def forecast_targets(ds, T, L):
    """TRUE forecasting target: y_full[start + T - 1 + L] (L weeks AFTER window end).

    This matches PFV evaluation (y_hat(t+L) vs y(t+L)). Previously used
    y_regression[:, L-1] which is window-INTERNAL week (autoencoding) — off by T weeks.
    """
    y_full = ds.y_regression_full  # [n_dates, H, W]
    H, W = ds.n_lat, ds.n_lon
    nW = len(ds.window_indices)
    y = np.full((nW, H, W), np.nan, dtype=np.float32)
    for i, s in enumerate(ds.window_indices):
        li = s + T - 1 + L
        if 0 <= li < ds.n_dates:
            y[i] = y_full[li]
    return y


def main():
    print("=" * 60)
    print("SCS XGBoost retraining (12 leads, causal data, train-only scaler)")
    print("=" * 60)
    print(f"Dataset: {SCS_DATASET_FILE}")
    print(f"Weights: {WEIGHT_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nLoading datasets...")
    split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                    val_years=(2011, 2015), test_years=(2016, 2019))
    ds_train = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="train",
                               dataset_file=SCS_DATASET_FILE, **split_kw)
    ds_val   = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val",
                               dataset_file=SCS_DATASET_FILE, **split_kw)
    ds_test  = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test",
                               dataset_file=SCS_DATASET_FILE, **split_kw)
    print(f"  train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")
    print(f"  grid={ds_train.n_lat}x{ds_train.n_lon}  C_dyn={ds_train.C_dyn}  C_stat={ds_train.C_stat}")

    X_tr, N_tr, H_tr, W_tr = build_features(ds_train)
    X_va, N_va, H_va, W_va = build_features(ds_val)
    X_te, N_te, H_te, W_te = build_features(ds_test)
    T = ds_train.history_weeks
    C_dyn = ds_train.C_dyn
    C_stat = ds_train.C_stat
    print(f"  Feature dim: {T}*{C_dyn} + {C_stat} = {T*C_dyn + C_stat}")

    all_results = {}
    for L in LEADS:
        t0 = time.time()
        # FORECASTING target: y_full[start + T - 1 + L] (L weeks AFTER window end)
        # matches PFV evaluation; old y_regression[:, L-1] was window-internal (off by T)
        y_tr = forecast_targets(ds_train, T, L).reshape(N_tr * H_tr * W_tr)
        y_va = forecast_targets(ds_val, T, L).reshape(N_va * H_va * W_va)
        y_te = forecast_targets(ds_test, T, L).reshape(N_te * H_te * W_te)

        m_tr = ds_train.label_mask.reshape(N_tr * H_tr * W_tr) & ~np.isnan(y_tr)
        m_va = ds_val.label_mask.reshape(N_va * H_va * W_va) & ~np.isnan(y_va)
        m_te = ds_test.label_mask.reshape(N_te * H_te * W_te) & ~np.isnan(y_te)

        X_tr_v = X_tr[m_tr]
        y_tr_v = np.where(m_tr, y_tr, 0.0).astype(np.float32)[m_tr]
        X_va_v = X_va[m_va]
        y_va_v = np.where(m_va, y_va, 0.0).astype(np.float32)[m_va]
        X_te_v = X_te[m_te]
        y_te_v = np.where(m_te, y_te, 0.0).astype(np.float32)[m_te]

        if X_tr_v.shape[0] > MAX_TR:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_tr_v.shape[0], MAX_TR, replace=False)
            X_tr_v, y_tr_v = X_tr_v[idx], y_tr_v[idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_v)   # train-only fit
        X_va_sc = scaler.transform(X_va_v)
        X_te_sc = scaler.transform(X_te_v)

        xgb = XGBRegressor(**XGB_PARAMS)
        xgb.fit(X_tr_sc, y_tr_v)
        pred_va = np.clip(xgb.predict(X_va_sc), 0, 1).astype(np.float32)
        pred_te = np.clip(xgb.predict(X_te_sc), 0, 1).astype(np.float32)

        def r2(p, t):
            if len(t) < 5: return float('nan')
            ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
        r2_val = r2(pred_va, y_va_v); r2_test = r2(pred_te, y_te_v)

        joblib.dump(xgb, WEIGHT_DIR / f"xgb_lead{L}.joblib")
        joblib.dump(scaler, WEIGHT_DIR / f"xgb_lead{L}_scaler.joblib")

        fit_time = time.time() - t0
        all_results[f"lead{L}"] = {
            "r2_val": r2_val, "r2_test": r2_test,
            "n_train": int(X_tr_v.shape[0]), "n_val": int(X_va_v.shape[0]),
            "n_test": int(X_te_v.shape[0]), "fit_time": fit_time,
        }
        print(f"  Lead {L:2d}: val_R2={r2_val:+.4f}  test_R2={r2_test:+.4f}  "
              f"n_tr={X_tr_v.shape[0]:6d}  t={fit_time:.1f}s")

    summary_path = WEIGHT_DIR / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"created": datetime.now().isoformat(),
                   "dataset": SCS_DATASET_FILE,
                   "xgb_params": XGB_PARAMS,
                   "results": all_results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved to {summary_path}")
    print(f"Weights saved to {WEIGHT_DIR}")
    print("\n=== SCS XGBoost RETRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
