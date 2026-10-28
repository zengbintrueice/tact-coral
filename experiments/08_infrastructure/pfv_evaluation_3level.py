# ==========================================================================
# Source: paper/a7_dual_output/pfv_evaluation_3level.py (verbatim except path fix)
# Experiment: 3-level Pure Forecasting Verification (pixel/classification/event)
# Paper: Methods (Statistical analysis), Reproducibility
# Purpose: sliding-window PFV metrics (RMSE, severe Brier) for a6 vs XGBoost
# Output: results/anchored_global_pfv.json
# ==========================================================================

# pfv_evaluation_3level.py — Pure Forecasting Verification (3-level)
#
#   PROBLEM: The original "All-Event Anchored" evaluation suffered from a
#     position-dependent artifact: a6 is ONLY supervised at position T-1
#     (predicts lead_weeks=4 target), but the anchored eval used ALL positions
#     (0..T-2 unsupervised). Severe prediction mean drifted from 0.05 (pos 0)
#     to 0.85 (pos T-1), making the anchored curves unreliable.
#
#   SOLUTION: Pure Forecasting Verification — the gold standard that matches
#     the training objective EXACTLY:
#     - At time t, model sees window [t-T+1, t] and predicts y(t+4)
#     - We verify: y_hat(t+4) vs y(t+4) actual observation
#     - No peak-anchoring, no position artifact, no training-eval mismatch
#
#   THREE EVALUATION LEVELS:
#     Level 1 (Pixel-level forecasting):
#       All valid pixels: RMSE(y_hat(t+4), y(t+4)), Brier, severe Brier
#     Level 2 (Severe classification):
#       For y > 0.8: CSI, F1, POD, FAR (binary severe-event detection)
#     Level 3 (Event-centered):
#       For each bleaching peak: y_hat(t_peak - 4) evaluates the model's
#       ability to predict the peak 4 weeks in advance
#
#   ANTI-LEAKAGE GUARANTEES:
#     - year-based split: test = 2016-2019, no temporal overlap with train
#     - sliding-window: each time point gets EXACTLY ONE verified prediction
#     - no event double-counting: each (t, i, j) verified once
#
#   Saves to results/anchored_global_pfv.json + pfv_classification.png
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' pfv_evaluation_3level.py

import os, sys, io, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib

# a7 shared paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import A7_DIR, PIECF_ROOT, A6_WEIGHT, XGB_WEIGHT_DIR, DATA_DIR, RESULTS_DIR

WEIGHT_DIR = XGB_WEIGHT_DIR

from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import get_year_range_for_mode

# lead_weeks=4 retrained weights (true forecasting, not autoencoding)
A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}

LEAD_WEEKS = 4       # matches training lead_weeks
SEVERE_THRESHOLD = 0.8  # y >= 0.8 → severe zone
PRED_THRESHOLD = 0.5    # prediction threshold for binary classification


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════
def rmse_calc(p, t, m):
    m = m > 0.5
    if m.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))


def brier_calc(p, t, m):
    m = m > 0.5
    if m.sum() < 5:
        return float("nan")
    return float(np.mean((p[m] - t[m]) ** 2))


def classification_metrics(pred, actual, severe_thresh, pred_thresh):
    """Compute CSI, POD, FAR, F1 for binary severe-event detection."""
    pred_bin = pred >= pred_thresh
    actual_bin = actual >= severe_thresh

    hits = np.sum(pred_bin & actual_bin)
    misses = np.sum(~pred_bin & actual_bin)
    false_alarms = np.sum(pred_bin & ~actual_bin)
    correct_negatives = np.sum(~pred_bin & ~actual_bin)

    denom_csi = hits + misses + false_alarms
    csi = float(hits / denom_csi) if denom_csi > 0 else float("nan")
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else float("nan")
    denom_f1 = 2 * hits + false_alarms + misses
    f1 = float(2 * hits / denom_f1) if denom_f1 > 0 else float("nan")
    accuracy = float((hits + correct_negatives) / len(pred)) if len(pred) > 0 else float("nan")

    return {
        "csi": csi, "pod": pod, "far": far, "f1": f1,
        "accuracy": accuracy,
        "n_hits": int(hits), "n_misses": int(misses),
        "n_false_alarms": int(false_alarms), "n_correct_negatives": int(correct_negatives),
        "n_total": int(len(pred)),
    }


# ═══════════════════════════════════════════════════════════════════
# Peak detection: find ALL local maxima in DHW time series per cell
# ═══════════════════════════════════════════════════════════════════
def find_all_peaks_full(y_full, min_prominence=0.02):
    """Find all local peak events in the FULL [n_dates, H, W] time series.

    Peaks are detected in absolute time (not per-window). A peak at absolute
    time t_peak can be directly indexed into forecast_map[t_peak - h].

    y_full: [n_dates, H, W] DHW values (NaN where no data)
    Returns peak_list: list of dicts with (t_peak, i, j, peak_value)
    """
    n_dates, H, W = y_full.shape
    peaks = []

    for i in range(H):
        for j in range(W):
            ts = y_full[:, i, j]
            valid = ~np.isnan(ts)
            if valid.sum() < 3:
                continue
            for t in range(1, n_dates - 1):
                if np.isnan(ts[t]) or np.isnan(ts[t - 1]) or np.isnan(ts[t + 1]):
                    continue
                if ts[t] > ts[t - 1] and ts[t] > ts[t + 1]:
                    left_min = ts[t]
                    for k in range(t - 1, -1, -1):
                        if np.isnan(ts[k]):
                            break
                        left_min = min(left_min, ts[k])
                        if ts[k] > ts[t]:
                            break
                    right_min = ts[t]
                    for k in range(t + 1, n_dates):
                        if np.isnan(ts[k]):
                            break
                        right_min = min(right_min, ts[k])
                        if ts[k] > ts[t]:
                            break
                    prominence = ts[t] - max(left_min, right_min)
                    if prominence >= min_prominence:
                        peaks.append({
                            't_peak': t,
                            'i': i,
                            'j': j,
                            'peak_value': float(ts[t]),
                            'prominence': float(prominence),
                        })
    return peaks


# ═══════════════════════════════════════════════════════════════════
# Prediction helpers
# ═══════════════════════════════════════════════════════════════════
def restore_a5():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_sliding_temporal(model, ds, device):
    """Sliding-window evaluation: each time point gets its own T-week window.

    For each valid time point t (T-1 <= t < n_dates):
      - Input window = dynamic[t-T+1 : t+1]  (T weeks of history)
      - Model output y_final[:, -1] = prediction for week t
      - Only position T-1 (supervised) is used -> no position artifact

    Returns forecast_map: [n_dates, H, W]  (NaN where window incomplete)
    """
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn

    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X_full = ds.X_dynamic_full  # [n_dates, H, W, C_dyn]

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


def predict_xgb_at_windows(ds, xgb_models, xgb_scalers, t_evals, T, H, W):
    """Compute XGBoost predictions at arbitrary absolute window-end times.

    For each t_eval in t_evals, builds the T-week window ending at t_eval
    and runs all 12 lead models. Returns dict: pred[t_eval, L, i, j].
    """
    C_dyn = ds.C_dyn
    C_stat = ds.C_stat
    X_full = ds.X_dynamic_full
    X_stat = ds.X_static[0]

    pred_map = np.full((ds.n_dates, 12, H, W), np.nan, dtype=np.float32)

    for t_eval in t_evals:
        if t_eval < T - 1 or t_eval >= ds.n_dates:
            continue
        s = t_eval - T + 1
        dyn_win = X_full[s:s + T]
        dyn_flat = dyn_win.transpose(1, 2, 0, 3).reshape(H * W, T * C_dyn)
        stat_flat = X_stat.reshape(H * W, C_stat)
        X = np.concatenate([dyn_flat, stat_flat], axis=1).astype(np.float32)

        for L in range(1, 13):
            scaler = xgb_scalers[L]
            model = xgb_models[L]
            X_s = scaler.transform(X)
            pred_te = np.clip(model.predict(X_s), 0, 1).astype(np.float32)
            pred_map[t_eval, L - 1, :, :] = pred_te.reshape(H, W)

    return pred_map


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Pure Forecasting Verification (3-level) — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))

# Load a6 (lead_weeks=4 retrained)
print("\nRestoring a6 sg_lam20 (13-ch, lead_weeks=4 retrained)...")
print(f"  Weight: {A5_MODEL}")
a5_model = restore_a5()
a5_model = a5_model.to(device)

# Load data (lead_weeks=4 for true forecasting)
print("\nLoading datasets...")
ds_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False, lead_weeks=4, **split_kw)
T, H, W = ds_test.history_weeks, ds_test.n_lat, ds_test.n_lon
n_dates = ds_test.n_dates
print(f"  T={T} H={H} W={W}  test={len(ds_test)}  n_dates={n_dates}")

# ═══════════════════════════════════════════════════════════════
# Sliding-window inference: forecast_map[n_dates, H, W]
# ═══════════════════════════════════════════════════════════════
print("\nGenerating sliding-window predictions (a6)...")
t0 = time.time()
forecast_map = predict_sliding_temporal(a5_model, ds_test, device)
print(f"  Done in {time.time()-t0:.1f}s  shape={forecast_map.shape}")

# ═══════════════════════════════════════════════════════════════
# LEVEL 1: Pixel-level Pure Forecasting Verification
# ═══════════════════════════════════════════════════════════════
# Gold standard: at time t, model predicts y(t+LEAD_WEEKS).
# We verify: y_hat(t+LEAD_WEEKS) vs y(t+LEAD_WEEKS) actual observation.
# Each (t, i, j) gets EXACTLY ONE verified prediction -> no double-counting.
print("\n" + "=" * 60)
print(f"LEVEL 1: Pixel-level Pure Forecasting Verification")
print(f"  y_hat(t+{LEAD_WEEKS}) vs y(t+{LEAD_WEEKS}) actual")
print("=" * 60)

y_full = ds_test.y_regression_full  # [n_dates, H, W]
year_range = get_year_range_for_mode(ds_test)

# a6
pfv_pred_a6, pfv_target_a6 = [], []
pfv_pred_a6_sv, pfv_target_a6_sv = [], []

for t in range(T - 1, n_dates):
    # Year filtering: only include time steps in the mode's year range
    if year_range is not None:
        year = int(ds_test.date_strings[t][:4])
        if not (year_range[0] <= year <= year_range[1]):
            continue
    t_target = t + LEAD_WEEKS
    if t_target >= n_dates:
        continue
    pred_t = forecast_map[t]
    target_t = y_full[t_target]
    valid = ~np.isnan(pred_t) & ~np.isnan(target_t)
    if valid.sum() < 1:
        continue
    pfv_pred_a6.extend(pred_t[valid].tolist())
    pfv_target_a6.extend(target_t[valid].tolist())
    sv_mask = valid & (target_t >= SEVERE_THRESHOLD)
    if sv_mask.sum() > 0:
        pfv_pred_a6_sv.extend(pred_t[sv_mask].tolist())
        pfv_target_a6_sv.extend(target_t[sv_mask].tolist())

pfv_pred_a6 = np.array(pfv_pred_a6, dtype=np.float32)
pfv_target_a6 = np.array(pfv_target_a6, dtype=np.float32)
pfv_pred_a6_sv = np.array(pfv_pred_a6_sv, dtype=np.float32)
pfv_target_a6_sv = np.array(pfv_target_a6_sv, dtype=np.float32)

a6_l1_rmse = rmse_calc(pfv_pred_a6, pfv_target_a6, np.ones_like(pfv_pred_a6))
a6_l1_brier = brier_calc(pfv_pred_a6, pfv_target_a6, np.ones_like(pfv_pred_a6))
a6_l1_sv_brier = brier_calc(pfv_pred_a6_sv, pfv_target_a6_sv, np.ones_like(pfv_pred_a6_sv))

print(f"\n  a6 Level 1 (Pixel-level, t+{LEAD_WEEKS}):")
print(f"    Global  RMSE={a6_l1_rmse:.4f}  Brier={a6_l1_brier:.5f}  n={len(pfv_pred_a6)}")
print(f"    Severe  Brier={a6_l1_sv_brier:.5f}  n_severe={len(pfv_pred_a6_sv)}")

# XGBoost lead-4 for fair comparison
print("\n  Loading XGBoost lead-4 model for comparison...")
xgb_model_l4 = joblib.load(WEIGHT_DIR / "xgb_lead4.joblib")
xgb_scaler_l4 = joblib.load(WEIGHT_DIR / "xgb_lead4_scaler.joblib")

C_dyn = ds_test.C_dyn
C_stat = ds_test.C_stat
X_stat = ds_test.X_static[0]

pfv_pred_xgb, pfv_target_xgb = [], []
pfv_pred_xgb_sv, pfv_target_xgb_sv = [], []

for t in range(T - 1, n_dates):
    # Year filtering
    if year_range is not None:
        year = int(ds_test.date_strings[t][:4])
        if not (year_range[0] <= year <= year_range[1]):
            continue
    t_target = t + LEAD_WEEKS
    if t_target >= n_dates:
        continue
    if t < T - 1 or t >= n_dates:
        continue
    s = t - T + 1
    dyn_win = ds_test.X_dynamic_full[s:s + T]
    dyn_flat = dyn_win.transpose(1, 2, 0, 3).reshape(H * W, T * C_dyn)
    stat_flat = X_stat.reshape(H * W, C_stat)
    X = np.concatenate([dyn_flat, stat_flat], axis=1).astype(np.float32)
    X_s = xgb_scaler_l4.transform(X)
    pred_t = np.clip(xgb_model_l4.predict(X_s), 0, 1).astype(np.float32).reshape(H, W)
    target_t = y_full[t_target]
    valid = ~np.isnan(pred_t) & ~np.isnan(target_t)
    if valid.sum() < 1:
        continue
    pfv_pred_xgb.extend(pred_t[valid].tolist())
    pfv_target_xgb.extend(target_t[valid].tolist())
    sv_mask = valid & (target_t >= SEVERE_THRESHOLD)
    if sv_mask.sum() > 0:
        pfv_pred_xgb_sv.extend(pred_t[sv_mask].tolist())
        pfv_target_xgb_sv.extend(target_t[sv_mask].tolist())

pfv_pred_xgb = np.array(pfv_pred_xgb, dtype=np.float32)
pfv_target_xgb = np.array(pfv_target_xgb, dtype=np.float32)
pfv_pred_xgb_sv = np.array(pfv_pred_xgb_sv, dtype=np.float32)
pfv_target_xgb_sv = np.array(pfv_target_xgb_sv, dtype=np.float32)

xgb_l1_rmse = rmse_calc(pfv_pred_xgb, pfv_target_xgb, np.ones_like(pfv_pred_xgb))
xgb_l1_brier = brier_calc(pfv_pred_xgb, pfv_target_xgb, np.ones_like(pfv_pred_xgb))
xgb_l1_sv_brier = brier_calc(pfv_pred_xgb_sv, pfv_target_xgb_sv, np.ones_like(pfv_pred_xgb_sv))

print(f"\n  xgb Level 1 (Pixel-level, lead={LEAD_WEEKS}):")
print(f"    Global  RMSE={xgb_l1_rmse:.4f}  Brier={xgb_l1_brier:.5f}  n={len(pfv_pred_xgb)}")
print(f"    Severe  Brier={xgb_l1_sv_brier:.5f}  n_severe={len(pfv_pred_xgb_sv)}")

level1_results = {
    "a6": {
        "rmse": float(a6_l1_rmse), "brier": float(a6_l1_brier),
        "severe_brier": float(a6_l1_sv_brier),
        "n_valid": int(len(pfv_pred_a6)), "n_severe": int(len(pfv_pred_a6_sv)),
    },
    "xgb_lead4": {
        "rmse": float(xgb_l1_rmse), "brier": float(xgb_l1_brier),
        "severe_brier": float(xgb_l1_sv_brier),
        "n_valid": int(len(pfv_pred_xgb)), "n_severe": int(len(pfv_pred_xgb_sv)),
    },
}

# ═══════════════════════════════════════════════════════════════
# LEVEL 2: Severe Classification Metrics
# ═══════════════════════════════════════════════════════════════
# Binary severe-event detection: y >= 0.8
# Metrics: CSI, POD, FAR, F1
print("\n" + "=" * 60)
print(f"LEVEL 2: Severe Classification (y >= {SEVERE_THRESHOLD})")
print("=" * 60)

a6_l2_cls = classification_metrics(pfv_pred_a6, pfv_target_a6, SEVERE_THRESHOLD, PRED_THRESHOLD)
xgb_l2_cls = classification_metrics(pfv_pred_xgb, pfv_target_xgb, SEVERE_THRESHOLD, PRED_THRESHOLD)

print(f"\n  a6 classification:")
print(f"    CSI={a6_l2_cls['csi']:.4f}  POD={a6_l2_cls['pod']:.4f}  FAR={a6_l2_cls['far']:.4f}  F1={a6_l2_cls['f1']:.4f}")
print(f"    Hits={a6_l2_cls['n_hits']}  Misses={a6_l2_cls['n_misses']}  FA={a6_l2_cls['n_false_alarms']}")
print(f"    n_total={a6_l2_cls['n_total']}")

print(f"\n  xgb classification:")
print(f"    CSI={xgb_l2_cls['csi']:.4f}  POD={xgb_l2_cls['pod']:.4f}  FAR={xgb_l2_cls['far']:.4f}  F1={xgb_l2_cls['f1']:.4f}")
print(f"    Hits={xgb_l2_cls['n_hits']}  Misses={xgb_l2_cls['n_misses']}  FA={xgb_l2_cls['n_false_alarms']}")
print(f"    n_total={xgb_l2_cls['n_total']}")

level2_results = {
    "a6": a6_l2_cls,
    "xgb_lead4": xgb_l2_cls,
}

# ═══════════════════════════════════════════════════════════════
# LEVEL 3: Event-centered Verification
# ═══════════════════════════════════════════════════════════════
# For each bleaching peak at (t_peak, i, j), evaluate the model's
# 4-week-ahead early warning: ŷ(t_peak - 4) at the same pixel.
# Anti-leakage: only peaks with t_peak >= T-1 + LEAD_WEEKS are used
# (so that ŷ(t_peak-4) has a valid sliding-window prediction AND
# the target y(t_peak) is in the test set).
print("\n" + "=" * 60)
print("LEVEL 3: Event-centered Verification (4-week-ahead warning)")
print("=" * 60)

peaks = find_all_peaks_full(y_full, min_prominence=0.02)
print(f"  Found {len(peaks)} peaks in test set")

# a6: for each peak, look up forecast_map[t_peak - LEAD_WEEKS]
ev_pred_a6, ev_target_a6 = [], []
ev_pred_a6_sv, ev_target_a6_sv = [], []

for pk in peaks:
    t_eval = pk['t_peak'] - LEAD_WEEKS
    i, j = pk['i'], pk['j']
    if t_eval < T - 1 or t_eval >= n_dates:
        continue
    pred_val = forecast_map[t_eval, i, j]
    if np.isnan(pred_val):
        continue
    ev_pred_a6.append(pred_val)
    ev_target_a6.append(pk['peak_value'])
    if pk['peak_value'] >= SEVERE_THRESHOLD:
        ev_pred_a6_sv.append(pred_val)
        ev_target_a6_sv.append(pk['peak_value'])

ev_pred_a6 = np.array(ev_pred_a6, dtype=np.float32)
ev_target_a6 = np.array(ev_target_a6, dtype=np.float32)
ev_pred_a6_sv = np.array(ev_pred_a6_sv, dtype=np.float32)
ev_target_a6_sv = np.array(ev_target_a6_sv, dtype=np.float32)

a6_l3_rmse = rmse_calc(ev_pred_a6, ev_target_a6, np.ones_like(ev_pred_a6))
a6_l3_brier = brier_calc(ev_pred_a6, ev_target_a6, np.ones_like(ev_pred_a6))
a6_l3_sv_brier = brier_calc(ev_pred_a6_sv, ev_target_a6_sv, np.ones_like(ev_pred_a6_sv))

print(f"\n  a6 event-centered (ŷ(t_peak-{LEAD_WEEKS}) vs peak):")
print(f"    Global  RMSE={a6_l3_rmse:.4f}  Brier={a6_l3_brier:.5f}  n_events={len(ev_pred_a6)}")
print(f"    Severe  Brier={a6_l3_sv_brier:.5f}  n_severe={len(ev_pred_a6_sv)}")

# XGBoost lead-4 event-centered
ev_pred_xgb, ev_target_xgb = [], []
ev_pred_xgb_sv, ev_target_xgb_sv = [], []

for pk in peaks:
    t_eval = pk['t_peak'] - LEAD_WEEKS
    i, j = pk['i'], pk['j']
    if t_eval < T - 1 or t_eval >= n_dates:
        continue
    s = t_eval - T + 1
    dyn_win = ds_test.X_dynamic_full[s:s + T]
    dyn_flat = dyn_win.transpose(1, 2, 0, 3).reshape(H * W, T * C_dyn)
    stat_flat = X_stat.reshape(H * W, C_stat)
    X = np.concatenate([dyn_flat, stat_flat], axis=1).astype(np.float32)
    X_s = xgb_scaler_l4.transform(X)
    pred_grid = np.clip(xgb_model_l4.predict(X_s), 0, 1).astype(np.float32).reshape(H, W)
    pred_val = pred_grid[i, j]
    if np.isnan(pred_val):
        continue
    ev_pred_xgb.append(pred_val)
    ev_target_xgb.append(pk['peak_value'])
    if pk['peak_value'] >= SEVERE_THRESHOLD:
        ev_pred_xgb_sv.append(pred_val)
        ev_target_xgb_sv.append(pk['peak_value'])

ev_pred_xgb = np.array(ev_pred_xgb, dtype=np.float32)
ev_target_xgb = np.array(ev_target_xgb, dtype=np.float32)
ev_pred_xgb_sv = np.array(ev_pred_xgb_sv, dtype=np.float32)
ev_target_xgb_sv = np.array(ev_target_xgb_sv, dtype=np.float32)

xgb_l3_rmse = rmse_calc(ev_pred_xgb, ev_target_xgb, np.ones_like(ev_pred_xgb))
xgb_l3_brier = brier_calc(ev_pred_xgb, ev_target_xgb, np.ones_like(ev_pred_xgb))
xgb_l3_sv_brier = brier_calc(ev_pred_xgb_sv, ev_target_xgb_sv, np.ones_like(ev_pred_xgb_sv))

print(f"\n  xgb event-centered (lead-{LEAD_WEEKS}):")
print(f"    Global  RMSE={xgb_l3_rmse:.4f}  Brier={xgb_l3_brier:.5f}  n_events={len(ev_pred_xgb)}")
print(f"    Severe  Brier={xgb_l3_sv_brier:.5f}  n_severe={len(ev_pred_xgb_sv)}")

level3_results = {
    "a6": {
        "rmse": float(a6_l3_rmse), "brier": float(a6_l3_brier),
        "severe_brier": float(a6_l3_sv_brier),
        "n_events": int(len(ev_pred_a6)), "n_severe": int(len(ev_pred_a6_sv)),
    },
    "xgb_lead4": {
        "rmse": float(xgb_l3_rmse), "brier": float(xgb_l3_brier),
        "severe_brier": float(xgb_l3_sv_brier),
        "n_events": int(len(ev_pred_xgb)), "n_severe": int(len(ev_pred_xgb_sv)),
    },
}

# Event-centered classification (Level 3 binary)
a6_l3_cls = classification_metrics(ev_pred_a6, ev_target_a6, SEVERE_THRESHOLD, PRED_THRESHOLD)
xgb_l3_cls = classification_metrics(ev_pred_xgb, ev_target_xgb, SEVERE_THRESHOLD, PRED_THRESHOLD)

print(f"\n  Event-centered classification:")
print(f"    a6:  CSI={a6_l3_cls['csi']:.4f}  POD={a6_l3_cls['pod']:.4f}  FAR={a6_l3_cls['far']:.4f}  F1={a6_l3_cls['f1']:.4f}")
print(f"    xgb: CSI={xgb_l3_cls['csi']:.4f}  POD={xgb_l3_cls['pod']:.4f}  FAR={xgb_l3_cls['far']:.4f}  F1={xgb_l3_cls['f1']:.4f}")

level3_results["a6"].update(a6_l3_cls)
level3_results["xgb_lead4"].update(xgb_l3_cls)

# ═══════════════════════════════════════════════════════════════
# Position Calibration Analysis (diagnostic for position artifact)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Position Calibration Analysis (diagnostic)")
print("=" * 60)

# Position calibration requires per-position model outputs which the
# sliding-window forecast_map does not store (only position T-1 is kept).
# This is intentional: Pure Forecasting Verification uses ONLY the
# supervised position T-1, so there is no position artifact to calibrate.
print("  (Pure Forecasting Verification uses only position T-1 — no artifact)")
print("  Position calibration deferred to dedicated script if needed.")

pos_cal = [{"position": p, "note": "deferred"} for p in range(T)]

# ═══════════════════════════════════════════════════════════════
# Save results JSON
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

pfv_results = {
    "method": "pure_forecasting_verification",
    "lead_weeks": LEAD_WEEKS,
    "severe_threshold": SEVERE_THRESHOLD,
    "pred_threshold": PRED_THRESHOLD,
    "model_weight": str(A5_MODEL),
    "level1_pixel": level1_results,
    "level2_classification": level2_results,
    "level3_event_centered": level3_results,
    "position_calibration": pos_cal,
    "timestamp": datetime.now().isoformat(),
}

out_json = RESULTS_DIR / "infrastructure/pfv_3level_metrics.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(pfv_results, f, indent=2, ensure_ascii=False)
print(f"  Saved: {out_json}")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY: Pure Forecasting Verification")
print("=" * 60)
print(f"\n  Level 1 (Pixel, t+{LEAD_WEEKS}):")
print(f"    a6   RMSE={a6_l1_rmse:.4f}  Brier={a6_l1_brier:.5f}  svBrier={a6_l1_sv_brier:.5f}")
print(f"    xgb  RMSE={xgb_l1_rmse:.4f}  Brier={xgb_l1_brier:.5f}  svBrier={xgb_l1_sv_brier:.5f}")
print(f"\n  Level 2 (Severe classification):")
print(f"    a6   CSI={a6_l2_cls['csi']:.4f}  POD={a6_l2_cls['pod']:.4f}  FAR={a6_l2_cls['far']:.4f}  F1={a6_l2_cls['f1']:.4f}")
print(f"    xgb  CSI={xgb_l2_cls['csi']:.4f}  POD={xgb_l2_cls['pod']:.4f}  FAR={xgb_l2_cls['far']:.4f}  F1={xgb_l2_cls['f1']:.4f}")
print(f"\n  Level 3 (Event-centered, ŷ(t_peak-{LEAD_WEEKS})):")
print(f"    a6   RMSE={a6_l3_rmse:.4f}  Brier={a6_l3_brier:.5f}  svBrier={a6_l3_sv_brier:.5f}")
print(f"    xgb  RMSE={xgb_l3_rmse:.4f}  Brier={xgb_l3_brier:.5f}  svBrier={xgb_l3_sv_brier:.5f}")
print(f"\n  Done.")
