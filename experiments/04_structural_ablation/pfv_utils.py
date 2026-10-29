# pfv_utils.py — Shared Pure Forecasting Verification utilities for a7
# LEGACY METRIC NAMING — see experiments/08_infrastructure/metric_semantics.py.
# The `brier` field written by this script is mean((pred - y_continuous)**2),
# i.e. a CONTINUOUS MSE, not the binary Brier score. It is kept only for
# backward compatibility. Canonical fields (mse_cont, score_reliability,
# brier_binary, ece_probability) live in
# results/qc/calibration_target_audit/metrics_v2.json. Do not use `brier`
# for new tables, figures or manuscript text.
#
#   Provides year-filtered PFV evaluation functions that correctly respect
#   the train/val/test split. Fixes the bug where predict_sliding_temporal()
#   + build_pfv_arrays() evaluated on ALL time steps regardless of mode.
#
#   Key functions:
#     - build_pfv_arrays(forecast_map, y_full, T, n_dates, year_range, date_strings)
#     - pixel_metrics(pred, target, severe_threshold)
#     - classification_metrics(pred, actual, severe_threshold, pred_threshold)
#
#   Usage:
#     from pfv_utils import build_pfv_arrays, pixel_metrics
#
#   Run:
#     (imported by other scripts)

import numpy as np
import pandas as pd


def build_pfv_arrays(forecast_map, y_full, T, n_dates, year_range=None, date_strings=None):
    """Build pixel-level t+LEAD_WEEKS verification arrays with year filtering.

    Args:
        forecast_map: [n_dates, H, W] model predictions
        y_full: [n_dates, H, W] ground truth
        T: history_weeks (window size)
        n_dates: total number of time steps
        year_range: optional (start_year, end_year) to filter time steps
        date_strings: optional list of date strings (YYYY-MM-DD) for year filtering

    Returns:
        preds: [N] array of predictions
        targets: [N] array of targets
        n_valid_steps: number of time steps included after filtering
    """
    LEAD_WEEKS = 4
    preds, targets = [], []
    n_valid_steps = 0

    for t in range(T - 1, n_dates):
        # Year filtering: only include time steps in the specified year range
        if year_range is not None and date_strings is not None:
            year = int(date_strings[t][:4])
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
        preds.extend(pred_t[valid].tolist())
        targets.extend(target_t[valid].tolist())
        n_valid_steps += 1

    return (np.array(preds, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            n_valid_steps)


def pixel_metrics(pred, target, severe_threshold=0.8):
    """Compute pixel-level RMSE, Brier, and severe Brier."""
    valid = ~np.isnan(pred) & ~np.isnan(target)
    pred = pred[valid]
    target = target[valid]

    if len(pred) == 0:
        return {"rmse": float("nan"), "brier": float("nan"),
                "severe_brier": float("nan"), "n_pixels": 0, "n_severe": 0}

    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    brier = float(np.mean((pred - target) ** 2))

    # Severe Brier: only where target >= severe_threshold
    severe_mask = target >= severe_threshold
    if severe_mask.sum() > 0:
        severe_brier = float(np.mean((pred[severe_mask] - target[severe_mask]) ** 2))
    else:
        severe_brier = float("nan")

    return {"rmse": rmse, "brier": brier, "severe_brier": severe_brier,
            "n_pixels": int(len(pred)), "n_severe": int(severe_mask.sum())}


def classification_metrics(pred, actual, severe_threshold, pred_thresh):
    """Compute CSI, POD, FAR, F1."""
    pred_bin = pred >= pred_thresh
    actual_bin = actual >= severe_threshold
    hits = int(np.sum(pred_bin & actual_bin))
    misses = int(np.sum(~pred_bin & actual_bin))
    false_alarms = int(np.sum(pred_bin & ~actual_bin))
    denom_csi = hits + misses + false_alarms
    csi = float(hits / denom_csi) if denom_csi > 0 else float("nan")
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else float("nan")
    denom_f1 = 2 * hits + false_alarms + misses
    f1 = float(2 * hits / denom_f1) if denom_f1 > 0 else float("nan")
    return {"csi": csi, "pod": pod, "far": far, "f1": f1,
            "hits": hits, "misses": misses, "false_alarms": false_alarms,
            "n_total": int(len(pred))}


def apply_year_filter_to_forecast(forecast_map, date_strings, year_range, lead_weeks=4):
    """Set forecast_map[t] = NaN for time steps whose PREDICTION year is outside year_range.

    For compute_auc-based scripts: forecast_map[t] is the prediction made at time t
    (using window x[t-T+1 : t]), which predicts y[t+lead_weeks].

    IMPORTANT: We filter by PREDICTION year (date_strings[t]), NOT target year.
    This is consistent with:
      - The year-based split in SCSPIECFDataset (which assigns windows by the
        year of the last window step = prediction time)
      - build_pfv_arrays() which also filters by date_strings[t]

    Filtering by target year would cause boundary leakage: e.g., a prediction
    made in Dec 2015 (val period) targeting Jan 2016 (test period) would be
    incorrectly included in test metrics.

    Args:
        forecast_map: [n_dates, H, W] array (modified in place)
        date_strings: list of date strings (YYYY-MM-DD)
        year_range: (start_year, end_year) tuple
        lead_weeks: lead time (default 4, unused but kept for API compatibility)

    Returns:
        n_kept: number of time steps retained
    """
    n_dates = forecast_map.shape[0]
    n_kept = 0
    for t in range(n_dates):
        if t >= len(date_strings):
            forecast_map[t] = np.nan
            continue
        year = int(date_strings[t][:4])  # Filter by PREDICTION year
        if not (year_range[0] <= year <= year_range[1]):
            forecast_map[t] = np.nan
        else:
            n_kept += 1
    return n_kept


def get_year_range_for_mode(ds):
    """Extract year range from dataset based on its mode.

    Args:
        ds: SCSPIECFDataset with year_based_split=True

    Returns:
        (start_year, end_year) tuple
    """
    if ds.mode == "val":
        return ds.val_years
    elif ds.mode == "test":
        return ds.test_years
    elif ds.mode == "train":
        return ds.train_years
    else:
        return None
