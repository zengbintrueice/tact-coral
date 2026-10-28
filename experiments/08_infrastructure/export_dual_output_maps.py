# ==========================================================================
# Source: paper/a7_dual_output/export_dual_output_maps.py (verbatim except path fix)
# Experiment: Dual-output interface (risk_score + risk_probability maps)
# Paper: Methods (Operational evaluation)
# Output: results/prediction_raw.nc, results/prediction_calibrated.nc
# ==========================================================================

# export_dual_output_maps.py — PIECF-a6 dual-output interface
#
#   Redefines a6's output from a single `y_final` to two semantically distinct
#   variables:
#
#     risk_score      = raw a6 output (S_raw) — for operational early warning
#     risk_probability = isotonic(S_raw)       — for probabilistic risk estimation
#
#   Saves:
#     results/prediction_raw.nc         — risk_score [n_dates, H, W]
#     results/prediction_calibrated.nc  — risk_probability [n_dates, H, W]
#
#   Both files include: time, lat, lon coordinates + land_mask + variable
#   attributes describing the semantic meaning of each output.
#
#   Run:
#     & 'D:\...\piecf_env\Scripts\python.exe' export_dual_output_maps.py

import os, sys, io, json, time, warnings, importlib, tempfile, shutil
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import torch
import xarray as xr
from sklearn.isotonic import IsotonicRegression

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


def predict_sliding_temporal(model, ds, device):
    """Sliding-window inference: each time point gets its own T-week window.

    Returns forecast_map: [n_dates, H, W] — raw a6 output (risk_score).
    """
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
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
            out = model(xd, xs, lm, scale=A5_CFG["scale"],
                        tau=A5_CFG["tau"], temp=A5_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
    return forecast_map


def build_pixel_arrays(forecast_map, y_full, T, n_dates,
                       year_range=None, date_strings=None):
    """Build pixel-level prediction and binary severe label arrays (year-filtered)."""
    preds, labels = [], []
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
        labels.extend((target_t[valid] >= SEVERE_THRESHOLD).tolist())
    return (np.array(preds, dtype=np.float32),
            np.array(labels, dtype=np.float32))


def save_netcdf(filepath, data, dates, lat, lon, land_mask, var_name, var_attrs):
    """Save a [n_dates, H, W] array as NetCDF with coordinates and metadata.

    Writes to system TEMP dir first, then moves to destination with retry.
    This avoids file-lock conflicts with sync services / antivirus.
    """
    ds = xr.Dataset(
        {
            var_name: (["time", "lat", "lon"], data.astype(np.float32), var_attrs),
            "land_mask": (["lat", "lon"], land_mask.astype(bool),
                          {"description": "True = land/invalid, False = ocean/valid"}),
        },
        coords={
            "time": np.arange(len(dates)),
            "time_date": ("time", dates),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": var_attrs.get("title", "PIECF-a6 prediction"),
            "created": datetime.now().isoformat(),
            "model": "a6_sg_lam20_lead4",
            "lead_weeks": LEAD_WEEKS,
            "severe_threshold": SEVERE_THRESHOLD,
        },
    )
    # Write to system TEMP dir (avoids lock on target dir from sync/AV)
    filepath = Path(filepath)
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"a7_{var_name}_{os.getpid()}_{int(time.time()*1000)}.nc"
    ds.to_netcdf(str(tmp_path))

    # Move to destination with retry (handles transient locks)
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            if dest.exists():
                os.remove(dest)
            shutil.move(str(tmp_path), str(dest))
            break
        except PermissionError:
            if attempt == max_attempts:
                # Last attempt: try a fresh unique filename
                dest = dest.with_suffix(f".v{attempt}.nc")
                shutil.move(str(tmp_path), str(dest))
            else:
                time.sleep(0.3)
    return ds


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"PIECF-a6 Dual-Output Interface — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_kw = dict(year_based_split=True, train_years=(2003, 2010),
                val_years=(2011, 2015), test_years=(2016, 2019))

print("\nRestoring a6...")
a5_model = restore_a5().to(device)

# ═══════════════════════════════════════════════════════════════
# Step 1: Fit isotonic regression on validation set
# ═══════════════════════════════════════════════════════════════
print("\n[Step 1] Fitting isotonic regression on validation set (2013-2015)...")
ds_val = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="val", random_split=False,
                         lead_weeks=4, **split_kw)
T, H, W = ds_val.history_weeks, ds_val.n_lat, ds_val.n_lon
n_dates_val = ds_val.n_dates
print(f"  T={T} H={H} W={W}  n_dates={n_dates_val}")

print("  Sliding-window inference (val)...")
forecast_map_val = predict_sliding_temporal(a5_model, ds_val, device)
val_preds, val_labels = build_pixel_arrays(
    forecast_map_val, ds_val.y_regression_full, T, n_dates_val,
    year_range=get_year_range_for_mode(ds_val), date_strings=ds_val.date_strings)
print(f"  val n_pixels={len(val_preds)}")

print("  Fitting Isotonic regression...")
iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso.fit(val_preds, val_labels)
print(f"  Isotonic fit complete.")

# ═══════════════════════════════════════════════════════════════
# Step 2: Inference on test set → dual output
# ═══════════════════════════════════════════════════════════════
print("\n[Step 2] Test set inference (2016-2019) → dual output...")
ds_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                          lead_weeks=4, **split_kw)
n_dates_test = ds_test.n_dates
print(f"  n_dates={n_dates_test}")

print("  Sliding-window inference (test)...")
forecast_map_test = predict_sliding_temporal(a5_model, ds_test, device)

# risk_score = raw a6 output (S_raw)
risk_score = forecast_map_test.copy()

# risk_probability = isotonic(S_raw)
print("  Applying isotonic calibration...")
# Isotonic regression cannot handle NaN — mask, apply, restore NaN
nan_mask = np.isnan(risk_score)
risk_probability = np.full_like(risk_score, np.nan, dtype=np.float32)
valid = ~nan_mask
if valid.sum() > 0:
    risk_probability[valid] = iso.predict(risk_score[valid].flatten()).astype(np.float32)

print(f"  risk_score:         shape={risk_score.shape}  "
      f"range=[{np.nanmin(risk_score):.4f}, {np.nanmax(risk_score):.4f}]")
print(f"  risk_probability:   shape={risk_probability.shape}  "
      f"range=[{np.nanmin(risk_probability):.4f}, {np.nanmax(risk_probability):.4f}]")

# Get coordinates from dataset
dates_test = ds_test.date_strings
lat = np.arange(H, dtype=np.float32)
lon = np.arange(W, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════
# Step 3: Save both outputs as NetCDF
# ═══════════════════════════════════════════════════════════════
print("\n[Step 3] Saving dual-output NetCDF files...")

# Output 1: prediction_raw.nc
raw_path = RESULTS_DIR / "infrastructure/prediction_raw.nc"
raw_attrs = {
    "long_name": "Raw severe thermal stress risk score",
    "units": "dimensionless",
    "description": (
        "Raw output of PIECF-a6 stop-gradient severe branch (S_raw). "
        "This is NOT a calibrated probability. "
        "Use with threshold optimization for operational early warning. "
        "Higher values indicate higher relative risk of severe thermal stress "
        "(y >= 0.8) in the 4-week forecast horizon."
    ),
    "title": "PIECF-a6 Raw Risk Score (S_raw)",
    "valid_range": [0.0, 1.0],
    "severe_threshold_recommended": 0.9,
    "forecast_lead_weeks": LEAD_WEEKS,
}
save_netcdf(raw_path, risk_score, dates_test, lat, lon, ds_val.land_mask,
            "risk_score", raw_attrs)
print(f"  Saved: {raw_path}")

# Output 2: prediction_calibrated.nc
cal_path = RESULTS_DIR / "infrastructure/prediction_calibrated.nc"
cal_attrs = {
    "long_name": "Calibrated severe thermal stress probability",
    "units": "probability",
    "description": (
        "Isotonic-calibrated probability P(severe | S_raw). "
        "Calibrated on validation set (2013-2015) using Isotonic regression. "
        "This IS a calibrated probability suitable for risk communication. "
        "Value 0.8 means ~80% observed frequency of severe thermal stress "
        "(y >= 0.8) in the 4-week forecast horizon."
    ),
    "title": "PIECF-a6 Calibrated Probability (P_cal = Iso(S_raw))",
    "valid_range": [0.0, 1.0],
    "calibration_method": "Isotonic regression",
    "calibration_val_years": "2013-2015",
    "forecast_lead_weeks": LEAD_WEEKS,
}
save_netcdf(cal_path, risk_probability, dates_test, lat, lon, ds_val.land_mask,
            "risk_probability", cal_attrs)
print(f"  Saved: {cal_path}")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DUAL-OUTPUT INTERFACE SUMMARY")
print("=" * 60)
print(f"""
  Model: PIECF-a6 (stop-gradient dual-head ConvLSTM)
  Weight: {A5_MODEL.name}
  Test period: 2016-2019 ({n_dates_test} weeks)

  Output 1 — Operational Early Warning
    Variable:      risk_score (S_raw)
    File:          prediction_raw.nc
    Purpose:       Which areas are most at risk?
    Processing:    threshold optimization (recommended thr=0.9)
    Metrics:       CSI, F1, POD, FAR

  Output 2 — Probabilistic Risk Estimation
    Variable:      risk_probability = Iso(S_raw)
    File:          prediction_calibrated.nc
    Purpose:       What is the probability of severe stress?
    Processing:    Isotonic calibration (val 2013-2015)
    Metrics:       ECE, Brier, reliability
""")
print("Done.")
