# ==========================================================================
# Source: paper/a7_dual_output/compute_train_normalization.py (verbatim except path fix)
# Experiment: Train-only normalization statistics
# Paper: Methods (normalization from SCS training period only)
# Output: results/train_only_norm_stats.json
# ==========================================================================

# -*- coding: utf-8 -*-
"""compute_train_normalization.py — Recompute SCS_CHANNEL_MEAN/STD from TRAIN period only.

Why: The original values in config.py were computed from the FULL 2003-2019 dataset,
which includes val (2011-2015) and test (2016-2019) periods. This is a normalization-
statistics leakage that mildly inflates cross-period generalization. We recompute using
only train-period (2003-2010) data to eliminate the leakage.

Inputs:
  - D:/piecf_data/scs_coral_bleaching_dataset_v9_1deg_raw.nc

Outputs:
  - prints new SCS_CHANNEL_MEAN / SCS_CHANNEL_STD (paste into config.py)
  - saves train_only_norm_stats.json for reproducibility

Notes:
  - Order of channels matches SCS_DYNAMIC_FEATURES + SCS_DERIVED_FEATURES in config.py
  - Uses the same derived-channel computation as aligned_dataset.py (_compute_derived_channels)
  - NaN handling: np.nanmean / np.nanstd over all valid ocean pixels
"""
import sys, io, os, json
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
SCS_FILE = DATA_DIR / "scs_coral_bleaching_dataset_v9_1deg_raw.nc"

# Train period. Production runs (all Steps 2-7) pass train_years=(2003, 2010)
# explicitly to create_scs_dataloaders; the loader's internal default
# (2003, 2012) is never used. Keep 2003-2010 here — must match the values
# every caller actually passes, NOT the loader default.
TRAIN_YEAR_MIN = 2003
TRAIN_YEAR_MAX = 2010

# Channel names (must match SCS_DYNAMIC_FEATURES + SCS_DERIVED_FEATURES)
RAW_DYNAMIC = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par', 'seasonal_sin',
               'wind_speed', 'ssr', 'sst_anomaly']
DERIVED = ['d_hotspot_dt', 'hotspot_cumsum_4w', 'hotspot_cumsum_12w', 'seasonal_cos']
ALL_CHANNELS = RAW_DYNAMIC + DERIVED


def compute_derived_channels(hotspot, date_strings):
    """Replicate aligned_dataset._compute_derived_channels exactly."""
    D = hotspot.shape[0]
    nan_mask = np.isnan(hotspot)
    hs_clean = np.where(nan_mask, 0.0, hotspot)

    # d_hotspot_dt: trailing difference
    d_hs = np.zeros_like(hotspot)
    d_hs[1:] = hs_clean[1:] - hs_clean[:-1]
    d_hs[nan_mask] = np.nan

    # hotspot_cumsum_4w / 12w: trailing sum
    cumsum = np.cumsum(hs_clean, axis=0)
    cs_4w = np.zeros_like(hotspot)
    cs_12w = np.zeros_like(hotspot)
    for i in range(D):
        if i < 4:
            cs_4w[i] = cumsum[i]
        else:
            cs_4w[i] = cumsum[i] - cumsum[i - 4]
        if i < 12:
            cs_12w[i] = cumsum[i]
        else:
            cs_12w[i] = cumsum[i] - cumsum[i - 12]
    cs_4w[nan_mask] = np.nan
    cs_12w[nan_mask] = np.nan

    # seasonal_cos
    from datetime import datetime as _dt
    woy = np.array([_dt.strptime(s, '%Y-%m-%d').isocalendar()[1]
                    for s in date_strings], dtype=np.float32)
    seasonal_cos = np.cos(2 * np.pi * woy / 52.0).astype(np.float32)
    sc_3d = np.broadcast_to(seasonal_cos[:, None, None], hotspot.shape).copy()

    return [d_hs, cs_4w, cs_12w, sc_3d]


def main():
    print("=" * 70)
    print("Recompute SCS_CHANNEL_MEAN/STD from TRAIN period only (2003-2010)")
    print("=" * 70)

    # Load dataset
    print(f"\n[1] Loading SCS dataset: {SCS_FILE}")
    ds = xr.open_dataset(SCS_FILE)
    times = pd.DatetimeIndex(ds.time.values)
    date_strings = [str(t)[:10] for t in times]

    # Train mask
    train_mask = (times.year >= TRAIN_YEAR_MIN) & (times.year <= TRAIN_YEAR_MAX)
    print(f"  Total weeks: {len(times)}")
    print(f"  Train weeks ({TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX}): {int(train_mask.sum())}")

    # Load raw dynamic channels
    raw_arrays = [ds[name].values.astype(np.float32) for name in RAW_DYNAMIC]
    hotspot_full = raw_arrays[2].copy()  # idx 2 = 'hotspot'

    # Compute derived channels over FULL time series (correct boundary handling)
    print("\n[2] Computing derived channels (d_hotspot_dt, cumsum_4w, cumsum_12w, seasonal_cos)")
    derived_arrays = compute_derived_channels(hotspot_full, date_strings)

    all_arrays = raw_arrays + derived_arrays
    ds.close()

    # Verify channel count
    assert len(all_arrays) == 13, f"Expected 13 channels, got {len(all_arrays)}"

    # Compute mean/std over TRAIN period only
    print(f"\n[3] Computing mean/std over TRAIN period only...")
    print(f"{'Channel':<25} {'Mean':>14} {'Std':>14} {'n_valid':>10}")
    print("-" * 65)
    means = []
    stds = []
    for name, arr in zip(ALL_CHANNELS, all_arrays):
        train_slice = arr[train_mask]
        valid = ~np.isnan(train_slice)
        n_valid = int(valid.sum())
        mu = float(np.nanmean(train_slice))
        sigma = float(np.nanstd(train_slice))
        means.append(mu)
        stds.append(sigma)
        print(f"{name:<25} {mu:>14.6f} {sigma:>14.6f} {n_valid:>10}")

    # Compare with current (leaky) values
    print("\n[4] Comparison with current (leaky, full-dataset) values:")
    CURRENT_MEAN = [28.0530, 0.3477, 0.1057, 0.4334, 0.6154, 0.0187, 3.8196, 16555252.7081, -0.0268,
                    0.0004, 0.4203, 1.2290, -0.0011]
    CURRENT_STD = [2.1987, 0.6272, 0.2650, 0.4886, 0.2017, 0.7046, 2.4716, 3569105.1996, 0.4716,
                   0.1645, 0.9289, 2.1538, 0.7094]
    print(f"{'Channel':<25} {'Old μ':>10} {'New μ':>10} {'Δ μ':>10} {'Old σ':>10} {'New σ':>10}")
    print("-" * 80)
    for i, name in enumerate(ALL_CHANNELS):
        dmu = means[i] - CURRENT_MEAN[i]
        print(f"{name:<25} {CURRENT_MEAN[i]:>10.4f} {means[i]:>10.4f} {dmu:>+10.4f} "
              f"{CURRENT_STD[i]:>10.4f} {stds[i]:>10.4f}")

    # Save JSON
    out = {
        "description": "Train-only (2003-2010) normalization stats for SCS v9_1deg_raw dataset",
        "train_period": [TRAIN_YEAR_MIN, TRAIN_YEAR_MAX],
        "n_train_weeks": int(train_mask.sum()),
        "channels": ALL_CHANNELS,
        "mean": means,
        "std": stds,
        "previous_leaky_mean": CURRENT_MEAN,
        "previous_leaky_std": CURRENT_STD,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = Path(__file__).resolve().parents[2] / "results" / "infrastructure/train_normalization_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[5] Saved: {out_path}")

    # Print paste-ready config.py snippet
    print("\n[6] Paste-ready snippet for config.py:")
    print(f"SCS_CHANNEL_MEAN = {[round(m, 6) for m in means]}")
    print(f"SCS_CHANNEL_STD = {[round(s, 6) for s in stds]}")


if __name__ == "__main__":
    main()
