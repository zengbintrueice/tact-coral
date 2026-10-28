#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test B — rebuild ERA5 SST anomaly with a CAUSAL (train-only) climatology.

The shipped SCS/GBR datasets' `sst_anomaly` channel is computed in the
builders against a monthly climatology averaged over the FULL ERA5 record
(SCS 1985-2020, GBR 1985-2020) — which includes the test years. That is a
genuine input-side leakage.

This script recomputes `sst_anomaly` for BOTH domains using a monthly
climatology built from ERA5 months with year <= 2010 (strictly before the SCS
validation years 2011-2015), applying the same regrid / causal monthly→weekly
interpolation / spatial IDW post-fill as the original builders, and writes
patched NetCDFs with ONLY the `sst_anomaly` variable replaced.

Self-validation: the full-period reconstruction is compared against the stored
`sst_anomaly` to prove the regrid/interp/fill replication is exact; only then
is the causal version written.

Outputs (in DATA_DIR):
  scs_coral_bleaching_dataset_v9_1deg_raw_causal_era5.nc
  gbr_coral_bleaching_dataset_v9_1deg_raw_causal_era5.nc
"""
import os, sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import DATA_DIR

# Causal climatology cutoff: strictly before SCS validation (2011). Applied to
# BOTH domains so the GBR zero-shot inputs use the same causal reference.
CLIM_YEAR_CUTOFF = 2010

DOMAINS = {
    "SCS": dict(
        src_nc=os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"\era5_monthly\era5_sst_monthly.nc",
        dataset_file='scs_coral_bleaching_dataset_v9_1deg_raw.nc',
        out_file='scs_coral_bleaching_dataset_v9_1deg_raw_causal_era5.nc',
        n_lat=21, n_lon=23,
        lats=np.array([3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5]),
        lons=np.array([100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5, 122.5]),
        time_start=np.datetime64('2003-01-05'), n_weeks=864,
    ),
    "GBR": dict(
        src_nc=os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"\era5_gbr\era5_sst_monthly.nc",
        dataset_file='gbr_coral_bleaching_dataset_v9_1deg_raw.nc',
        out_file='gbr_coral_bleaching_dataset_v9_1deg_raw_causal_era5.nc',
        n_lat=15, n_lon=12,
        lats=np.arange(-24, -9, 1.0) + 0.5,   # [-23.5, ..., -9.5]
        lons=np.arange(142, 154, 1.0) + 0.5,  # [142.5, ..., 153.5]
        time_start=np.datetime64('1985-04-01'), n_weeks=1935,
    ),
}


def cell_bounds(centers):
    centers = np.asarray(centers, dtype=float)
    d = np.abs(np.diff(centers).mean()) / 2
    return centers - d, centers + d


def build_regrid_mask(src_lat, src_lon, lat_lo, lat_hi, lon_lo, lon_hi):
    src_lat = np.asarray(src_lat); src_lon = np.asarray(src_lon)
    flipped = src_lat[0] > src_lat[-1]
    m = (src_lat >= lat_lo) & (src_lat < lat_hi)
    if flipped:
        m = m[::-1]
    n = (src_lon >= lon_lo) & (src_lon < lon_hi)
    return m[:, None] & n[None, :]


def regrid_monthly_to_grid(src_monthly, masks_dict, N_LAT, N_LON):
    T = src_monthly.shape[0]
    n_src = src_monthly[0].size
    flat = src_monthly.reshape(T, n_src)
    out = np.full((T, N_LAT, N_LON), np.nan, dtype=np.float32)
    for (i, j), mask in masks_dict.items():
        idx = np.where(mask.ravel())[0]
        vals = flat[:, idx]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            n_valid = np.isfinite(vals).sum(axis=1)
            s = np.nansum(vals, axis=1)
            valid_t = n_valid > 0
            out[valid_t, i, j] = (s[valid_t] / n_valid[valid_t]).astype(np.float32)
    return out


def monthly_to_weekly_interp(monthly, monthly_time, target_weeks, method='causal'):
    t_weekly = np.array([pd.Timestamp(w).toordinal() for w in target_weeks], dtype=np.float64)
    T_m, H, W = monthly.shape
    n_weeks = len(target_weeks)
    flat_monthly = monthly.reshape(T_m, H * W)
    flat_out = np.full((n_weeks, H * W), np.nan, dtype=np.float32)
    if method == 'causal':
        month_avail = np.array([
            pd.Timestamp(m).to_period('M').to_timestamp(how='end').toordinal()
            for m in monthly_time
        ], dtype=np.float64)
        idx = np.searchsorted(month_avail, t_weekly, side='right') - 1
        valid_week = idx >= 0
        for k in range(H * W):
            ts_ff = pd.Series(flat_monthly[:, k]).ffill().values
            col = np.full(n_weeks, np.nan, dtype=np.float32)
            col[valid_week] = ts_ff[idx[valid_week]]
            flat_out[:, k] = col
        return flat_out.reshape(n_weeks, H, W)
    t_monthly = np.array([pd.Timestamp(m).toordinal() for m in monthly_time], dtype=np.float64)
    for k in range(H * W):
        ts = flat_monthly[:, k]
        valid = np.isfinite(ts)
        if valid.sum() < 2:
            if valid.sum() == 1:
                flat_out[:, k] = ts[valid][0]
            continue
        flat_out[:, k] = np.interp(t_weekly, t_monthly[valid], ts[valid])
    return flat_out.reshape(n_weeks, H, W)


def fill_nan_cells_spatial(data, ocean_mask, k=5):
    original_ndim = data.ndim
    if data.ndim == 2:
        data = data[None, :, :]
    T, H, W = data.shape
    valid_source = ocean_mask.copy()
    if original_ndim == 3:
        valid_source &= np.isfinite(data).any(axis=0)
    else:
        valid_source &= np.isfinite(data[0])
    if original_ndim == 3:
        targets = ocean_mask & np.isnan(data).all(axis=0)
    else:
        targets = ocean_mask & np.isnan(data[0])
    n_targets = int(targets.sum()); n_sources = int(valid_source.sum())
    if n_targets == 0 or n_sources == 0:
        return data.squeeze() if original_ndim == 2 else data
    src_coords = np.argwhere(valid_source).astype(np.float32)
    tgt_coords = np.argwhere(targets).astype(np.float32)
    tree = cKDTree(src_coords)
    dists, indices = tree.query(tgt_coords, k=min(k, n_sources))
    if k == 1:
        dists = dists[:, None]; indices = indices[:, None]
    weights = 1.0 / (dists ** 2 + 1e-8)
    out = data.copy()
    for t in range(T):
        for ti in range(n_targets):
            i, j = int(tgt_coords[ti, 0]), int(tgt_coords[ti, 1])
            src_idxs = indices[ti]; w = weights[ti]
            src_vals = np.array([out[t, int(src_coords[s, 0]), int(src_coords[s, 1])] for s in src_idxs])
            valid_v = np.isfinite(src_vals)
            if valid_v.any():
                out[t, i, j] = np.average(src_vals[valid_v], weights=w[valid_v])
    return out.squeeze() if original_ndim == 2 else out


def compute_anomaly_weekly(dom, clim_year_cutoff):
    """Regrid raw ERA5 monthly SST for this domain, subtract per-month
    climatology over months with year <= cutoff, interp to weekly (causal)."""
    ds = xr.open_dataset(dom['src_nc'])
    era5_sst_monthly = ds['sst'].values.astype(np.float32)
    era5_lats = ds['latitude'].values
    era5_lons = ds['longitude'].values
    time_var = 'valid_time' if 'valid_time' in ds else 'time'
    era5_time = pd.DatetimeIndex(ds[time_var].values)
    ds.close()

    masks = {}
    LAT_LO, LAT_HI = cell_bounds(dom['lats'])
    LON_LO, LON_HI = cell_bounds(dom['lons'])
    for i in range(dom['n_lat']):
        for j in range(dom['n_lon']):
            m = build_regrid_mask(era5_lats, era5_lons, LAT_LO[i], LAT_HI[i], LON_LO[j], LON_HI[j])
            if m.any():
                masks[i, j] = m

    era5_sst_regrid = regrid_monthly_to_grid(era5_sst_monthly, masks, dom['n_lat'], dom['n_lon'])

    years = era5_time.year.values
    months = era5_time.month.values
    clim = np.full((12, dom['n_lat'], dom['n_lon']), np.nan, dtype=np.float32)
    for m in range(1, 13):
        mm = (months == m) & (years <= clim_year_cutoff)
        if mm.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                clim[m-1] = np.nanmean(era5_sst_regrid[mm], axis=0)
    clim_3d = np.stack([clim[m-1] for m in months])
    anomaly = era5_sst_regrid - clim_3d
    del era5_sst_regrid, clim_3d, clim

    weekly_t = dom['time_start'] + np.arange(dom['n_weeks']) * np.timedelta64(7, 'D')
    return monthly_to_weekly_interp(anomaly, era5_time, weekly_t)


def run_domain(name, dom):
    print(f"\n{'='*60}\nDomain {name}: {dom['dataset_file']}\n{'='*60}")
    src_ds_path = DATA_DIR / dom['dataset_file']

    print(f"[1/3] full-period anomaly (validation)...", flush=True)
    full_anomaly = compute_anomaly_weekly(dom, clim_year_cutoff=3000)  # 3000 → all months
    print(f"[2/3] causal (<= {CLIM_YEAR_CUTOFF}) anomaly...", flush=True)
    causal_anomaly = compute_anomaly_weekly(dom, clim_year_cutoff=CLIM_YEAR_CUTOFF)

    ds_src = xr.open_dataset(src_ds_path)
    ocean_mask = ds_src['ocean_mask'].values.astype(bool)
    stored = ds_src['sst_anomaly'].values.astype(np.float32)
    n_weeks_file = ds_src.sizes['time']
    ds_src.close()

    if len(full_anomaly) != n_weeks_file:
        raise RuntimeError(f"[{name}] week count mismatch: {len(full_anomaly)} vs file {n_weeks_file}")
    full_filled = fill_nan_cells_spatial(full_anomaly, ocean_mask, k=5).astype(np.float32)
    causal_filled = fill_nan_cells_spatial(causal_anomaly, ocean_mask, k=5).astype(np.float32)

    both = np.isfinite(stored) & np.isfinite(full_filled)
    diff = np.abs(stored[both] - full_filled[both])
    max_err = float(diff.max()) if both.any() else float('nan')
    print(f"[3/3] validation: n={both.sum()}, max_err={max_err:.5f}")
    if both.any() and max_err > 1e-3:
        raise RuntimeError(f"[{name}] replication mismatch (max_err={max_err:.5f}) — causal file NOT written")
    print(f"  ✓ {name} replication matches stored sst_anomaly")

    out_path = DATA_DIR / dom['out_file']
    ds_out = xr.open_dataset(src_ds_path)
    ds_out['sst_anomaly'].values = causal_filled
    ds_out.attrs['sst_anomaly_climatology'] = (
        f'causal train-only (months with year<={CLIM_YEAR_CUTOFF}), '
        f'rebuilt by experiments/01_training/rebuild_causal_era5.py')
    enc = {z: {'zlib': True, 'complevel': 5, 'dtype': 'float32'} if ds_out[z].dtype == np.float32 else {}
           for z in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=enc)
    ds_out.close()
    print(f"  wrote {out_path}")
    print(f"  shift (causal-full): mean={np.nanmean(causal_filled-full_filled):+.4f} "
          f"std={np.nanstd(causal_filled-full_filled):.4f} °C")


def main():
    only = sys.argv[1].upper() if len(sys.argv) > 1 else None
    for name, dom in DOMAINS.items():
        if only and name != only:
            continue
        run_domain(name, dom)


if __name__ == "__main__":
    main()
