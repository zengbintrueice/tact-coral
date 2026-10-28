#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caribbean 1° Raw-from-Source Dataset Builder (TSCF zero-shot target domain)
===========================================================================

Builds a 1° (22×32) dataset for the Caribbean / Wider Tropical Atlantic directly
from raw data sources, mirroring build_v9_1deg_from_raw.py (SCS) and
build_gbr_1deg_from_raw.py (GBR). NEW file — the SCS/GBR builders are untouched.

Caribbean grid: lat 4..26 (22 cells, centers 4.5..25.5), lon -91..-59 (32 cells,
centers -90.5..-58.5)
Time: 2003-01-05 ~ 2020-12-29, weekly resolution (940 weeks). CRW 5km inputs
2002-01..2020-12 (84-day DHW + 24-week window lookback before week 0 is covered).

CRITICAL: Jensen-aware DHW — computed at native 5km resolution (clip + 84-day
rolling sum), THEN spatial-block-averaged to 1°.

Data sources:
  - CRW 5km daily SST/SSTA/Hotspot (crw_5km_caribbean/, 2002-2020, lon -180..180)
  - OC-CCI v6 monthly chlorophyll (GLOBAL chlorophyll/ dir, subset to bbox)
  - MODIS monthly PAR (coastwatch_par_caribbean/, real data — NOT climatology)
  - ERA5 monthly SST/wind/SSR (era5_caribbean/, 2003-2020)
  - ETOPO1 bathymetry (global), WOA23 salinity (global), IBTrACS cyclone (global)

Output:
  - D:\\piecf_crw_data\\caribbean_coral_bleaching_dataset_v9_1deg_raw.nc
  - 13 dynamic + 5 static + 3 target + 2 mask = 23 variables, shape [940, 22, 32]

Notes / decisions (documented, honest):
  - ERA5 Caribbean starts 2003-01; weeks 0-3 (2003-01) carry NaN for
    wind_speed/ssr/sst_anomaly (no completed Dec-2002 month available). The
    zero-shot evaluation only uses test years 2017-2020, so this is harmless.
  - Cyclone frequency computed from global IBTrACS (storm-center day counts in
    2003-2020 per 1° cell, /18 y, z-scored); falls back to 0 if IBTrACS fails.
  - PAR min-max params (18.20, 56.63) reused from SCS so the frozen SCS
    normalization scale is identical.
  - static features are NOT consumed by the model (a6 ignores x_static);
    they are carried for dataset schema parity.

Run:
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' data/source_builders/build_caribbean_1deg_from_raw.py
"""
import os, sys, glob, warnings
import numpy as np
import xarray as xr
import pandas as pd
import re
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

from jensen_dhw import (
    compute_dhw_5km, build_aggregation_matrix, aggregate_to_1deg, dhw_to_target,
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
DNS = os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r""
CRW_DIR = os.path.join(DNS, 'crw_5km_caribbean')
CHL_DIR = os.path.join(DNS, 'chlorophyll')
PAR_DIR = os.path.join(DNS, 'coastwatch_par_caribbean')
ERA5_DIR = os.path.join(DNS, 'era5_caribbean')
BATHY_PATH = os.path.join(DNS, 'bathymetry', 'ETOPO1_Ice_g_gmt4.grd')
SAL_PATH = os.path.join(DNS, 'salinity', 'WOA23_salinity_annual.nc')
CYCLONE_PATH = os.path.join(DNS, 'cyclone', 'IBTrACS.ALL.v04r00.nc')
OUTPUT_DIR = r'D:\piecf_crw_data'  # ASCII: netCDF4 cannot write under Chinese paths
OUTPUT_NC = os.path.join(OUTPUT_DIR, 'caribbean_coral_bleaching_dataset_v9_1deg_raw.nc')

# ─────────────────────────────────────────────────────────────
# 1° target grid (cell centers, 22×32)
# ─────────────────────────────────────────────────────────────
N_LAT, N_LON = 22, 32
LATS = np.arange(4, 26, 1.0) + 0.5      # [4.5, 5.5, ..., 25.5]
LONS = np.arange(-91, -59, 1.0) + 0.5   # [-90.5, -89.5, ..., -58.5]

# Time: weekly from 2003-01-05 through 2020-12-29 (940 weeks)
WEEKLY = np.timedelta64(7, 'D')
TIME_START = np.datetime64('2003-01-05')
TIME_END = np.datetime64('2020-12-31')
N_WEEKS = int((TIME_END - TIME_START) / WEEKLY) + 1   # 940
WEEKLY_T = TIME_START + np.arange(N_WEEKS) * WEEKLY
tsp = pd.DatetimeIndex(WEEKLY_T)

# ─────────────────────────────────────────────────────────────
# CRW 5km raw grid (source) — Caribbean region
# lat: 26.025..4.025 (441 pts, DESCENDING), lon: -90.975..-58.975 (641, asc)
# ─────────────────────────────────────────────────────────────
CRW_LAT = np.arange(26.025, 4.0 - 0.025, -0.05)   # 26.025..4.025, 441 pts, desc
CRW_LON = np.arange(-90.975, -58.975 + 0.025, 0.05)  # -90.975..-58.975, 641 pts


def cell_bounds(centers):
    centers = np.asarray(centers, dtype=float)
    d = np.abs(np.diff(centers).mean()) / 2
    return centers - d, centers + d


LAT_LO, LAT_HI = cell_bounds(LATS)   # 4.0..26.0
LON_LO, LON_HI = cell_bounds(LONS)   # -91.0..-59.0


def build_regrid_mask(src_lat, src_lon, lat_lo, lat_hi, lon_lo, lon_hi):
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    flipped = src_lat[0] > src_lat[-1]
    m = (src_lat >= lat_lo) & (src_lat < lat_hi)
    if flipped:
        m = m[::-1]
    n = (src_lon >= lon_lo) & (src_lon < lon_hi)
    return m[:, None] & n[None, :]


def monthly_to_weekly_interp(monthly, monthly_time, target_weeks, method='causal'):
    """Causal (default) month-end forward-fill; same semantics as SCS builder."""
    t_weekly = np.array([pd.Timestamp(w).toordinal() for w in target_weeks], dtype=np.float64)
    T_m, H, W = monthly.shape
    n_weeks = len(target_weeks)
    flat_monthly = monthly.reshape(T_m, H * W)
    flat_out = np.full((n_weeks, H * W), np.nan, dtype=np.float32)
    if method == 'causal':
        month_avail = np.array([
            pd.Timestamp(m).to_period('M').to_timestamp(how='end').toordinal()
            for m in monthly_time], dtype=np.float64)
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
    n_targets = int(targets.sum())
    n_sources = int(valid_source.sum())
    if n_targets == 0 or n_sources == 0:
        return data.squeeze() if original_ndim == 2 else data
    src_coords = np.argwhere(valid_source).astype(np.float32)
    tgt_coords = np.argwhere(targets).astype(np.float32)
    tree = cKDTree(src_coords)
    dists, indices = tree.query(tgt_coords, k=min(k, n_sources))
    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]
    weights = 1.0 / (dists ** 2 + 1e-8)
    out = data.copy()
    for t in range(T):
        for ti in range(n_targets):
            i, j = int(tgt_coords[ti, 0]), int(tgt_coords[ti, 1])
            src_idxs = indices[ti]
            w = weights[ti]
            src_vals = np.array([out[t, int(src_coords[s, 0]), int(src_coords[s, 1])] for s in src_idxs])
            valid_v = np.isfinite(src_vals)
            if valid_v.any():
                out[t, i, j] = np.average(src_vals[valid_v], weights=w[valid_v])
    return out.squeeze() if original_ndim == 2 else out


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
print("=" * 60)
print("Caribbean 1° Dataset Builder (RAW from sources, Jensen-aware DHW)")
print("=" * 60)
print(f"Grid: {N_LAT}x{N_LON}, Time: {N_WEEKS} weeks")
print(f"Time range: {str(TIME_START)[:10]} to {str(WEEKLY_T[-1])[:10]}")
print(f"CRW 5km grid: {len(CRW_LAT)}x{len(CRW_LON)} (Caribbean)")
print()

# Build regrid masks on CRW Caribbean grid
print("STEP 0: Building regrid masks and CSR aggregation matrix...")
flat_lookup_std = np.arange(len(CRW_LAT) * len(CRW_LON)).reshape(len(CRW_LAT), len(CRW_LON))
cell_flat_indices_std = []
cell_ij_std = []
for i in range(N_LAT):
    for j in range(N_LON):
        m = build_regrid_mask(CRW_LAT, CRW_LON, LAT_LO[i], LAT_HI[i], LON_LO[j], LON_LO[j] + 1.0)
        if m.any():
            cell_flat_indices_std.append(flat_lookup_std[m])
            cell_ij_std.append((i, j))
n_cells_std = len(cell_flat_indices_std)
print(f"  Built {n_cells_std} regrid masks (CRW Caribbean grid)")

rows_csr, cols_csr = [], []
for cell_k, fidx in enumerate(cell_flat_indices_std):
    for pix in fidx:
        rows_csr.append(cell_k)
        cols_csr.append(pix)
M_csr = csr_matrix((np.ones(len(rows_csr), dtype=np.float32),
                    (rows_csr, cols_csr)),
                   shape=(n_cells_std, len(CRW_LAT) * len(CRW_LON)))
cell_size = np.asarray(M_csr.sum(axis=1)).ravel().astype(np.float32)
print(f"  CSR matrix: {M_csr.shape}, nnz={M_csr.nnz}")

# ═════════════════════════════════════════════════════════════
# STEP 1: Regrid CRW 5km daily → 1° weekly block-avg
# ═════════════════════════════════════════════════════════════
print("\nSTEP 1: Regridding CRW 5km daily (SSTA + Hotspot + SST) → 1° weekly block-avg...")
ssta_by_month, hs_by_month, sst_by_month = {}, {}, {}
for fp in sorted(glob.glob(os.path.join(CRW_DIR, 'crw_sstanomaly_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        ssta_by_month[ym.group(1)] = fp
for fp in sorted(glob.glob(os.path.join(CRW_DIR, 'crw_hotspot_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        hs_by_month[ym.group(1)] = fp
for fp in sorted(glob.glob(os.path.join(CRW_DIR, 'crw_sst_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        sst_by_month[ym.group(1)] = fp
print(f"  SSTA months: {len(ssta_by_month)}, Hotspot: {len(hs_by_month)}, SST: {len(sst_by_month)}")

months_weekly = set()
for w in range(N_WEEKS):
    for dt64 in [WEEKLY_T[w], WEEKLY_T[w] + np.timedelta64(6, 'D')]:
        dt = pd.Timestamp(dt64)
        months_weekly.add(f"{dt.year}{dt.month:02d}")
months_weekly = sorted(months_weekly)
all_crw_months = sorted(set(list(ssta_by_month.keys()) + list(hs_by_month.keys())))
months_needed = sorted(set(months_weekly) | set(all_crw_months))
print(f"  Weekly window: {len(months_weekly)} months, all CRW: {len(all_crw_months)}, union: {len(months_needed)}")

ssta_sum = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
ssta_cnt = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)
hs_sum = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
hs_cnt = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)
sst_sum = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
sst_cnt = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)

# Daily axis for DHW: start ~100 days before week 0 (84-day DHW lookback)
TOTAL_START = WEEKLY_T[0] - np.timedelta64(100, 'D')
TOTAL_END = np.datetime64('2020-12-31')
DAILY_T = np.arange(TOTAL_START, TOTAL_END + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
N_DAILY = len(DAILY_T)
t0_daily = pd.Timestamp(TOTAL_START)
print(f'  Daily hotspot grid: [{N_DAILY} days x {N_LAT}x{N_LON}] = ~{N_DAILY * N_LAT * N_LON * 4 / 1e6:.0f} MB')

WIN = 84
hs_daily_5km = np.full((N_DAILY, len(CRW_LAT), len(CRW_LON)), np.nan, dtype=np.float32)

n_months_done = 0
for month_str in months_needed:
    if month_str not in ssta_by_month:
        continue
    try:
        ds_ssta = xr.open_dataset(ssta_by_month[month_str])
        ssta_daily = ds_ssta['CRW_SSTANOMALY'].values
        ssta_time = pd.DatetimeIndex(ds_ssta['time'].values)
        ds_ssta.close()

        hs_daily = None
        if month_str in hs_by_month:
            ds_hs = xr.open_dataset(hs_by_month[month_str])
            hs_daily = ds_hs['CRW_HOTSPOT'].values
            ds_hs.close()

        sst_daily = None
        if month_str in sst_by_month:
            ds_sst = xr.open_dataset(sst_by_month[month_str])
            sst_daily = ds_sst['CRW_SST'].values
            ds_sst.close()

        n_days = len(ssta_time)
        ssta_flat = ssta_daily.reshape(n_days, -1).astype(np.float64)
        ssta_valid = np.isfinite(ssta_flat)
        ssta_flat[~ssta_valid] = 0.0
        if hs_daily is not None:
            hs_flat = hs_daily.reshape(n_days, -1).astype(np.float64)
            hs_valid = np.isfinite(hs_flat)
            hs_flat[~hs_valid] = 0.0
        sst_flat = sst_valid = None
        if sst_daily is not None:
            sst_flat = sst_daily.reshape(n_days, -1).astype(np.float64)
            sst_valid = np.isfinite(sst_flat)
            sst_flat[~sst_valid] = 0.0

        t0 = pd.Timestamp(TIME_START)
        day_offsets = np.array([(pd.Timestamp(dt) - t0).days for dt in ssta_time])
        day_weeks = day_offsets // 7
        day_in_window = (day_weeks >= 0) & (day_weeks < N_WEEKS)

        for cell_k, ((i, j), fidx) in enumerate(zip(cell_ij_std, cell_flat_indices_std)):
            w_idx = day_weeks[day_in_window]
            day_sums = ssta_flat[:, fidx].sum(axis=1)
            day_counts = ssta_valid[:, fidx].sum(axis=1)
            np.add.at(ssta_sum[:, i, j], w_idx, day_sums[day_in_window])
            np.add.at(ssta_cnt[:, i, j], w_idx, day_counts[day_in_window])
            if hs_daily is not None:
                day_hs_sums = hs_flat[:, fidx].sum(axis=1)
                day_hs_counts = hs_valid[:, fidx].sum(axis=1)
                np.add.at(hs_sum[:, i, j], w_idx, day_hs_sums[day_in_window])
                np.add.at(hs_cnt[:, i, j], w_idx, day_hs_counts[day_in_window])
            if sst_daily is not None:
                day_sst_sums = sst_flat[:, fidx].sum(axis=1)
                day_sst_counts = sst_valid[:, fidx].sum(axis=1)
                np.add.at(sst_sum[:, i, j], w_idx, day_sst_sums[day_in_window])
                np.add.at(sst_cnt[:, i, j], w_idx, day_sst_counts[day_in_window])

        if hs_daily is not None:
            hs_2d = hs_daily.reshape(n_days, len(CRW_LAT), len(CRW_LON))
            for d in range(n_days):
                dt = ssta_time[d]
                g_offset = (pd.Timestamp(dt) - t0_daily).days
                if g_offset < 0 or g_offset >= N_DAILY:
                    continue
                hs_daily_5km[g_offset] = hs_2d[d][::-1]  # flip to ascending lat
            del hs_2d, hs_daily

        n_months_done += 1
        if n_months_done % 50 == 0 or n_months_done == len(months_needed):
            print(f"    Processed {n_months_done}/{len(months_needed)} months")
    except Exception as e:
        print(f"    Month {month_str} failed: {e}")

ssta_window = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
hotspot_raw = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
sst_window = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
valid_ssta = ssta_cnt > 0
valid_hs = hs_cnt > 0
valid_sst = sst_cnt > 0
ssta_window[valid_ssta] = (ssta_sum[valid_ssta] / ssta_cnt[valid_ssta]).astype(np.float32)
hotspot_raw[valid_hs] = (hs_sum[valid_hs] / hs_cnt[valid_hs]).astype(np.float32)
sst_window[valid_sst] = (sst_sum[valid_sst] / sst_cnt[valid_sst]).astype(np.float32)
print(f"  SSTA: NaN frac={np.isnan(ssta_window).mean():.4f}")
print(f"  Hotspot: range=[{np.nanmin(hotspot_raw):.3f}, {np.nanmax(hotspot_raw):.3f}]")
print(f"  SST: NaN frac={np.isnan(sst_window).mean():.4f}, range=[{np.nanmin(sst_window):.3f}, {np.nanmax(sst_window):.3f}]")

hotspot_window = np.full_like(hotspot_raw, np.nan)
hotspot_window[valid_hs] = np.clip(hotspot_raw[valid_hs], 0, None)

weeks_of_year = np.array([t.isocalendar()[1] for t in tsp])
seasonal_sin_window = np.broadcast_to(
    (np.sin(2 * np.pi * weeks_of_year / 52).astype(np.float32))[:, None, None],
    (N_WEEKS, N_LAT, N_LON)).copy()
seasonal_cos_window = np.broadcast_to(
    (np.cos(2 * np.pi * weeks_of_year / 52).astype(np.float32))[:, None, None],
    (N_WEEKS, N_LAT, N_LON)).copy()

ssta_nan_frac = np.isnan(ssta_window).mean(axis=0)
ssta_all_nan = ssta_nan_frac > 0.9999
print(f"  SSTA all-NaN cells: {ssta_all_nan.sum()} / {N_LAT * N_LON}")

del M_csr, cell_size

# ═════════════════════════════════════════════════════════════
# STEP 2: Jensen-aware DHW targets from NATIVE 5km
# ═════════════════════════════════════════════════════════════
# Memory-safe cell-local computation: for each 1° cell, run compute_dhw_5km
# only on that cell's ~400 5km pixels (rolling 84-day sum per pixel), then mean
# the cell's valid pixels per day. Result is bit-identical to the CSR-matrix
# aggregation (same pixel membership, same mean of valid pixels), but peak
# memory is ~pixels-per-cell instead of the full 5km grid.
print("\nSTEP 2: Computing Jensen-aware DHW targets via jensen_dhw.py (cell-local)...")
CRW_NLON = len(CRW_LON)

dhw_daily_1deg = np.full((N_DAILY, n_cells_std), np.nan, dtype=np.float32)
for cell_k, fidx in enumerate(cell_flat_indices_std):
    la = fidx // CRW_NLON
    lo = fidx % CRW_NLON
    cell_hs = hs_daily_5km[:, la, lo]                # [N_DAILY, n_pix] f32
    cell_dhw = compute_dhw_5km(cell_hs)              # rolling 84d per pixel
    valid = np.isfinite(cell_dhw)
    n_valid = valid.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = np.nansum(np.where(valid, cell_dhw, 0.0), axis=1)
    ok = n_valid > 0
    dhw_daily_1deg[ok, cell_k] = (s[ok] / n_valid[ok]).astype(np.float32)
    if (cell_k + 1) % 100 == 0 or cell_k == n_cells_std - 1:
        print(f"    DHW cell {cell_k + 1}/{n_cells_std}")
del hs_daily_5km
print(f"    1° daily DHW: range=[{np.nanmin(dhw_daily_1deg):.4f}, {np.nanmax(dhw_daily_1deg):.4f}]")

dhw_weekly_degdays = np.full((N_WEEKS, n_cells_std), np.nan, dtype=np.float32)
for w in range(N_WEEKS):
    day_start = (pd.Timestamp(WEEKLY_T[w]) - t0_daily).days
    day_end = min(day_start + 7, N_DAILY)
    if day_start < 0:
        continue
    window = dhw_daily_1deg[day_start:day_end]
    valid_per_cell = np.isfinite(window).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dhw_weekly_degdays[w] = np.where(
            valid_per_cell > 0,
            np.nansum(window, axis=0) / valid_per_cell,
            np.nan).astype(np.float32)
del dhw_daily_1deg
dhw_weekly = dhw_weekly_degdays / 7.0
del dhw_weekly_degdays
print(f"    DHW weekly: range=[{np.nanmin(dhw_weekly):.4f}, {np.nanmax(dhw_weekly):.4f}]")

y_regression = dhw_to_target(dhw_weekly, scale=10.0)
y_warning = dhw_to_target(dhw_weekly, scale=8.0)
y_classification = np.where(np.isnan(dhw_weekly), np.nan,
                            (dhw_weekly >= 4.0).astype(np.float32))
y_regression = y_regression.reshape(N_WEEKS, N_LAT, N_LON)
y_warning = y_warning.reshape(N_WEEKS, N_LAT, N_LON)
y_classification = y_classification.reshape(N_WEEKS, N_LAT, N_LON)
ssta_all_nan_3d = np.broadcast_to(ssta_all_nan[None, :, :], y_regression.shape)
for arr in [y_regression, y_warning, y_classification]:
    arr[ssta_all_nan_3d] = np.nan
valid_targets = np.isfinite(y_regression)
n_pos = int((y_classification[valid_targets] >= 1).sum())
print(f"  y_regression: range=[{np.nanmin(y_regression):.4f}, {np.nanmax(y_regression):.4f}]")
print(f"  y_classification positive rate: {n_pos}/{int(valid_targets.sum())}")

# ═════════════════════════════════════════════════════════════
# STEP 3: Chlorophyll from OC-CCI v6 global (subset to Caribbean)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 3: Rebuilding chlorophyll from OC-CCI v6.0 global...")
chl_files = sorted(glob.glob(os.path.join(CHL_DIR, 'oc_cci_chlor_a_*_fv6.0.nc')))
if not chl_files:
    raise RuntimeError('no OC-CCI files found')
print(f"  Found {len(chl_files)} OC-CCI monthly files")
ds_chl0 = xr.open_dataset(chl_files[0])
slat = ds_chl0['lat'].values
slon = ds_chl0['lon'].values
ds_chl0.close()
flip_chl = slat[0] > slat[-1]

lat_mask = (slat >= LAT_LO[0] - 0.5) & (slat <= LAT_HI[-1] + 0.5)
lon_mask = (slon >= LON_LO[0] - 0.5) & (slon <= LON_HI[-1] + 0.5)
slat_sub = slat[lat_mask]
slon_sub = slon[lon_mask]
lat_idx_chl = np.where(lat_mask)[0]
lon_idx_chl = np.where(lon_mask)[0]
print(f"  OC-CCI subset: {len(slat_sub)}x{len(slon_sub)} (from {len(slat)}x{len(slon)})")

slat_sub_arr = np.asarray(slat_sub)
slon_sub_arr = np.asarray(slon_sub)
subset_shape = (len(slat_sub_arr), len(slon_sub_arr))
flat_lookup_sub = np.arange(subset_shape[0] * subset_shape[1]).reshape(subset_shape)
chl_indices = {}
for i in range(N_LAT):
    ml = (slat_sub_arr >= LAT_LO[i]) & (slat_sub_arr < LAT_HI[i])
    if flip_chl:
        ml = ml[::-1]
    for j in range(N_LON):
        mo = (slon_sub_arr >= LON_LO[j]) & (slon_sub_arr < LON_HI[j])
        m = ml[:, None] & mo[None, :]
        if m.any():
            chl_indices[i, j] = flat_lookup_sub[m]

mvals = {}
for idx, fp in enumerate(chl_files):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if not ym:
        continue
    ym_s = ym.group(1)
    y, mo = int(ym_s[:4]), int(ym_s[4:])
    try:
        ds = xr.open_dataset(fp)
        ca_full = ds['chlor_a'].squeeze().values
        ds.close()
        if ca_full.ndim != 2:
            continue
        ca = ca_full[lat_idx_chl[0]:lat_idx_chl[-1] + 1,
                     lon_idx_chl[0]:lon_idx_chl[-1] + 1]
        ca_flat = ca.ravel()
        regrid = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
        for (i, j), fidx in chl_indices.items():
            v = ca_flat[fidx]
            v = v[np.isfinite(v)]
            if len(v) > 0:
                regrid[i, j] = np.mean(v)
        mvals[(y, mo)] = regrid
        if (idx + 1) % 25 == 0:
            print(f"    Processed {idx+1}/{len(chl_files)} files")
    except Exception as e:
        print(f"    file failed {os.path.basename(fp)}: {e}")
print(f"    Total months regridded: {len(mvals)}")
mvals_keys = sorted(mvals.keys())
monthly_stack = np.full((len(mvals_keys), N_LAT, N_LON), np.nan, dtype=np.float32)
for idx, key in enumerate(mvals_keys):
    monthly_stack[idx] = mvals[key]
monthly_times = [pd.Timestamp(f"{y}-{m:02d}-15") for (y, m) in mvals_keys]
chl_window = monthly_to_weekly_interp(monthly_stack, monthly_times, WEEKLY_T)

for t in range(N_WEEKS):
    chl_window[t][ssta_all_nan] = np.nan
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    chlorophyll = np.where(np.isnan(chl_window), np.nan,
                           np.log1p(np.maximum(chl_window, 0))).astype(np.float32)
print(f"  Chlorophyll: range=[{np.nanmin(chlorophyll):.4f}, {np.nanmax(chlorophyll):.4f}], NaN frac={np.isnan(chlorophyll).mean():.4f}")

# ═════════════════════════════════════════════════════════════
# STEP 4: PAR (MODIS monthly, Caribbean real data)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 4: Rebuilding PAR from MODIS monthly (Caribbean)...")
par_files = sorted(glob.glob(os.path.join(PAR_DIR, 'modis_par_??????.nc')))
if not par_files:
    par_window = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
    print("  WARNING: No MODIS PAR files; all NaN")
else:
    print(f"  Found {len(par_files)} MODIS monthly PAR files")
    ds_par0 = xr.open_dataset(par_files[0])
    plat = ds_par0['latitude'].values
    plon = ds_par0['longitude'].values
    ds_par0.close()
    flip_par = plat[0] > plat[-1]
    plat_arr = np.asarray(plat)
    plon_arr = np.asarray(plon)
    par_shape = (len(plat_arr), len(plon_arr))
    flat_lookup_par = np.arange(par_shape[0] * par_shape[1]).reshape(par_shape)
    par_indices = {}
    for i in range(N_LAT):
        ml = (plat_arr >= LAT_LO[i]) & (plat_arr < LAT_HI[i])
        if flip_par:
            ml = ml[::-1]
        for j in range(N_LON):
            mo = (plon_arr >= LON_LO[j]) & (plon_arr < LON_HI[j])
            m = ml[:, None] & mo[None, :]
            if m.any():
                par_indices[i, j] = flat_lookup_par[m]
    mvals_par = {}
    for idx, fp in enumerate(par_files):
        bn = os.path.basename(fp).replace('modis_par_', '').replace('.nc', '')
        y, mo = int(bn[:4]), int(bn[4:6])
        try:
            ds = xr.open_dataset(fp)
            pp = ds['par'].squeeze().values
            ds.close()
            if pp.ndim != 2:
                continue
            pp_flat = pp.ravel()
            regrid = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
            for (i, j), fidx in par_indices.items():
                v = pp_flat[fidx]
                v = v[np.isfinite(v)]
                if len(v) > 0:
                    regrid[i, j] = np.mean(v)
            mvals_par[(y, mo)] = regrid
        except Exception as e:
            print(f"    file failed {os.path.basename(fp)}: {e}")
    print(f"    Total months regridded: {len(mvals_par)}")
    mvals_par_keys = sorted(mvals_par.keys())
    monthly_stack_par = np.full((len(mvals_par_keys), N_LAT, N_LON), np.nan, dtype=np.float32)
    for idx, key in enumerate(mvals_par_keys):
        monthly_stack_par[idx] = mvals_par[key]
    monthly_times_par = [pd.Timestamp(f"{y}-{m:02d}-15") for (y, m) in mvals_par_keys]
    par_window = monthly_to_weekly_interp(monthly_stack_par, monthly_times_par, WEEKLY_T)

for t in range(N_WEEKS):
    par_window[t][ssta_all_nan] = np.nan
PAR_MIN, PAR_MAX = 18.20, 56.63   # SCS-consistent params (frozen SCS normalization scale)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    par_norm = ((par_window - PAR_MIN) / (PAR_MAX - PAR_MIN)).astype(np.float32)
    par_norm = np.clip(par_norm, 0, 1)
    par_norm[np.isnan(par_window)] = np.nan
print(f"  PAR: range=[{np.nanmin(par_norm):.4f}, {np.nanmax(par_norm):.4f}], NaN frac={np.isnan(par_norm).mean():.4f}")

# ═════════════════════════════════════════════════════════════
# STEP 5: ERA5 wind/ssr/sst_anomaly (Caribbean monthly)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 5: ERA5 wind/ssr/sst_anomaly (Caribbean)...")
era5_sst_ds = xr.open_dataset(os.path.join(ERA5_DIR, 'era5_sst_monthly.nc'))
era5_wind_ds = xr.open_dataset(os.path.join(ERA5_DIR, 'era5_wind_monthly.nc'))
era5_ssr_ds = xr.open_dataset(os.path.join(ERA5_DIR, 'era5_ssr_monthly.nc'))
era5_lats = era5_sst_ds['latitude'].values
era5_lons = era5_sst_ds['longitude'].values
era5_sst_monthly = era5_sst_ds['sst'].values.astype(np.float32)
u10 = era5_wind_ds['u10'].values.astype(np.float32)
v10 = era5_wind_ds['v10'].values.astype(np.float32)
era5_wind_monthly = np.sqrt(u10 ** 2 + v10 ** 2)
era5_ssr_monthly = era5_ssr_ds['ssr'].values.astype(np.float32)
time_var = 'valid_time' if 'valid_time' in era5_sst_ds else 'time'
era5_monthly_time = pd.DatetimeIndex(era5_sst_ds[time_var].values)
era5_sst_ds.close()
era5_wind_ds.close()
era5_ssr_ds.close()

era5_masks = {}
for i in range(N_LAT):
    for j in range(N_LON):
        m = build_regrid_mask(era5_lats, era5_lons, LAT_LO[i], LAT_HI[i], LON_LO[j], LON_HI[j])
        if m.any():
            era5_masks[i, j] = m

def regrid_monthly_to_grid(src_monthly, masks_dict, label=""):
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

era5_sst_regrid = regrid_monthly_to_grid(era5_sst_monthly, era5_masks)
era5_wind_regrid = regrid_monthly_to_grid(era5_wind_monthly, era5_masks)
era5_ssr_regrid = regrid_monthly_to_grid(era5_ssr_monthly, era5_masks)

era5_months = era5_monthly_time.month.values
era5_years = era5_monthly_time.year.values
clim = np.full((12, N_LAT, N_LON), np.nan, dtype=np.float32)
for m in range(1, 13):
    mm = (era5_months == m) & (era5_years <= 2010)   # causal: train-only basis
    if mm.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clim[m - 1] = np.nanmean(era5_sst_regrid[mm], axis=0)
clim_3d = np.stack([clim[m - 1] for m in era5_months])
era5_sst_anomaly_regrid = era5_sst_regrid - clim_3d
del era5_sst_regrid, clim_3d, clim

wind_speed = monthly_to_weekly_interp(era5_wind_regrid, era5_monthly_time, WEEKLY_T)
ssr = monthly_to_weekly_interp(era5_ssr_regrid, era5_monthly_time, WEEKLY_T)
sst_anomaly = monthly_to_weekly_interp(era5_sst_anomaly_regrid, era5_monthly_time, WEEKLY_T)
print(f"  wind_speed: range=[{np.nanmin(wind_speed):.4f}, {np.nanmax(wind_speed):.4f}]")
print(f"  ssr:        range=[{np.nanmin(ssr):.4f}, {np.nanmax(ssr):.4f}]")
print(f"  sst_anomaly: range=[{np.nanmin(sst_anomaly):.4f}, {np.nanmax(sst_anomaly):.4f}]")

# ═════════════════════════════════════════════════════════════
# STEP 6: Static features (ETOPO1 global / WOA23 global / IBTrACS / bathy-DTS)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 6: Regridding static features to 1° grid (Caribbean)...")
ds_topo = xr.open_dataset(BATHY_PATH)
topo_z = ds_topo['z'].values
topo_y = ds_topo['y'].values
topo_x = ds_topo['x'].values
ds_topo.close()
topo_lat_step = 180.0 / (len(topo_y) - 1)
topo_lon_step = 360.0 / (len(topo_x) - 1)

bathy_raw = np.zeros((N_LAT, N_LON), dtype=np.float32)
land_fraction = np.zeros((N_LAT, N_LON), dtype=np.float32)
for i in range(N_LAT):
    li_lo = int(np.clip(np.floor((LAT_LO[i] - topo_y[0]) / topo_lat_step), 0, len(topo_y) - 1))
    li_hi = int(np.clip(np.ceil((LAT_HI[i] - topo_y[0]) / topo_lat_step) + 1, 0, len(topo_y)))
    for j in range(N_LON):
        oi_lo = int(np.clip(np.floor((LON_LO[j] - topo_x[0]) / topo_lon_step), 0, len(topo_x) - 1))
        oi_hi = int(np.clip(np.ceil((LON_HI[j] - topo_x[0]) / topo_lon_step) + 1, 0, len(topo_x)))
        block = topo_z[li_lo:li_hi, oi_lo:oi_hi]
        valid = block[np.isfinite(block)]
        if len(valid) > 0:
            bathy_raw[i, j] = np.mean(valid)
            land_fraction[i, j] = np.sum(valid > 0) / len(valid)

ocean_pre = land_fraction < 0.5
bathy = np.zeros_like(bathy_raw)
bathy[ocean_pre] = np.log1p(-bathy_raw[ocean_pre]).astype(np.float32)

from scipy.ndimage import distance_transform_edt
land_mask = ~ocean_pre
dist_grid = distance_transform_edt(ocean_pre.astype(np.float32))
dist_to_shore_raw = dist_grid.astype(np.float32)
dist_to_shore = np.log1p(dist_to_shore_raw)

# --- Cyclone frequency from global IBTrACS (2003-2020 storm-center days) ---
print("  Cyclone frequency from IBTrACS (global)...")
cyclone_raw = np.zeros((N_LAT, N_LON), dtype=np.float32)
try:
    ds_cy = xr.open_dataset(CYCLONE_PATH)
    cy_lat = ds_cy['lat'].values.astype(np.float32).ravel()   # (storm, time)
    cy_lon = ds_cy['lon'].values.astype(np.float32).ravel()
    cy_time = ds_cy['time'].values.ravel()                    # datetime64
    ds_cy.close()
    yr = pd.DatetimeIndex(cy_time).year.values.astype(np.float32)
    ok = np.isfinite(cy_lat) & np.isfinite(cy_lon) & (yr >= 2003) & (yr <= 2020)
    cy_lat, cy_lon = cy_lat[ok], cy_lon[ok]
    for i in range(N_LAT):
        ml = (cy_lat >= LAT_LO[i]) & (cy_lat < LAT_HI[i])
        for j in range(N_LON):
            mo = (cy_lon >= LON_LO[j]) & (cy_lon < LON_HI[j])
            cyclone_raw[i, j] = float(np.sum(ml & mo)) / 18.0   # storm-center days / yr
    print(f"    IBTrACS points in box: {int(ok.sum())}, per-yr max={cyclone_raw.max():.2f}")
except Exception as e:
    print(f"    IBTrACS failed ({type(e).__name__}: {e}) — using 0 like GBR")
    cyclone_raw = np.zeros((N_LAT, N_LON), dtype=np.float32)
cmean, cstd = cyclone_raw.mean(), cyclone_raw.std()
cyclone_freq = ((cyclone_raw - cmean) / cstd).astype(np.float32) if cstd > 0 else cyclone_raw

# --- Salinity from WOA23 global annual climatology (surface) ---
print("  Loading WOA23 global salinity...")
ds_sal = xr.open_dataset(SAL_PATH, decode_times=False)
sal_lat = np.asarray(ds_sal['lat'].values)
sal_lon = np.asarray(ds_sal['lon'].values)
sal_data = np.array(ds_sal['s_an'].values)
ds_sal.close()
# WOA23 s_an shape may be (depth, lat, lon) or (time, depth, lat, lon)
if sal_data.ndim == 4:
    sal_surface = sal_data[0, 0]       # (time=0, depth=0)
elif sal_data.ndim == 3:
    sal_surface = sal_data[0]          # (depth=0)
else:
    sal_surface = sal_data.squeeze()
    if sal_surface.ndim == 3:
        sal_surface = sal_surface[0]
flip_sal = sal_lat[0] > sal_lat[-1]
salinity = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
for i in range(N_LAT):
    ml = (sal_lat >= LAT_LO[i]) & (sal_lat < LAT_HI[i])
    if flip_sal:
        ml = ml[::-1]
    for j in range(N_LON):
        mo = (sal_lon >= LON_LO[j]) & (sal_lon < LON_HI[j])
        m = ml[:, None] & mo[None, :]
        if m.any():
            block = sal_surface[m]
            valid = block[np.isfinite(block)]
            if len(valid) > 0:
                salinity[i, j] = np.mean(valid)

print(f"  bathymetry: range=[{np.nanmin(bathy):.4f}, {np.nanmax(bathy):.4f}]")
print(f"  dist_to_shore: range=[{np.nanmin(dist_to_shore):.4f}, {np.nanmax(dist_to_shore):.4f}]")
print(f"  ocean cells: {ocean_pre.sum()}")
print(f"  salinity NaN frac: {np.isnan(salinity).mean():.4f}")

salinity = fill_nan_cells_spatial(salinity, ocean_pre, k=5)
print(f"  salinity post-fill NaN frac: {np.isnan(salinity).mean():.4f}")

# ═════════════════════════════════════════════════════════════
# STEP 7: Ocean mask + spatial fills
# ═════════════════════════════════════════════════════════════
print("\nSTEP 7: Building ocean mask and filling dynamic gaps...")
ocean_mask = ocean_pre & ~ssta_all_nan
land_mask_out = ~ocean_mask
print(f'  ocean cells: {ocean_mask.sum()}, land cells: {land_mask_out.sum()}')

# Exclude finite-y-on-land from targets (aligned with training label_mask
# = finite y & ocean). Land cells that happen to carry SSTA data would
# otherwise keep a finite target; the model never saw them at training time.
n_y_land = int(np.isfinite(y_regression)[:, land_mask_out].sum())
for arr in [y_regression, y_warning, y_classification]:
    arr[:, land_mask_out] = np.nan
print(f'  [land-mask] finite-y-on-land EXCLUDED: {n_y_land} cells-weeks')

for arr in [wind_speed, ssr, sst_anomaly]:
    for t in range(N_WEEKS):
        arr[t][ssta_all_nan] = np.nan
for t in range(N_WEEKS):
    sst_window[t][ssta_all_nan] = np.nan

print("  Filling sst_anomaly NaN ocean cells...")
sst_anomaly = fill_nan_cells_spatial(sst_anomaly, ocean_pre & ~ssta_all_nan, k=5)
print("  Filling chlorophyll NaN ocean cells...")
chlorophyll = fill_nan_cells_spatial(chlorophyll, ocean_pre & ~ssta_all_nan, k=5)
print("  Filling PAR NaN ocean cells...")
par_norm = fill_nan_cells_spatial(par_norm, ocean_pre & ~ssta_all_nan, k=5)

# ═════════════════════════════════════════════════════════════
# STEP 8: Derived channels (d_hotspot_dt, cumsum_4w, cumsum_12w)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 8: Computing derived channels...")
d_hotspot_dt = np.full_like(hotspot_window, np.nan)
d_hotspot_dt[1:] = hotspot_window[1:] - hotspot_window[:-1]

cumsum_4w = np.full_like(hotspot_window, np.nan)
cumsum_12w = np.full_like(hotspot_window, np.nan)
for w in range(N_WEEKS):
    win4 = hotspot_window[max(0, w - 3):w + 1]
    win12 = hotspot_window[max(0, w - 11):w + 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v4 = np.isfinite(win4).sum(axis=0)
        v12 = np.isfinite(win12).sum(axis=0)
        cumsum_4w[w] = np.where(v4 > 0, np.nansum(win4, axis=0), np.nan).astype(np.float32)
        cumsum_12w[w] = np.where(v12 > 0, np.nansum(win12, axis=0), np.nan).astype(np.float32)

print(f"  d_hotspot_dt: range=[{np.nanmin(d_hotspot_dt):.4f}, {np.nanmax(d_hotspot_dt):.4f}]")
print(f"  cumsum_4w: range=[{np.nanmin(cumsum_4w):.4f}, {np.nanmax(cumsum_4w):.4f}]")
print(f"  cumsum_12w: range=[{np.nanmin(cumsum_12w):.4f}, {np.nanmax(cumsum_12w):.4f}]")

# ═════════════════════════════════════════════════════════════
# STEP 9: Write output NetCDF
# ═════════════════════════════════════════════════════════════
print("\nSTEP 9: Writing output NetCDF...")
data_vars = {
    'sst': xr.DataArray(sst_window.astype('float32'), dims=['time', 'lat', 'lon']),
    'ssta': xr.DataArray(ssta_window.astype('float32'), dims=['time', 'lat', 'lon']),
    'hotspot': xr.DataArray(hotspot_window, dims=['time', 'lat', 'lon']),
    'chlorophyll': xr.DataArray(chlorophyll, dims=['time', 'lat', 'lon']),
    'par': xr.DataArray(par_norm, dims=['time', 'lat', 'lon']),
    'seasonal_sin': xr.DataArray(seasonal_sin_window.astype('float32'), dims=['time', 'lat', 'lon']),
    'wind_speed': xr.DataArray(wind_speed, dims=['time', 'lat', 'lon']),
    'ssr': xr.DataArray(ssr, dims=['time', 'lat', 'lon']),
    'sst_anomaly': xr.DataArray(sst_anomaly, dims=['time', 'lat', 'lon']),
    'd_hotspot_dt': xr.DataArray(d_hotspot_dt, dims=['time', 'lat', 'lon']),
    'hotspot_cumsum_4w': xr.DataArray(cumsum_4w, dims=['time', 'lat', 'lon']),
    'hotspot_cumsum_12w': xr.DataArray(cumsum_12w, dims=['time', 'lat', 'lon']),
    'seasonal_cos': xr.DataArray(seasonal_cos_window.astype('float32'), dims=['time', 'lat', 'lon']),
    'y_regression': xr.DataArray(y_regression, dims=['time', 'lat', 'lon']),
    'y_warning': xr.DataArray(y_warning, dims=['time', 'lat', 'lon']),
    'y_classification': xr.DataArray(y_classification, dims=['time', 'lat', 'lon']),
    'land_mask': xr.DataArray(np.broadcast_to(land_mask_out.astype('float32'),
                                              (N_WEEKS, N_LAT, N_LON)).copy(), dims=['time', 'lat', 'lon']),
    'ocean_mask': xr.DataArray(ocean_mask.astype('float32'), dims=['lat', 'lon']),
    'bathymetry': xr.DataArray(bathy, dims=['lat', 'lon']),
    'dist_to_shore': xr.DataArray(dist_to_shore, dims=['lat', 'lon']),
    'cyclone_frequency': xr.DataArray(cyclone_freq, dims=['lat', 'lon']),
    'salinity': xr.DataArray(salinity, dims=['lat', 'lon']),
}
ds_out = xr.Dataset(
    data_vars,
    coords={'time': WEEKLY_T.astype('datetime64[ns]'), 'lat': LATS, 'lon': LONS},
    attrs={
        'title': 'Caribbean 1° coral bleaching dataset (built from raw sources, Jensen-aware DHW)',
        'history': 'Built by data/source_builders/build_caribbean_1deg_from_raw.py',
        'dynamic_channels': 'sst(CRW raw), ssta, hotspot, chlorophyll, par, seasonal_sin, wind_speed, ssr, sst_anomaly, d_hotspot_dt, hotspot_cumsum_4w, hotspot_cumsum_12w, seasonal_cos (13)',
        'static_channels': 'bathymetry, dist_to_shore, cyclone_frequency, salinity, ocean_mask (5)',
        'targets': 'y_regression=clip(DHW/10,0,1), y_warning=clip(DHW/8,0,1), y_classification=I(DHW>=4)',
        'DHW_definition': '84-day rolling sum of clip(hotspot,0,+inf) at NATIVE 5km, then block-avg to 1° (Jensen-aware)',
        'par_source': 'MODIS monthly (coastwatch_par_caribbean), min-max [0,1] with SCS params (18.20, 56.63)',
        'era5_source': 'ERA5 Caribbean monthly (era5_caribbean); sst_anomaly climatology basis years<=2010 (causal)',
        'cyclone_source': 'IBTrACS global 2003-2020 storm-center days/yr, z-scored',
        'creation_date': str(np.datetime64('today')),
    },
)
encode = {z: {'zlib': True, 'complevel': 5, 'dtype': 'float32'}
          for z in data_vars if data_vars[z].dtype == np.float32}
for z in data_vars:
    if z not in encode:
        encode[z] = {'dtype': data_vars[z].dtype}
ds_out.to_netcdf(OUTPUT_NC, encoding=encode)
mb = os.path.getsize(OUTPUT_NC) / 1024 / 1024
print(f"\nWrote {OUTPUT_NC} ({mb:.1f} MB)")

# ═════════════════════════════════════════════════════════════
# STEP 10: Validation
# ═════════════════════════════════════════════════════════════
print("\nSTEP 10: Validation...")
print(f"{'Variable':22s} {'Shape':20s} {'NaN%':>8s} {'Min':>10s} {'Max':>10s}")
print("-" * 74)
for name in data_vars:
    v = data_vars[name].values
    fin = v[np.isfinite(v)]
    if v.ndim == 3:
        row = f"{np.isnan(v).mean()*100:7.2f}" if len(fin) else "  100.00"
        mn = f"{np.nanmin(v):.4f}" if len(fin) else "-"
        mx = f"{np.nanmax(v):.4f}" if len(fin) else "-"
    else:
        row = f"{np.isnan(v).mean()*100:7.2f}" if len(fin) else "  100.00"
        mn = f"{np.nanmin(v):.4f}" if len(fin) else "-"
        mx = f"{np.nanmax(v):.4f}" if len(fin) else "-"
    print(f"{name:22s} {str(v.shape):20s} {row:>8s} {mn:>10s} {mx:>10s}")
om = data_vars['ocean_mask'].values
print(f"\nOcean cells (1°): {int(om.sum())}, Land cells: {int((~om.astype(bool)).sum())}")
print(f"\n=== CARIBBEAN BUILD COMPLETE ===")
