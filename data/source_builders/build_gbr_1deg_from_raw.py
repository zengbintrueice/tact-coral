#!/usr/bin/env python3
"""
GBR 1° Raw-from-Source Dataset Builder
========================================

Builds a 1° (15×12) dataset for the Great Barrier Reef (GBR) directly from
raw data sources, using the SAME pipeline as build_v9_1deg_from_raw.py (SCS).

GBR grid: lat -24..-10 (15 cells), lon 142..154 (12 cells)
Time: 1985-04-01 ~ 2020-12-31, weekly resolution (~1935 weeks)

CRITICAL: Jensen-aware DHW — compute at native 5km resolution (clip + 84-day
rolling sum), THEN spatial-block-average to 1°. This preserves thermal stress
peaks (Jensen's inequality).

Data sources:
  - CRW 5km daily SST/SSTA/Hotspot (from crw_5km_gbr/)
  - OC-CCI v6 monthly chlorophyll GBR subset → weekly causal (month-end forward-fill, NO future leakage), log1p
  - ETOPO1 bathymetry GBR subset
  - WOA23 salinity GBR subset
  - PAR: latitude-based climatology (MODIS not available for GBR)
  - ERA5: optional (if era5_gbr/ exists), else fill with SCS-channel-mean (=0 norm)

Output:
  - data/gbr_coral_bleaching_dataset_v9_1deg_raw.nc
  - 13 dynamic + 5 static + 3 target + 2 mask = 23 variables
  - Shape: [~1935, 15, 12]
"""
import os, sys, glob, warnings
import numpy as np
import xarray as xr
import pandas as pd
import re
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# Jensen-aware DHW target construction (independent reusable module,
# bundled locally in this folder — self-contained in the LTTS release)
from jensen_dhw import (
    compute_dhw_5km, build_aggregation_matrix, aggregate_to_1deg, dhw_to_target,
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Source via ASCII junction D:\piecf_dns → new_sources.
# MUST use ASCII path: netCDF4's C library cannot open .nc files under Chinese
# paths on Windows (FileNotFoundError even though the file exists).
DNS = os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"" if os.path.isdir(os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"") else os.path.join(
    PROJECT_ROOT, 'backups', 'piecf_a6_nophys_20260718', 'a6_xgb_experiment', 'data', 'new_sources')
CRW_GBR_DIR = os.path.join(DNS, 'crw_5km_gbr')
CHL_GBR_DIR = os.path.join(DNS, 'chlorophyll_gbr')
BATHY_GBR_PATH = os.path.join(DNS, 'bathymetry_gbr', 'bathy_gbr.nc')
SAL_GBR_PATH = os.path.join(DNS, 'salinity_gbr', 'salinity_gbr.nc')
ERA5_GBR_DIR = os.path.join(DNS, 'era5_gbr')
# Output via ASCII junction — netCDF4 cannot write .nc under Chinese paths on Windows
OUTPUT_DIR = r'D:\piecf_crw_data'  # ASCII: netCDF4 cannot write .nc under Chinese paths
OUTPUT_NC = os.path.join(OUTPUT_DIR, 'gbr_coral_bleaching_dataset_v9_1deg_raw.nc')

# ─────────────────────────────────────────────────────────────
# 1° target grid (cell centers, 15×12)
# ─────────────────────────────────────────────────────────────
N_LAT, N_LON = 15, 12
LATS = np.arange(-24, -9, 1.0) + 0.5  # [-23.5, -22.5, ..., -9.5]
LONS = np.arange(142, 154, 1.0) + 0.5  # [142.5, 143.5, ..., 153.5]

# Time: weekly from 1985-04-01 (need pre-2003 lookback for DHW)
N_WEEKS = 1935  # ~37 years * 52.25
WEEKLY = np.timedelta64(7, 'D')
TIME_START = np.datetime64('1985-04-01')
WEEKLY_T = TIME_START + np.arange(N_WEEKS) * WEEKLY
tsp = pd.DatetimeIndex(WEEKLY_T)

# ─────────────────────────────────────────────────────────────
# CRW 5km raw grid (source) — GBR region
# lat: -23.975..-8.975, 301 pts, descending
# lon: 142.025..154.025, 241 pts, ascending
# ─────────────────────────────────────────────────────────────
CRW_LAT = np.arange(-8.975, -24.0 - 0.025, -0.05)  # -8.975..-23.975, 301 pts, descending
CRW_LON = np.arange(142.025, 154.025 + 0.025, 0.05)  # 142.025..154.025, 241 pts

def cell_bounds(centers):
    """Compute cell edges from cell centers (assumes uniform spacing)."""
    centers = np.asarray(centers, dtype=float)
    d = np.abs(np.diff(centers).mean()) / 2
    return centers - d, centers + d

# 1° target grid cell boundaries
LAT_LO, LAT_HI = cell_bounds(LATS)  # -24.0..-9.0
LON_LO, LON_HI = cell_bounds(LONS)  # 142.0..154.0

def build_regrid_mask(src_lat, src_lon, lat_lo, lat_hi, lon_lo, lon_hi):
    """Build boolean mask mapping source grid to target cell (i, j)."""
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    flipped = src_lat[0] > src_lat[-1]  # descending latitude (CRW)
    m = (src_lat >= lat_lo) & (src_lat < lat_hi)
    if flipped:
        m = m[::-1]
    n = (src_lon >= lon_lo) & (src_lon < lon_hi)
    return m[:, None] & n[None, :]


def monthly_to_weekly_interp(monthly, monthly_time, target_weeks, method='causal'):
    """Interpolate monthly means to weekly resolution.

    method='causal' (default — NO future leakage):
        A monthly mean is an after-the-fact statistic: it averages days LATER
        than any week inside that month. So each month is treated as available
        only after the month ENDS, and week w takes the most recent FULLY-
        COMPLETED month (forward-fill / step hold) — never the current or a
        future month. Per-cell forward-fill on the monthly series handles gaps
        (e.g. OC-CCI cloud mask) using only past valid months.
        Cost: features lag real time by ~2-6 weeks (inherent to monthly data).

    method='linear' (legacy — HAS future leakage):
        Linear interpolation between monthly timestamps. Weeks in early/mid
        month borrow the current month's full mean (and possibly next month),
        so input features carry mild future information. Kept for ablation.
    """
    t_weekly = np.array([pd.Timestamp(w).toordinal() for w in target_weeks], dtype=np.float64)
    T_m, H, W = monthly.shape
    n_weeks = len(target_weeks)
    flat_monthly = monthly.reshape(T_m, H * W)
    flat_out = np.full((n_weeks, H * W), np.nan, dtype=np.float32)

    if method == 'causal':
        # Availability timestamp = last day of that month
        month_avail = np.array([
            pd.Timestamp(m).to_period('M').to_timestamp(how='end').toordinal()
            for m in monthly_time
        ], dtype=np.float64)
        # Most recent completed month (month_avail <= week) via forward-fill
        idx = np.searchsorted(month_avail, t_weekly, side='right') - 1  # (n_weeks,)
        valid_week = idx >= 0
        for k in range(H * W):
            ts_ff = pd.Series(flat_monthly[:, k]).ffill().values  # past-only fill
            col = np.full(n_weeks, np.nan, dtype=np.float32)
            col[valid_week] = ts_ff[idx[valid_week]]
            flat_out[:, k] = col
        return flat_out.reshape(n_weeks, H, W)

    # legacy linear interpolation (future-leaking)
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
    """Fill NaN cells with inverse-distance-weighted mean of nearest valid ocean neighbors."""
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


def build_par_climatology_weekly():
    """Build a simple latitude-based weekly PAR climatology for GBR.

    PAR is primarily a function of latitude and day-of-year. We use a simple
    model: PAR(lat, doy) ≈ PAR_max * max(0, sin(lat_factor)) * clear_sky_factor.

    For a zero-shot cross-region test, this provides seasonal structure.
    Min-max normalized to [0, 1] using the same SCS params.
    """
    PAR_MIN, PAR_MAX = 18.20, 56.63

    doy = np.array([t.dayofyear for t in tsp])
    lat_abs = np.abs(LATS)  # [9.5, 10.5, ..., 23.5]

    # Simple model: peak PAR at summer solstice (doy~172 for SH = July),
    # scaled by latitude (lower PAR at higher latitudes).
    par = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float32)
    for i, la in enumerate(lat_abs):
        # Southern hemisphere: summer ~ Dec-Feb, peak ~ doy 355 (Dec 21)
        seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * (doy - 355) / 365)
        lat_factor = np.cos(np.radians(la - 15))  # peak ~15°S
        par[:, i, :] = PAR_MIN + (PAR_MAX - PAR_MIN) * seasonal[:, None] * lat_factor

    par_norm = np.clip((par - PAR_MIN) / (PAR_MAX - PAR_MIN), 0, 1)
    return par_norm.astype(np.float32)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
print("=" * 60)
print("GBR 1° Dataset Builder (RAW from sources, Jensen-aware DHW)")
print("=" * 60)
print(f"Grid: {N_LAT}x{N_LON}, Time: {N_WEEKS} weeks ({N_LAT*N_LON} cells)")
print(f"Time range: {str(TIME_START)[:10]} to {str(WEEKLY_T[-1])[:10]}")
print(f"CRW 5km grid: {len(CRW_LAT)}×{len(CRW_LON)} (GBR)")
print()

# Build regrid masks (for the GBR 301×241 grid)
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
print(f"  Built {n_cells_std} regrid masks (GBR 301×241 grid)")

# CSR sparse aggregation matrix: 5km pixels → 1° cells
print("  Building CSR sparse aggregation matrix (5km → 1°)...")
rows_csr, cols_csr = [], []
for cell_k, fidx in enumerate(cell_flat_indices_std):
    for pix in fidx:
        rows_csr.append(cell_k); cols_csr.append(pix)
M_csr = csr_matrix((np.ones(len(rows_csr), dtype=np.float32),
                    (rows_csr, cols_csr)),
                   shape=(n_cells_std, len(CRW_LAT) * len(CRW_LON)))
cell_size = np.asarray(M_csr.sum(axis=1)).ravel().astype(np.float32)
print(f"  CSR matrix: {M_csr.shape}, nnz={M_csr.nnz}, cell_size: min={cell_size.min():.0f} max={cell_size.max():.0f}")

# ═════════════════════════════════════════════════════════════
# STEP 1: Regrid CRW 5km daily → 1° weekly block-avg
# ═════════════════════════════════════════════════════════════
print("\nSTEP 1: Regridding CRW 5km daily (SSTA + Hotspot + SST) → 1° weekly block-avg...")

# Collect CRW month files
ssta_by_month, hs_by_month, sst_by_month = {}, {}, {}
for fp in sorted(glob.glob(os.path.join(CRW_GBR_DIR, 'crw_sstanomaly_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        ssta_by_month[ym.group(1)] = fp
for fp in sorted(glob.glob(os.path.join(CRW_GBR_DIR, 'crw_hotspot_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        hs_by_month[ym.group(1)] = fp
for fp in sorted(glob.glob(os.path.join(CRW_GBR_DIR, 'crw_sst_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        sst_by_month[ym.group(1)] = fp

print(f"  SSTA months: {len(ssta_by_month)}, Hotspot months: {len(hs_by_month)}, SST months: {len(sst_by_month)}")

# Months needed for weekly window
months_weekly = set()
for w in range(N_WEEKS):
    for dt64 in [WEEKLY_T[w], WEEKLY_T[w] + np.timedelta64(6, 'D')]:
        dt = pd.Timestamp(dt64)
        months_weekly.add(f"{dt.year}{dt.month:02d}")
months_weekly = sorted(months_weekly)

# All CRW months (for daily hotspot/DHW accumulation)
all_crw_months = sorted(set(list(ssta_by_month.keys()) + list(hs_by_month.keys())))
months_needed = sorted(set(months_weekly) | set(all_crw_months))
print(f"  Weekly window: {len(months_weekly)} months, all CRW: {len(all_crw_months)}, union: {len(months_needed)}")

# Weekly accumulation arrays
ssta_sum = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
ssta_cnt = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)
hs_sum   = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
hs_cnt   = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)
sst_sum  = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
sst_cnt  = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)

# Daily time axis for DHW
TOTAL_START = np.datetime64('1985-04-01')
TOTAL_END   = np.datetime64('2020-12-31')
DAILY_T = np.arange(TOTAL_START, TOTAL_END + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
N_DAILY = len(DAILY_T)
t0_daily = pd.Timestamp(TOTAL_START)

print(f'  Daily hotspot grid: [{N_DAILY} days × {N_LAT}×{N_LON}] = ~{N_DAILY * N_LAT * N_LON * 4 / 1e6:.0f} MB')

# Jensen-aware DHW: accumulate ALL daily 5km hotspot, then compute via jensen_dhw.py
WIN = 84  # CRW standard (kept for metadata)
hs_daily_5km = np.full((N_DAILY, len(CRW_LAT), len(CRW_LON)), np.nan, dtype=np.float32)

n_months_done = 0
for month_str in months_needed:
    if month_str not in ssta_by_month:
        continue
    try:
        # Load SSTA
        ds_ssta = xr.open_dataset(ssta_by_month[month_str])
        ssta_daily = ds_ssta['CRW_SSTANOMALY'].values
        ssta_time = pd.DatetimeIndex(ds_ssta['time'].values)
        ds_ssta.close()

        # Load Hotspot
        hs_daily = None
        if month_str in hs_by_month:
            ds_hs = xr.open_dataset(hs_by_month[month_str])
            hs_daily = ds_hs['CRW_HOTSPOT'].values
            ds_hs.close()

        # Load SST
        sst_daily = None
        if month_str in sst_by_month:
            ds_sst = xr.open_dataset(sst_by_month[month_str])
            sst_daily = ds_sst['CRW_SST'].values
            ds_sst.close()

        n_days = len(ssta_time)

        # Flat arrays
        ssta_flat = ssta_daily.reshape(n_days, -1).astype(np.float64)
        ssta_valid = np.isfinite(ssta_flat)
        ssta_flat[~ssta_valid] = 0.0

        if hs_daily is not None:
            hs_flat = hs_daily.reshape(n_days, -1).astype(np.float64)
            hs_valid = np.isfinite(hs_flat)
            hs_flat[~hs_valid] = 0.0

        sst_flat = None
        sst_valid = None
        if sst_daily is not None:
            sst_flat = sst_daily.reshape(n_days, -1).astype(np.float64)
            sst_valid = np.isfinite(sst_flat)
            sst_flat[~sst_valid] = 0.0

        # Week assignment
        t0 = pd.Timestamp(TIME_START)
        day_offsets = np.array([(pd.Timestamp(dt) - t0).days for dt in ssta_time])
        day_weeks = day_offsets // 7
        day_in_window = (day_weeks >= 0) & (day_weeks < N_WEEKS)

        # Weekly block-avg accumulation
        for cell_k, ((i, j), fidx) in enumerate(zip(cell_ij_std, cell_flat_indices_std)):
            w_idx = day_weeks[day_in_window]

            # SSTA
            day_sums = ssta_flat[:, fidx].sum(axis=1)
            day_counts = ssta_valid[:, fidx].sum(axis=1)
            np.add.at(ssta_sum[:, i, j], w_idx, day_sums[day_in_window])
            np.add.at(ssta_cnt[:, i, j], w_idx, day_counts[day_in_window])

            # Hotspot
            if hs_daily is not None:
                day_hs_sums = hs_flat[:, fidx].sum(axis=1)
                day_hs_counts = hs_valid[:, fidx].sum(axis=1)
                np.add.at(hs_sum[:, i, j], w_idx, day_hs_sums[day_in_window])
                np.add.at(hs_cnt[:, i, j], w_idx, day_hs_counts[day_in_window])

            # SST
            if sst_daily is not None:
                day_sst_sums = sst_flat[:, fidx].sum(axis=1)
                day_sst_counts = sst_valid[:, fidx].sum(axis=1)
                np.add.at(sst_sum[:, i, j], w_idx, day_sst_sums[day_in_window])
                np.add.at(sst_cnt[:, i, j], w_idx, day_sst_counts[day_in_window])

        # Jensen-aware DHW: accumulate daily 5km hotspot for later processing
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

# Compute weekly means
ssta_window = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
hotspot_raw = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
sst_window = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)

valid_ssta = ssta_cnt > 0
valid_hs = hs_cnt > 0
valid_sst = sst_cnt > 0
ssta_window[valid_ssta] = (ssta_sum[valid_ssta] / ssta_cnt[valid_ssta]).astype(np.float32)
hotspot_raw[valid_hs] = (hs_sum[valid_hs] / hs_cnt[valid_hs]).astype(np.float32)
sst_window[valid_sst] = (sst_sum[valid_sst] / sst_cnt[valid_sst]).astype(np.float32)

print(f"  SSTA: shape={ssta_window.shape}, NaN frac={np.isnan(ssta_window).mean():.4f}")
print(f"  Hotspot (raw): range=[{np.nanmin(hotspot_raw):.3f}, {np.nanmax(hotspot_raw):.3f}]")
print(f"  CRW SST: shape={sst_window.shape}, NaN frac={np.isnan(sst_window).mean():.4f}")
print(f"    SST range=[{np.nanmin(sst_window):.3f}, {np.nanmax(sst_window):.3f}]")

# Clip hotspot to [0, inf)
hotspot_window = np.full_like(hotspot_raw, np.nan)
hotspot_window[valid_hs] = np.clip(hotspot_raw[valid_hs], 0, None)

# Seasonal_sin
weeks_of_year = np.array([t.isocalendar()[1] for t in tsp])
seasonal_sin_window = np.sin(2 * np.pi * weeks_of_year / 52).astype(np.float32)
seasonal_sin_window = np.broadcast_to(seasonal_sin_window[:, None, None], (N_WEEKS, N_LAT, N_LON)).copy()

# Seasonal_cos
seasonal_cos_window = np.cos(2 * np.pi * weeks_of_year / 52).astype(np.float32)
seasonal_cos_window = np.broadcast_to(seasonal_cos_window[:, None, None], (N_WEEKS, N_LAT, N_LON)).copy()

# SSTA all-NaN cell mask
ssta_nan_frac = np.isnan(ssta_window).mean(axis=0)
ssta_all_nan = ssta_nan_frac > 0.9999
print(f"  SSTA all-NaN cells: {ssta_all_nan.sum()} / {N_LAT * N_LON}")

del M_csr, cell_size
print("  CSR aggregation matrix freed")

# ═════════════════════════════════════════════════════════════
# STEP 2: Generate targets from NATIVE 5km DHW (Jensen-aware)
# Uses the independent jensen_dhw.py module.
# ═════════════════════════════════════════════════════════════
print("\nSTEP 2: Computing Jensen-aware DHW targets via jensen_dhw.py...")

# 2a. Native-5km rolling DHW
print("  2a. Computing native-5km DHW (compute_dhw_5km)...")
dhw_5km = compute_dhw_5km(hs_daily_5km)
del hs_daily_5km
print(f"    5km DHW: range=[{np.nanmin(dhw_5km):.4f}, {np.nanmax(dhw_5km):.4f}] deg-days")

# 2b. Build aggregation matrix (5km → 1°)
print("  2b. Building 5km→1° aggregation matrix (build_aggregation_matrix)...")
lat_edges = np.concatenate([LAT_LO, [LAT_HI[-1]]])
lon_edges = np.concatenate([LON_LO, [LON_HI[-1]]])
M_jensen, cell_size_jensen = build_aggregation_matrix(CRW_LAT, CRW_LON, lat_edges, lon_edges)
n_cells = M_jensen.shape[0]
print(f"    Aggregation matrix: {M_jensen.shape}, {N_LAT}×{N_LON}={N_LAT*N_LON} cells")

# 2c. Aggregate each day: 5km → 1°
print("  2c. Aggregating 5km→1° daily DHW (aggregate_to_1deg)...")
dhw_daily_1deg = np.full((N_DAILY, n_cells), np.nan, dtype=np.float32)
for d in range(N_DAILY):
    dhw_daily_1deg[d] = aggregate_to_1deg(dhw_5km[d].ravel(), M_jensen, cell_size_jensen)
del dhw_5km, M_jensen, cell_size_jensen
print(f"    1° daily DHW: range=[{np.nanmin(dhw_daily_1deg):.4f}, {np.nanmax(dhw_daily_1deg):.4f}] deg-days")

# 2d. Daily → weekly (7-day mean)
print("  2d. Aggregating daily → weekly...")
dhw_weekly_degdays = np.full((N_WEEKS, n_cells), np.nan, dtype=np.float32)
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
            np.nan
        ).astype(np.float32)
del dhw_daily_1deg

# 2e. degree-days → degree-weeks
dhw_weekly = dhw_weekly_degdays / 7.0
del dhw_weekly_degdays
print(f"    DHW (weekly, deg-weeks): range=[{np.nanmin(dhw_weekly):.4f}, {np.nanmax(dhw_weekly):.4f}]")

# 2f. Generate targets via jensen_dhw.dhw_to_target
print("  2f. Generating targets (dhw_to_target)...")
y_regression = dhw_to_target(dhw_weekly, scale=10.0)
y_warning = dhw_to_target(dhw_weekly, scale=8.0)
y_classification = np.where(np.isnan(dhw_weekly), np.nan,
                            (dhw_weekly >= 4.0).astype(np.float32))

# 2g. Reshape [N_WEEKS, n_cells] → [N_WEEKS, N_LAT, N_LON]
y_regression = y_regression.reshape(N_WEEKS, N_LAT, N_LON)
y_warning = y_warning.reshape(N_WEEKS, N_LAT, N_LON)
y_classification = y_classification.reshape(N_WEEKS, N_LAT, N_LON)

# Apply SSTA all-NaN cell mask
ssta_all_nan_3d = np.broadcast_to(ssta_all_nan[None, :, :], y_regression.shape)
for arr in [y_regression, y_warning, y_classification]:
    arr[ssta_all_nan_3d] = np.nan

valid_targets = np.isfinite(y_regression)
n_pos = int((y_classification[valid_targets] >= 1).sum())
n_total = int(valid_targets.sum())
print(f"  y_regression:     range=[{np.nanmin(y_regression):.4f}, {np.nanmax(y_regression):.4f}]")
print(f"  y_warning:        range=[{np.nanmin(y_warning):.4f}, {np.nanmax(y_warning):.4f}]")
print(f"  y_classification: positive rate={n_pos}/{n_total} = {n_pos/n_total*100:.2f}%")

# ═════════════════════════════════════════════════════════════
# STEP 3: Chlorophyll from OC-CCI v6 GBR subset
# ═════════════════════════════════════════════════════════════
print("\nSTEP 3: Rebuilding chlorophyll from OC-CCI v6.0 GBR subset...")
chl_files = sorted(glob.glob(os.path.join(CHL_GBR_DIR, 'oc_cci_chlor_a_*_fv6.0.nc')))

if not chl_files:
    chlorophyll = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
    print("  WARNING: No OC-CCI GBR files; all NaN")
else:
    print(f"  Found {len(chl_files)} OC-CCI GBR monthly files")
    # Open first VALID file to get source grid (skip empty/corrupt files)
    slat = slon = None
    for fp0 in chl_files:
        try:
            ds_chl0 = xr.open_dataset(fp0)
            tlat = np.array(ds_chl0['lat']); tlon = np.array(ds_chl0['lon'])
            ds_chl0.close()
            if tlat.size > 0 and tlon.size > 0:
                slat, slon = tlat, tlon
                break
        except Exception:
            continue
    if slat is None:
        raise ValueError('No valid OC-CCI GBR file found for grid detection')
    flip_chl = slat[0] > slat[-1]

    # Build regrid FLAT INDICES on the subsetted grid
    subset_shape = (len(slat), len(slon))
    flat_lookup_sub = np.arange(subset_shape[0] * subset_shape[1]).reshape(subset_shape)

    chl_indices = {}
    for i in range(N_LAT):
        ml = (slat >= LAT_LO[i]) & (slat < LAT_HI[i])
        if flip_chl:
            ml = ml[::-1]
        for j in range(N_LON):
            mo = (slon >= LON_LO[j]) & (slon < LON_HI[j])
            m = ml[:, None] & mo[None, :]
            if m.any():
                chl_indices[i, j] = flat_lookup_sub[m]

    mvals = {}
    for idx, fp in enumerate(chl_files):
        ym = re.search(r'(\d{6})', os.path.basename(fp))
        if not ym:
            continue
        ym_s = ym.group(1); y = int(ym_s[:4]); mo = int(ym_s[4:])
        try:
            ds = xr.open_dataset(fp)
            ca = ds['chlor_a'].squeeze().values
            ds.close()
            if ca.ndim != 2:
                continue
            ca_flat = ca.ravel()
            regrid = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
            for (i, j), fidx in chl_indices.items():
                v = ca_flat[fidx]
                v = v[np.isfinite(v)]
                if len(v) > 0:
                    regrid[i, j] = np.mean(v)
            mvals[(y, mo)] = regrid
            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx+1}/{len(chl_files)} files")
        except Exception as e:
            print(f"    file failed {os.path.basename(fp)}: {e}")

    print(f"    Total months regridded: {len(mvals)}")

    # Build monthly stack for linear interpolation
    mvals_keys = sorted(mvals.keys())
    n_months = len(mvals_keys)
    monthly_stack = np.full((n_months, N_LAT, N_LON), np.nan, dtype=np.float32)
    for idx, key in enumerate(mvals_keys):
        monthly_stack[idx] = mvals[key]

    monthly_times = [pd.Timestamp(f"{y}-{m:02d}-15") for (y, m) in mvals_keys]
    chl_window = monthly_to_weekly_interp(monthly_stack, monthly_times, WEEKLY_T)

# Apply SSTA-NaN mask
for t in range(N_WEEKS):
    chl_window[t][ssta_all_nan] = np.nan

# log1p transform
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    chlorophyll = np.where(np.isnan(chl_window), np.nan,
                           np.log1p(np.maximum(chl_window, 0))).astype(np.float32)

print(f"  Chlorophyll: range=[{np.nanmin(chlorophyll):.4f}, {np.nanmax(chlorophyll):.4f}], NaN frac={np.isnan(chlorophyll).mean():.4f}")

# ═════════════════════════════════════════════════════════════
# STEP 4: PAR (latitude-based climatology for GBR)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 4: Building PAR climatology (MODIS not available for GBR)...")
par_norm = build_par_climatology_weekly()
for t in range(N_WEEKS):
    par_norm[t][ssta_all_nan] = np.nan
print(f"  PAR: range=[{np.nanmin(par_norm):.4f}, {np.nanmax(par_norm):.4f}], NaN frac={np.isnan(par_norm).mean():.4f}")

# ═════════════════════════════════════════════════════════════
# STEP 5: ERA5 wind/ssr/sst_anomaly (optional — fill with 0 if unavailable)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 5: ERA5 wind/ssr/sst_anomaly...")

era5_available = os.path.isdir(ERA5_GBR_DIR) and len(glob.glob(os.path.join(ERA5_GBR_DIR, '*.nc'))) >= 3

if era5_available:
    print(f"  ERA5 GBR data found in {ERA5_GBR_DIR}")
    # Load ERA5 monthly files
    era5_sst_ds = xr.open_dataset(os.path.join(ERA5_GBR_DIR, 'era5_sst_monthly.nc'))
    era5_wind_ds = xr.open_dataset(os.path.join(ERA5_GBR_DIR, 'era5_wind_monthly.nc'))
    era5_ssr_ds = xr.open_dataset(os.path.join(ERA5_GBR_DIR, 'era5_ssr_monthly.nc'))

    era5_lats = np.array(era5_sst_ds['latitude'])
    era5_lons = np.array(era5_sst_ds['longitude'])

    era5_sst_monthly = era5_sst_ds['sst'].values.astype(np.float32)
    u10 = era5_wind_ds['u10'].values.astype(np.float32)
    v10 = era5_wind_ds['v10'].values.astype(np.float32)
    era5_wind_monthly = np.sqrt(u10**2 + v10**2)
    era5_ssr_monthly = era5_ssr_ds['ssr'].values.astype(np.float32)

    # ERA5 files use 'time' dimension (renamed from 'valid_time' during concat)
    era5_time_var = 'valid_time' if 'valid_time' in era5_sst_ds else 'time'
    era5_monthly_time = pd.DatetimeIndex(era5_sst_ds[era5_time_var].values)

    era5_sst_ds.close()
    era5_wind_ds.close()
    era5_ssr_ds.close()

    # Regrid ERA5 → 1°
    print("  Regridding ERA5 monthly → 1°...")
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

    era5_sst_regrid = regrid_monthly_to_grid(era5_sst_monthly, era5_masks, "era5_sst")
    era5_wind_regrid = regrid_monthly_to_grid(era5_wind_monthly, era5_masks, "era5_wind")
    era5_ssr_regrid = regrid_monthly_to_grid(era5_ssr_monthly, era5_masks, "era5_ssr")

    # ERA5 SST → anomaly
    print("  Computing ERA5 SST climatology → anomaly...")
    era5_months = era5_monthly_time.month.values
    clim = np.full((12, N_LAT, N_LON), np.nan, dtype=np.float32)
    for m in range(1, 13):
        month_mask = era5_months == m
        if month_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                clim[m-1] = np.nanmean(era5_sst_regrid[month_mask], axis=0)
    clim_3d = np.stack([clim[m-1] for m in era5_months])
    era5_sst_anomaly_regrid = era5_sst_regrid - clim_3d
    del era5_sst_regrid, clim_3d, clim

    print("  Interpolating ERA5 monthly → weekly...")
    wind_speed = monthly_to_weekly_interp(era5_wind_regrid, era5_monthly_time, WEEKLY_T)
    ssr = monthly_to_weekly_interp(era5_ssr_regrid, era5_monthly_time, WEEKLY_T)
    sst_anomaly = monthly_to_weekly_interp(era5_sst_anomaly_regrid, era5_monthly_time, WEEKLY_T)

    print(f"  wind_speed: range=[{np.nanmin(wind_speed):.4f}, {np.nanmax(wind_speed):.4f}]")
    print(f"  ssr:        range=[{np.nanmin(ssr):.4f}, {np.nanmax(ssr):.4f}]")
    print(f"  sst_anomaly: range=[{np.nanmin(sst_anomaly):.4f}, {np.nanmax(sst_anomaly):.4f}]")
else:
    print(f"  WARNING: ERA5 GBR data not found at {ERA5_GBR_DIR}")
    print("  Filling wind_speed, ssr, sst_anomaly with NaN (will be filled with channel mean = 0 after z-score norm)")
    wind_speed = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
    ssr = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
    sst_anomaly = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)

# ═════════════════════════════════════════════════════════════
# STEP 6: Static features — regrid from GBR subsets
# ═════════════════════════════════════════════════════════════
print("\nSTEP 6: Regridding static features to 1° grid (GBR)...")

# --- ETOPO1 bathymetry (GBR subset) ---
print("  Loading ETOPO1 GBR bathymetry...")
ds_topo = xr.open_dataset(BATHY_GBR_PATH)
topo_z = np.array(ds_topo['z'])
topo_y = np.array(ds_topo['y'])  # lat
topo_x = np.array(ds_topo['x'])  # lon
ds_topo.close()

topo_lat_step = np.abs(np.diff(topo_y).mean())
topo_lon_step = np.abs(np.diff(topo_x).mean())

bathy_raw = np.zeros((N_LAT, N_LON), dtype=np.float32)
land_fraction = np.zeros((N_LAT, N_LON), dtype=np.float32)

for i in range(N_LAT):
    li_lo = int(np.clip(np.floor((LAT_LO[i] - topo_y.min()) / topo_lat_step), 0, len(topo_y) - 1))
    li_hi = int(np.clip(np.ceil((LAT_HI[i] - topo_y.min()) / topo_lat_step) + 1, 0, len(topo_y)))
    for j in range(N_LON):
        oi_lo = int(np.clip(np.floor((LON_LO[j] - topo_x.min()) / topo_lon_step), 0, len(topo_x) - 1))
        oi_hi = int(np.clip(np.ceil((LON_HI[j] - topo_x.min()) / topo_lon_step) + 1, 0, len(topo_x)))
        block = topo_z[li_lo:li_hi, oi_lo:oi_hi]
        valid = block[np.isfinite(block)]
        if len(valid) > 0:
            bathy_raw[i, j] = np.mean(valid)
            land_fraction[i, j] = np.sum(valid > 0) / len(valid)

# Bathymetry transform: land → 0, ocean → log1p(-depth)
ocean_pre = land_fraction < 0.5
bathy = np.zeros_like(bathy_raw)
bathy[ocean_pre] = np.log1p(-bathy_raw[ocean_pre]).astype(np.float32)

# --- Distance to shore (from bathymetry: distance from ocean cell to nearest land) ---
print("  Computing distance to shore from bathymetry...")
from scipy.ndimage import distance_transform_edt
land_mask = ~ocean_pre
# Distance from each ocean cell to nearest land cell (in grid units)
dist_grid = distance_transform_edt(ocean_pre.astype(np.float32))
dist_to_shore_raw = dist_grid.astype(np.float32) * 1.0  # ~1° grid units
dist_to_shore = np.log1p(dist_to_shore_raw)

# --- Cyclone frequency (not available for GBR — use 0) ---
print("  Cyclone frequency: not available for GBR, using 0")
cyclone_freq = np.zeros((N_LAT, N_LON), dtype=np.float32)

# --- Salinity (from WOA23 GBR subset) ---
print("  Loading WOA23 GBR salinity...")
ds_sal = xr.open_dataset(SAL_GBR_PATH, decode_times=False)
sal_lat = np.array(ds_sal.lat)
sal_lon = np.array(ds_sal.lon)
sal_data = np.array(ds_sal['s_an'])  # (depth, lat, lon) or (lat, lon)
ds_sal.close()

# WOA23 s_an shape: (depth, lat, lon) — take surface (depth=0)
if sal_data.ndim == 3:
    sal_surface = sal_data[0]  # surface layer
elif sal_data.ndim == 2:
    sal_surface = sal_data
else:
    sal_surface = sal_data.squeeze()
    if sal_surface.ndim == 3:
        sal_surface = sal_surface[0]

# Regrid WOA23 → 1° (block-avg)
# Initialize with NaN (not 0) so that cells with no valid WOA23 data
# are treated as missing and filled by spatial IDW below.
salinity = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
flip_sal = sal_lat[0] > sal_lat[-1]
sal_lat_arr = np.asarray(sal_lat)
sal_lon_arr = np.asarray(sal_lon)

for i in range(N_LAT):
    ml = (sal_lat_arr >= LAT_LO[i]) & (sal_lat_arr < LAT_HI[i])
    if flip_sal:
        ml = ml[::-1]
    for j in range(N_LON):
        mo = (sal_lon_arr >= LON_LO[j]) & (sal_lon_arr < LON_HI[j])
        m = ml[:, None] & mo[None, :]
        if m.any():
            block = sal_surface[m]
            valid = block[np.isfinite(block)]
            if len(valid) > 0:
                salinity[i, j] = np.mean(valid)

print(f"  bathymetry: range=[{np.nanmin(bathy):.4f}, {np.nanmax(bathy):.4f}]")
print(f"  dist_to_shore: range=[{np.nanmin(dist_to_shore):.4f}, {np.nanmax(dist_to_shore):.4f}]")
print(f"  ocean cells (land_fraction<0.5): {ocean_pre.sum()}")
print(f"  salinity: NaN frac={np.isnan(salinity).mean():.4f}")

# Fill salinity NaN ocean cells with spatial IDW
salinity = fill_nan_cells_spatial(salinity, ocean_pre, k=5)
print(f"  salinity post-fill: NaN frac={np.isnan(salinity).mean():.4f}")

# Fill sst_anomaly all-NaN ocean cells with spatial IDW
print("  Filling sst_anomaly NaN ocean cells...")
sst_anomaly = fill_nan_cells_spatial(sst_anomaly, ocean_pre & ~ssta_all_nan, k=5)
print(f"    sst_anomaly post-fill: range=[{np.nanmin(sst_anomaly):.4f}, {np.nanmax(sst_anomaly):.4f}], "
      f"NaN frac={np.isnan(sst_anomaly).mean():.4f}")

# Fill chlorophyll & PAR all-NaN ocean cells with spatial IDW
print("  Filling chlorophyll NaN ocean cells...")
chlorophyll = fill_nan_cells_spatial(chlorophyll, ocean_pre & ~ssta_all_nan, k=5)
print(f"    chlorophyll post-fill: NaN frac={np.isnan(chlorophyll).mean():.4f}")

print("  Filling PAR NaN ocean cells...")
par_norm = fill_nan_cells_spatial(par_norm, ocean_pre & ~ssta_all_nan, k=5)
print(f"    PAR post-fill: NaN frac={np.isnan(par_norm).mean():.4f}")

# ═════════════════════════════════════════════════════════════
# STEP 7: Build ocean mask
# ═════════════════════════════════════════════════════════════
print("\nSTEP 7: Building ocean mask...")

ocean_mask = ocean_pre & ~ssta_all_nan
land_mask_out = ~ocean_mask

print(f'  ocean cells: {ocean_mask.sum()}, land cells: {land_mask_out.sum()}')

# Apply NaN mask to dynamic ERA5 variables at cells where all-NaN in SSTA
for arr in [wind_speed, ssr, sst_anomaly]:
    for t in range(N_WEEKS):
        arr[t][ssta_all_nan] = np.nan

# Also mask SST at all-NaN cells
for t in range(N_WEEKS):
    sst_window[t][ssta_all_nan] = np.nan

print(f'  Post-masking wind NaN frac: {np.isnan(wind_speed).mean():.4f}')
print(f'  Post-masking SST NaN frac: {np.isnan(sst_window).mean():.4f}')

# ═════════════════════════════════════════════════════════════
# STEP 8: Derived channels (d_hotspot_dt, cumsum_4w, cumsum_12w)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 8: Computing derived channels...")

# d_hotspot_dt: weekly difference of hotspot
d_hotspot_dt = np.full_like(hotspot_window, np.nan)
d_hotspot_dt[1:] = hotspot_window[1:] - hotspot_window[:-1]

# cumsum_4w: 4-week rolling sum of hotspot
cumsum_4w = np.full_like(hotspot_window, np.nan)
for w in range(N_WEEKS):
    start = max(0, w - 3)
    window = hotspot_window[start:w+1]
    valid_per_cell = np.isfinite(window).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cumsum_4w[w] = np.where(
            valid_per_cell > 0,
            np.nansum(window, axis=0),
            np.nan
        ).astype(np.float32)

# cumsum_12w: 12-week rolling sum of hotspot
cumsum_12w = np.full_like(hotspot_window, np.nan)
for w in range(N_WEEKS):
    start = max(0, w - 11)
    window = hotspot_window[start:w+1]
    valid_per_cell = np.isfinite(window).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cumsum_12w[w] = np.where(
            valid_per_cell > 0,
            np.nansum(window, axis=0),
            np.nan
        ).astype(np.float32)

print(f"  d_hotspot_dt: range=[{np.nanmin(d_hotspot_dt):.4f}, {np.nanmax(d_hotspot_dt):.4f}]")
print(f"  cumsum_4w: range=[{np.nanmin(cumsum_4w):.4f}, {np.nanmax(cumsum_4w):.4f}]")
print(f"  cumsum_12w: range=[{np.nanmin(cumsum_12w):.4f}, {np.nanmax(cumsum_12w):.4f}]")

# ═════════════════════════════════════════════════════════════
# STEP 9: Write output NetCDF
# ═════════════════════════════════════════════════════════════
print("\nSTEP 9: Writing output NetCDF...")

data_vars = {
    'sst':                   xr.DataArray(sst_window.astype('float32'), dims=['time','lat','lon']),
    'ssta':                  xr.DataArray(ssta_window.astype('float32'), dims=['time','lat','lon']),
    'hotspot':               xr.DataArray(hotspot_window, dims=['time','lat','lon']),
    'chlorophyll':           xr.DataArray(chlorophyll, dims=['time','lat','lon']),
    'par':                   xr.DataArray(par_norm, dims=['time','lat','lon']),
    'seasonal_sin':          xr.DataArray(seasonal_sin_window.astype('float32'), dims=['time','lat','lon']),
    'wind_speed':            xr.DataArray(wind_speed, dims=['time','lat','lon']),
    'ssr':                   xr.DataArray(ssr, dims=['time','lat','lon']),
    'sst_anomaly':           xr.DataArray(sst_anomaly, dims=['time','lat','lon']),
    'd_hotspot_dt':          xr.DataArray(d_hotspot_dt, dims=['time','lat','lon']),
    'hotspot_cumsum_4w':     xr.DataArray(cumsum_4w, dims=['time','lat','lon']),
    'hotspot_cumsum_12w':    xr.DataArray(cumsum_12w, dims=['time','lat','lon']),
    'seasonal_cos':          xr.DataArray(seasonal_cos_window.astype('float32'), dims=['time','lat','lon']),
    'y_regression':          xr.DataArray(y_regression, dims=['time','lat','lon']),
    'y_warning':             xr.DataArray(y_warning, dims=['time','lat','lon']),
    'y_classification':      xr.DataArray(y_classification, dims=['time','lat','lon']),
    'land_mask':             xr.DataArray(np.broadcast_to(land_mask_out.astype('float32'), (N_WEEKS, N_LAT, N_LON)).copy(), dims=['time','lat','lon']),
    'ocean_mask':            xr.DataArray(ocean_mask.astype('float32'), dims=['lat','lon']),
    'bathymetry':            xr.DataArray(bathy, dims=['lat','lon']),
    'dist_to_shore':         xr.DataArray(dist_to_shore, dims=['lat','lon']),
    'cyclone_frequency':     xr.DataArray(cyclone_freq, dims=['lat','lon']),
    'salinity':              xr.DataArray(salinity, dims=['lat','lon']),
}

ds_out = xr.Dataset(
    data_vars,
    coords={
        'time': WEEKLY_T.astype('datetime64[ns]'),
        'lat':  LATS,
        'lon':  LONS,
    },
    attrs={
        'title': 'GBR 1° coral bleaching dataset (built from raw sources, Jensen-aware DHW)',
        'history': 'Built by src/data/build_gbr_1deg_from_raw.py',
        'dynamic_channels': 'sst(CRW raw), ssta, hotspot, chlorophyll, par, seasonal_sin, wind_speed, ssr, sst_anomaly, d_hotspot_dt, hotspot_cumsum_4w, hotspot_cumsum_12w, seasonal_cos (13)',
        'static_channels': 'bathymetry, dist_to_shore, cyclone_frequency, salinity, ocean_mask (5)',
        'targets': 'y_regression=clip(DHW/10,0,1), y_warning=clip(DHW/8,0,1), y_classification=I(DHW>=4)',
        'DHW_definition': '84-day rolling sum of clip(hotspot,0,+inf) at NATIVE 5km, then block-avg to 1° (Jensen-aware)',
        'par_source': 'latitude-based climatology (MODIS not available for GBR)',
        'era5_source': 'ERA5 GBR if available, else NaN (filled with channel mean after z-score)',
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
    if v.ndim == 3:
        fin = v[np.isfinite(v)]
        row = f"{np.isnan(v).mean()*100:7.2f}" if len(fin) else "  100.00"
        mn = f"{np.nanmin(v):.4f}" if len(fin) else "-"
        mx = f"{np.nanmax(v):.4f}" if len(fin) else "-"
        print(f"{name:22s} {str(v.shape):20s} {row:>8s} {mn:>10s} {mx:>10s}")
    else:
        fin = v[np.isfinite(v)]
        mn = f"{np.nanmin(v):.4f}" if len(fin) else "-"
        mx = f"{np.nanmax(v):.4f}" if len(fin) else "-"
        nanp = f"{np.isnan(v).mean()*100:.2f}" if np.isnan(v).any() else "0.00"
        print(f"{name:22s} {str(v.shape):20s} {nanp:>8s} {mn:>10s} {mx:>10s}")

om = data_vars['ocean_mask'].values
n_ocean = int(om.sum())
n_land = int((~om.astype(bool)).sum())
print(f"\nOcean cells (1°): {n_ocean}, Land cells: {n_land}")
print(f"\n=== GBR BUILD COMPLETE ===")
