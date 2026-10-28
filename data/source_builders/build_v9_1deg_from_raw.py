#!/usr/bin/env python3
"""
PIECF v9 → 1° Raw-from-Source Dataset Builder
===============================================

Builds a 1° (21×23) dataset DIRECTLY from raw data sources (no nearest-neighbor
upsampling from the 2° v9_final2.nc). Adds CRW SST as a new 9th dynamic channel.

Key differences from build_v9_consolidated.py:
  1. STEP 1: Also regrids CRW SST (absolute temperature, °C) alongside SSTA/Hotspot
  2. STEP 2: Generates targets DIRECTLY from DHW (rolling 12-week positive Hotspot sum)
             instead of nearest-neighbor regridding from v9_final2 (2°)
  3. Output file renamed to scs_coral_bleaching_dataset_v9_1deg_raw.nc

Data sources
------------
- CRW 5km daily SST/SSTA/Hotspot (monthly .nc files, 1985-2020)
- OC-CCI v6 monthly chlorophyll → weekly FF, log1p
- MODIS monthly PAR → weekly FF, min-max [0,1]
- ERA5 monthly SST/wind/SSR → weekly FF
- ETOPO1 bathymetry + IBTrACS cyclone + WOA23 salinity

New output
----------
- 18 variables (vs 17 in v9_consolidated_1deg): adds 'sst' (CRW raw SST, °C)
- 9 dynamic channels (vs 8): sst, ssta, hotspot, chlorophyll, par, seasonal_sin,
  wind_speed, ssr, sst_anomaly
- Targets derived from DHW computed directly from regridded weekly hotspot

Usage:
    'piecf_env/Scripts/python.exe' src/data/build_v9_1deg_from_raw.py
"""
import os, sys, glob, warnings
import numpy as np
import xarray as xr
import pandas as pd
import re
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DNS = os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r""         # raw/pre-processed sources
PIECF_DATA = r'D:\L1_project_root' if os.path.isdir(r'D:\L1_project_root') else PROJECT_ROOT
OUTPUT_DIR = PIECF_DATA
# LTTS_CRW: write the final derived dataset into this repo's data/ (self-contained)
OUT_DIR = r'D:\piecf_crw_data'  # ASCII: netCDF4 cannot write .nc under Chinese paths

# ─────────────────────────────────────────────────────────────
# 1° target grid (cell centers, 21×23)
# ─────────────────────────────────────────────────────────────
N_LAT, N_LON = 21, 23
LATS = np.array([3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5])
LONS = np.array([100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5, 122.5])

# Time: 864 weeks × 7 days, starting 2003-01-05
N_WEEKS = 864
WEEKLY = np.timedelta64(7, 'D')
TIME_START = np.datetime64('2003-01-05')
WEEKLY_T = TIME_START + np.arange(N_WEEKS) * WEEKLY
tsp = pd.DatetimeIndex(WEEKLY_T)

# ─────────────────────────────────────────────────────────────
# CRW 5km raw grid (source) — SSTA/Hotspot grid
# ─────────────────────────────────────────────────────────────
CRW_LAT = np.arange(24.025, 3.0 - 0.025, -0.05)  # 24.025..3.025, 421 pts, descending
CRW_LON = np.arange(100.025, 123.025 + 0.025, 0.05)  # 100.025..123.025, 461 pts

# SST file grid variants:
#   Old grid (198504-199109): lat 25.025..1.975 (462 pts), lon 99.025..124.025 (501 pts)
#   New grid (199110-202012): lat 24.025..3.025 (421 pts), lon 100.025..123.025 (461 pts)
# We detect at runtime based on shape.

def cell_bounds(centers):
    """Compute cell edges from cell centers (assumes uniform spacing)."""
    centers = np.asarray(centers, dtype=float)
    d = np.abs(np.diff(centers).mean()) / 2
    return centers - d, centers + d

# 1° target grid cell boundaries
LAT_LO, LAT_HI = cell_bounds(LATS)  # 3.0..24.0 at 1° steps
LON_LO, LON_HI = cell_bounds(LONS)  # 100.0..123.0 at 1° steps

# Build regrid index: for each (i,j) target cell, boolean mask into CRW source
# CRW lat is descending (24→3), so we handle the flip
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

    Args:
        monthly: (T_months, H, W) array.
        monthly_time: monthly timestamps (any day-of-month; mapped to month-end
            internally for 'causal').
        target_weeks: weekly timestamps.

    Returns:
        (N_weeks, H, W) float32 array.
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
                flat_out[:, k] = ts[valid][0]  # fallback: constant fill
            continue
        flat_out[:, k] = np.interp(t_weekly, t_monthly[valid], ts[valid])

    return flat_out.reshape(n_weeks, H, W)


def fill_nan_cells_spatial(data, ocean_mask, k=5):
    """Fill NaN cells with inverse-distance-weighted mean of nearest valid ocean neighbors.

    For 2D arrays (static features like salinity): fills NaN cells once.
    For 3D arrays (time-varying like sst_anomaly): fills NaN at every timestep.

    Args:
        data: 2D (H, W) or 3D (T, H, W) array.
        ocean_mask: boolean (H, W) — cells eligible as interpolation sources.
        k: number of nearest neighbors.

    Returns:
        Array of same shape as input with NaN cells filled.
    """
    original_ndim = data.ndim
    if data.ndim == 2:
        data = data[None, :, :]  # (1, H, W)
    T, H, W = data.shape

    # Sources: ocean cells with data at ANY timestep (3D) or at the single slice (2D)
    valid_source = ocean_mask.copy()
    if original_ndim == 3:
        valid_source &= np.isfinite(data).any(axis=0)
    else:
        valid_source &= np.isfinite(data[0])

    # Targets: ocean cells that are entirely NaN
    if original_ndim == 3:
        targets = ocean_mask & np.isnan(data).all(axis=0)
    else:
        targets = ocean_mask & np.isnan(data[0])

    n_targets = int(targets.sum())
    n_sources = int(valid_source.sum())

    if n_targets == 0 or n_sources == 0:
        print(f"    No spatial fill needed (targets={n_targets}, sources={n_sources})")
        return data.squeeze() if original_ndim == 2 else data

    print(f"    Filling {n_targets} NaN cells from {n_sources} valid neighbors (k={k})")

    # KDTree over source coordinates (lat_idx, lon_idx)
    src_coords = np.argwhere(valid_source).astype(np.float32)
    tgt_coords = np.argwhere(targets).astype(np.float32)

    tree = cKDTree(src_coords)
    dists, indices = tree.query(tgt_coords, k=min(k, n_sources))

    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    # Pre-compute IDW weights (shared across timesteps)
    weights = 1.0 / (dists ** 2 + 1e-8)  # (N_targets, k)

    # Apply to each timestep
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


print("=" * 60)
print("PIECF v9 1° Dataset Builder (RAW from sources)")
print("=" * 60)
print(f"Grid: {N_LAT}x{N_LON}, Time: {N_WEEKS} weeks")
print(f"Time range: {str(TIME_START)[:10]} to {str(WEEKLY_T[-1])[:10]}")
print(f"Key additions vs v9_consolidated: +CRW SST, -NN target regrid (DHW direct)")
print()

# ═════════════════════════════════════════════════════════════
# STEP 1: Regrid CRW 5km daily (SSTA + Hotspot + SST) → 1° weekly block-avg
#
# Vectorized algorithm with np.add.at.
# SST files use TWO different grids (older 462x501, newer 421x461).
# We detect the grid shape per-file at runtime and select the right flat_lookup.
# ═════════════════════════════════════════════════════════════
print("STEP 1: Regridding CRW 5km daily (SSTA + Hotspot + SST) → 1° weekly block-avg...")

# Build regrid masks (for the standard 421x461 SSTA/Hotspot grid)
print("  Building regrid masks (standard 421×461 grid)...")
flat_lookup_std = np.arange(421 * 461).reshape(421, 461)
cell_flat_indices_std = []  # list of flat-index arrays, one per target cell
cell_ij_std = []  # (i, j) for each entry
for i in range(N_LAT):
    for j in range(N_LON):
        m = build_regrid_mask(CRW_LAT, CRW_LON, LAT_LO[i], LAT_HI[i], LON_LO[j], LON_HI[j])
        if m.any():
            cell_flat_indices_std.append(flat_lookup_std[m])
            cell_ij_std.append((i, j))
n_cells_std = len(cell_flat_indices_std)
print(f"  Built {n_cells_std} regrid masks (standard 421×461 grid)")

# CSR sparse aggregation matrix: 5km pixels → 1° cells (for native DHW spatial avg)
print("  Building CSR sparse aggregation matrix (5km → 1°)...")
rows_csr, cols_csr = [], []
for cell_k, fidx in enumerate(cell_flat_indices_std):
    for pix in fidx:
        rows_csr.append(cell_k); cols_csr.append(pix)
M_csr = csr_matrix((np.ones(len(rows_csr), dtype=np.float32),
                    (rows_csr, cols_csr)),
                   shape=(n_cells_std, len(CRW_LAT) * len(CRW_LON)))
cell_size = np.asarray(M_csr.sum(axis=1)).ravel().astype(np.float32)  # [n_cells_std]
del rows_csr, cols_csr
print(f"  CSR matrix: {M_csr.shape}, nnz={M_csr.nnz}, cell_size: min={cell_size.min():.0f} max={cell_size.max():.0f} mean={cell_size.mean():.0f}")

# Regrid masks for the OLD SST grid (462x501)
OLD_SST_LAT = np.arange(25.025, 2.0 - 0.025, -0.05)  # 25.025..1.975, 462 pts
OLD_SST_LON = np.arange(99.025, 124.025 + 0.025, 0.05)  # 99.025..124.025, 501 pts
flat_lookup_old_sst = np.arange(len(OLD_SST_LAT) * len(OLD_SST_LON)).reshape(len(OLD_SST_LAT), len(OLD_SST_LON))
cell_flat_indices_sst = []
cell_ij_sst = []
for i in range(N_LAT):
    for j in range(N_LON):
        m = build_regrid_mask(OLD_SST_LAT, OLD_SST_LON, LAT_LO[i], LAT_HI[i], LON_LO[j], LON_HI[j])
        if m.any():
            cell_flat_indices_sst.append(flat_lookup_old_sst[m])
            cell_ij_sst.append((i, j))
n_cells_sst = len(cell_flat_indices_sst)
print(f"  Built {n_cells_sst} regrid masks (old 462×501 SST grid)")

# Accumulate weekly sums and counts: [N_WEEKS, N_LAT, N_LON]
ssta_sum = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
ssta_cnt = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)
hs_sum   = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
hs_cnt   = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)
sst_sum  = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.float64)
sst_cnt  = np.zeros((N_WEEKS, N_LAT, N_LON), dtype=np.int32)

# For DHW (CRW standard): compute daily 1° hotspot daily series, then 84-day rolling sum.
# We need all daily data from 1985-04 to 2020-12 for the rolling sum.
# Memory: ~13,149 days × 483 cells × 4 bytes = ~25 MB (float32) — acceptable.
# We accumulate ALL daily 5km hotspot → 1° daily block-avg.

# Daily time axis: 1985-04-01 to 2020-12-31
TOTAL_START = np.datetime64('1985-04-01')
TOTAL_END   = np.datetime64('2020-12-31')
DAILY_T = np.arange(TOTAL_START, TOTAL_END + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
N_DAILY = len(DAILY_T)
t0_daily = pd.Timestamp(TOTAL_START)

print(f'  Daily hotspot grid allocated: [{N_DAILY} days × {N_LAT}×{N_LON}] = ~{N_DAILY * N_LAT * N_LON * 4 / 1e6:.0f} MB')

# Native-5km rolling window: CRW-standard 84-day rolling sum at NATIVE 5km resolution
# (Jensen-aware: clip + rolling sum BEFORE spatial averaging preserves peaks)
WIN = 84
# CRW-standard DHW: only HotSpot ≥ 1 °C contributes to the 84-day accumulation
DHW_CLIP_LOWER = 1.0
slope_buf = np.zeros((WIN, len(CRW_LAT), len(CRW_LON)), dtype=np.float32)  # clipped hotspot per day
nan_buf   = np.zeros((WIN, len(CRW_LAT), len(CRW_LON)), dtype=np.bool_)   # valid mask per day
buf_n = 0  # days in buffer (caps at WIN)

# 1° daily DHW accumulation (float32 sufficient precision for DHW ≈ 0–250 deg-days)
dhw_daily_acc = np.zeros((N_DAILY, N_LAT, N_LON), dtype=np.float32)
dhw_daily_cnt = np.zeros((N_DAILY, N_LAT, N_LON), dtype=np.int32)
print(f"  5km rolling window buffers allocated: slope_buf={slope_buf.nbytes/1e6:.1f} MB, nan_buf={nan_buf.nbytes/1e6:.1f} MB")

# Month files for CRW SSTA, Hotspot, and SST
ssta_by_month, hs_by_month, sst_by_month = {}, {}, {}

for fp in sorted(glob.glob(os.path.join(DNS, 'crw_5km_sst', 'crw_sstanomaly_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        ssta_by_month[ym.group(1)] = fp

for fp in sorted(glob.glob(os.path.join(DNS, 'crw_5km_sst', 'crw_hotspot_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        hs_by_month[ym.group(1)] = fp

for fp in sorted(glob.glob(os.path.join(DNS, 'crw_5km_sst', 'crw_sst_*.nc'))):
    ym = re.search(r'(\d{6})', os.path.basename(fp))
    if ym:
        sst_by_month[ym.group(1)] = fp

# Determine which months are needed for:
#  (a) weekly accumulation (864 weeks, 2003-2019)
#  (b) daily hotspot accumulation — need ALL months for CRW files (1985-2020)
#      since DHW rolling 84-day sum needs pre-2003 lookback (but mostly we just
#      need 84 days before 2003-01-05, which is ~2002-10; we load all for simplicity)

# Weekly window: known months_needed
months_weekly = set()
for w in range(N_WEEKS):
    for dt64 in [WEEKLY_T[w], WEEKLY_T[w] + np.timedelta64(6, 'D')]:
        dt = pd.Timestamp(dt64)
        months_weekly.add(f"{dt.year}{dt.month:02d}")
months_weekly = sorted(months_weekly)

# All CRW months (for daily hotspot accumulation)
all_crw_months = sorted(set(list(ssta_by_month.keys()) + list(hs_by_month.keys())))
print(f"  Weekly window: {len(months_weekly)} months, CRW files available: {len(all_crw_months)} months")

# Process union of weekly + all CRW months (so we capture daily data before 2003)
months_needed = sorted(set(months_weekly) | set(all_crw_months))
print(f"  Processing {len(months_needed)} months of CRW daily data...")
n_months_done = 0

for month_str in months_needed:
    if month_str not in ssta_by_month:
        continue
    try:
        # Load SSTA daily data
        ds_ssta = xr.open_dataset(ssta_by_month[month_str])
        ssta_daily = ds_ssta['CRW_SSTANOMALY'].values  # [n_days, 421, 461]
        ssta_time = pd.DatetimeIndex(ds_ssta['time'].values)
        ds_ssta.close()

        # Load Hotspot
        if month_str in hs_by_month:
            ds_hs = xr.open_dataset(hs_by_month[month_str])
            hs_daily = ds_hs['CRW_HOTSPOT'].values
            ds_hs.close()
        else:
            hs_daily = None

        # Load CRW SST (absolute temperature)
        sst_daily = None
        if month_str in sst_by_month:
            ds_sst = xr.open_dataset(sst_by_month[month_str])
            sst_daily = ds_sst['CRW_SST'].values  # [n_days, H, W]
            ds_sst.close()

        n_days = len(ssta_time)

        # Convert to flat [n_days, 421*461], tracking NaN
        ssta_flat = ssta_daily.reshape(n_days, -1).astype(np.float64)
        ssta_valid = np.isfinite(ssta_flat)
        ssta_flat[~ssta_valid] = 0.0

        if hs_daily is not None:
            hs_flat = hs_daily.reshape(n_days, -1).astype(np.float64)
            hs_valid = np.isfinite(hs_flat)
            hs_flat[~hs_valid] = 0.0

        # For SST: detect grid shape and choose appropriate flat_lookup
        sst_flat = None
        sst_valid = None
        sst_cell_flat_indices = None
        sst_cell_ij = None
        if sst_daily is not None:
            if sst_daily.shape[1] == 421 and sst_daily.shape[2] == 461:
                # Standard grid (matches SSTA/Hotspot) — use identical lookup
                sst_flat = sst_daily.reshape(n_days, -1).astype(np.float64)
                sst_valid = np.isfinite(sst_flat)
                sst_flat[~sst_valid] = 0.0
                sst_cell_flat_indices = cell_flat_indices_std
                sst_cell_ij = cell_ij_std
            elif sst_daily.shape[1] == 462 and sst_daily.shape[2] == 501:
                # Old, larger grid
                sst_flat = sst_daily.reshape(n_days, -1).astype(np.float64)
                sst_valid = np.isfinite(sst_flat)
                sst_flat[~sst_valid] = 0.0
                sst_cell_flat_indices = cell_flat_indices_sst
                sst_cell_ij = cell_ij_sst
            else:
                # Unexpected shape: try the standard lookup (best-effort)
                warnings.warn(f"SST month {month_str}: unexpected shape {sst_daily.shape}, using standard regrid")
                sst_flat = sst_daily.reshape(n_days, -1).astype(np.float64)
                sst_valid = np.isfinite(sst_flat)
                sst_flat[~sst_valid] = 0.0
                sst_cell_flat_indices = cell_flat_indices_std
                sst_cell_ij = cell_ij_std

        # For each day, find target week
        t0 = pd.Timestamp(TIME_START)
        day_offsets = np.array([(pd.Timestamp(dt) - t0).days for dt in ssta_time])
        day_weeks = day_offsets // 7
        day_in_window = (day_weeks >= 0) & (day_weeks < N_WEEKS)

        # For each target cell, compute sum over source pixels (vectorized using np.add.at)
        for cell_k, ((i, j), fidx) in enumerate(zip(cell_ij_std, cell_flat_indices_std)):
            # SSTA
            day_sums = ssta_flat[:, fidx].sum(axis=1)
            day_counts = ssta_valid[:, fidx].sum(axis=1)
            w_idx = day_weeks[day_in_window]
            np.add.at(ssta_sum[:, i, j], w_idx, day_sums[day_in_window])
            np.add.at(ssta_cnt[:, i, j], w_idx, day_counts[day_in_window])

            # Hotspot
            if hs_daily is not None:
                day_hs_sums = hs_flat[:, fidx].sum(axis=1)
                day_hs_counts = hs_valid[:, fidx].sum(axis=1)
                np.add.at(hs_sum[:, i, j], w_idx, day_hs_sums[day_in_window])
                np.add.at(hs_cnt[:, i, j], w_idx, day_hs_counts[day_in_window])

        # SST accumulation (using the appropriate cell index set)
        if sst_daily is not None:
            for cell_k, ((i, j), fidx) in enumerate(zip(sst_cell_ij, sst_cell_flat_indices)):
                day_sst_sums = sst_flat[:, fidx].sum(axis=1)
                day_sst_counts = sst_valid[:, fidx].sum(axis=1)
                w_idx = day_weeks[day_in_window]
                np.add.at(sst_sum[:, i, j], w_idx, day_sst_sums[day_in_window])
                np.add.at(sst_cnt[:, i, j], w_idx, day_sst_counts[day_in_window])

        # Native-5km rolling DHW (CRW standard): HotSpot ≥ 1 °C, 84-day rolling
        # sum at NATIVE 5km, full-window validity, THEN spatial mean to 1°.
        # Preserves thermal-stress peaks (Jensen-aware). Matches jensen_dhw.py.
        if hs_daily is not None:
            # Pre-compute CRW-clipped hotspot (only HotSpot ≥ 1 °C contributes)
            # and NaN mask for this month (vectorized)
            hs_2d = hs_daily.reshape(n_days, len(CRW_LAT), len(CRW_LON))
            hs_nan = np.isfinite(hs_2d)
            hs_clipped = np.where(hs_nan & (hs_2d >= DHW_CLIP_LOWER),
                                  hs_2d, 0.0).astype(np.float32)

            for d in range(n_days):
                dt = ssta_time[d]
                g_offset = (pd.Timestamp(dt) - t0_daily).days
                if g_offset < 0 or g_offset >= N_DAILY:
                    continue

                buf_k = g_offset % WIN
                slope_buf[buf_k] = hs_clipped[d]
                nan_buf[buf_k]   = hs_nan[d]
                buf_n = min(buf_n + 1, WIN)

                if g_offset < WIN - 1:
                    continue  # buffer not yet full → no complete 84-day window

                # === Compute 5km DHW for this day ===
                # Extract 84 days in chronological order (descending global_day → ascending)
                idx = [(g_offset - k) % WIN for k in range(WIN)]
                window_slope = slope_buf[idx]   # [84, 421, 461]
                window_nan   = nan_buf[idx]

                dhw_5km = window_slope.sum(axis=0)        # [421, 461] deg-days
                valid_5km = window_nan.sum(axis=0)         # [421, 461] valid-day count

                # CRW-strict: only 5 km pixels with the FULL 84-day window count.
                # Partial windows treat missing days as 0 and would bias DHW low.
                full_window = (valid_5km == WIN)           # [421, 461] bool
                dhw_5km_flat = np.where(full_window.ravel(), dhw_5km.ravel(), np.nan)
                full_flat = full_window.ravel().astype(np.float32)

                # === Native 5km → 1° block mean over full-window pixels ===
                # (matches jensen_dhw.aggregate_to_1deg: mean over valid pixels)
                cell_sum = M_csr.dot(np.nan_to_num(dhw_5km_flat, nan=0.0))  # [n_cells]
                cell_cnt = M_csr.dot(full_flat)                              # full-window pixels/cell
                cell_mean = np.divide(cell_sum, cell_cnt,
                                      out=np.full_like(cell_sum, np.nan),
                                      where=cell_cnt > 0)

                # Accumulate into 1° daily
                for cell_k, (i, j) in enumerate(cell_ij_std):
                    if np.isfinite(cell_mean[cell_k]):
                        dhw_daily_acc[g_offset, i, j] = cell_mean[cell_k]
                        dhw_daily_cnt[g_offset, i, j] = 1  # mark as valid
            del hs_2d, hs_nan, hs_clipped, hs_daily

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

# Clip hotspot to [0, inf) for feature output (matches v9 convention)
# np.clip preserves NaN values (NaN compared with >= 0 is False → stays NaN)
hotspot_window = np.full_like(hotspot_raw, np.nan)
hotspot_window[valid_hs] = np.clip(hotspot_raw[valid_hs], 0, None)

# Seasonal_sin (deterministic — no NaN)
weeks_of_year = np.array([t.isocalendar()[1] for t in tsp])
seasonal_sin_window = np.sin(2 * np.pi * weeks_of_year / 52).astype(np.float32)
seasonal_sin_window = np.broadcast_to(seasonal_sin_window[:, None, None], (N_WEEKS, N_LAT, N_LON)).copy()

# Pre-compute SSTA all-NaN cell mask
ssta_nan_frac = np.isnan(ssta_window).mean(axis=0)
ssta_all_nan = ssta_nan_frac > 0.9999
print(f"  SSTA all-NaN cells: {ssta_all_nan.sum()} / {N_LAT * N_LON}")

# Free 5km rolling window buffers (no longer needed after STEP 1)
del slope_buf, nan_buf, M_csr, cell_size
print("  5km rolling window buffers freed")

# ═════════════════════════════════════════════════════════════
# STEP 2: Generate targets from NATIVE 5km DHW (Plan B)
#
# DHW computed at native 5km resolution (STEP 1), then block-averaged to 1°.
# This preserves thermal stress peaks (Jensen's inequality aware).
#
# DHW unit: degree-days (84-day sum of clipped hotspot, °C)
# 1° weekly: mean of 7 daily DHW values
# y targets require degree-weeks (divide by 7). Thresholds:
#   Alert Level 1 = 4 deg-weeks (CRW standard)
#   Alert Level 2 = 8 deg-weeks
# ═════════════════════════════════════════════════════════════
print("\nSTEP 2: Aggregating native-5km DHW -> weekly -> targets...")

# Build 1° daily DHW from accumulation arrays
dhw_daily_1deg = np.full((N_DAILY, N_LAT, N_LON), np.nan, dtype=np.float32)
valid_daily = dhw_daily_cnt > 0
dhw_daily_1deg[valid_daily] = dhw_daily_acc[valid_daily]
del dhw_daily_acc, dhw_daily_cnt, valid_daily

print(f"  1° daily DHW: valid cells = {np.isfinite(dhw_daily_1deg).sum()}, "
      f"range=[{np.nanmin(dhw_daily_1deg):.4f}, {np.nanmax(dhw_daily_1deg):.4f}] deg-days")

# Aggregate daily → weekly (7-day mean)
print("  Aggregating daily DHW -> weekly (7-day mean)...")
dhw_weekly_degdays = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
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

print(f"  DHW (weekly, deg-days): range=[{np.nanmin(dhw_weekly_degdays):.4f}, {np.nanmax(dhw_weekly_degdays):.4f}]")

# Convert degree-days → degree-weeks (CRW standard for thresholds)
#  Alert Level 1 = 4 deg-weeks, Alert Level 2 = 8 deg-weeks
dhw_weekly = dhw_weekly_degdays / 7.0
print(f"  DHW (weekly, deg-weeks): range=[{np.nanmin(dhw_weekly):.4f}, {np.nanmax(dhw_weekly):.4f}]")
del dhw_weekly_degdays

# Generate targets (CRW standard scale: thresholds in degree-weeks)
y_regression     = np.clip(dhw_weekly / 10.0, 0, 1)
y_warning        = np.clip(dhw_weekly /  8.0, 0, 1)
y_classification = np.where(np.isnan(dhw_weekly), np.nan,
                            (dhw_weekly >= 4).astype(np.float32))

# Apply SSTA all-NaN cell mask (broadcast 2D → 3D)
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
# STEP 3: Rebuild chlorophyll (OC-CCI v6 monthly → weekly FF at 1°)
# ═════════════════════════════════════════════════════════════
print("\nSTEP 3: Rebuilding chlorophyll from OC-CCI v6.0 monthly (1° grid)...")
chl_file_dir = os.path.join(DNS, 'chlorophyll')
chl_files = sorted(glob.glob(os.path.join(chl_file_dir, 'oc_cci_chlor_a_*_fv6.0.nc')))

if not chl_files:
    print("  WARNING: No OC-CCI files; using v3 fallback (log1p transformed)")
    ds3tmp = xr.open_dataset(os.path.join(PIECF_DATA, 'scs_coral_bleaching_dataset_v3.nc'))
    chl_window = ds3tmp['chlorophyll'].values[939:939 + N_WEEKS]
    ds3tmp.close()
else:
    print(f"  Found {len(chl_files)} OC-CCI monthly files")
    # Open first file to get source grid
    ds_chl0 = xr.open_dataset(chl_files[0])
    slat = ds_chl0['lat'].values; slon = ds_chl0['lon'].values
    ds_chl0.close()
    flip_chl = slat[0] > slat[-1]

    # OPTIMIZATION: OC-CCI is global (4320×8640). Subset to our region first.
    lat_mask = (slat >= 2.5) & (slat <= 24.5)
    lon_mask = (slon >= 99.5) & (slon <= 123.5)
    slat_sub = slat[lat_mask]
    slon_sub = slon[lon_mask]
    lat_idx_chl = np.where(lat_mask)[0]
    lon_idx_chl = np.where(lon_mask)[0]
    print(f"  OC-CCI subset: {len(slat_sub)}×{len(slon_sub)} (from {len(slat)}×{len(slon)})")

    # Build regrid FLAT INDICES on the SUBSETTED grid
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
        ym_s = ym.group(1); y = int(ym_s[:4]); mo = int(ym_s[4:])
        try:
            ds = xr.open_dataset(fp)
            ca_full = ds['chlor_a'].squeeze().values
            ds.close()
            if ca_full.ndim != 2:
                continue
            ca = ca_full[lat_idx_chl[0]:lat_idx_chl[-1]+1, lon_idx_chl[0]:lon_idx_chl[-1]+1]
            ca_flat = ca.ravel()
            regrid = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
            for (i, j), fidx in chl_indices.items():
                v = ca_flat[fidx]
                v = v[np.isfinite(v)]
                if len(v) > 0:
                    regrid[i, j] = np.mean(v)
            mvals[(y, mo)] = regrid
            if (idx + 1) % 20 == 0:
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

    # Monthly center time (assume mid-month = day 15)
    monthly_times = [pd.Timestamp(f"{y}-{m:02d}-15") for (y, m) in mvals_keys]

    # Causal monthly → weekly (default: month-end forward-fill, NO future leakage)
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

# (Chlorophyll NaN ocean cells will be filled in STEP 6 after ocean_mask is built)

# ═════════════════════════════════════════════════════════════
# STEP 4: Rebuild PAR (MODIS monthly → weekly FF at 1°, min-max [0,1])
# ═════════════════════════════════════════════════════════════
print("\nSTEP 4: Rebuilding PAR from MODIS monthly → weekly FF (1° grid)...")
par_dir = os.path.join(DNS, 'coastwatch_par')
par_files = sorted(glob.glob(os.path.join(par_dir, 'modis_par_??????.nc')))

if not par_files:
    par_window = np.full((N_WEEKS, N_LAT, N_LON), np.nan, dtype=np.float32)
    print("  WARNING: No MODIS PAR files; all NaN")
else:
    print(f"  Found {len(par_files)} MODIS monthly PAR files")
    ds_par0 = xr.open_dataset(par_files[0])
    plat = ds_par0['latitude'].values; plon = ds_par0['longitude'].values
    ds_par0.close()

    # Build regrid FLAT INDICES for MODIS → 1° target grid
    flip_par = plat[0] > plat[-1]
    plat_arr = np.asarray(plat); plon_arr = np.asarray(plon)
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
        y = int(bn[:4]); mo = int(bn[4:6])
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
            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx+1}/{len(par_files)} files")
        except Exception as e:
            print(f"    file failed {os.path.basename(fp)}: {e}")

    print(f"    Total months regridded: {len(mvals_par)}")

    # Build monthly stack for linear interpolation
    mvals_par_keys = sorted(mvals_par.keys())
    n_months_par = len(mvals_par_keys)
    monthly_stack_par = np.full((n_months_par, N_LAT, N_LON), np.nan, dtype=np.float32)
    for idx, key in enumerate(mvals_par_keys):
        monthly_stack_par[idx] = mvals_par[key]

    # Monthly center time (assume mid-month = day 15)
    monthly_times_par = [pd.Timestamp(f"{y}-{m:02d}-15") for (y, m) in mvals_par_keys]

    # Linear interpolation monthly → weekly (avoids FF step artifacts)
    par_window = monthly_to_weekly_interp(monthly_stack_par, monthly_times_par, WEEKLY_T)

# Apply SSTA-NaN mask
for t in range(N_WEEKS):
    par_window[t][ssta_all_nan] = np.nan

# Min-max normalization [0, 1] using the v9-consistent params (18.20, 56.63)
PAR_MIN, PAR_MAX = 18.20, 56.63
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    par_norm = ((par_window - PAR_MIN) / (PAR_MAX - PAR_MIN)).astype(np.float32)
    par_norm = np.clip(par_norm, 0, 1)
    par_norm[np.isnan(par_window)] = np.nan

print(f"  PAR: range=[{np.nanmin(par_norm):.4f}, {np.nanmax(par_norm):.4f}], NaN frac={np.isnan(par_norm).mean():.4f}")

# (PAR NaN ocean cells will be filled in STEP 6 after ocean_mask is built)

# ═════════════════════════════════════════════════════════════
# STEP 5: ERA5 wind/ssr/sst_anomaly
# Load raw ERA5 monthly (121×141 at 0.25°) → regrid to 1° → forward-fill to weekly
# ═════════════════════════════════════════════════════════════
print("\nSTEP 5: Loading ERA5 monthly → regrid to 1° → weekly FF...")

# Load raw ERA5 monthly files
era5_sst_ds = xr.open_dataset(os.path.join(DNS, 'era5_monthly', 'era5_sst_monthly.nc'))
era5_wind_ds = xr.open_dataset(os.path.join(DNS, 'era5_monthly', 'era5_wind_monthly.nc'))
era5_ssr_ds = xr.open_dataset(os.path.join(DNS, 'era5_monthly', 'era5_ssr_monthly.nc'))

era5_lats = era5_sst_ds['latitude'].values    # 121 pts, 0-30N
era5_lons = era5_sst_ds['longitude'].values  # 141 pts, 95-130E

# ERA5 monthly data [216, 121, 141]
era5_sst_monthly = era5_sst_ds['sst'].values.astype(np.float32)
# Wind speed = sqrt(u10² + v10²)
u10 = era5_wind_ds['u10'].values.astype(np.float32)
v10 = era5_wind_ds['v10'].values.astype(np.float32)
era5_wind_monthly = np.sqrt(u10**2 + v10**2)
era5_ssr_monthly = era5_ssr_ds['ssr'].values.astype(np.float32)

# Monthly time axis
era5_monthly_time = pd.DatetimeIndex(era5_sst_ds['valid_time'].values)

era5_sst_ds.close()
era5_wind_ds.close()
era5_ssr_ds.close()

# Regrid ERA5 0.25° → 1° using same regrid_masks logic
print("  Regridding ERA5 monthly 0.25° → 1°...")
era5_masks = {}
for i in range(N_LAT):
    for j in range(N_LON):
        m = build_regrid_mask(era5_lats, era5_lons, LAT_LO[i], LAT_HI[i], LON_LO[j], LON_HI[j])
        if m.any():
            era5_masks[i, j] = m

def regrid_monthly_to_grid(src_monthly, masks_dict, N_LAT, N_LON, label=""):
    """Regrid [T, H_s, W_s] → [T, N_LAT, N_LON] using pre-computed boolean masks."""
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

era5_sst_regrid = regrid_monthly_to_grid(era5_sst_monthly, era5_masks, N_LAT, N_LON, "era5_sst")
era5_wind_regrid = regrid_monthly_to_grid(era5_wind_monthly, era5_masks, N_LAT, N_LON, "era5_wind")
era5_ssr_regrid = regrid_monthly_to_grid(era5_ssr_monthly, era5_masks, N_LAT, N_LON, "era5_ssr")

# ═════════════════════════════════════════════════════════════
# Convert ERA5 SST (Kelvin) → SST anomaly (°C)
# ERA5 stores SST in Kelvin (~300 K). Compute monthly climatology
# from the full 1985-2020 record, subtract to get true anomaly.
# ═════════════════════════════════════════════════════════════
print("  Computing ERA5 SST monthly climatology (1985-2020) → anomaly...")
era5_months = era5_monthly_time.month.values  # 1..216
clim = np.full((12, N_LAT, N_LON), np.nan, dtype=np.float32)
for m in range(1, 13):
    month_mask = era5_months == m
    if month_mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clim[m-1] = np.nanmean(era5_sst_regrid[month_mask], axis=0)
# Subtract climatology (broadcast over time)
clim_3d = np.stack([clim[m-1] for m in era5_months])  # [T, N_LAT, N_LON]
era5_sst_anomaly_regrid = era5_sst_regrid - clim_3d
print(f"    ERA5 SST climatology subtracted: raw range=[{np.nanmin(era5_sst_regrid):.2f}, {np.nanmax(era5_sst_regrid):.2f}] K")
print(f"    ERA5 SST anomaly range=[{np.nanmin(era5_sst_anomaly_regrid):.2f}, {np.nanmax(era5_sst_anomaly_regrid):.2f}] °C")
del era5_sst_regrid, clim_3d, clim

print("  Interpolating ERA5 monthly → weekly (linear, no FF)...")
wind_speed = monthly_to_weekly_interp(era5_wind_regrid, era5_monthly_time, WEEKLY_T)
ssr = monthly_to_weekly_interp(era5_ssr_regrid, era5_monthly_time, WEEKLY_T)
sst_anomaly = monthly_to_weekly_interp(era5_sst_anomaly_regrid, era5_monthly_time, WEEKLY_T)

print(f"  wind_speed: range=[{np.nanmin(wind_speed):.4f}, {np.nanmax(wind_speed):.4f}]")
print(f"  ssr:        range=[{np.nanmin(ssr):.4f}, {np.nanmax(ssr):.4f}]")
print(f"  sst_anomaly: range=[{np.nanmin(sst_anomaly):.4f}, {np.nanmax(sst_anomaly):.4f}]")

# ═════════════════════════════════════════════════════════════
# STEP 6: Static features — regrid from raw sources to 1° (21×23)
# Sources: ETOPO1 (1 arc-min), IBTrACS cyclone, WOA23 salinity
# ═════════════════════════════════════════════════════════════
print("\nSTEP 6: Regridding static features to 1° grid...")

# --- ETOPO1 bathymetry ---
ds_topo = xr.open_dataset(os.path.join(DNS, 'bathymetry', 'ETOPO1_Ice_g_gmt4.grd'))
topo_z = ds_topo['z'].values  # (10801, 21601) global, y ascending
topo_y = ds_topo['y'].values  # lat -90..90, 10801 pts, ascending
topo_x = ds_topo['x'].values  # lon -180..180, 21601 pts, ascending
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

# Bathymetry transform: land → 0, ocean → log1p(-depth)
ocean_pre = land_fraction < 0.5
bathy = np.zeros_like(bathy_raw)
bathy[ocean_pre] = np.log1p(-bathy_raw[ocean_pre]).astype(np.float32)

# --- Distance to shore (from pre-built 2° → NN to 1°) ---
static_npz_old = np.load(os.path.join(DNS, 'static_features_20x15.npz'))
dist_to_shore_raw_2deg = static_npz_old['distance_to_shore'].astype(np.float32)
lats_2deg_full = np.linspace(4.5, 22.5, 20)
lons_2deg_full = np.linspace(101.97, 121.03, 15)
del static_npz_old

lat_idx_nn = np.array([np.argmin(np.abs(lats_2deg_full - lat)) for lat in LATS])
lon_idx_nn = np.array([np.argmin(np.abs(lons_2deg_full - lon)) for lon in LONS])
dist_to_shore_raw = np.zeros((N_LAT, N_LON), dtype=np.float32)
for i, si in enumerate(lat_idx_nn):
    for j, sj in enumerate(lon_idx_nn):
        dist_to_shore_raw[i, j] = dist_to_shore_raw_2deg[si, sj]

dist_to_shore = np.log1p(dist_to_shore_raw)

# --- Cyclone frequency (from pre-computed 2° → NN to 1°) ---
cyclone_raw_2deg = np.load(os.path.join(DNS, 'static_features_20x15.npz'))['cyclone_frequency'].astype(np.float32)
cyclone_raw = np.zeros((N_LAT, N_LON), dtype=np.float32)
for i, si in enumerate(lat_idx_nn):
    for j, sj in enumerate(lon_idx_nn):
        cyclone_raw[i, j] = cyclone_raw_2deg[si, sj]

cmean, cstd = cyclone_raw.mean(), cyclone_raw.std()
cyclone_freq = ((cyclone_raw - cmean) / cstd).astype(np.float32)

# --- Salinity (from WOA23 annual climatology, regrid 2°→1° via NN) ---
sal_2deg = np.load(os.path.join(DNS, 'salinity', 'woa23_salinity_v3_20x15.npz'))['salinity'].astype(np.float32)
salinity = np.zeros((N_LAT, N_LON), dtype=np.float32)
for i, si in enumerate(lat_idx_nn):
    for j, sj in enumerate(lon_idx_nn):
        salinity[i, j] = sal_2deg[si, sj]

print(f"  bathymetry: range=[{np.nanmin(bathy):.4f}, {np.nanmax(bathy):.4f}]")
print(f"  dist_to_shore: range=[{np.nanmin(dist_to_shore):.4f}, {np.nanmax(dist_to_shore):.4f}]")
print(f"  cyclone_freq: mean={cmean:.3f}, std={cstd:.3f}")
print(f"  ocean cells (land_fraction<0.5): {ocean_pre.sum()}")
print(f"  salinity: NaN frac={np.isnan(salinity).mean():.4f}")

# Fill salinity NaN ocean cells with spatial IDW from neighbors (WOA23 coverage gaps)
salinity = fill_nan_cells_spatial(salinity, ocean_pre, k=5)
print(f"  salinity post-fill: NaN frac={np.isnan(salinity).mean():.4f}")

# Fill sst_anomaly all-NaN ocean cells with spatial IDW from neighbors
# (these cells fall outside ERA5 0.25° regrid coverage)
print("  Filling sst_anomaly NaN ocean cells...")
sst_anomaly = fill_nan_cells_spatial(sst_anomaly, ocean_pre & ~ssta_all_nan, k=5)
print(f"    sst_anomaly post-fill: range=[{np.nanmin(sst_anomaly):.4f}, {np.nanmax(sst_anomaly):.4f}], "
      f"NaN frac={np.isnan(sst_anomaly).mean():.4f}")

# Fill chlorophyll & PAR all-NaN ocean cells with spatial IDW from neighbors
# (some coastal cells miss OC-CCI / MODIS coverage)
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

# Ocean mask: land_fraction < 0.5 AND has some SSTA coverage
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
print(f'  Chlorophyll NaN frac: {np.isnan(chlorophyll).mean():.4f}')
print(f'  PAR NaN frac: {np.isnan(par_norm).mean():.4f}')

# ═════════════════════════════════════════════════════════════
# STEP 8: Write output NetCDF
# ═════════════════════════════════════════════════════════════
print("\nSTEP 8: Writing output NetCDF...")

OUT_NC = os.path.join(OUT_DIR, 'scs_coral_bleaching_dataset_v9_1deg_raw.nc')

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
        'title': 'PIECF v9 1° (built from raw sources, DHW direct targets)',
        'history': 'Built by src/data/build_v9_1deg_from_raw.py',
        'dynamic_channels': 'sst(CRW raw), ssta, hotspot, chlorophyll, par, seasonal_sin, wind_speed, ssr, sst_anomaly (9)',
        'static_channels': 'bathymetry, dist_to_shore, cyclone_frequency, salinity, ocean_mask (5)',
        'targets': 'y_regression=clip(DHW/10,0,1), y_warning=clip(DHW/8,0,1), y_classification=I(DHW>=4)',
        'DHW_definition': '12-week rolling sum of clip(hotspot, 0, +inf) computed at 1° weekly resolution',
        'target_generation': 'DIRECT from DHW (no nearest-neighbor from 2°)',
        'sst_source': 'CRW 5km daily SST → 1° weekly block-avg (°C, raw)',
        'chlorophyll_source': 'OC-CCI v6.0 monthly → weekly causal (month-end forward-fill, NO future leakage), log1p',
        'par_source': 'MODIS monthly → weekly causal (month-end forward-fill, NO future leakage), min-max [0,1] (min=18.20, max=56.63)',
        'era5_source': 'ERA5 monthly 0.25° → 1° regrid → weekly causal (month-end forward-fill, NO future leakage)',
        'salinity_postfill': 'WOA23 annual → NaN ocean cells filled via spatial IDW (k=5)',
        'sst_anomaly_postfill': 'ERA5 skin temp → all-NaN ocean cells filled via spatial IDW (k=5)',
        'ocean_mask': 'land_fraction < 0.5 AND NOT ssta-all-NaN',
        'creation_date': str(np.datetime64('today')),
    },
)

encode = {z: {'zlib': True, 'complevel': 5, 'dtype': 'float32'}
          for z in data_vars if data_vars[z].dtype == np.float32}
for z in data_vars:
    if z not in encode:
        encode[z] = {'dtype': data_vars[z].dtype}

ds_out.to_netcdf(OUT_NC, encoding=encode)
mb = os.path.getsize(OUT_NC) / 1024 / 1024
print(f"\nWrote {OUT_NC} ({mb:.1f} MB)")

# ═════════════════════════════════════════════════════════════
# STEP 9: Validation
# ═════════════════════════════════════════════════════════════
print("\nSTEP 9: Validation...")
print(f"{'Variable':20s} {'Shape':20s} {'NaN%':>8s} {'Min':>10s} {'Max':>10s}")
print("-" * 72)
for name in data_vars:
    v = data_vars[name].values
    if v.ndim == 3:
        fin = v[np.isfinite(v)]
        row = f"{np.isnan(v).mean()*100:7.2f}" if len(fin) else "  100.00"
        mn = f"{np.nanmin(v):.4f}" if len(fin) else "-"
        mx = f"{np.nanmax(v):.4f}" if len(fin) else "-"
        print(f"{name:20s} {str(v.shape):20s} {row:>8s} {mn:>10s} {mx:>10s}")
    else:
        fin = v[np.isfinite(v)]
        mn = f"{np.nanmin(v):.4f}" if len(fin) else "-"
        mx = f"{np.nanmax(v):.4f}" if len(fin) else "-"
        nanp = f"{np.isnan(v).mean()*100:.2f}" if np.isnan(v).any() else "0.00"
        print(f"{name:20s} {str(v.shape):20s} {nanp:>8s} {mn:>10s} {mx:>10s}")

# Ocean mask stats
om = data_vars['ocean_mask'].values
n_ocean = int(om.sum())
n_land = int((~om.astype(bool)).sum())
print(f"\nOcean cells (1°): {n_ocean}, Land cells: {n_land}")

n_train = int((864 - 24 + 1) * 0.70)
print(f"Training pairs: {n_ocean} ocean × {n_train} windows = {n_ocean * n_train:,}")

# Cross-check: new DHW targets vs v9_final2 NN targets (diagnostic only;
# the dataset NC is already written above — skip gracefully if v9_final2
# is not available on this machine)
V9F2_PATH = os.path.join(OUTPUT_DIR, 'scs_coral_bleaching_dataset_v9_final2.nc')
if not os.path.exists(V9F2_PATH):
    print("  (v9_final2 not available; cross-check diagnostics skipped)")
    print("\n=== BUILD COMPLETE ===")
    sys.exit(0)
print("\n--- Cross-check: new DHW targets vs v9_final2 NN targets ---")
ds9 = xr.open_dataset(V9F2_PATH)
lats_ref = ds9['lat'].values
lons_ref = ds9['lon'].values
lat_idx_chk = np.array([np.argmin(np.abs(lats_ref - lat)) for lat in LATS])
lon_idx_chk = np.array([np.argmin(np.abs(lons_ref - lon)) for lon in LONS])

for vname in ['y_regression', 'y_warning', 'y_classification']:
    v_new = ds_out[vname].values
    v_ref_raw = ds9[vname].values
    min_t = min(v_new.shape[0], v_ref_raw.shape[0])
    v_ref = np.full_like(v_new, np.nan)
    for i, si in enumerate(lat_idx_chk):
        for j, sj in enumerate(lon_idx_chk):
            v_ref[:min_t, i, j] = v_ref_raw[:min_t, si, sj]
    v_new_cmp = v_new[:min_t]
    v_ref_cmp = v_ref[:min_t]
    valid = np.isfinite(v_new_cmp) & np.isfinite(v_ref_cmp)
    if valid.sum() > 0:
        d = np.abs(v_new_cmp[valid] - v_ref_cmp[valid])
        # R²
        ss_res = np.sum((v_new_cmp[valid] - v_ref_cmp[valid])**2)
        ss_tot = np.sum((v_ref_cmp[valid] - v_ref_cmp[valid].mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        print(f"  {vname:20s}: max_diff={d.max():.6f}, mean_diff={d.mean():.6f}, R2={r2:.6f}")
    else:
        print(f"  {vname:20s}: no overlap")

# Cross-check features vs v9_final2
print("\n--- Cross-check features vs v9_final2 (2° NN regrid) ---")
for vname in ['ssta', 'hotspot', 'chlorophyll', 'par', 'seasonal_sin', 'wind_speed', 'ssr', 'sst_anomaly']:
    if vname not in ds_out.data_vars or vname not in ds9.data_vars:
        continue
    v_new = ds_out[vname].values
    v_ref_raw = ds9[vname].values
    min_t = min(v_new.shape[0], v_ref_raw.shape[0])
    v_ref = np.full_like(v_new, np.nan)
    for i, si in enumerate(lat_idx_chk):
        for j, sj in enumerate(lon_idx_chk):
            v_ref[:min_t, i, j] = v_ref_raw[:min_t, si, sj]
    v_new_cmp = v_new[:min_t]
    v_ref_cmp = v_ref[:min_t]
    valid = np.isfinite(v_new_cmp) & np.isfinite(v_ref_cmp)
    if valid.sum() > 0:
        d = np.abs(v_new_cmp[valid] - v_ref_cmp[valid])
        print(f"  {vname:20s}: max_diff={d.max():.6f}, mean_diff={d.mean():.6f}, match_1e-5={np.mean(d < 1e-5):.4f}")
    else:
        print(f"  {vname:20s}: no overlap")

ds9.close()
print("\n=== BUILD COMPLETE ===")
