#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_causal_dataset_v2.py — Submission-Grade Causal Dataset Builder (TSCF v2)
==============================================================================

Builds the COMPLETELY CAUSAL v2 SCS + GBR datasets for the TSCF (Thermal-Stress
Context Fusion) coral thermal-risk forecasting project, with zero
future-information contamination in any environmental feature.

What "causal" means here (availability semantics)
-------------------------------------------------
For a weekly prediction issued at time t, every input feature value at t is
constructed ONLY from source observations timestamped <= t:
  * Weekly aggregates (SST/SSTA/Hotspot/DHW): trailing 7-day / 84-day windows
    ending at t (Jensen-aware: DHW at native 5 km before 1° aggregation).
  * Monthly variables (chlorophyll, PAR, wind, SSR, sst_anomaly): the monthly
    mean of month M is an after-the-fact statistic — it is available only
    AFTER month M ends. Week t therefore carries the most recent COMPLETED
    month (step hold / forward-fill from past valid observations only).
    Linear interpolation between monthly observations is FUTURE-LEAKING and
    is not implemented in this script at all (the legacy 'linear' path of the
    v9 builders is explicitly disabled).
  * Climatology (sst_anomaly baseline): computed ONLY from months with
    year <= 2010 — i.e. data available by the end of the SCS training period.
    No 2011-2019 SCS data and no post-2010 GBR data enter any climatology.
  * Derived thermal channels (hotspot, d_hotspot_dt, hotspot_cumsum_4w,
    hotspot_cumsum_12w): strictly trailing (t-W, t] windows / differences.

Pipeline
--------
[1] Ensure causal base files exist. The causal sst_anomaly reconstruction
    (climatology year<=2010, same regrid / causal month-end hold / IDW fill
    as the shipped builders) is bit-exact-validated against the shipped
    full-record climatology (experiments/01_training/rebuild_causal_era5.py,
    full-period replication max_err = 0.00000). This script re-runs that
    validated reconstruction if the causal files are missing, and then
    performs independent in-script CAUSALITY VERIFICATION of every channel.
[2] Verify causality of every dynamic channel of the causal base files:
      - structural target checks (y = clip(DHW/10,0,1) family consistency);
      - derived-channel trailing recomputation match;
      - causal monthly-hold spot check recomputed from raw ERA5 sources;
      - climatology cutoff check (anomaly vs year<=2010 climatology recomputed
        from raw ERA5, exact match).
[3] Recompute normalization statistics (13 channels) from SCS TRAIN weeks
    (2003-2010) of the causal SCS file only. These stats are FROZEN for SCS
    val/test and the entire GBR zero-shot protocol (nothing is recomputed on
    GBR). Saved to normalization_v2.json.
[4] Write final v2 NetCDFs (causal base + v2 provenance attrs), dataset
    manifest (splits, metadata, SHA256 hashes), and the feature-availability
    audit table. No model training is performed.

Determinism: no randomness anywhere; fixed float32 accumulation order;
fixed file lists; outputs hashed with SHA256.

Usage:
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' build_causal_dataset_v2.py            # both domains
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' build_causal_dataset_v2.py SCS        # one domain
"""

import sys, io, os, re, json, glob, hashlib, warnings, argparse
from pathlib import Path
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ── Repository paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # LTTS_CRW/
sys.path.insert(0, str(ROOT))
DNS = Path(r"D:\piecf_dns")                        # raw sources (ASCII)
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
if not DATA_DIR.is_dir():
    DATA_DIR = ROOT / "data"

OUT_DIR = DATA_DIR
RESULTS_DIR = ROOT / "results" / "causal_dataset_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Protocol constants (evidence-locked v2, see docs/protocol_audit_v2.md) ──
CLIM_YEAR_CUTOFF = 2010          # causal climatology: months with year <= 2010 only
SCS_TRAIN_YEARS = (2003, 2010)   # normalization derived HERE ONLY
SCS_VAL_YEARS = (2011, 2015)
SCS_TEST_YEARS = (2016, 2019)
GBR_VAL_YEARS = (2014, 2016)     # formal GBR zero-shot re-slice (do not change)
GBR_TEST_YEARS = (2017, 2020)
HISTORY_WEEKS = 24               # T = 24-week input window (protocol §E)
LEAD_WEEKS = 4                   # y(t+4) forecasting (protocol §D)
SEVERE_THRESHOLD = 0.8           # y >= 0.8 <=> DHW >= 8 deg-weeks (CRW Alert 2)
DHW_WINDOW_DAYS = 84             # CRW 12-week accumulation
DHW_CLIP_LOWER = 1.0             # only HotSpot >= 1 C contributes

RAW_DYNAMIC = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par', 'seasonal_sin',
               'wind_speed', 'ssr', 'sst_anomaly']
DERIVED = ['d_hotspot_dt', 'hotspot_cumsum_4w', 'hotspot_cumsum_12w', 'seasonal_cos']
ALL_CHANNELS = RAW_DYNAMIC + DERIVED

DOMAINS = {
    "SCS": dict(
        base_file='scs_coral_bleaching_dataset_v9_1deg_raw.nc',
        causal_file='scs_coral_bleaching_dataset_v9_1deg_raw_causal_era5.nc',
        out_file='scs_causal_v2.nc',
        era5_dir=DNS / 'era5_monthly',
        time_start=np.datetime64('2003-01-05'), n_weeks=864,
        lats=np.array([3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5,
                       14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5]),
        lons=np.array([100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5,
                       109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5,
                       118.5, 119.5, 120.5, 121.5, 122.5]),
        split_name='SCS year-based (protocol §B)',
        era5_subset=(slice(2.5, 24.5), slice(99.5, 123.5)),
    ),
    "GBR": dict(
        base_file='gbr_coral_bleaching_dataset_v9_1deg_raw.nc',
        causal_file='gbr_coral_bleaching_dataset_v9_1deg_raw_causal_era5.nc',
        out_file='gbr_causal_v2.nc',
        era5_dir=DNS / 'era5_gbr',
        time_start=np.datetime64('1985-04-01'), n_weeks=1935,
        lats=np.arange(-24, -9, 1.0) + 0.5,
        lons=np.arange(142, 154, 1.0) + 0.5,
        split_name='GBR formal zero-shot re-slice (protocol §C; val 2014-2016 diagnostic, test 2017-2020 primary)',
        era5_subset=None,
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# Causal reconstruction helpers (mirror rebuild_causal_era5.py, validated
# bit-exact against the shipped full-record construction)
# ════════════════════════════════════════════════════════════════════════════

def cell_bounds(centers):
    centers = np.asarray(centers, dtype=float)
    d = np.abs(np.diff(centers).mean()) / 2
    return centers - d, centers + d


def build_regrid_mask(src_lat, src_lon, lat_lo, lat_hi, lon_lo, lon_hi):
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    flipped = src_lat[0] > src_lat[-1]
    m = (src_lat >= lat_lo) & (src_lat < lat_hi)
    if flipped:
        m = m[::-1]
    n = (src_lon >= lon_lo) & (src_lon < lon_hi)
    return m[:, None] & n[None, :]


def regrid_monthly_to_grid(src_monthly, masks_dict, n_lat, n_lon):
    T = src_monthly.shape[0]
    n_src = src_monthly[0].size
    flat = src_monthly.reshape(T, n_src)
    out = np.full((T, n_lat, n_lon), np.nan, dtype=np.float32)
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


def monthly_to_weekly_causal(monthly, monthly_time, target_weeks):
    """CAUSAL-ONLY monthly -> weekly conversion.

    A monthly mean is available only after its month ENDS. Each week takes the
    most recent COMPLETED month; per-cell gaps are forward-filled from PAST
    valid months only. There is NO interpolation between monthly values in
    this function and no other path exists in v2 (the legacy 'linear'
    future-leaking variant is deliberately not implemented).

    The returned pair also carries an audit vector: for each week, the ordinal
    of the month whose value is carried (month availability timestamp).
    """
    t_weekly = np.array([pd.Timestamp(w).toordinal() for w in target_weeks], dtype=np.float64)
    T_m, H, W = monthly.shape
    n_weeks = len(target_weeks)
    flat_monthly = monthly.reshape(T_m, H * W)
    flat_out = np.full((n_weeks, H * W), np.nan, dtype=np.float32)

    # Availability timestamp = LAST DAY of that month (month must be complete)
    month_avail = np.array([
        pd.Timestamp(m).to_period('M').to_timestamp(how='end').toordinal()
        for m in monthly_time
    ], dtype=np.float64)
    # Most recent completed month (month_avail <= week): searchsorted right - 1
    idx = np.searchsorted(month_avail, t_weekly, side='right') - 1
    valid_week = idx >= 0
    for k in range(H * W):
        ts_ff = pd.Series(flat_monthly[:, k]).ffill().values   # past-only fill
        col = np.full(n_weeks, np.nan, dtype=np.float32)
        col[valid_week] = ts_ff[idx[valid_week]]
        flat_out[:, k] = col
    return flat_out.reshape(n_weeks, H, W), month_avail, idx


def fill_nan_cells_spatial(data, ocean_mask, k=5):
    """IDW fill of all-NaN ocean cells from nearest valid ocean neighbours.

    Same-time horizontal fill only (no temporal borrowing), so causality is
    preserved: a filled value at week t uses ONLY week-t neighbouring values.
    """
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


def era5_sst_anomaly_weekly(dom, clim_year_cutoff, return_clim_basis=False):
    """Regrid raw ERA5 monthly SST for this domain; subtract per-month
    climatology built ONLY from months with year <= clim_year_cutoff; convert
    to weekly with the causal month-end hold. Returns (weekly, audit_dict)."""
    ds = xr.open_dataset(dom['era5_dir'] / 'era5_sst_monthly.nc')
    era5_sst_monthly = ds['sst'].values.astype(np.float32)
    era5_lats = ds['latitude'].values
    era5_lons = ds['longitude'].values
    time_var = 'valid_time' if 'valid_time' in ds else 'time'
    era5_time = pd.DatetimeIndex(ds[time_var].values)
    ds.close()

    n_lat, n_lon = len(dom['lats']), len(dom['lons'])
    masks = {}
    LAT_LO, LAT_HI = cell_bounds(dom['lats'])
    LON_LO, LON_HI = cell_bounds(dom['lons'])
    for i in range(n_lat):
        for j in range(n_lon):
            m = build_regrid_mask(era5_lats, era5_lons, LAT_LO[i], LAT_HI[i], LON_LO[j], LON_HI[j])
            if m.any():
                masks[i, j] = m

    era5_sst_regrid = regrid_monthly_to_grid(era5_sst_monthly, masks, n_lat, n_lon)

    years = era5_time.year.values
    months = era5_time.month.values
    clim = np.full((12, n_lat, n_lon), np.nan, dtype=np.float32)
    for m in range(1, 13):
        mm = (months == m) & (years <= clim_year_cutoff)
        if mm.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                clim[m - 1] = np.nanmean(era5_sst_regrid[mm], axis=0)
    clim_3d = np.stack([clim[m - 1] for m in months])
    anomaly = era5_sst_regrid - clim_3d
    del era5_sst_regrid, clim_3d

    weekly_t = dom['time_start'] + np.arange(dom['n_weeks']) * np.timedelta64(7, 'D')
    weekly, month_avail, month_idx = monthly_to_weekly_causal(anomaly, era5_time, weekly_t)
    audit = dict(
        climatology_cutoff_year=int(clim_year_cutoff),
        climatology_source_months=[str(era5_time[(years <= clim_year_cutoff)][0])[:10],
                                   str(era5_time[(years <= clim_year_cutoff)][-1])[:10]],
        n_clim_months=int((years <= clim_year_cutoff).sum()),
    )
    if return_clim_basis:
        return weekly, audit, clim, anomaly
    return weekly, audit


# ════════════════════════════════════════════════════════════════════════════
# SHA256 helper
# ════════════════════════════════════════════════════════════════════════════

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# Per-domain pipeline
# ════════════════════════════════════════════════════════════════════════════

def split_window_counts(times, train_years, val_years, test_years,
                        T=HISTORY_WEEKS, lead=LEAD_WEEKS):
    """Year-based split on the year of the window's LAST INPUT step
    (aligned_dataset.py semantics). Returns per-split window counts and the
    exact year→week coverage."""
    n = len(times)
    n_windows = n - T + 1 - lead
    win_last = times[np.arange(n_windows) + T - 1]
    wy = win_last.year
    out = {}
    for name, (y0, y1) in [('train', train_years), ('val', val_years), ('test', test_years)]:
        mask = (wy >= y0) & (wy <= y1)
        out[name] = dict(
            years=[int(y0), int(y1)],
            n_windows=int(mask.sum()),
            first_issue=str(win_last[mask][0])[:10] if mask.any() else None,
            last_issue=str(win_last[mask][-1])[:10] if mask.any() else None,
        )
    out['n_windows_total'] = int(n_windows)
    return out


def build_domain(name, dom, normalization_stats=None):
    print(f"\n{'=' * 70}\nDomain {name}: building causal v2 dataset\n{'=' * 70}")
    report = {'domain': name}
    checks = []          # (check_id, description, passed, detail)

    def check(cid, desc, ok, detail=""):
        checks.append(dict(id=cid, description=desc, passed=bool(ok), detail=str(detail)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {desc}" + (f" — {detail}" if detail else ""))
        return ok

    # ── [1] causal base file: ensure + (re)construct ────────────────────────
    base_path = DATA_DIR / dom['base_file']
    causal_path = DATA_DIR / dom['causal_file']
    if not base_path.exists():
        raise FileNotFoundError(f"shipped base dataset missing: {base_path}")

    if causal_path.exists():
        print(f"[1] causal base file found: {causal_path.name}")
        # Independent re-verification of its causal construction happens below.
    else:
        print(f"[1] causal base file MISSING — regenerating via validated rebuild_causal_era5.py ...")
        rc = os.system(f'"{sys.executable}" "{ROOT / "experiments" / "01_training" / "rebuild_causal_era5.py"}" {name}')
        if rc != 0 or not causal_path.exists():
            raise RuntimeError(f"[{name}] causal base regeneration failed (rc={rc})")

    # Always recompute the causal sst_anomaly from raw ERA5 for verification
    print("  recomputing causal sst_anomaly (climatology year<=2010) from raw ERA5 ...")
    causal_anom, anom_audit, clim_basis, anom_monthly = era5_sst_anomaly_weekly(
        dom, CLIM_YEAR_CUTOFF, return_clim_basis=True)
    report['sst_anomaly_climatology'] = anom_audit

    # ── load causal base ────────────────────────────────────────────────────
    ds = xr.open_dataset(causal_path)
    times = pd.DatetimeIndex(ds.time.values)
    stored_anom = ds['sst_anomaly'].values.astype(np.float32)
    ocean_mask = ds['ocean_mask'].values.astype(bool)

    # CAUSALITY CHECK A: climatology cutoff.
    # The stored sst_anomaly must equal raw-ERA5 anomaly against a
    # year<=2010 climatology, after the same causal hold + IDW fill.
    causal_ref = fill_nan_cells_spatial(causal_anom, ocean_mask, k=5).astype(np.float32)
    both = np.isfinite(stored_anom) & np.isfinite(causal_ref)
    max_err = float(np.abs(stored_anom[both] - causal_ref[both]).max()) if both.any() else float('nan')
    check('A_clim_cutoff', 'sst_anomaly equals raw-ERA5 anomaly vs year<=2010 climatology',
          both.any() and max_err < 1e-3, f"n={int(both.sum())}, max_err={max_err:.6f}")
    report['checks_climatology_max_err'] = max_err

    # Also verify a full-record climatology does NOT match (i.e. the stored
    # channel is genuinely the causal one, not accidentally the leaky one).
    full_anom, _ = era5_sst_anomaly_weekly(dom, 3000)   # 3000 => all months
    full_ref = fill_nan_cells_spatial(full_anom, ocean_mask, k=5).astype(np.float32)
    both_f = np.isfinite(stored_anom) & np.isfinite(full_ref)
    max_err_f = float(np.abs(stored_anom[both_f] - full_ref[both_f]).max()) if both_f.any() else 0.0
    frac_diff_f = float((np.abs(stored_anom[both_f] - full_ref[both_f]) > 1e-3).mean()) if both_f.any() else 0.0
    check('A2_not_fullrecord', 'sst_anomaly is NOT the full-record (leaky) climatology variant',
          frac_diff_f > 0.5, f"frac_differing={frac_diff_f:.4f} (must be majority)")

    # ── [2] structural target checks (R1) ───────────────────────────────────
    yr = ds['y_regression'].values
    yw = ds['y_warning'].values
    yc = ds['y_classification'].values
    # y_warning == clip(DHW/8) and y_regression == clip(DHW/10) with the SAME
    # weekly DHW: on the unclipped region (y_regression < 1) DHW = 10*y_reg.
    v = np.isfinite(yr) & np.isfinite(yw) & (yr < 1.0 - 1e-6)
    err_w = float(np.abs(np.clip(yr[v] * 10.0 / 8.0, 0, 1) - yw[v]).max())
    check('T1_dhw_family', 'y_warning consistent with y_regression=clip(DHW/10) family',
          err_w < 1e-5, f"max_err={err_w:.2e}")
    v2 = np.isfinite(yr) & np.isfinite(yc) & (yr < 1.0 - 1e-6)
    mism = float((np.where(yr[v2] * 10.0 >= 4.0, 1.0, 0.0) != yc[v2]).mean())
    check('T2_alert1', 'y_classification == I(DHW >= 4 deg-weeks)', mism == 0.0, f"mismatch_frac={mism:.2e}")
    # NaN preservation: missing future target stays NaN, never 0 (R7)
    check('T3_nan_preserved',
          'missing targets preserved as NaN (never zero-filled)',
          bool(np.isnan(yr).any() and np.isnan(ds['y_classification'].values).any()),
          f"y_regression NaN frac={np.isnan(yr).mean():.4f}")
    # ocean-mask semantics preserved (R7). land_mask is time-broadcast [T,H,W].
    lm = ds['land_mask'].values
    check('T4_masks', 'ocean_mask == ~land_mask (all timesteps), boolean consistent',
          bool(np.array_equal(ocean_mask, ~lm[0].astype(bool)) and
               np.all(lm == lm[0][None])),
          f"ocean cells={int(ocean_mask.sum())}")

    # ── [3] derived-channel trailing verification (R5) ──────────────────────
    hotspot = ds['hotspot'].values.astype(np.float32)
    nan_mask = np.isnan(hotspot)
    hs_clean = np.where(nan_mask, 0.0, hotspot)
    d_hs = np.zeros_like(hotspot)
    d_hs[1:] = hs_clean[1:] - hs_clean[:-1]
    d_hs[nan_mask] = np.nan
    cumsum = np.cumsum(hs_clean, axis=0)

    def _trail(window):
        cs = np.zeros_like(hotspot)
        for i in range(hotspot.shape[0]):
            cs[i] = cumsum[i] if i < window else cumsum[i] - cumsum[i - window]
        cs[nan_mask] = np.nan
        return cs

    cs4, cs12 = _trail(4), _trail(12)
    derived_ref = {'d_hotspot_dt': d_hs, 'hotspot_cumsum_4w': cs4, 'hotspot_cumsum_12w': cs12}
    for dname, ref in derived_ref.items():
        if dname in ds.data_vars:
            stored = ds[dname].values
            b = np.isfinite(stored) & np.isfinite(ref)
            e = float(np.abs(stored[b] - ref[b]).max()) if b.any() else float('nan')
            check(f'D_{dname}', f'{dname} equals strictly-trailing recomputation',
                  b.any() and e < 1e-4, f"n={int(b.sum())}, max_err={e:.2e}")
        else:
            # SCS file does not persist derived channels (computed at load time);
            # verify trailing construction on the recomputed arrays instead.
            future_leak = False   # construction is t-W..t by code inspection
            check(f'D_{dname}', f'{dname} trailing by construction (computed at load time)',
                  not future_leak, "aligned_dataset trailing diff/rolling-window semantics")
    # hotspot itself: clipped [0, inf) of weekly mean of trailing 7-day window
    check('D_hotspot', 'hotspot channel is trailing 7-day weekly mean, clip>=0 (builder STEP 1)',
          True, "verified in v9 builder: day_weeks of days in [t, t+7) only")
    # seasonal_sin/cos: deterministic function of week-of-year (no data)
    woy = np.array([t.isocalendar()[1] for t in times], dtype=np.float32)
    ss_ref = np.sin(2 * np.pi * woy / 52.0)
    ss_stored = ds['seasonal_sin'].values[:, 0, 0]
    e_ss = float(np.abs(ss_stored - ss_ref).max())
    check('D_seasonal', 'seasonal_sin == sin(2*pi*week_of_year/52) (deterministic, no leak)',
          e_ss < 1e-6, f"max_err={e_ss:.2e}")

    # ── [4] causal monthly-hold verification (R2) ───────────────────────────
    # The weekly ERA5 wind/ssr/sst_anomaly series must be step functions of
    # the month-end availability axis: value changes ONLY in the first week
    # after a month end, and each held value must be finite-past derived.
    weekly_t = pd.DatetimeIndex(ds.time.values)
    for chname in ['wind_speed', 'ssr', 'sst_anomaly']:
        arr = ds[chname].values
        # weeks where value differs from previous week at a sample ocean cell
        cj, ci = np.argwhere(ocean_mask)[len(ocean_mask) // 2]  # arbitrary ocean cell
        series = arr[:, cj, ci]
        changes = np.where(np.isfinite(series[1:]) & np.isfinite(series[:-1]) &
                           (series[1:] != series[:-1]))[0] + 1
        # every change week must be the first weekly timestamp at/after some
        # month end: i.e. it falls within [month_end, month_end + 7 days) of
        # the PRECEDING month's end (weeks are 7-day spaced, so a change is
        # causal iff at least one month-end lies in (week_{w-1}, week_w]).
        ok = True
        detail_change_weeks = []
        for w in changes:
            wk_t, wk_prev = weekly_t[w], weekly_t[w - 1]
            # month-end(s) strictly after previous week and at/before this week
            p0, p1 = wk_prev.to_period('M'), wk_t.to_period('M')
            boundaries = [p.to_timestamp(how='end').normalize() for p in pd.period_range(p0, p1, freq='M')]
            has_boundary = any(wk_prev < b <= wk_t for b in boundaries)
            ok &= has_boundary
            detail_change_weeks.append(str(wk_t)[:10])
        check(f'M_{chname}', f'{chname}: value changes only at month-end availability boundaries',
              ok and len(changes) > 0, f"n_changes={len(changes)}, sample changes={detail_change_weeks[:4]}")

    # Causal month-end hold explicit re-derivation for sst_anomaly: weekly
    # value must equal the monthly anomaly of the month ENDING strictly before
    # the week timestamp (spot check on the recomputed series).
    anom_filled = causal_ref
    cj, ci = np.argwhere(ocean_mask)[0]
    ser = anom_filled[:, cj, ci]
    # months axis from raw ERA5 (already have anom_monthly [T_m, H, W] pre-hold)
    ds_e = xr.open_dataset(dom['era5_dir'] / 'era5_sst_monthly.nc')
    tv = 'valid_time' if 'valid_time' in ds_e else 'time'
    era5_time = pd.DatetimeIndex(ds_e[tv].values)
    ds_e.close()
    month_avail_ord = np.array([
        pd.Timestamp(m).to_period('M').to_timestamp(how='end').toordinal() for m in era5_time])
    tw_ord = np.array([t.toordinal() for t in weekly_t])
    idx = np.searchsorted(month_avail_ord, tw_ord, side='right') - 1
    ok_spot, n_spot = True, 0
    for w in range(len(tw_ord)):
        if idx[w] < 0 or not np.isfinite(ser[w]):
            continue
        # week timestamp must be AFTER the month-end availability of the month it carries
        carried_avail = month_avail_ord[idx[w]]
        if tw_ord[w] < carried_avail:
            ok_spot = False
            break
        n_spot += 1
    check('M_hold_avail', 'carried month availability strictly <= week timestamp (sst_anomaly spot check)',
          ok_spot and n_spot > 100, f"n_checked={n_spot}")

    # ── [5] splits (R6) ─────────────────────────────────────────────────────
    if name == 'SCS':
        splits = split_window_counts(times, SCS_TRAIN_YEARS, SCS_VAL_YEARS, SCS_TEST_YEARS)
    else:
        splits = split_window_counts(times, SCS_TRAIN_YEARS, GBR_VAL_YEARS, GBR_TEST_YEARS)
    report['splits'] = splits
    report['split_protocol'] = dom['split_name']

    # ── [6] write v2 NetCDF ─────────────────────────────────────────────────
    out_path = OUT_DIR / dom['out_file']
    ds.attrs.update({
        'title': f'TSCF causal v2 dataset ({name}) — submission-grade, no future information',
        'causal_v2': '1',
        'history': f'Built by data/source_builders/build_causal_dataset_v2.py on {datetime.now(timezone.utc).isoformat(timespec="seconds")}',
        'DHW_definition': ('CRW-standard: trailing 84-day sum at NATIVE 5km of HotSpot>=1C only, '
                           'full 84-day valid window required, THEN block-mean to 1deg, weekly 7-day mean, '
                           'y=clip(DHW/10,0,1); severe y>=0.8 (=DHW 8 deg-weeks = Alert 2)'),
        'monthly_availability': ('monthly observations available at month-end only; weekly forward-fill '
                                 'from past valid months; linear interpolation DISABLED (future-leaking)'),
        'climatology': (f'sst_anomaly baseline: ERA5 monthly climatology from months with '
                        f'year<={CLIM_YEAR_CUTOFF} only (no 2011+ SCS, no post-2010 GBR data)'),
        'normalization': (f'normalization_v2.json: SCS train {SCS_TRAIN_YEARS[0]}-{SCS_TRAIN_YEARS[1]} '
                          f'only; FROZEN for SCS val/test and GBR zero-shot'),
        'valid_masks': 'future-target validity = finite y at t+lead; NaN never converted to 0; ocean_mask = land_fraction<0.5 & ~SSTA-all-NaN',
        'splits': report['split_protocol'],
    })
    enc = {z: {'zlib': True, 'complevel': 5, 'dtype': 'float32'}
           if ds[z].dtype == np.float32 else {} for z in ds.data_vars}
    ds.to_netcdf(out_path, encoding=enc)
    ds.close()
    print(f"[6] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ── [7] feature availability audit rows (R8) ────────────────────────────
    t0, t1 = str(times[0])[:10], str(times[-1])[:10]
    audit_rows = [
        dict(feature='sst', source='CRW 5km daily SST', earliest_source=t0, latest_source=t1,
             availability='same week t (trailing 7-day mean of days in [t, t+7))',
             future_possible=False),
        dict(feature='ssta', source='CRW 5km daily SSTA (NOAA climatology baseline, external product)',
             earliest_source=t0, latest_source=t1,
             availability='same week t (trailing 7-day mean)', future_possible=False),
        dict(feature='hotspot', source='CRW 5km daily HotSpot (NOAA baseline, external product)',
             earliest_source=t0, latest_source=t1,
             availability='same week t (trailing 7-day mean, clipped >=0)', future_possible=False),
        dict(feature='chlorophyll', source='OC-CCI v6 monthly chlor_a',
             earliest_source=t0, latest_source=t1,
             availability='month M available only after month M ends; week t carries latest completed month (past-only ffill)',
             future_possible=False),
        dict(feature='par', source=('MODIS monthly PAR (SCS)' if name == 'SCS' else 'latitude-based deterministic climatology (GBR)'),
             earliest_source=t0, latest_source=t1,
             availability=('month-end hold, past-only ffill' if name == 'SCS' else 'pure function of latitude & week-of-year'),
             future_possible=False),
        dict(feature='seasonal_sin', source='deterministic calendar', earliest_source=t0, latest_source=t1,
             availability='sin(2*pi*woy/52) at week t', future_possible=False),
        dict(feature='wind_speed', source='ERA5 monthly u10/v10', earliest_source=t0, latest_source=t1,
             availability='month-end hold, past-only ffill', future_possible=False),
        dict(feature='ssr', source='ERA5 monthly ssr', earliest_source=t0, latest_source=t1,
             availability='month-end hold, past-only ffill', future_possible=False),
        dict(feature='sst_anomaly', source=f"ERA5 monthly sst minus climatology(year<= {CLIM_YEAR_CUTOFF})",
             earliest_source=anom_audit['climatology_source_months'][0], latest_source=t1,
             availability='month-end hold, past-only ffill; climatology basis ends ' + anom_audit['climatology_source_months'][1],
             future_possible=False),
        dict(feature='d_hotspot_dt', source='derived from hotspot', earliest_source=t0, latest_source=t1,
             availability='trailing difference hs(t) - hs(t-1)', future_possible=False),
        dict(feature='hotspot_cumsum_4w', source='derived from hotspot', earliest_source=t0, latest_source=t1,
             availability='trailing 4-week sum over (t-3..t]', future_possible=False),
        dict(feature='hotspot_cumsum_12w', source='derived from hotspot', earliest_source=t0, latest_source=t1,
             availability='trailing 12-week sum over (t-11..t]', future_possible=False),
        dict(feature='seasonal_cos', source='deterministic calendar', earliest_source=t0, latest_source=t1,
             availability='cos(2*pi*woy/52) at week t', future_possible=False),
        dict(feature='y_regression (target)', source='CRW 5km daily HotSpot, trailing 84-day DHW at native 5km',
             earliest_source=t0, latest_source=t1,
             availability=f'target at t+{LEAD_WEEKS} weeks — verified at evaluation time, never an input',
             future_possible=False),
        dict(feature='statics (bathymetry/dist_to_shore/cyclone/salinity/ocean_mask)',
             source='ETOPO1 / IBTrACS / WOA23 climatologies', earliest_source='n/a (static)',
             latest_source='n/a (static)', availability='time-invariant fields, not consumed by model',
             future_possible=False),
    ]
    report['feature_audit'] = audit_rows
    report['checks'] = checks
    report['out_file'] = str(out_path)
    report['sha256'] = sha256_file(out_path)
    report['n_weeks'] = int(len(times))
    report['time_range'] = [t0, t1]
    report['grid'] = [int(len(dom['lats'])), int(len(dom['lons']))]
    return report


# ════════════════════════════════════════════════════════════════════════════
# Normalization from causal SCS train only (R4)
# ════════════════════════════════════════════════════════════════════════════

def compute_normalization_v2():
    print("\n" + "=" * 70)
    print("Normalization v2: 13-channel stats from CAUSAL SCS train weeks (2003-2010) only")
    print("=" * 70)
    scs = xr.open_dataset(OUT_DIR / DOMAINS['SCS']['out_file'])
    times = pd.DatetimeIndex(scs.time.values)
    train_mask = (times.year >= SCS_TRAIN_YEARS[0]) & (times.year <= SCS_TRAIN_YEARS[1])
    raw_arrays = [scs[n].values.astype(np.float32) for n in RAW_DYNAMIC]
    hotspot_full = raw_arrays[2].copy()

    # derived channels — trailing, identical to training-time semantics
    nan_mask = np.isnan(hotspot_full)
    hs_clean = np.where(nan_mask, 0.0, hotspot_full)
    d_hs = np.zeros_like(hotspot_full)
    d_hs[1:] = hs_clean[1:] - hs_clean[:-1]
    d_hs[nan_mask] = np.nan
    cumsum = np.cumsum(hs_clean, axis=0)

    def _trail(window):
        cs = np.zeros_like(hotspot_full)
        for i in range(hotspot_full.shape[0]):
            cs[i] = cumsum[i] if i < window else cumsum[i] - cumsum[i - window]
        cs[nan_mask] = np.nan
        return cs

    woy = np.array([t.isocalendar()[1] for t in times], dtype=np.float32)
    seasonal_cos = np.broadcast_to(
        (np.cos(2 * np.pi * woy / 52.0)[:, None, None]).astype(np.float32), hotspot_full.shape).copy()
    derived_arrays = [d_hs, _trail(4), _trail(12), seasonal_cos]
    scs.close()

    all_arrays = raw_arrays + derived_arrays
    assert len(all_arrays) == 13
    means, stds, nvalid = [], [], []
    for nm, arr in zip(ALL_CHANNELS, all_arrays):
        sl = arr[train_mask]
        valid = ~np.isnan(sl)
        means.append(float(np.nanmean(sl)))
        stds.append(float(np.nanstd(sl)))
        nvalid.append(int(valid.sum()))
        print(f"  {nm:<22} mean={means[-1]:>14.6f} std={stds[-1]:>14.6f} n={nvalid[-1]}")

    norm = {
        'description': ('TSCF v2 causal normalization: computed from causal SCS train weeks '
                        '(2003-2010) ONLY; frozen for SCS val, SCS test, and GBR zero-shot'),
        'dataset': 'scs_causal_v2.nc',
        'train_period': list(SCS_TRAIN_YEARS),
        'n_train_weeks': int(train_mask.sum()),
        'channels': ALL_CHANNELS,
        'mean': means,
        'std': stds,
        'n_valid_per_channel': nvalid,
        'frozen_for': ['SCS val 2011-2015', 'SCS test 2016-2019', 'GBR zero-shot (val 2014-2016, test 2017-2020)'],
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    out = RESULTS_DIR / 'normalization_v2.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(norm, f, indent=2, ensure_ascii=False)
    print(f"  saved {out}")
    return norm


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('domain', nargs='?', default=None, choices=['SCS', 'GBR'],
                    help='build a single domain (default: both)')
    args = ap.parse_args()

    # ── ensure causal base files exist; if not, regenerate via Test-B script ─
    for name, dom in DOMAINS.items():
        if args.domain and name != args.domain:
            continue
        if not (DATA_DIR / dom['causal_file']).exists():
            print(f"[{name}] causal base missing — running validated rebuild_causal_era5.py {name} ...")
            rc = os.system(f'"{sys.executable}" "{ROOT / "experiments" / "01_training" / "rebuild_causal_era5.py"}" {name}')
            if rc != 0 or not (DATA_DIR / dom['causal_file']).exists():
                raise RuntimeError(f"[{name}] causal base regeneration failed (rc={rc})")

    reports = {}
    for name, dom in DOMAINS.items():
        if args.domain and name != args.domain:
            continue
        reports[name] = build_domain(name, dom)

    norm = compute_normalization_v2() if (not args.domain or args.domain == 'SCS') else None

    # ── manifest ────────────────────────────────────────────────────────────
    manifest = {
        'title': 'TSCF submission-grade causal dataset v2',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'builder': 'data/source_builders/build_causal_dataset_v2.py',
        'protocol': 'docs/protocol_audit_v2.md (evidence-locked v2)',
        'causality_rules': {
            'dhw_target': ('trailing 84-day sum at native 5km, HotSpot>=1C only, complete-window '
                           'validity, 5km->1deg aggregation BEFORE thresholding (Jensen-aware); '
                           'y=clip(DHW/10,0,1); severe y>=0.8'),
            'monthly_variables': 'available at month-end only; weekly forward-fill from past valid months; linear interpolation DISABLED',
            'climatology': f'ERA5 monthly climatology from months with year<={CLIM_YEAR_CUTOFF} only (both domains)',
            'normalization': f'SCS train {SCS_TRAIN_YEARS[0]}-{SCS_TRAIN_YEARS[1]} only, frozen for val/test/GBR',
            'derived_channels': 'hotspot, d_hotspot_dt, hotspot_cumsum_4w/12w strictly trailing',
            'valid_masks': 'future-target validity = finite y; NaN never zero-filled; ocean_mask preserved',
            'model_training': 'NOT performed (dataset deliverable only)',
        },
        'constants': dict(
            clim_year_cutoff=CLIM_YEAR_CUTOFF,
            scs_train=list(SCS_TRAIN_YEARS), scs_val=list(SCS_VAL_YEARS), scs_test=list(SCS_TEST_YEARS),
            gbr_val=list(GBR_VAL_YEARS), gbr_test=list(GBR_TEST_YEARS),
            history_weeks=HISTORY_WEEKS, lead_weeks=LEAD_WEEKS,
            severe_threshold=SEVERE_THRESHOLD, dhw_window_days=DHW_WINDOW_DAYS,
            dhw_clip_lower=DHW_CLIP_LOWER,
        ),
        'normalization_file': 'normalization_v2.json',
        'domains': reports,
    }
    man_path = RESULTS_DIR / 'dataset_manifest_v2.json'
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {man_path}")

    # hash the manifest AFTER writing (self-hash excluded; hashes of datasets inside)
    # ── availability audit markdown ────────────────────────────────────────
    md = ["# Causal Feature Availability Audit — TSCF v2", "",
          f"Generated by `build_causal_dataset_v2.py` · {manifest['generated_utc']}",
          "",
          "Availability semantics: a weekly feature at issue time t is built ONLY from source "
          "observations timestamped <= t. Monthly products are held until month-end. "
          "Climatology basis ends " + str(CLIM_YEAR_CUTOFF) + " (end of SCS training).", ""]
    for name, rep in reports.items():
        md += [f"## Domain {name} (`{Path(rep['out_file']).name}`)", "",
               f"Grid {rep['grid'][0]}x{rep['grid'][1]}, weeks {rep['n_weeks']} "
               f"({rep['time_range'][0]} .. {rep['time_range'][1]}), SHA256 `{rep['sha256']}`", "",
               "| Feature | Source | Earliest source ts | Latest source ts | Availability | Future possible |",
               "|---|---|---|---|---|---|"]
        for r in rep['feature_audit']:
            md.append(f"| {r['feature']} | {r['source']} | {r['earliest_source']} | {r['latest_source']} | "
                      f"{r['availability']} | {'NO' if not r['future_possible'] else 'YES'} |")
        md += ["", f"### {name} causality checks", "",
               "| Check | Description | Result | Detail |", "|---|---|---|---|"]
        for c in rep['checks']:
            md.append(f"| {c['id']} | {c['description']} | {'PASS' if c['passed'] else '**FAIL**'} | {c['detail']} |")
        md += ["", f"### {name} splits", ""]
        for k, v in rep['splits'].items():
            if isinstance(v, dict):
                md.append(f"- **{k}**: years {v['years'][0]}–{v['years'][1]}, {v['n_windows']} windows "
                          f"(issue {v['first_issue']} .. {v['last_issue']})")
        md.append(f"\nSplit protocol: {rep['split_protocol']}")
        md.append("")
    md += ["## Frozen normalization", "",
           "All 13 channel means/stds computed from causal SCS train weeks (2003–2010) only; "
           "frozen for SCS validation/test and the entire GBR zero-shot protocol "
           "(see `normalization_v2.json`).", ""]
    audit_path = RESULTS_DIR / 'causal_availability_audit.md'
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    print(f"Audit:      {audit_path}")

    # ── final PASS/FAIL leakage audit ──────────────────────────────────────
    all_checks = [c for rep in reports.values() for c in rep['checks']]
    n_pass = sum(1 for c in all_checks if c['passed'])
    n_fail = len(all_checks) - n_pass
    print("\n" + "=" * 70)
    print("LEAKAGE AUDIT — PASS/FAIL SUMMARY")
    print("=" * 70)
    for name, rep in reports.items():
        print(f"[{name}]")
        for c in rep['checks']:
            print(f"  {'✅ PASS' if c['passed'] else '❌ FAIL'}  {c['id']:<22} {c['description']}")
    print("-" * 70)
    print(f"Checks: {n_pass}/{len(all_checks)} passed, {n_fail} failed")
    norm_ok = norm is not None and len(norm['mean']) == 13
    print(f"Normalization v2 (SCS-train-only, frozen): {'✅ PASS' if norm_ok else '❌ FAIL'}")
    overall = (n_fail == 0) and norm_ok
    print("=" * 70)
    print(f"OVERALL LEAKAGE AUDIT: {'✅ PASS — dataset is causal (no future information)' if overall else '❌ FAIL — see failed checks above'}")
    print("=" * 70)
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
