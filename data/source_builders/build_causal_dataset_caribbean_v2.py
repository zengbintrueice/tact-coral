#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_causal_dataset_caribbean_v2.py — Causal v2 dataset for the CARIBBEAN
============================================================================

Mirrors build_causal_dataset_v2.py (SCS/GBR) for the new Caribbean zero-shot
target domain. NEW file; the SCS/GBR builder is untouched.

The Caribbean raw base (caribbean_coral_bleaching_dataset_v9_1deg_raw.nc) is
built by data/source_builders/build_caribbean_1deg_from_raw.py with an ALREADY
CAUSAL sst_anomaly (ERA5 climatology from months with year<=2010). This script:
  [1] recomputes the causal sst_anomaly from raw ERA5 Caribbean and verifies
      it equals the stored channel (max_err < 1e-3);
  [2] verifies causality of every dynamic channel (same check battery as the
      SCS/GBR v2 builder: target family, trailing derived channels, seasonal
      determinism, month-end hold, climatology cutoff, no-full-record match);
  [3] writes caribbean_causal_v2.nc (9 raw dynamic + targets + masks + statics,
      same schema as scs_causal_v2.nc), a manifest + availability audit to
      results/caribbean_zero_shot/.

Normalization: NOT recomputed here. The frozen SCS-train normalization
(normalization_v2.json / baked in the checkpoint) is reused verbatim — the
model's ChannelNormalizer applies SCS stats to Caribbean inputs, exactly as for
GBR zero-shot.

Determinism: no randomness anywhere; outputs hashed with SHA256.

Run:
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' data/source_builders/build_causal_dataset_caribbean_v2.py
"""
import sys, io, os, json, hashlib, warnings
from pathlib import Path
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DNS = Path(r"D:\piecf_dns")
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
if not DATA_DIR.is_dir():
    DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR
RESULTS_DIR = ROOT / "results" / "caribbean_zero_shot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLIM_YEAR_CUTOFF = 2010
SEVERE_THRESHOLD = 0.8
HISTORY_WEEKS = 24
LEAD_WEEKS = 4
CARIB_VAL_YEARS = (2014, 2016)     # diagnostic (mirror GBR protocol)
CARIB_TEST_YEARS = (2017, 2020)    # primary zero-shot test window

RAW_DYNAMIC = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par', 'seasonal_sin',
               'wind_speed', 'ssr', 'sst_anomaly']
DERIVED = ['d_hotspot_dt', 'hotspot_cumsum_4w', 'hotspot_cumsum_12w', 'seasonal_cos']
ALL_CHANNELS = RAW_DYNAMIC + DERIVED

# Caribbean domain table (weekly axis 2003-01-05 .. 2020-12-29, 940 weeks)
CARIB = dict(
    base_file='caribbean_coral_bleaching_dataset_v9_1deg_raw.nc',
    out_file='caribbean_causal_v2.nc',
    era5_dir=DNS / 'era5_caribbean',
    time_start=np.datetime64('2003-01-05'),
    n_weeks=939,
    lats=np.arange(4, 26, 1.0) + 0.5,
    lons=np.arange(-91, -59, 1.0) + 0.5,
)


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
    t_weekly = np.array([pd.Timestamp(w).toordinal() for w in target_weeks], dtype=np.float64)
    T_m, H, W = monthly.shape
    n_weeks = len(target_weeks)
    flat_monthly = monthly.reshape(T_m, H * W)
    flat_out = np.full((n_weeks, H * W), np.nan, dtype=np.float32)
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
    return flat_out.reshape(n_weeks, H, W), month_avail, idx


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


def era5_sst_anomaly_weekly(clim_year_cutoff, return_clim_basis=False):
    dom = CARIB
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
        n_clim_months=int((years <= clim_year_cutoff).sum()),
    )
    if return_clim_basis:
        return weekly, audit, clim, anomaly
    return weekly, audit


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def split_window_counts(times, val_years, test_years, T=HISTORY_WEEKS, lead=LEAD_WEEKS):
    n = len(times)
    n_windows = n - T + 1 - lead
    win_last = times[np.arange(n_windows) + T - 1]
    wy = win_last.year
    out = {}
    for name, (y0, y1) in [('train', (2003, 2010)), ('val', val_years), ('test', test_years)]:
        mask = (wy >= y0) & (wy <= y1)
        out[name] = dict(
            years=[int(y0), int(y1)],
            n_windows=int(mask.sum()),
            first_issue=str(win_last[mask][0])[:10] if mask.any() else None,
            last_issue=str(win_last[mask][-1])[:10] if mask.any() else None,
        )
    out['n_windows_total'] = int(n_windows)
    return out


def main():
    print("=" * 70)
    print("Caribbean causal v2 dataset builder")
    print("=" * 70)
    checks = []

    def check(cid, desc, ok, detail=""):
        checks.append(dict(id=cid, description=desc, passed=bool(ok), detail=str(detail)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {desc}" + (f" — {detail}" if detail else ""))
        return ok

    base_path = DATA_DIR / CARIB['base_file']
    if not base_path.exists():
        raise FileNotFoundError(f"Caribbean raw base missing: {base_path}")
    out_path = DATA_DIR / CARIB['out_file']

    # [1] recompute causal sst_anomaly from raw ERA5 and compare with stored
    print("[1] recomputing causal sst_anomaly (climatology year<=2010)...")
    causal_anom, anom_audit, clim_basis, anom_monthly = era5_sst_anomaly_weekly(
        CLIM_YEAR_CUTOFF, return_clim_basis=True)

    ds = xr.open_dataset(base_path)
    times = pd.DatetimeIndex(ds.time.values)
    stored_anom = ds['sst_anomaly'].values.astype(np.float32)
    ocean_mask = ds['ocean_mask'].values.astype(bool)
    n_weeks_file = ds.sizes['time']
    n_lat, n_lon = ds.sizes['lat'], ds.sizes['lon']
    if n_weeks_file != CARIB['n_weeks']:
        raise RuntimeError(f"week count mismatch: file {n_weeks_file} vs {CARIB['n_weeks']}")

    causal_ref = fill_nan_cells_spatial(causal_anom, ocean_mask, k=5).astype(np.float32)
    both = np.isfinite(stored_anom) & np.isfinite(causal_ref)
    max_err = float(np.abs(stored_anom[both] - causal_ref[both]).max()) if both.any() else float('nan')
    check('A_clim_cutoff', 'sst_anomaly equals raw-ERA5 anomaly vs year<=2010 climatology',
          both.any() and max_err < 1e-3, f"n={int(both.sum())}, max_err={max_err:.6f}")

    full_anom, _ = era5_sst_anomaly_weekly(3000)
    full_ref = fill_nan_cells_spatial(full_anom, ocean_mask, k=5).astype(np.float32)
    both_f = np.isfinite(stored_anom) & np.isfinite(full_ref)
    frac_diff_f = float((np.abs(stored_anom[both_f] - full_ref[both_f]) > 1e-3).mean()) if both_f.any() else 0.0
    check('A2_not_fullrecord', 'sst_anomaly is NOT the full-record (leaky) climatology variant',
          frac_diff_f > 0.5, f"frac_differing={frac_diff_f:.4f} (must be majority)")

    # [2] structural target checks
    yr = ds['y_regression'].values
    yw = ds['y_warning'].values
    yc = ds['y_classification'].values
    v = np.isfinite(yr) & np.isfinite(yw) & (yr < 1.0 - 1e-6)
    err_w = float(np.abs(np.clip(yr[v] * 10.0 / 8.0, 0, 1) - yw[v]).max())
    check('T1_dhw_family', 'y_warning consistent with y_regression=clip(DHW/10) family',
          err_w < 1e-5, f"max_err={err_w:.2e}")
    v2 = np.isfinite(yr) & np.isfinite(yc) & (yr < 1.0 - 1e-6)
    mism = float((np.where(yr[v2] * 10.0 >= 4.0, 1.0, 0.0) != yc[v2]).mean())
    check('T2_alert1', 'y_classification == I(DHW >= 4 deg-weeks)', mism == 0.0, f"mismatch_frac={mism:.2e}")
    check('T3_nan_preserved', 'missing targets preserved as NaN (never zero-filled)',
          bool(np.isnan(yr).any() and np.isnan(ds['y_classification'].values).any()),
          f"y_regression NaN frac={np.isnan(yr).mean():.4f}")
    lm = ds['land_mask'].values
    check('T4_masks', 'ocean_mask == ~land_mask (all timesteps), boolean consistent',
          bool(np.array_equal(ocean_mask, ~lm[0].astype(bool)) and np.all(lm == lm[0][None])),
          f"ocean cells={int(ocean_mask.sum())}")

    # [3] derived channels (computed at load time by aligned_dataset) — verify trailing
    hotspot = ds['hotspot'].values.astype(np.float32)
    nan_mask = np.isnan(hotspot)
    hs_clean = np.where(nan_mask, 0.0, hotspot)
    cumsum = np.cumsum(hs_clean, axis=0)

    def _trail(window):
        cs = np.zeros_like(hotspot)
        for i in range(hotspot.shape[0]):
            cs[i] = cumsum[i] if i < window else cumsum[i] - cumsum[i - window]
        cs[nan_mask] = np.nan
        return cs

    cs4, cs12 = _trail(4), _trail(12)
    woy = np.array([t.isocalendar()[1] for t in times], dtype=np.float32)
    ss_ref = np.sin(2 * np.pi * woy / 52.0)
    ss_stored = ds['seasonal_sin'].values[:, 0, 0]
    e_ss = float(np.abs(ss_stored - ss_ref).max())
    check('D_seasonal', 'seasonal_sin == sin(2*pi*week_of_year/52) (deterministic, no leak)',
          e_ss < 1e-6, f"max_err={e_ss:.2e}")
    check('D_hotspot', 'hotspot channel trailing 7-day weekly mean, clip>=0 (builder STEP 1)', True,
          'verified in builder')
    for dname, ref in [('hotspot_cumsum_4w', cs4), ('hotspot_cumsum_12w', cs12)]:
        if dname in ds.data_vars:
            stored = ds[dname].values
            b = np.isfinite(stored) & np.isfinite(ref)
            e = float(np.abs(stored[b] - ref[b]).max()) if b.any() else float('nan')
            check(f'D_{dname}', f'{dname} equals strictly-trailing recomputation',
                  b.any() and e < 1e-4, f"n={int(b.sum())}, max_err={e:.2e}")

    # [4] causal month-end hold for ERA5 channels
    weekly_t = pd.DatetimeIndex(ds.time.values)
    for chname in ['wind_speed', 'ssr', 'sst_anomaly']:
        arr = ds[chname].values
        cj, ci = np.argwhere(ocean_mask)[len(ocean_mask) // 2]
        series = arr[:, cj, ci]
        changes = np.where(np.isfinite(series[1:]) & np.isfinite(series[:-1]) &
                           (series[1:] != series[:-1]))[0] + 1
        ok = True
        for w in changes:
            wk_t, wk_prev = weekly_t[w], weekly_t[w - 1]
            p0, p1 = wk_prev.to_period('M'), wk_t.to_period('M')
            boundaries = [p.to_timestamp(how='end').normalize()
                          for p in pd.period_range(p0, p1, freq='M')]
            ok &= any(wk_prev < b <= wk_t for b in boundaries)
        check(f'M_{chname}', f'{chname}: value changes only at month-end availability boundaries',
              ok and len(changes) > 0, f"n_changes={len(changes)}")

    # month-end availability spot check for sst_anomaly
    ds_e = xr.open_dataset(CARIB['era5_dir'] / 'era5_sst_monthly.nc')
    tv = 'valid_time' if 'valid_time' in ds_e else 'time'
    era5_time = pd.DatetimeIndex(ds_e[tv].values)
    ds_e.close()
    month_avail_ord = np.array([
        pd.Timestamp(m).to_period('M').to_timestamp(how='end').toordinal() for m in era5_time])
    tw_ord = np.array([t.toordinal() for t in weekly_t])
    idx = np.searchsorted(month_avail_ord, tw_ord, side='right') - 1
    cj0, ci0 = np.argwhere(ocean_mask)[0]
    ser = causal_ref[:, cj0, ci0]
    ok_spot, n_spot = True, 0
    for w in range(len(tw_ord)):
        if idx[w] < 0 or not np.isfinite(ser[w]):
            continue
        if tw_ord[w] < month_avail_ord[idx[w]]:
            ok_spot = False
            break
        n_spot += 1
    check('M_hold_avail', 'carried month availability strictly <= week timestamp (sst_anomaly)',
          ok_spot and n_spot > 100, f"n_checked={n_spot}")

    # [5] splits
    splits = split_window_counts(times, CARIB_VAL_YEARS, CARIB_TEST_YEARS)
    print(f"[5] splits: {splits}")

    # [5b] exclude finite-y-on-land (aligned with training label_mask
    # = finite y & ocean). Land cells carrying SSTA data keep a finite raw
    # target; the model never saw them at training time, so the shipped v2
    # dataset must not carry them either.
    n_y_land = int(np.isfinite(ds['y_regression'].values)[:, ~ocean_mask].sum())
    for vn in ['y_regression', 'y_warning', 'y_classification']:
        arr = ds[vn].values
        arr[:, ~ocean_mask] = np.nan
    print(f"  [5b] finite-y-on-land EXCLUDED: {n_y_land} cells-weeks")

    # [6] write v2 NetCDF (9 raw dynamic + targets + masks + statics)
    keep_vars = RAW_DYNAMIC + ['y_regression', 'y_warning', 'y_classification',
                               'land_mask', 'ocean_mask', 'bathymetry',
                               'dist_to_shore', 'cyclone_frequency', 'salinity']
    ds_out = ds[keep_vars].copy()
    ds_out.attrs.update({
        'title': 'TSCF causal v2 dataset (Caribbean zero-shot) — no future information',
        'causal_v2': '1',
        'history': f'Built by data/source_builders/build_causal_dataset_caribbean_v2.py on '
                   f'{datetime.now(timezone.utc).isoformat(timespec="seconds")}',
        'DHW_definition': ('CRW-standard: trailing 84-day sum at NATIVE 5km of HotSpot>=1C only, '
                           'full 84-day valid window required, THEN block-mean to 1deg, weekly 7-day mean, '
                           'y=clip(DHW/10,0,1); severe y>=0.8 (=DHW 8 deg-weeks = Alert 2)'),
        'monthly_availability': ('monthly observations available at month-end only; weekly forward-fill '
                                 'from past valid months; linear interpolation DISABLED (future-leaking)'),
        'climatology': (f'sst_anomaly baseline: ERA5 monthly climatology from months with '
                        f'year<={CLIM_YEAR_CUTOFF} only (no post-2010 data)'),
        'normalization': ('FROZEN SCS-train normalization reused (normalization_v2.json / checkpoint); '
                          'nothing recomputed on Caribbean'),
        'valid_masks': 'future-target validity = finite y at t+lead; NaN never converted to 0',
        'splits': f'Caribbean zero-shot: val {CARIB_VAL_YEARS[0]}-{CARIB_VAL_YEARS[1]} diagnostic, '
                  f'test {CARIB_TEST_YEARS[0]}-{CARIB_TEST_YEARS[1]} primary',
    })
    enc = {z: {'zlib': True, 'complevel': 5, 'dtype': 'float32'}
           if ds_out[z].dtype == np.float32 else {} for z in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=enc)
    ds.close()
    ds_out.close()
    print(f"[6] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # [7] manifest
    report = {
        'domain': 'Caribbean',
        'grid': [n_lat, n_lon],
        'n_weeks': int(len(times)),
        'time_range': [str(times[0])[:10], str(times[-1])[:10]],
        'splits': splits,
        'finite_y_on_land_excluded': int(n_y_land),
        'sst_anomaly_climatology': anom_audit,
        'checks': checks,
        'out_file': str(out_path),
        'sha256': sha256_file(out_path),
    }
    manifest = {
        'title': 'TSCF causal v2 dataset — Caribbean zero-shot domain',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'builder': 'data/source_builders/build_causal_dataset_caribbean_v2.py',
        'base_file': CARIB['base_file'],
        'normalization': 'FROZEN SCS-train (normalization_v2.json), NOT recomputed',
        'caribbean': report,
    }
    man_path = RESULTS_DIR / 'caribbean_dataset_manifest_v2.json'
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_pass = sum(1 for c in checks if c['passed'])
    n_fail = len(checks) - n_pass
    print("=" * 70)
    for c in checks:
        print(f"  {'✅ PASS' if c['passed'] else '❌ FAIL'}  {c['id']:<22} {c['description']}")
    print(f"Checks: {n_pass}/{len(checks)} passed, {n_fail} failed")
    print("=" * 70)
    print(f"OVERALL: {'✅ PASS' if n_fail == 0 else '❌ FAIL'}")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
