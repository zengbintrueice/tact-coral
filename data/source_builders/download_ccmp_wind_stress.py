"""
REMSS CCMP v3.1 Wind Stress Download Script
==========================================
Downloads CCMP daily wind data (no auth), subsets to SCS,
computes wind stress via bulk formula, saves monthly means.
Uses 4-thread parallel download.

Run: python src/data/download_ccmp_wind_stress.py
"""

import os, sys, time, shutil, ssl, urllib.request, urllib.error
import tempfile, calendar, re, threading
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'backups', 'piecf_a6_nophys_20260718', 'a6_xgb_experiment', 'data', 'new_sources', 'ccmp_wind')
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = date(1993, 1, 1)
END_DATE   = date(2020, 12, 31)

LAT_LO, LAT_HI = 3.0, 25.0
LON_LO, LON_HI = 99.0, 124.0

BASE_URL = 'https://data.remss.com/ccmp/v03.1'
RHO_AIR = 1.225
DRAG_COEFF = 0.0013

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python urllib'}


def make_url(d):
    return '%s/Y%04d/M%02d/CCMP_Wind_Analysis_%04d%02d%02d_V03.1_L4.nc' % (
        BASE_URL, d.year, d.month, d.year, d.month, d.day)


def download_to_bytes(target_date):
    url = make_url(target_date)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
            data = resp.read()
        if len(data) < 1000000:
            return None, 'too_small'
        return data, None
    except urllib.error.HTTPError as e:
        return None, 'HTTP_%d' % e.code
    except Exception as e:
        return None, '%s_%s' % (type(e).__name__, str(e)[:30])


def process_bytes(data_bytes, tmpdir):
    import numpy as np
    import xarray as xr
    tmp_nc = os.path.join(tmpdir, 'in.nc')
    with open(tmp_nc, 'wb') as f:
        f.write(data_bytes)
    ds = xr.open_dataset(tmp_nc, chunks={'time': 1})
    lat, lon = ds['latitude'].values, ds['longitude'].values
    li = (lat >= LAT_LO) & (lat <= LAT_HI)
    lo = (lon >= LON_LO) & (lon <= LON_HI)
    u = np.where(li, np.where(lo, ds['uwnd'].values, np.nan), np.nan)
    # u shape: (time, lat, lon)
    u_sub = ds['uwnd'].where(xr.DataArray(li, dims='latitude')).where(
        xr.DataArray(lo, dims='longitude'))
    v_sub = ds['vwnd'].where(xr.DataArray(li, dims='latitude')).where(
        xr.DataArray(lo, dims='longitude'))
    ws = np.sqrt(u_sub.values**2 + v_sub.values**2)
    stress = RHO_AIR * DRAG_COEFF * ws**2  # (time, lat_sub, lon_sub)
    result = xr.Dataset({
        'wind_stress': (['time', 'latitude', 'longitude'], stress),
        'wind_speed':  (['time', 'latitude', 'longitude'], ws),
    }, coords={
        'time': ds.time.values,
        'latitude': lat[li],
        'longitude': lon[lo],
    })
    ds.close()
    return result


def save_monthly(days_list, year, month, lats, lons):
    import numpy as np
    import xarray as xr
    stacked = np.stack(days_list, axis=0)
    monthly_mean = np.nanmean(stacked, axis=0)
    out_path = os.path.join(DATA_DIR, 'ccmp_wind_stress_%04d%02d.nc' % (year, month))
    ds = xr.Dataset({
        'wind_stress': (['latitude', 'longitude'], monthly_mean),
    }, coords={'latitude': lats, 'longitude': lons})
    ds['wind_stress'].attrs = {
        'long_name': 'Wind stress magnitude (monthly mean)',
        'units': 'Pa',
        'source': 'CCMP v3.1 (REMSS), rho_air=%.3f, Cd=%.4f' % (RHO_AIR, DRAG_COEFF),
        'days_used': len(days_list),
    }
    ds.to_netcdf(out_path)
    ds.close()
    return out_path


N_WORKERS = 4
_progress_lock = threading.Lock()
_progress = {'ok': 0, 'fail': 0, 'bytes': 0}


def download_one_day(d):
    """Download + process one day, returns (date, stress_2d_or_None, lats, lons)."""
    import numpy as np
    data_bytes, err = download_to_bytes(d)
    if not data_bytes:
        return d, None, None, None
    tmpdir = tempfile.mkdtemp()
    try:
        ds_day = process_bytes(data_bytes, tmpdir)
        daily_stress = np.mean(ds_day['wind_stress'].values, axis=0)
        return d, daily_stress, ds_day.latitude.values, ds_day.longitude.values
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def process_month(year, month, existing_set):
    """Download one month in parallel, save monthly composite. Skips if exists."""
    ym = '%04d%02d' % (year, month)
    with _progress_lock:
        skip = ym in existing_set
    if skip:
        return ym, 'skip', 0

    n_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, n_days + 1)]

    acc = []
    lats = lons = None
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(download_one_day, d): d for d in days}
        for future in as_completed(futures):
            try:
                d, stress, la, lo = future.result()
                if stress is not None:
                    acc.append(stress)
                    if lats is None:
                        lats, lons = la, lo
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1

    if acc and lats is not None:
        save_monthly(acc, year, month, lats, lons)
        return ym, 'saved', ok
    return ym, 'no_data', ok


def main():
    import numpy as np
    print('=' * 60, flush=True)
    print('REMSS CCMP v3.1 Wind Stress Download (parallel %d threads)' % N_WORKERS, flush=True)
    print('  Time: %s ~ %s' % (START_DATE, END_DATE), flush=True)
    print('  Spatial: %.1f-%.1fN, %.1f-%.1fE' % (LAT_LO, LAT_HI, LON_LO, LON_HI), flush=True)
    print('  Save: %s' % DATA_DIR, flush=True)
    print('=' * 60, flush=True)

    existing = set()
    for f in os.listdir(DATA_DIR):
        m = re.match(r'ccmp_wind_stress_(\d{6})\.nc$', f)
        if m:
            existing.add(m.group(1))
    print('Existing monthly composites: %d' % len(existing), flush=True)

    # Build month list
    months = []
    y, m = START_DATE.year, START_DATE.month
    while True:
        if date(y, m, 1) > END_DATE:
            break
        months.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1

    total_months = len(months)
    print('Months to process: %d' % total_months, flush=True)

    start_time = time.time()
    saved = skipped = no_data = 0

    for i, (y, m) in enumerate(months, 1):
        ym, status, n_ok = process_month(y, m, existing)
        if status == 'saved':
            saved += 1
        elif status == 'skip':
            skipped += 1
        else:
            no_data += 1

        if i % 10 == 0 or i == total_months:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (total_months - i) / max(rate, 0.001)
            print('  [%d/%d] saved:%d skip:%d nodata:%d | %.2f month/s ETA:%.1fh' % (
                i, total_months, saved, skipped, no_data, rate * 86400, eta / 3600), flush=True)

    print(flush=True)
    print('=' * 60, flush=True)
    print('Done in %.1f hours' % ((time.time() - start_time) / 3600), flush=True)
    print('  Saved: %d | Skipped: %d | No-data: %d' % (saved, skipped, no_data), flush=True)
    print('=' * 60, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
