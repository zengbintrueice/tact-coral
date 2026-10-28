"""
CoastWatch ERDDAP MODIS Aqua PAR Download
===========================================
Download MODIS Aqua daily PAR from NOAA CoastWatch ERDDAP - no auth needed.
Product: erdMH1par01day (MODIS Aqua PAR, 4km daily)
Time range: 2003-01 ~ 2020-12
Spatial: South China Sea with buffer (3-25N, 99-124E)
Strategy: daily -> monthly mean -> monthly NetCDF.
Run:
    python src/data/download_coastwatch_par.py
"""

import os, sys, time, io, ssl, urllib.request, urllib.error, re
from datetime import date, timedelta
from collections import defaultdict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "backups", "piecf_a6_nophys_20260718", "a6_xgb_experiment", "data", "new_sources", "coastwatch_par")
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = date(2003, 1, 1)
END_DATE = date(2020, 12, 31)
LAT_RANGE = "[(3):(25)]"
LON_RANGE = "[(99):(124)]"
BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1par01day.nc"

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python urllib"}


def make_url(target_date):
    ds = target_date.strftime("%Y-%m-%dT12:00:00Z")
    return BASE_URL + "?par[(" + ds + ")]" + LAT_RANGE + LON_RANGE


def download_day(target_date):
    url = make_url(target_date)
    fname = target_date.strftime("modis_par_%Y%m%d.nc")
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 100000:
        return True, fpath, None
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
            data = resp.read()
        if len(data) < 50000:
            return False, None, "Too small (" + str(len(data)) + " bytes)"
        if data[:4] != bytes([0x89, 72, 68, 70]):
            if data[:3] != b"CDF":
                return False, None, "Not NetCDF magic"
        with open(fpath, "wb") as f:
            f.write(data)
        return True, fpath, None
    except urllib.error.HTTPError as e:
        return False, None, "HTTP " + str(e.code)
    except urllib.error.URLError as e:
        return False, None, "URLError: " + str(e.reason)
    except Exception as e:
        return False, None, type(e).__name__ + ": " + str(e)


def aggregate_month(year, month):
    import numpy as np
    import xarray as xr
    prefix = "modis_par_" + str(year) + str(month).zfill(2)
    pat = prefix + r"\d{8}\.nc$"
    daily_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(prefix) and re.match(pat, f)
    ])
    if not daily_files:
        return None, "no daily files"
    all_data = []
    lats = lons = None
    for fname in daily_files:
        fpath = os.path.join(DATA_DIR, fname)
        try:
            ds = xr.open_dataset(fpath)
            if "par" not in ds.variables:
                ds.close()
                continue
            par = ds["par"].values
            if lats is None:
                lats = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
                lons = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
            all_data.append(par)
            ds.close()
        except Exception:
            continue
    if not all_data:
        return None, "no valid daily data"
    stacked = np.stack(all_data, axis=0)
    stacked = np.where(stacked > 9e36, np.nan, stacked)
    monthly_mean = np.nanmean(stacked, axis=0)
    out_path = os.path.join(DATA_DIR, prefix + ".nc")
    ds_out = xr.Dataset(
        {"par": (["latitude", "longitude"], monthly_mean)},
        coords={"latitude": lats, "longitude": lons}
    )
    ds_out["par"].attrs = {
        "long_name": "Photosynthetically Active Radiation (monthly mean)",
        "units": "einstein m^-2 day^-1",
        "source": "MODIS Aqua via CoastWatch ERDDAP (erdMH1par01day)",
        "days_used": len(all_data),
    }
    ds_out.to_netcdf(out_path)
    ds_out.close()
    return out_path, None


def main():
    print("=" * 60)
    print("CoastWatch ERDDAP MODIS Aqua PAR Download")
    print("  Time:", START_DATE, "~", END_DATE)
    print("  Spatial: LAT", LAT_RANGE + ", LON", LON_RANGE)
    print("  Save:", DATA_DIR)
    print("=" * 60)

    d = START_DATE
    all_dates = []
    while d <= END_DATE:
        all_dates.append(d)
        d += timedelta(days=1)
    print("Total days in range:", len(all_dates))

    existing_monthly = set()
    for f in os.listdir(DATA_DIR):
        m = re.match(r"modis_par_(\d{6})\.nc$", f)
        if m:
            existing_monthly.add(m.group(1))
    print("Existing monthly composites:", len(existing_monthly))

    days_needed = defaultdict(list)
    for d in all_dates:
        ym = d.strftime("%Y%m")
        if ym not in existing_monthly:
            days_needed[ym].append(d)

    months_to_download = sorted(days_needed.keys())
    days_to_download = sum(len(v) for v in days_needed.values())
    print("Days to download:", days_to_download, "across", len(months_to_download), "months")
    if days_to_download == 0:
        print("All months already aggregated. Done.")
        return 0

    ok = skip = fail = 0
    start_time = time.time()
    flat_dates = [d for ym in months_to_download for d in days_needed[ym]]
    for i, d in enumerate(flat_dates, 1):
        success, fpath, err = download_day(d)
        if success:
            if fpath is not None:
                skip += 1
            else:
                ok += 1
        else:
            fail += 1
        if i % 100 == 0 or i == days_to_download:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (days_to_download - i) / rate if rate > 0 else 0
            print("  [%d/%d] ok:%d skip:%d fail:%d | %.1f/s ETA:%dmin" % (i, days_to_download, ok, skip, fail, rate, eta/60))
        if success:
            time.sleep(0.05)

    print()
    print("Download phase done. ok:%d skip:%d fail:%d" % (ok, skip, fail))
    print("=" * 60)
    print()
    print("Aggregating daily -> monthly composites...")
    daily_by_month = defaultdict(list)
    for f in os.listdir(DATA_DIR):
        m = re.match(r"modis_par_(\d{6})\d{2}\.nc$", f)
        if m:
            daily_by_month[m.group(1)].append(f)

    m_ok = m_fail = m_skip = 0
    for ym in sorted(daily_by_month.keys()):
        y, m_num = int(ym[:4]), int(ym[4:6])
        out_name = "modis_par_" + ym + ".nc"
        out_path = os.path.join(DATA_DIR, out_name)
        if os.path.exists(out_path):
            m_skip += 1
            continue
        result, err = aggregate_month(y, m_num)
        if result:
            m_ok += 1
            if m_ok % 20 == 0:
                print("  monthly composites:", m_ok)
        else:
            m_fail += 1
            print("  FAIL", ym + ":", err)

    print()
    print("Monthly composites: %d new, %d existing, %d failed" % (m_ok, m_skip, m_fail))
    print("Total monthly files:", m_ok + m_skip)
    total_time = (time.time() - start_time) / 60
    print("Total time: %.1f minutes" % total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
