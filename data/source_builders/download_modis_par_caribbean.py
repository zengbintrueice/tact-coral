#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODIS Aqua PAR Download — CARIBBEAN (CoastWatch ERDDAP, monthly means)
=======================================================================
Based on download_coastwatch_par_gbr.py (成功经验) with Caribbean bbox.

Product: erdMH1par01day (MODIS Aqua PAR, 4km daily)
Time range: 2003-01 ~ 2020-12 (216 months, mirrors SCS modis_par_YYYYMM files)
Output   : D:\piecf_dns\coastwatch_par_caribbean\modis_par_YYYYMM.nc
           (monthly MEAN over daily values, >9e36 fill → NaN, no time dim —
           same schema as the SCS files the v9 builder consumes)

优化: 与 GBR 脚本逐日请求不同, 本脚本按月发出单个 ERDDAP 时间区间请求
(一月一次, ~216 次), 端侧做日值 nanmean — 输出与 GBR 逐日聚合结果一致,
请求数减少 ~97%。

⚠️ 经度约定自动探测 (与 download_crw_5km_caribbean.py 相同设计):
   CoastWatch ERDDAP 全球 4km 产品经度可能为 -180..180 或 0..360。
   启动时用加勒比海中心 (15N, 75W) 单日 5x5 小方块 preflight 自动选择:
     -180..180 : lon (-92):(-58)
     0..360    : lon (268):(302)
   两种探测都失败则退出, 绝不在约定未知时全量下载。

Run:
    & piecf_env\Scripts\python.exe download_modis_par_caribbean.py
"""
import os
import io
import ssl
import sys
import json
import time
import calendar
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta

import numpy as np
import xarray as xr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = os.environ.get('PAR_OUT_DIR', os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"\coastwatch_par_caribbean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = date(2003, 1, 1)
END_DATE = date(2020, 12, 31)

LAT_RANGE = "[(3):(27)]"          # Caribbean 4:26 + 1° buffer
LON_CANDIDATES = {
    '-180_180': "[(-92):(-58)]",
    '0_360': "[(268):(302)]",
}
BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1par01day.nc"

# preflight: 加勒比海中心单日 5x5
PROBE_LAT = "[(14.9):(15.1)]"
PROBE_LON = {'-180_180': "[(-75.1):(-74.9)]", '0_360': "[(284.9):(285.1)]"}
PROBE_DAY = date(2015, 6, 15)

MIN_FILE_SIZE = 20_000
REQUEST_TIMEOUT = 240
RETRY_WAIT = 5

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python urllib"}


def _fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=REQUEST_TIMEOUT) as resp:
        data = resp.read()
    if len(data) < 500:
        raise RuntimeError(f'response too small ({len(data)} bytes)')
    if data[:4] != bytes([0x89, 72, 68, 70]) and data[:3] != b"CDF":
        raise RuntimeError("Not NetCDF magic")
    return data


def detect_lon_convention():
    """Preflight: 单日 5x5 探测确定经度约定。返回 (convention, lon_range)。"""
    ds_s = PROBE_DAY.strftime("%Y-%m-%dT12:00:00Z")
    for conv, lon_probe in PROBE_LON.items():
        url = BASE_URL + "?par[(" + ds_s + ")]" + PROBE_LAT + lon_probe
        try:
            data = _fetch(url)
            ds = xr.open_dataset(io.BytesIO(data))
            lon_c = [c for c in ds.coords if c in ('longitude', 'lon')][0]
            lon_vals = ds[lon_c].values
            par = ds['par'].values if 'par' in ds else None
            ds.close()
            finite = float(np.nanmean(par)) if par is not None and np.isfinite(par).any() \
                else float('nan')
            expect = {'-180_180': (-92.0, -58.0), '0_360': (268.0, 302.0)}[conv]
            ok = (lon_vals.min() >= expect[0] - 5) and (lon_vals.max() <= expect[1] + 5)
            print(f'  probe {conv}: lon returned '
                  f'[{float(lon_vals.min()):.2f}, {float(lon_vals.max()):.2f}] '
                  f'par_mean={finite:.2f} range_ok={ok}')
            if ok:
                return conv, LON_CANDIDATES[conv]
        except Exception as e:
            print(f'  probe {conv}: FAILED {type(e).__name__}: {e}')
    return None, None


def month_iter():
    d = START_DATE
    while d <= END_DATE:
        yield d.year, d.month
        d = date(d.year + (d.month == 12), 1 if d.month == 12 else d.month + 1, 1)


def download_month(year, month, lon_range):
    """One ERDDAP request for the whole month's daily values; nanmean → monthly file."""
    prefix = f"modis_par_{year}{month:02d}"
    out_path = os.path.join(OUTPUT_DIR, prefix + ".nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > MIN_FILE_SIZE:
        return 'SKIP'

    last_day = calendar.monthrange(year, month)[1]
    t0 = datetime(year, month, 1, 12).strftime("%Y-%m-%dT12:00:00Z")
    t1 = datetime(year, month, last_day, 12).strftime("%Y-%m-%dT12:00:00Z")
    url = BASE_URL + f"?par[({t0}):1:({t1})]" + LAT_RANGE + lon_range

    data = _fetch(url)
    ds = xr.open_dataset(io.BytesIO(data))
    if 'par' not in ds:
        ds.close()
        return 'FAIL: no par variable'
    par = ds['par'].values
    lat_c = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_c = 'longitude' if 'longitude' in ds.coords else 'lon'
    lats, lons = ds[lat_c].values, ds[lon_c].values
    ds.close()

    par = par.astype(np.float64)
    par = np.where(par > 9e36, np.nan, par)
    if not np.isfinite(par).any():
        return 'FAIL: all fill/NaN'
    monthly_mean = np.nanmean(par, axis=0)

    ds_out = xr.Dataset(
        {"par": ([lat_c, lon_c], monthly_mean.astype(np.float32))},
        coords={lat_c: lats, lon_c: lons})
    ds_out["par"].attrs = {
        "long_name": "Photosynthetically Active Radiation (monthly mean, Caribbean)",
        "units": "einstein m^-2 day^-1",
        "source": "MODIS Aqua via CoastWatch ERDDAP (erdMH1par01day)",
    }
    ds_out.to_netcdf(out_path + ".tmp.nc")
    os.replace(out_path + ".tmp.nc", out_path)
    return f'OK {os.path.getsize(out_path)/1024:.0f}KB'


def main():
    print("=" * 60)
    print("MODIS Aqua PAR Downloader — CARIBBEAN (monthly means)")
    print("=" * 60)
    print(f'   Server : coastwatch.pfeg.noaa.gov (erdMH1par01day)')
    print(f'   Period : 2003-01 ~ 2020-12 (monthly)')
    print(f'   Region : lat 3:27  lon -92:-58 (Caribbean)')
    print(f'   Output : {OUTPUT_DIR}')
    print()
    print('[preflight] 经度约定探测 (15N, 75W 单日 5x5)...')
    conv, lon_range = detect_lon_convention()
    if conv is None:
        print('\n❌ 经度约定探测失败 — 未开始全量下载。检查网络后重试。')
        sys.exit(2)
    print(f'[preflight] 使用约定 {conv}: lon {lon_range}')
    with open(os.path.join(OUTPUT_DIR, 'download_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'region': 'Caribbean', 'lat_range': LAT_RANGE,
                   'lon_range': lon_range, 'lon_convention': conv,
                   'server': BASE_URL, 'period': '2003-2020',
                   'started': datetime.now().isoformat()}, f, indent=2)

    months = list(month_iter())
    n_total = len(months)
    n_ok = n_skip = n_fail = 0
    fail_log = []
    t0 = time.time()
    for i, (year, month) in enumerate(months, 1):
        for attempt in range(20):
            try:
                status = download_month(year, month, lon_range)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    OSError, RuntimeError, Exception) as e:
                status = f'FAIL: {type(e).__name__}: {str(e)[:80]}'
                time.sleep(RETRY_WAIT)
        if status == 'SKIP':
            n_skip += 1
        elif status.startswith('OK'):
            n_ok += 1
            if n_ok % 12 == 0 or i == n_total:
                rate = (n_ok) / ((time.time() - t0) / 60) if time.time() > t0 else 0
                eta = (n_total - n_ok - n_skip) / rate if rate > 0 else 0
                print(f'  [{i}/{n_total}] {year}-{month:02d} {status}  '
                      f'{rate:.1f}/min  ETA {eta:.0f}min', flush=True)
        else:
            n_fail += 1
            fail_log.append(f'{year}-{month:02d}: {status}')
            print(f'  GIVEUP {year}-{month:02d}: {status}', flush=True)

    print()
    print("=" * 60)
    print(f"CARIBBEAN PAR 下载完成: OK={n_ok} SKIP={n_skip} GIVEUP={n_fail}")
    print(f"Files on disk: {len(os.listdir(OUTPUT_DIR))} in {OUTPUT_DIR}")
    if fail_log:
        print("Failed months (re-run to retry):")
        for line in fail_log:
            print(f"  {line}")
    print("=" * 60)


if __name__ == '__main__':
    main()
