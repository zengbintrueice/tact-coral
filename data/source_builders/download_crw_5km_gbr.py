"""
Download NOAA CRW 5km daily SST for Great Barrier Reef (GBR) (1985-2020)
多线程并行下载: 6 个变量各自独立线程, 线程内按月份顺序下载, 跨线程并发可降低平均延迟。
跳过已下载文件以支持续传, 无限重试失败任务。

Based on download_crw_5km_all.py with GBR-specific bbox and upwell server.

Output naming:
    crw_sst_YYYYMM.nc, crw_sstanomaly_YYYYMM.nc,
    crw_dhw_YYYYMM.nc, crw_baa_YYYYMM.nc, crw_baa_7d_max_YYYYMM.nc,
    crw_hotspot_YYYYMM.nc
"""
import os
import sys
import ssl
import time
import urllib.request
import http.client
import socket
import threading
from datetime import datetime
from urllib.error import HTTPError, URLError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'backups', 'piecf_a6_nophys_20260718', 'a6_xgb_experiment', 'data', 'new_sources', 'crw_5km_gbr')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GBR 区域: lat 9-24°S + 缓冲, lon 142-154°E
BASE_URL = 'https://upwell.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.nc'

VARS = ['CRW_SST', 'CRW_SSTANOMALY', 'CRW_HOTSPOT', 'CRW_DHW', 'CRW_BAA', 'CRW_BAA_7D_MAX']

LAT_RANGE = '(-24):(-9)'
LON_RANGE = '(142):(154)'

MIN_FILE_SIZE = 50_000
REQUEST_TIMEOUT = 240
RETRY_WAIT = 3
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36')

_lock = threading.Lock()
_done = 0
_ok = 0
_skip = 0
_out_of_range = 0
_fail = 0
_total = 0
_start_time = 0

# SSL context for upwell.pfeg.noaa.gov (bypasses certificate verification)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _log(msg):
    with _lock:
        print(msg, flush=True)


def build_url(year, month, var):
    t0 = datetime(year, month, 1, 12, 0, 0)
    t_end = datetime(year + 1, 1, 1, 12, 0, 0) if month == 12 else datetime(year, month + 1, 1, 12, 0, 0)
    t0s = t0.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1s = t_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    return (f'{BASE_URL}?{var}%5B({t0s}):1:({t1s})%5D'
            f'%5B{LAT_RANGE}%5D%5B{LON_RANGE}%5D')


def download_one(year, month, var):
    """Download single (year, month, var).  One-shot; caller handles retry."""
    prefix = var.lower()
    outfile = os.path.join(OUTPUT_DIR, f'{prefix}_{year}{month:02d}.nc')

    if os.path.exists(outfile) and os.path.getsize(outfile) > MIN_FILE_SIZE:
        return 'SKIP'

    url = build_url(year, month, var)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=_SSL_CTX)
        data = resp.read()
        if len(data) < 1000:
            if os.path.exists(outfile):
                os.remove(outfile)
            return 'FAIL: too_small'
        with open(outfile, 'wb') as f:
            f.write(data)
        return f'OK {len(data)/1024/1024:.1f}MB'
    except HTTPError as e:
        if e.code == 404:
            return 'OUT_OF_RANGE'
        return f'FAIL: HTTP {e.code}'
    except (URLError, TimeoutError, OSError,
            http.client.IncompleteRead, http.client.HTTPException,
            ConnectionResetError, socket.timeout, EOFError) as e:
        if os.path.exists(outfile):
            try:
                os.remove(outfile)
            except OSError:
                pass
        return f'FAIL: {type(e).__name__}'


def _pipe_variable(var):
    """按年份-月份顺序下载某个变量的所有数据, 某月失败即重试不前进。"""
    global _ok, _skip, _out_of_range, _fail, _done
    for year in range(1985, 2021):
        for month in range(1, 13):
            while True:
                status = download_one(year, month, var)

                if status == 'SKIP':
                    with _lock:
                        _skip += 1
                        _done += 1
                    break

                elif status.startswith('OK'):
                    with _lock:
                        _ok += 1
                        _done += 1
                        elapsed = time.time() - _start_time
                        rate = _ok / (elapsed / 60) if elapsed > 0 else 0
                        remaining = _total - _ok - _skip - _out_of_range
                        eta = remaining / rate if rate > 0 else 0
                    _log(f'[{_ok}/{_total-_skip}] {var} {year}-{month:02d} {status}'
                          f'  {rate:.2f} ok/min  ETA {eta:.0f}min  {elapsed/60:.0f}min')
                    break

                elif status == 'OUT_OF_RANGE':
                    with _lock:
                        _out_of_range += 1
                        _done += 1
                    break

                else:
                    with _lock:
                        _fail += 1
                        n = _fail
                    _log(f'  RETRY {var} {year}-{month:02d}: {status} (#{n})')
                    time.sleep(RETRY_WAIT)


def main():
    global _total, _start_time
    _total = len(VARS) * 36 * 12

    print("=" * 60)
    print("CRW 5km Parallel Downloader — GBR  (1 thread per variable, 6 vars concurrently)")
    print("=" * 60)
    print(f'   Server    : upwell.pfeg.noaa.gov')
    print(f'   Variables : {VARS}')
    print(f'   Period    : 1985-01 ~ 2020-12  (monthly)')
    print(f'   Region    : lat -24:-9  lon 142:154  (GBR)')
    print(f'   Output    : {OUTPUT_DIR}')
    print(f'   Total jobs: {_total}')
    print()
    _start_time = time.time()

    threads = []
    for v in VARS:
        t = threading.Thread(target=_pipe_variable, args=(v,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - _start_time
    print()
    print("=" * 60)
    print("GBR 下载完成!")
    print("=" * 60)
    print(f'   OK           : {_ok}')
    print(f'   SKIP (cached): {_skip}')
    print(f'   OUT_OF_RANGE : {_out_of_range}')
    print(f'   Time         : {elapsed/3600:.2f} h')
    print(f'   Avg rate     : {_ok/(elapsed/60):.2f} success/min')
    print(f'   Files on disk: {len(os.listdir(OUTPUT_DIR))} in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
