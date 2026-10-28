#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download NOAA CRW 5km daily SST for the CARIBBEAN (1985-2020)
多线程并行下载: 6 个变量各自独立线程, 线程内按月份顺序下载。
跳过已下载文件以支持续传, 无限重试失败任务。

Based on download_crw_5km_gbr.py with Caribbean bbox.

⚠️ 经度约定自动探测: upwell ERDDAP 的 NOAA_DHW 数据集经度可能为
   -180..180 或 0..360。启动时先用加勒比海中心 (15N, 75W) 一个
   5x5 小方块做 preflight 请求, 解析返回 NetCDF 的实际经度坐标与
   数据有效性, 自动选择正确约定后再开始全量下载。两种约定分别:
     -180..180 : lon (-91):(-59)
     0..360    : lon (269):(301)
   若两种 preflight 都失败(服务器不可达/数据缺失), 脚本退出并给出
   明确提示 — 绝不在约定未知的情形下开始全量下载。

Output naming (same as GBR/SCS):
    crw_sst_YYYYMM.nc, crw_sstanomaly_YYYYMM.nc,
    crw_dhw_YYYYMM.nc, crw_baa_YYYYMM.nc, crw_baa_7d_max_YYYYMM.nc,
    crw_hotspot_YYYYMM.nc
"""
import os
import sys
import ssl
import io
import json
import time
import urllib.request
import http.client
import socket
import threading
from datetime import datetime
from urllib.error import HTTPError, URLError

import numpy as np

# 强制 UTF-8 输出 (Windows 控制台默认 GBK, 打印 emoji 会 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = os.environ.get('CRW_OUT_DIR', os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"\crw_5km_caribbean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Caribbean 区域: lat 5-25°N + 缓冲 (4-26), lon 91-59°W + 缓冲 ──
BASE_URL = 'https://upwell.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.nc'

VARS = ['CRW_SST', 'CRW_SSTANOMALY', 'CRW_HOTSPOT', 'CRW_DHW', 'CRW_BAA', 'CRW_BAA_7D_MAX']

LAT_RANGE = '(4):(26)'
LON_CANDIDATES = {
    '-180_180': '(-91):(-59)',
    '0_360': '(269):(301)',
}
# preflight 探测点: 加勒比海中心海面点 (非陆地)
PROBE_LAT = '(14.9):(15.1)'
PROBE_LON = {'-180_180': '(-75.1):(-74.9)', '0_360': '(284.9):(285.1)'}
PROBE_TIME = '(2015-06-15T12:00:00Z)'

MIN_FILE_SIZE = 50_000
REQUEST_TIMEOUT = 240
RETRY_WAIT_BASE = 10      # 指数退避起点(秒): 服务器繁忙时不要太激进
RETRY_WAIT_MAX = 300      # 退避封顶 5 分钟
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


def _fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=_SSL_CTX)
    data = resp.read()
    if len(data) < 500:
        raise RuntimeError(f'response too small ({len(data)} bytes)')
    return data


def detect_lon_convention():
    """Preflight: 用加勒比海中心小方块确定经度约定。
    返回 (convention, lon_range)。408/URLError 等瞬时故障带重试+退避
    (与下载循环一致), 避免服务器繁忙时误判为约定未知。"""
    import xarray as xr
    for conv, lon_probe in PROBE_LON.items():
        url = (f'{BASE_URL}?CRW_SST%5B{PROBE_TIME}%5D'
               f'%5B{PROBE_LAT}%5D%5B{lon_probe}%5D')
        for attempt in range(6):
            try:
                data = _fetch(url)
                break
            except (HTTPError, URLError, TimeoutError, OSError) as e:
                code = getattr(e, 'code', type(e).__name__)
                wait = min(RETRY_WAIT_BASE * 2 ** attempt, RETRY_WAIT_MAX)
                print(f'  probe {conv}: {code} (attempt {attempt + 1}, '
                      f'wait {wait:.0f}s)')
                time.sleep(wait)
        else:
            print(f'  probe {conv}: FAILED after 6 attempts')
            continue
        try:
            ds = xr.open_dataset(io.BytesIO(data))
            lon_vals = ds[[c for c in ds.coords if c in ('longitude', 'lon')][0]].values
            sst = ds['CRW_SST'].values if 'CRW_SST' in ds else None
            ds.close()
            finite = float(np.nanmean(sst)) if sst is not None and np.isfinite(sst).any() \
                else float('nan')
            # 判据: 返回坐标与请求区间一致(而非被错误解释到地球另一侧)
            expect = {'-180_180': (-91.0, -59.0), '0_360': (269.0, 301.0)}[conv]
            lo_ok = (lon_vals.min() >= expect[0] - 5) and (lon_vals.max() <= expect[1] + 5)
            print(f'  probe {conv}: lon returned '
                  f'[{float(lon_vals.min()):.2f}, {float(lon_vals.max()):.2f}] '
                  f'sst_mean={finite:.2f} range_ok={lo_ok}')
            if lo_ok:
                return conv, LON_CANDIDATES[conv]
        except Exception as e:
            print(f'  probe {conv}: parse FAILED {type(e).__name__}: {e}')
    return None, None


def build_url(year, month, var, lon_range):
    t0 = datetime(year, month, 1, 12, 0, 0)
    t_end = datetime(year + 1, 1, 1, 12, 0, 0) if month == 12 else datetime(year, month + 1, 1, 12, 0, 0)
    t0s = t0.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1s = t_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    return (f'{BASE_URL}?{var}%5B({t0s}):1:({t1s})%5D'
            f'%5B{LAT_RANGE}%5D%5B{lon_range}%5D')


def download_one(year, month, var, lon_range):
    """Download single (year, month, var).  One-shot; caller handles retry."""
    prefix = var.lower()
    outfile = os.path.join(OUTPUT_DIR, f'{prefix}_{year}{month:02d}.nc')

    if os.path.exists(outfile) and os.path.getsize(outfile) > MIN_FILE_SIZE:
        return 'SKIP'

    url = build_url(year, month, var, lon_range)
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


def _pipe_variable(var, lon_range):
    """按年份-月份顺序下载某个变量的所有数据, 某月失败即重试不前进。

    年份范围 2002-2020 (非 GBR 的 1985-2020): causal v2 时间轴从 2003-01-05
    开始, 84 天 DHW 回溯 + 24 周输入窗口只需 ~2002-11 起的 CRW 数据;
    2002-01 起留出安全余量。磁盘约束: 加勒比盒 22°x32° 每月 SST 文件
    ~67MB, 全期 36 年约 125GB 超出磁盘余量, 19 年约 66GB。"""
    global _ok, _skip, _out_of_range, _fail, _done
    consecutive_fails = 0
    for year in range(2002, 2021):
        for month in range(1, 13):
            while True:
                status = download_one(year, month, var, lon_range)

                if status == 'SKIP':
                    with _lock:
                        _skip += 1
                        _done += 1
                    break

                elif status.startswith('OK'):
                    consecutive_fails = 0
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
                    consecutive_fails += 1
                    wait = min(RETRY_WAIT_BASE * 2 ** (consecutive_fails - 1),
                               RETRY_WAIT_MAX)
                    with _lock:
                        _fail += 1
                        n = _fail
                    _log(f'  RETRY {var} {year}-{month:02d}: {status} (#{n})'
                         f'  wait {wait:.0f}s')
                    time.sleep(wait)


def main():
    global _total, _start_time
    _total = len(VARS) * 19 * 12

    print("=" * 60)
    print("CRW 5km Parallel Downloader — CARIBBEAN (1 thread per variable)")
    print("=" * 60)
    print(f'   Server    : upwell.pfeg.noaa.gov (NOAA_DHW)')
    print(f'   Variables : {VARS}')
    print(f'   Period    : 2002-01 ~ 2020-12  (monthly files, daily data)')
    print(f'   Region    : lat 4:26  lon -91:-59  (Caribbean)')
    print(f'   Output    : {OUTPUT_DIR}')
    print(f'   Total jobs: {_total}')
    print()
    print('[preflight] 经度约定探测 (15N, 75W 5x5 方块)...')
    conv, lon_range = detect_lon_convention()
    if conv is None:
        print()
        print('❌ 经度约定探测失败: 服务器不可达或数据缺失。')
        print('   未开始全量下载 (绝不在约定未知时下载)。')
        print('   请检查网络后重试; 若服务器长期不可用, 考虑 ')
        print('   coralreefwatch.noaa.gov 官方 5km v3.1 月度文件源。')
        sys.exit(2)
    print(f'[preflight] 使用约定 {conv}: lon {lon_range}')
    with open(os.path.join(OUTPUT_DIR, 'download_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'region': 'Caribbean', 'lat_range': LAT_RANGE,
                   'lon_range': lon_range, 'lon_convention': conv,
                   'server': BASE_URL, 'vars': VARS,
                   'period': '1985-2020',
                   'started': datetime.now().isoformat()}, f, indent=2)
    print()
    _start_time = time.time()

    threads = []
    for v in VARS:
        t = threading.Thread(target=_pipe_variable, args=(v, lon_range), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - _start_time
    print()
    print("=" * 60)
    print("CARIBBEAN 下载完成!")
    print("=" * 60)
    print(f'   OK           : {_ok}')
    print(f'   SKIP (cached): {_skip}')
    print(f'   OUT_OF_RANGE : {_out_of_range}')
    print(f'   Time         : {elapsed/3600:.2f} h')
    print(f'   Avg rate     : {_ok/(elapsed/60):.2f} success/min')
    print(f'   Files on disk: {len(os.listdir(OUTPUT_DIR))} in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
