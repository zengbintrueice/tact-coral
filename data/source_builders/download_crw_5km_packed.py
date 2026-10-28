"""
Download NOAA CRW 5km daily products via PACKED (multi-variable) requests.

每个请求打包 3 个变量一次下载 (SST+SSTA+DHW / HOTSPOT+BAA+BAA_7D_MAX),
将请求数从 6 次/月 降到 2 次/月 (减少 2/3 请求量), 大幅降低 NOAA 服务器
压力与 408 频率。下载后在本地拆分为单变量文件, 磁盘结构与既有
crw_<var>_YYYYMM.nc 命名完全一致, 断点续传 (已存在文件自动跳过)。

Usage:
  # 东海+黄海 全量 1985-2024 (默认)
  python download_crw_5km_packed.py --region es_ys

  # 南海 2021-2024 补下 (补齐测试期缺口)
  python download_crw_5km_packed.py --region scs --start 2021 --end 2024

  # 自定义年份区间
  python download_crw_5km_packed.py --region es_ys --start 2020 --end 2024
"""
import os
import sys
import time
import urllib.request
import http.client
import socket
import threading
from datetime import datetime
from urllib.error import HTTPError, URLError

# --- force IPv4 -------------------------------------------------------------
# coastwatch.pfeg.noaa.gov is unreachable over IPv6 from some networks (SYN
# hangs); resolve and connect over IPv4 explicitly.  Default is enabled.
FORCE_IPV4 = os.environ.get('CRW_FORCE_IPV4', '1') != '0'

if FORCE_IPV4:
    import ssl

    def _ipv4_connect(self, *a, **kw):
        """IPv4-only connect, then do the normal TLS handshake."""
        host = self.host
        # (a) resolve host -> IPv4 addresses only
        addrs = [res for res in socket.getaddrinfo(
            host, self.port, socket.AF_INET, socket.SOCK_STREAM)]
        if not addrs:
            raise OSError(f'no IPv4 address for {host}')
        # (b) connect TCP to the first reachable IPv4 addr
        sock = None
        for af, socktype, proto, canon, sa in addrs:
            s = socket.socket(af, socktype, proto)
            try:
                s.settimeout(self.timeout)
                s.connect(sa)
                sock = s
                break
            except OSError:
                s.close()
        if sock is None:
            raise OSError(f'IPv4 connect failed for {host}')
        self.sock = sock
        # (c) replay the normal TLS wrap (httpsclient does this after connect)
        if isinstance(self, http.client.HTTPSConnection) and self._context:
            try:
                self.sock = self._context.wrap_socket(self.sock,
                                                      server_hostname=host)
            except Exception:
                self.sock.close()
                self.sock = None
                raise

    http.client.HTTPSConnection.connect = _ipv4_connect

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 数据源可通过环境变量 CRW_BASE_URL 覆盖 (默认 coastwatch;
# 故障时可切到官方镜像 https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.nc)
BASE_URL = os.environ.get(
    'CRW_BASE_URL',
    'https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.nc'
)

# lon360 数据集 (经度 0..360) 用 CRW_LON_OFFSET 翻译负经度区域:
# natl 的 (-61):(-19) -> (299):(341)。默认 0 = 不翻译。
LON_OFFSET = int(os.environ.get('CRW_LON_OFFSET', '0'))


def _lon_translate(lon_range):
    """对 '(a):(b)' 经度区间施加 CRW_LON_OFFSET (用于 lon360 数据集)."""
    if not LON_OFFSET:
        return lon_range
    lo, hi = lon_range.split(':')
    return (f'({int(lo.strip("()")) + LON_OFFSET})'
            f':({int(hi.strip("()")) + LON_OFFSET})')

# --- 变量打包: 每个 PACK 一次请求下载 3 个变量 ---
PACKS = [
    ['CRW_SST', 'CRW_SSTANOMALY', 'CRW_DHW'],
    ['CRW_HOTSPOT', 'CRW_BAA', 'CRW_BAA_7D_MAX'],
]

# --- 区域定义 ---
REGIONS = {
    # 南海 (含 buffer)
    'scs': {
        'lat': '(3):(24)',
        'lon': '(100):(123)',
        'out': os.environ.get("PIECF_RAW_DIR", r"D:\projects\PINN-OA") + r"\data\raw\crw_5km_sst",
    },
    # 东海+黄海 (含 buffer)
    'es_ys': {
        'lat': '(24):(42)',
        'lon': '(119):(130)',
        'out': os.environ.get("PIECF_RAW_DIR", r"D:\projects\PINN-OA") + r"\data\raw\crw_5km_es_ys",
    },
    # 北大西洋副热带环流 (含 buffer; 西经用负值索引, 已验证可下载)
    'natl': {
        'lat': '(19):(41)',
        'lon': '(-61):(-19)',
        'out': os.environ.get("PIECF_RAW_DIR", r"D:\projects\PINN-OA") + r"\data\raw\crw_5km_natl",
    },
}

MIN_FILE_SIZE = 50_000
# 超时可用环境变量调大: 镜像带宽较低, 大包传输可能逼近 600s
REQUEST_TIMEOUT = int(os.environ.get('CRW_REQUEST_TIMEOUT', '600'))
RETRY_WAIT = 5
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


def _log(msg):
    with _lock:
        print(msg, flush=True)


def build_url(year, month, vars_, lat_range, lon_range):
    """构建多变量打包请求 URL (逗号连接多个变量)."""
    t0 = datetime(year, month, 1, 12, 0, 0)
    t_end = datetime(year + 1, 1, 1, 12, 0, 0) if month == 12 else datetime(year, month + 1, 1, 12, 0, 0)
    t0s = t0.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1s = t_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    segs = []
    for v in vars_:
        segs.append(f'{v}%5B({t0s}):1:({t1s})%5D%5B{lat_range}%5D%5B{_lon_translate(lon_range)}%5D')
    return f'{BASE_URL}?' + ','.join(segs)


def split_pack(pack_nc, vars_, out_dir, year, month):
    """把打包的 nc 文件拆分为单变量文件 crw_<var>_YYYYMM.nc. 返回成功拆分的数量."""
    import netCDF4

    n_ok = 0
    src = netCDF4.Dataset(pack_nc)
    for v in vars_:
        outfile = os.path.join(out_dir, f'{v.lower()}_{year}{month:02d}.nc')
        # 已存在则跳过 (断点续传)
        if os.path.exists(outfile) and os.path.getsize(outfile) > MIN_FILE_SIZE:
            continue
        dst = netCDF4.Dataset(outfile, 'w', format='NETCDF4')
        for dim_name, dim in src.dimensions.items():
            dst.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)
        # 坐标变量
        for coord in ('time', 'latitude', 'longitude'):
            cv = src.variables[coord]
            cnew = dst.createVariable(coord, cv.dtype, cv.dimensions)
            for a in cv.ncattrs():
                setattr(cnew, a, cv.getncattr(a))
            cnew[:] = cv[:]
        # 该变量 (注意 _FillValue 必须在创建时指定)
        var = src.variables[v]
        fill = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
        vnew = dst.createVariable(v, var.dtype, var.dimensions,
                                  zlib=True, complevel=4, fill_value=fill)
        for a in var.ncattrs():
            if a == '_FillValue':
                continue
            setattr(vnew, a, var.getncattr(a))
        vnew[:] = var[:]
        dst.close()
        n_ok += 1
    src.close()
    return n_ok


def _fetch_status(url):
    """Return '200' | 'tiny' | '<code>' | 'error' for a URL, 30s cap."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        return '200' if len(data) >= 1000 else 'tiny'
    except HTTPError as e:
        return str(e.code)
    except Exception:
        return 'error'


def _probe_missing(year, month, vars_, lat_range, lon_range):
    """Return True only if the server genuinely lacks this month (real gap).

    Two very different things produce a 404 on the packed request:
      (a) the month is a real gap (predates the product, product hole);
      (b) the whole dataset is temporarily unavailable (observed 2026-09-04:
          PACIOOS removed dhw_5km for a while -> every request 404).
    A target-month 404 is only trusted after a CONTROL request to a known-valid
    time slice (2015-06-15) at the same cell also returns data.  If the control
    also 404s, the dataset itself is down -> report NOT missing and let the
    retry loop wait for recovery (otherwise the run would skip every month).
    """
    t = datetime(year, month, 15, 12, 0, 0).strftime('%Y-%m-%dT%H:%M:%SZ')
    lat0 = lat_range.split(':')[0]
    lon0 = _lon_translate(lon_range).split(':')[0]
    segs = [f'{v}%5B({t}):1:({t})%5D%5B{lat0}%5D%5B{lon0}%5D' for v in vars_]
    url = f'{BASE_URL}?' + ','.join(segs)
    if _fetch_status(url) != '404':
        return False  # target month responds -> data exists -> not missing
    # target 404 -> distinguish real gap from dataset outage via control probe
    tc = datetime(2015, 6, 15, 12, 0, 0).strftime('%Y-%m-%dT%H:%M:%SZ')
    ctrl = (f'{BASE_URL}?CRW_SST%5B({tc}):1:({tc})%5D'
            f'%5B{lat0}%5D%5B{lon0}%5D')
    if _fetch_status(ctrl) == '404':
        return False  # dataset itself unavailable -> transient, retry
    return True  # control OK, target 404 -> genuine gap


def download_pack(year, month, vars_, lat_range, lon_range, out_dir):
    """下载一个月的一个变量包 (3变量), 拆分后删除临时包文件."""
    tag = f'{year}-{month:02d}'
    prefix = '_'.join(v.lower() for v in vars_)

    # 断点续传: 该月该包所有单变量文件都已存在则跳过
    all_exist = all(
        os.path.exists(os.path.join(out_dir, f'{v.lower()}_{year}{month:02d}.nc'))
        and os.path.getsize(os.path.join(out_dir, f'{v.lower()}_{year}{month:02d}.nc')) > MIN_FILE_SIZE
        for v in vars_
    )
    if all_exist:
        return 'SKIP'

    pack_tmp = os.path.join(out_dir, f'_pack_{prefix}_{year}{month:02d}.nc')
    url = build_url(year, month, vars_, lat_range, lon_range)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        data = resp.read()
        if len(data) < 1000:
            return 'FAIL: too_small'
        with open(pack_tmp, 'wb') as f:
            f.write(data)
        n = split_pack(pack_tmp, vars_, out_dir, year, month)
        try:
            os.remove(pack_tmp)
        except OSError:
            pass
        return f'OK {len(data)/1024/1024:.1f}MB x{n}'
    except HTTPError as e:
        if e.code == 404:
            # 404 may be transient (mirror overload) OR a genuine data gap.
            # Verify with a tiny probe before skipping permanently.
            if _probe_missing(year, month, vars_, lat_range, lon_range):
                return 'OUT_OF_RANGE'
            return 'FAIL: HTTP 404 (probe OK -> transient)'
        return f'FAIL: HTTP {e.code}'
    except (URLError, TimeoutError, OSError,
            http.client.IncompleteRead, http.client.HTTPException,
            ConnectionResetError, socket.timeout, EOFError) as e:
        if os.path.exists(pack_tmp):
            try:
                os.remove(pack_tmp)
            except OSError:
                pass
        return f'FAIL: {type(e).__name__}'


def _pipe_region(region_cfg, start_year, end_year):
    """按月份顺序下载一个区域所有变量包."""
    global _ok, _skip, _fail, _done, _out_of_range
    out_dir = region_cfg['out']
    os.makedirs(out_dir, exist_ok=True)
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for vars_ in PACKS:
                while True:
                    status = download_pack(year, month, vars_,
                                           region_cfg['lat'], region_cfg['lon'], out_dir)
                    if status == 'SKIP':
                        with _lock:
                            _skip += 1
                            _done += 1
                        break
                    elif status == 'OUT_OF_RANGE':
                        # 该月无数据 (如 CRW 1985-01~03), 跳过不重试
                        with _lock:
                            _out_of_range += 1
                            _done += 1
                        _log(f'  OUT_OF_RANGE {year}-{month:02d} {vars_[0].lower()}+ (skip)')
                        break
                    elif status.startswith('OK'):
                        with _lock:
                            _ok += 1
                            _done += 1
                            elapsed = time.time() - _start_time
                            rate = _ok / (elapsed / 60) if elapsed > 0 else 0
                            remaining = _total - _ok - _skip
                            eta = remaining / rate if rate > 0 else 0
                        _log(f'[{_ok}/{_total-_skip}] {year}-{month:02d} {vars_[0].lower()}+ '
                             f'{status}  {rate:.2f} pack/min  ETA {eta:.0f}min  {elapsed/60:.0f}min')
                        break
                    else:
                        with _lock:
                            _fail += 1
                            n = _fail
                        _log(f'  RETRY {year}-{month:02d} {vars_[0].lower()}: {status} (#{n})')
                        time.sleep(RETRY_WAIT)


def _worker(job, region_cfg):
    """线程池 worker: 处理单个 (year, month, pack) 任务, 失败无限重试."""
    global _ok, _skip, _fail, _done, _out_of_range
    year, month, vars_ = job
    out_dir = region_cfg['out']
    while True:
        status = download_pack(year, month, vars_,
                               region_cfg['lat'], region_cfg['lon'], out_dir)
        if status == 'SKIP':
            with _lock:
                _skip += 1
                _done += 1
            return
        elif status == 'OUT_OF_RANGE':
            with _lock:
                _out_of_range += 1
                _done += 1
            return
        elif status.startswith('OK'):
            with _lock:
                _ok += 1
                _done += 1
                elapsed = time.time() - _start_time
                rate = _ok / (elapsed / 60) if elapsed > 0 else 0
                remaining = _total - _ok - _skip
                eta = remaining / rate if rate > 0 else 0
            _log(f'[{_ok}/{_total-_skip}] {year}-{month:02d} {vars_[0].lower()}+ '
                 f'{status}  {rate:.2f} pack/min  ETA {eta:.0f}min  {elapsed/60:.0f}min')
            return
        else:
            with _lock:
                _fail += 1
                n = _fail
            _log(f'  RETRY {year}-{month:02d} {vars_[0].lower()}: {status} (#{n})')
            time.sleep(RETRY_WAIT)


def _run_parallel(region_cfg, start_year, end_year, workers):
    """多线程并行: 每个任务 = (year, month, pack), ThreadPoolExecutor 并发."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(y, m, v) for y in range(start_year, end_year + 1)
            for m in range(1, 13) for v in PACKS]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, j, region_cfg) for j in jobs]
        for f in futures:
            f.result()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CRW 5km packed (multi-var) downloader')
    parser.add_argument('--region', choices=['scs', 'es_ys', 'natl'], default='es_ys',
                        help='region: scs=South China Sea, es_ys=East China Sea+Yellow Sea, '
                             'natl=North Atlantic subtropical gyre')
    parser.add_argument('--start', type=int, default=1985, help='start year (default 1985)')
    parser.add_argument('--end', type=int, default=2024, help='end year (default 2024)')
    parser.add_argument('--workers', type=int, default=1,
                        help='parallel download threads (default 1 = serial; try 2-3)')
    args = parser.parse_args()

    global _total, _start_time
    n_years = args.end - args.start + 1
    _total = n_years * 12 * len(PACKS)

    cfg = REGIONS[args.region]
    print("=" * 60)
    print(f"CRW 5km PACKED Downloader — {args.region}  (workers={args.workers})")
    print("=" * 60)
    print(f'   Packs      : {len(PACKS)} per month (was 6 requests)')
    print(f'   Period     : {args.start}-01 ~ {args.end}-12')
    print(f'   Region     : lat {cfg["lat"]}  lon {cfg["lon"]}')
    print(f'   Output     : {cfg["out"]}')
    print(f'   Total packs: {_total}  (was {_total * 3})')
    print()
    _start_time = time.time()

    if args.workers > 1:
        _run_parallel(cfg, args.start, args.end, args.workers)
    else:
        _pipe_region(cfg, args.start, args.end)

    elapsed = time.time() - _start_time
    print()
    print("=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f'   OK           : {_ok}')
    print(f'   SKIP (cached): {_skip}')
    print(f'   OUT_OF_RANGE : {_out_of_range}')
    print(f'   Time         : {elapsed/3600:.2f} h')
    print(f'   Avg rate     : {_ok/(elapsed/60):.2f} pack/min')


if __name__ == '__main__':
    main()
