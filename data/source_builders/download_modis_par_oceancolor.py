"""
MODIS Aqua PAR (Photosynthetically Active Radiation) 下载脚本
=================================================================
从 NASA OceanColor 下载 MODIS Aqua Level 3 月合成 PAR 数据。

产品: AQUA_MODIS.YYYYMM.L3m.MN.PAR.par_4km.nc
时间范围: 2002-07 (Aqua 卫星发射) ~ 2020-12
空间范围: 全球 4km 分辨率

注意:
  - 需要 NASA Earthdata Login (URS 认证)
  - 首次运行需使用 src/data/setup_earthdata_auth.py 配置认证
  - 注册: https://urs.earthdata.nasa.gov/

运行方式:
    python src/data/download_modis_par_oceancolor.py
"""

import os
import sys
import time
import io
import ssl
import urllib.request
import urllib.error
import re

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'backups', 'piecf_a6_nophys_20260718', 'a6_xgb_experiment', 'data', 'new_sources', 'modis_par')
os.makedirs(DATA_DIR, exist_ok=True)

# Time range (MODIS Aqua: 2002-07 onwards)
START_YEAR, START_MONTH = 2002, 7
END_YEAR, END_MONTH = 2020, 12

# NASA OceanColor base URL for direct download
BASE_URL = 'https://oceandata.sci.gsfc.nasa.gov/ob/getfile/'

# File size threshold (PAR files ~150 MB)
MIN_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# SSL context (NASA uses standard certs, no bypass needed)
ssl_ctx = ssl.create_default_context()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Mozilla/5.0 (compatible; Python urllib)"
    ),
}


def parse_netrc(netrc_path=None):
    """Parse ~/.netrc for Earthdata credentials."""
    if netrc_path is None:
        netrc_path = os.path.expanduser('~/.netrc')
    if not os.path.exists(netrc_path):
        return None, None
    try:
        import netrc
        n = netrc.netrc(netrc_path)
        # Look for machine urs.earthdata.nasa.gov
        auth = n.authenticators('urs.earthdata.nasa.gov')
        if auth:
            return auth[0], auth[2]  # login, password
    except Exception:
        pass
    return None, None


def make_url(year, month):
    """Build NASA OceanColor download URL for monthly PAR composite."""
    yyyymm = f"{year}{month:02d}"
    filename = f"A{yyyymm}.L3m.MN.PAR.par_4km.nc"
    return f"{BASE_URL}{filename}", filename


def download_with_auth(url, filepath, username, password, timeout=600):
    """Download file with Earthdata URS authentication."""
    import base64
    auth_str = f"{username}:{password}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    headers = dict(HEADERS)
    headers['Authorization'] = f'Basic {auth_b64}'

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=timeout) as resp:
            data = resp.read()

        # Validate
        if len(data) < MIN_FILE_SIZE:
            return False, f"Too small ({len(data)} bytes)"

        # Check NetCDF magic
        if data[:4] != b'\x89HDF':
            return False, f"Not NetCDF (magic: {data[:4]})"

        with open(filepath, 'wb') as f:
            f.write(data)
        return True, None
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("=" * 60)
    print("MODIS Aqua PAR Download (NASA OceanColor)")
    print(f"Time range: {START_YEAR}-{START_MONTH:02d} ~ {END_YEAR}-{END_MONTH:02d}")
    print(f"Save to: {DATA_DIR}")
    print("=" * 60)

    # Get Earthdata credentials
    username, password = parse_netrc()
    if not username:
        print("\nERROR: Earthdata credentials not found in ~/.netrc")
        print("\nPlease create ~/.netrc with the following content:")
        print("  machine urs.earthdata.nasa.gov")
        print("  login YOUR_USERNAME")
        print("  password YOUR_PASSWORD")
        print("\nThen run: chmod 600 ~/.netrc")
        print("\nRegister at: https://urs.earthdata.nasa.gov/user/register")
        return 1

    print(f"Using Earthdata user: {username[:3]}***")

    # Build month list
    months = []
    y, m = START_YEAR, START_MONTH
    while (y, m) <= (END_YEAR, END_MONTH):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    total = len(months)
    print(f"Total months: {total}")

    success = 0
    skip = 0
    fail = 0
    start_time = time.time()

    for i, (year, month) in enumerate(months, 1):
        url, filename = make_url(year, month)
        filepath = os.path.join(DATA_DIR, filename)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > MIN_FILE_SIZE:
            skip += 1
            if i % 20 == 0 or i == total:
                print(f"[{i}/{total}] Progress: {skip} skipped, {success} ok, {fail} fail")
            continue

        ok, err = download_with_auth(url, filepath, username, password)
        if ok:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[{i}/{total}] {year}-{month:02d} OK ({size_mb:.1f} MB)")
            success += 1
        else:
            print(f"[{i}/{total}] {year}-{month:02d} FAIL: {err}")
            fail += 1

    print("\n" + "=" * 60)
    print(f"Download complete in {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Success: {success}  |  Skipped: {skip}  |  Failed: {fail}")
    print("=" * 60)

    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
