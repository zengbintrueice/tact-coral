"""
OC-CCI v6.0 Chlorophyll-a 下载脚本
ESA Ocean Colour Climate Change Initiative v6.0

下载南海区域 (3.5–23.5°N, 100.5–122.5°E) 月合成 chlorophyll-a 数据
时间范围: 1997-09 ~ 2020-12 (覆盖项目时间窗 + 前 2 年冷启动)

Source: CEDA Archive (Centre for Environmental Data Analysis)
URL: https://data.ceda.ac.uk/neodc/esacci/ocean_colour/data/v6.0-release/geographic/netcdf/chlor_a/monthly/v6.0/

Note:
  - 文件约 183MB/月 (4km 分辨率, 全球)
  - 需要 SSL verify=False (CEDA 证书链问题)
  - 添加 User-Agent 绕过
"""

import os
import sys
import time
import ssl
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backups" / "piecf_a6_nophys_20260718" / "a6_xgb_experiment" / "data" / "new_sources" / "chlorophyll"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CEDA OC-CCI v6.0 BASE URL
BASE_URL = (
    "https://data.ceda.ac.uk/neodc/esacci/ocean_colour/data/"
    "v6.0-release/geographic/netcdf/chlor_a/monthly/v6.0"
)

# Time range
START_YEAR, START_MONTH = 1997, 9   # OC-CCI starts Sept 1997
END_YEAR, END_MONTH = 2020, 12

# Download settings
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds
TIMEOUT = 600     # 10 minutes per file (183 MB)
MIN_FILE_SIZE = 100 * 1024 * 1024  # 100 MB minimum for valid file

# SSL context (CEDA has certificate chain issues)
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def month_range(start_y, start_m, end_y, end_m):
    """Generate (year, month) tuples from start to end inclusive."""
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def make_url(year: int, month: int) -> str:
    """Build download URL for a given year/month."""
    yyyymm = f"{year}{month:02d}"
    filename = (
        f"ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-"
        f"{yyyymm}-fv6.0.nc"
    )
    return f"{BASE_URL}/{year}/{filename}?download=1"


def make_filepath(year: int, month: int) -> Path:
    """Build local file save path."""
    yyyymm = f"{year}{month:02d}"
    return DATA_DIR / f"oc_cci_chlor_a_{yyyymm}_fv6.0.nc"


def download_file(url: str, filepath: Path) -> bool:
    """Download single file with retry logic. Returns True on success."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=TIMEOUT) as resp:
                data = resp.read()

            # Validate
            if len(data) < MIN_FILE_SIZE:
                print(f"  [Attempt {attempt}] File too small ({len(data)} bytes), retrying...")
                time.sleep(RETRY_DELAY)
                continue

            # Check NetCDF magic bytes
            if data[:4] != b"\x89HDF":
                print(f"  [Attempt {attempt}] Not a valid NetCDF file (magic: {data[:4]}), retrying...")
                time.sleep(RETRY_DELAY)
                continue

            filepath.write_bytes(data)
            size_mb = len(data) / (1024 * 1024)
            print(f"  Downloaded: {filepath.name} ({size_mb:.1f} MB)")
            return True

        except urllib.error.HTTPError as e:
            print(f"  [Attempt {attempt}] HTTP {e.code}: {e.reason}")
            if e.code == 404:
                return False  # File doesn't exist, don't retry
            time.sleep(RETRY_DELAY)

        except urllib.error.URLError as e:
            print(f"  [Attempt {attempt}] URL Error: {e.reason}")
            time.sleep(RETRY_DELAY * 2)

        except Exception as e:
            print(f"  [Attempt {attempt}] Error: {type(e).__name__}: {e}")
            time.sleep(RETRY_DELAY)

    return False


def main():
    # Force UTF-8 output
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    months = list(month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH))
    total = len(months)

    print("=" * 60)
    print("OC-CCI v6.0 Chlorophyll-a Download")
    print(f"Time range: {START_YEAR}-{START_MONTH:02d} ~ {END_YEAR}-{END_MONTH:02d}")
    print(f"Total months: {total}")
    print(f"Estimated size: {total * 183 / 1024:.1f} GB")
    print(f"Save to: {DATA_DIR}")
    print("=" * 60)

    success = 0
    skip = 0
    fail = 0

    start_time = time.time()

    for i, (year, month) in enumerate(months, 1):
        filepath = make_filepath(year, month)

        # Skip if already downloaded
        if filepath.exists() and filepath.stat().st_size > MIN_FILE_SIZE:
            skip += 1
            print(f"[{i}/{total}] SKIP (exists): {filepath.name}")
            continue

        url = make_url(year, month)
        elapsed = time.time() - start_time
        eta = (elapsed / max(success + fail, 1)) * (total - i + 1)
        print(f"[{i}/{total}] Downloading {year}-{month:02d}... (ETA: {eta / 60:.0f} min)")

        if download_file(url, filepath):
            success += 1
        else:
            fail += 1
            print(f"  FAILED: {year}-{month:02d}")

    print("=" * 60)
    print(f"Download complete in {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Success: {success}  |  Skipped: {skip}  |  Failed: {fail}")
    print("=" * 60)

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
