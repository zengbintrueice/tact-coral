"""
ERA5 PAR + Wind Stress 下载脚本
=================================
从 ECMWF Copernicus Climate Data Store (CDS) 下载 ERA5 月平均再分析数据:
  - surface_solar_radiation_downwards (ssrd): 地表向下太阳辐射 (PAR 代理)
  - mean_eastward_turbulent_surface_stress: 东向湍流表面应力 (风应力分量)
  - mean_northward_turbulent_surface_stress: 北向湍流表面应力 (风应力分量)

时间范围: 1985-01 ~ 2020-12 (对齐项目)
空间范围: 南海区域 (3.5°N~23.5°N, 100.5°E~122.5°E) + 外围缓冲

警告: 需要 CDS API credentials (~/.cdsapirc):
    url: https://cds.climate.copernicus.eu/api/v2
    key: {UID}:{API_KEY}
    申请: https://cds.climate.copernicus.eu/user/register

运行方式:
    python src/data/download_era5_par_windstress.py
"""

import os
import sys
import time
import glob
import io
import warnings

warnings.filterwarnings('ignore')

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'backups', 'piecf_a6_nophys_20260718', 'a6_xgb_experiment', 'data', 'new_sources', 'era5')
os.makedirs(DATA_DIR, exist_ok=True)

# Time range
START_YEAR = 1985
END_YEAR = 2020

# Spatial: SCS + buffer (0.25° resolution, so make it slightly larger)
AREA = [25.5, 99.5, 2.5, 124.5]   # North, West, South, East
RESOLUTION = 0.25                 # ERA5 native resolution

# Variables
VARIABLES = [
    'surface_solar_radiation_downwards',           # PAR proxy (J m^-2)
    'mean_eastward_turbulent_surface_stress',       # Wind stress eastward (N m^-2 s)
    'mean_northward_turbulent_surface_stress',      # Wind stress northward (N m^-2 s)
]

# Output format
FORMAT = 'netcdf'


def download_year(year, output_path):
    """Download one year of ERA5 monthly data."""
    import cdsapi

    client = cdsapi.Client()

    client.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': VARIABLES,
            'year': [str(year)],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
            ],
            'time': '00:00',
            'area': AREA,
            'grid': [RESOLUTION, RESOLUTION],
            'format': FORMAT,
        },
        output_path
    )
    return True


def main():
    print("=" * 60)
    print("ERA5 PAR + Wind Stress Download")
    print(f"  Years: {START_YEAR}-{END_YEAR}")
    print(f"  Area: {AREA}")
    print(f"  Variables: {VARIABLES}")
    print(f"  Save to: {DATA_DIR}")
    print("=" * 60)

    # Check CDS API credentials
    cdsapirc = os.path.expanduser('~/.cdsapirc')
    if not os.path.exists(cdsapirc):
        print(f"\nERROR: CDS API credentials not found at {cdsapirc}")
        print("Please create this file with the following content:")
        print("  url: https://cds.climate.copernicus.eu/api/v2")
        print("  key: YOUR_UID:YOUR_API_KEY")
        print("\nTo get credentials:")
        print("  1. Register at https://cds.climate.copernicus.eu/user/register")
        print("  2. Find your UID and API key at https://cds.climate.copernicus.eu/api-how-to")
        print("  3. Accept the ERA5 license terms at the CDS website")
        return 1

    years_to_download = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = os.path.join(DATA_DIR, f'era5_surf_strd_stress_{year}.nc')
        if not os.path.exists(path) or os.path.getsize(path) < 100000:
            years_to_download.append((year, path))

    if not years_to_download:
        print("All years already downloaded!")
        return 0

    print(f"\nYears to download: {len(years_to_download)} (skipping cached)")

    success = 0
    fail = 0
    start_time = time.time()

    for i, (year, path) in enumerate(years_to_download, 1):
        elapsed = time.time() - start_time
        eta = (elapsed / max(success, 1)) * (len(years_to_download) - i + 1) if success > 0 else 0
        print(f"\n[{i}/{len(years_to_download)}] Downloading {year} (ETA: {eta / 60:.0f} min)...")

        try:
            download_year(year, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Saved: {os.path.basename(path)} ({size_mb:.1f} MB)")
            success += 1
            # Polite delay for CDS API rate limits
            time.sleep(0.5)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            fail += 1
            # Remove failed partial download
            if os.path.exists(path):
                os.remove(path)

    print("\n" + "=" * 60)
    print(f"Download complete in {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Success: {success}  |  Failed: {fail}")
    print("=" * 60)

    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
