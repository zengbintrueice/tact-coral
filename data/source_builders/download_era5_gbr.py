"""
ERA5 GBR 下载脚本
==================
从 ECMWF Copernicus Climate Data Store (CDS) 下载 ERA5 月平均再分析数据
for the Great Barrier Reef (GBR) region:
  - sea_surface_temperature → sst (用于计算 sst_anomaly)
  - 10m_u_component_of_wind + 10m_v_component_of_wind → u10/v10 (→ wind_speed)
  - surface_net_solar_radiation → ssr

时间范围: 1985-01 ~ 2020-12
空间范围: GBR 区域 (lat -25:-8, lon 141:155) + 缓冲

输出 (3 个文件):
  - data/new_sources/era5_gbr/era5_sst_monthly.nc
  - data/new_sources/era5_gbr/era5_wind_monthly.nc
  - data/new_sources/era5_gbr/era5_ssr_monthly.nc

策略: 按年下载 (避免 CDS 大请求超时), 然后拼接为单文件。

警告: 需要 CDS API credentials (~/.cdsapirc):
    url: https://cds.climate.copernicus.eu/api/v2
    key: {UID}:{API_KEY}
    申请: https://cds.climate.copernicus.eu/user/register

运行方式:
    python src/data/download_era5_gbr.py
"""
import os
import sys
import time
import glob
import tempfile
import io
import warnings

warnings.filterwarnings('ignore')

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'backups', 'piecf_a6_nophys_20260718', 'a6_xgb_experiment', 'data', 'new_sources', 'era5_gbr')
os.makedirs(DATA_DIR, exist_ok=True)

# Time range
START_YEAR = 1985
END_YEAR = 2020

# Spatial: GBR + buffer (0.25° resolution)
# GBR target: lat -25:-8, lon 141:155
AREA = [-8, 141, -25, 155]   # North, West, South, East
RESOLUTION = 0.25            # ERA5 native resolution

# 3 variable groups → 3 separate output files
VARIABLE_GROUPS = {
    'era5_sst_monthly': ['sea_surface_temperature'],
    'era5_wind_monthly': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    'era5_ssr_monthly': ['surface_net_solar_radiation'],
}

FORMAT = 'netcdf'
TMP_DIR = tempfile.gettempdir()


def download_year(year, variables, output_path):
    """Download one year of ERA5 monthly data for given variables."""
    import cdsapi

    client = cdsapi.Client()

    client.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': variables,
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


def concat_year_files(year_files, output_path):
    """Concatenate per-year ERA5 files into a single monthly file.

    CDS ERA5 monthly files have dimensions (valid_time=12, lat, lon) where
    'valid_time' holds the 12 monthly values. We concatenate along
    'valid_time', then rename → 'time' for a clean (432, lat, lon) result.

    Writes to temp ASCII path first, then moves to final path (avoids
    Chinese-path netCDF4 PermissionError).
    """
    import xarray as xr

    if not year_files:
        return False

    year_files_sorted = sorted(year_files)
    datasets = [xr.open_dataset(f) for f in year_files_sorted]

    # Concatenate along 'valid_time' → (432, lat, lon)
    combined = xr.concat(datasets, dim='valid_time')
    combined = combined.sortby('valid_time')

    # Rename 'valid_time' → 'time' for clean indexing
    combined = combined.rename({'valid_time': 'time'})

    # Write to temp ASCII path, then move (avoids Chinese path netCDF4 issue)
    tmp_out = os.path.join(TMP_DIR, os.path.basename(output_path))
    combined.to_netcdf(tmp_out)
    combined.close()
    for ds in datasets:
        ds.close()

    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(tmp_out, output_path)
    return True


def main():
    print("=" * 60)
    print("ERA5 GBR Download")
    print(f"  Years: {START_YEAR}-{END_YEAR}")
    print(f"  Area: {AREA} (N, W, S, E)")
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

    # Check if final output files already exist (skip if all 3 present and valid)
    all_cached = True
    for fname in VARIABLE_GROUPS:
        out_path = os.path.join(DATA_DIR, f'{fname}.nc')
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100000:
            all_cached = False
            break
    if all_cached:
        print("\nAll 3 output files already exist and are valid. Skip download.")
        return 0

    # Build download list: per-year files for each variable group
    tasks = []
    for fname, variables in VARIABLE_GROUPS.items():
        out_path = os.path.join(DATA_DIR, f'{fname}.nc')
        # If final file already exists, skip this group
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            print(f"  SKIP (cached): {fname}.nc")
            continue

        years_needed = []
        for year in range(START_YEAR, END_YEAR + 1):
            path = os.path.join(TMP_DIR, f'era5_gbr_{fname}_{year}.nc')
            if not os.path.exists(path) or os.path.getsize(path) < 50000:
                years_needed.append((year, path))
        if years_needed:
            tasks.append((fname, variables, years_needed, out_path))

    if not tasks:
        print("\nAll year-files cached. Proceeding to concatenation...")
    else:
        total_years = sum(len(years) for _, _, years, _ in tasks)
        print(f"\nVariable groups to download: {len(tasks)} ({total_years} year-files total)")
        for fname, variables, years, _ in tasks:
            print(f"  {fname}: {variables} — {len(years)} years")

        success = 0
        fail = 0
        start_time = time.time()

        for fname, variables, years, out_path in tasks:
            print(f"\n{'=' * 60}")
            print(f"Downloading: {fname} ({variables})")
            print(f"{'=' * 60}")
            for i, (year, path) in enumerate(years, 1):
                elapsed = time.time() - start_time
                remaining = total_years - success - fail
                eta = (elapsed / max(success, 1)) * remaining if success > 0 else 0
                print(f"  [{i}/{len(years)}] {year} (ETA: {eta / 60:.0f} min)...")

                try:
                    download_year(year, variables, path)
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    print(f"    Saved: {os.path.basename(path)} ({size_mb:.1f} MB)")
                    success += 1
                    # Polite delay for CDS API rate limits
                    time.sleep(0.5)
                except Exception as e:
                    print(f"    FAILED: {type(e).__name__}: {e}")
                    fail += 1
                    # Remove failed partial download
                    if os.path.exists(path):
                        os.remove(path)

        print("\n" + "=" * 60)
        print(f"Download phase complete in {(time.time() - start_time) / 60:.1f} minutes")
        print(f"  Success: {success}  |  Failed: {fail}")
        print("=" * 60)

        if fail > 0:
            print(f"\nWARNING: {fail} years failed. Concatenating available years...")

    # Concatenation phase: merge per-year files into single output files
    print("\n" + "=" * 60)
    print("Concatenation phase: merging year-files into single files")
    print("=" * 60)

    for fname in VARIABLE_GROUPS:
        out_path = os.path.join(DATA_DIR, f'{fname}.nc')
        # Check if already cached
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            print(f"  SKIP (cached): {fname}.nc")
            continue

        # Collect all year-files for this group
        year_files = []
        for year in range(START_YEAR, END_YEAR + 1):
            path = os.path.join(TMP_DIR, f'era5_gbr_{fname}_{year}.nc')
            if os.path.exists(path) and os.path.getsize(path) > 50000:
                year_files.append(path)

        if not year_files:
            print(f"  SKIP: no year-files found for {fname}")
            continue

        print(f"  Concatenating {len(year_files)} year-files for {fname}...")
        try:
            concat_year_files(year_files, out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"    Saved: {fname}.nc ({size_mb:.1f} MB)")

            # Clean up year-files after successful concatenation
            for f in year_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            print(f"    Cleaned up {len(year_files)} year-files")
        except Exception as e:
            print(f"    CONCAT FAILED: {type(e).__name__}: {e}")

    # Final verification
    print("\n" + "=" * 60)
    print("Final verification:")
    all_ok = True
    for fname in VARIABLE_GROUPS:
        out_path = os.path.join(DATA_DIR, f'{fname}.nc')
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"  OK: {fname}.nc ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING: {fname}.nc")
            all_ok = False
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
