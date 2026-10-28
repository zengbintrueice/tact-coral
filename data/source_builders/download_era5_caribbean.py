"""
ERA5 Caribbean 下载脚本
========================
从 ECMWF Copernicus Climate Data Store (CDS) 下载 ERA5 月平均再分析数据
for the Caribbean region:
  - sea_surface_temperature → sst (用于计算 sst_anomaly)
  - 10m_u/v_component_of_wind → wind_speed
  - surface_net_solar_radiation → ssr

时间范围: 2003-01 ~ 2020-12 (216 个月, 与 SCS era5_monthly 完全一致;
causal v2 的 sst_anomaly 气候态只用 year<=2010 月份, 2003 起已足够,
2002 年 CRW 侧的 DHW 回溯不依赖 ERA5)
空间范围: Caribbean (lat 4:26, lon -91:-59) + 缓冲 (AREA = [26, -91, 4, -59])

输出 (D:\piecf_dns\era5_caribbean\):
  - era5_sst_monthly.nc
  - era5_wind_monthly.nc
  - era5_ssr_monthly.nc

策略: 按年下载 (避免 CDS 大请求超时), 然后拼接为单文件。
基于 download_era5_gbr.py (成功经验) 改包围盒与年份。

运行方式:
    & piecf_env\Scripts\python.exe download_era5_caribbean.py
"""
import os
import sys
import time
import shutil
import tempfile
import io
import warnings

warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"\era5_caribbean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 2003
END_YEAR = 2020

# Spatial: Caribbean + buffer (0.25° resolution), AREA = [North, West, South, East]
AREA = [26, -91, 4, -59]
RESOLUTION = 0.25

VARIABLE_GROUPS = {
    'era5_sst_monthly': ['sea_surface_temperature'],
    'era5_wind_monthly': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    'era5_ssr_monthly': ['surface_net_solar_radiation'],
}

FORMAT = 'netcdf'
TMP_DIR = tempfile.gettempdir()


def download_year(year, variables, output_path):
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
    """Concatenate per-year ERA5 files into a single monthly file
    (valid_time → time; temp ASCII path then move)."""
    import xarray as xr

    if not year_files:
        return False

    year_files_sorted = sorted(year_files)
    datasets = [xr.open_dataset(f) for f in year_files_sorted]

    combined = xr.concat(datasets, dim='valid_time')
    combined = combined.sortby('valid_time')
    combined = combined.rename({'valid_time': 'time'})

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
    print("ERA5 Caribbean Download")
    print(f"  Years: {START_YEAR}-{END_YEAR}")
    print(f"  Area: {AREA} (N, W, S, E)")
    print(f"  Save to: {OUTPUT_DIR}")
    print("=" * 60)

    cdsapirc = os.path.expanduser('~/.cdsapirc')
    if not os.path.exists(cdsapirc):
        print(f"\nERROR: CDS API credentials not found at {cdsapirc}")
        return 1

    all_cached = True
    for fname in VARIABLE_GROUPS:
        out_path = os.path.join(OUTPUT_DIR, f'{fname}.nc')
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100000:
            all_cached = False
            break
    if all_cached:
        print("\nAll 3 output files already exist and are valid. Skip download.")
        return 0

    tasks = []
    for fname, variables in VARIABLE_GROUPS.items():
        out_path = os.path.join(OUTPUT_DIR, f'{fname}.nc')
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            print(f"  SKIP (cached): {fname}.nc")
            continue
        years_needed = []
        for year in range(START_YEAR, END_YEAR + 1):
            path = os.path.join(TMP_DIR, f'era5_caribbean_{fname}_{year}.nc')
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
                print(f"  [{i}/{len(years)}] {year} (ETA: {eta / 60:.0f} min)...", flush=True)

                try:
                    download_year(year, variables, path)
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    print(f"    Saved: {os.path.basename(path)} ({size_mb:.1f} MB)", flush=True)
                    success += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                    fail += 1
                    if os.path.exists(path):
                        os.remove(path)

        print("\n" + "=" * 60)
        print(f"Download phase complete in {(time.time() - start_time) / 60:.1f} minutes")
        print(f"  Success: {success}  |  Failed: {fail}")
        print("=" * 60)

        if fail > 0:
            print(f"\nWARNING: {fail} years failed. Re-run the script to retry them "
                  f"(year-files are resumable).")

    print("\n" + "=" * 60)
    print("Concatenation phase: merging year-files into single files")
    print("=" * 60)

    for fname in VARIABLE_GROUPS:
        out_path = os.path.join(OUTPUT_DIR, f'{fname}.nc')
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            print(f"  SKIP (cached): {fname}.nc")
            continue

        year_files = []
        for year in range(START_YEAR, END_YEAR + 1):
            path = os.path.join(TMP_DIR, f'era5_caribbean_{fname}_{year}.nc')
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
            for f in year_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            print(f"    Cleaned up {len(year_files)} year-files")
        except Exception as e:
            print(f"    CONCAT FAILED: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Final verification:")
    all_ok = True
    for fname in VARIABLE_GROUPS:
        out_path = os.path.join(OUTPUT_DIR, f'{fname}.nc')
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
