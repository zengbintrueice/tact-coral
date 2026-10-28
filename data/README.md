# Data — shipped derived datasets + raw-source reproduction chain

## What ships in this repository

| File | Size | Role |
|---|---|---|
| `scs_coral_bleaching_dataset_v9_1deg_raw.nc` | 5.5 MB | SCS source domain (train/val/test), 21×23 1° grid, weekly 2003–2020 |
| `gbr_coral_bleaching_dataset_v9_1deg_raw.nc` | 4.8 MB | GBR zero-shot target domain, 15×12 1° grid, weekly 1985–2020 |
| `china_bco_dmo_bleaching.csv` | 3.9 MB | GCBD independent validation (van Woesik & Kratochwill 2022, Sci. Data 9, 20 — via BCO-DMO) |
| `gcbd_global_bleaching_environmental.csv` | 7.2 MB | FULL global GCBD (27,025 sample events 1980–2020), built by `source_builders/download_gcbd_global.py` from the author's SQLite snapshot (BCO-DMO DOI 10.26008/1912/bco-dmo.773466.2); per-region availability counts in `results/gcbd_global_region_counts/` |
| `caribbean_coral_bleaching_dataset_v9_1deg_raw.nc` | 10.4 MB | CARIB zero-shot target domain (derived 1° dataset, same build chain as SCS/GBR) |
| `scs_causal_v2.nc` | 4.7 MB | **Primary SCS causal-v2 tensors** read by the experiment scripts (`dataset_file="scs_causal_v2.nc"`) |
| `gbr_causal_v2.nc` | 4.4 MB | **Primary GBR causal-v2 tensors** read by the experiment scripts (`dataset_file="gbr_causal_v2.nc"`) |
| `caribbean_causal_v2.nc` | 7.6 MB | CARIB causal v2 tensors (sha256-locked in `results/caribbean_zero_shot/caribbean_dataset_manifest_v2.json`) |

The NetCDF files listed above are the **derived datasets read by the experiments**,
not the raw sources. They are small enough to ship with the release, so the
repository is **fully self-contained for evaluation**: a fresh clone can re-run
every experiment without fetching any external data (the GCBD CSV was already
shipped; `paths.py` falls back to `./data` when `D:\piecf_crw_data` is absent).

## How the NetCDF files are produced (raw → derived)

The 1° datasets are **built from multi-terabyte public satellite sources**,
not downloaded as-is. Reproduction chain:

```
public raw sources (NOAA CRW 5km, ESA OC-CCI, CoastWatch/MODIS PAR, ERA5, ...)   [multi-TB]
        │  step 0: data/source_builders/download_*.py   (fetch raw files)
        ▼
raw source directory  new_sources/  (~180 GB on the author machine)
        │  step 1: build_v9_1deg_from_raw.py   → scs_coral_bleaching_dataset_v9_1deg_raw.nc
        │           build_gbr_1deg_from_raw.py → gbr_coral_bleaching_dataset_v9_1deg_raw.nc
        │           (DHW target via jensen_dhw.py: Jensen-aware clip + 84-day
        │            rolling sum at 5 km native resolution, then block-average to 1°)
        ▼
data/*.nc   (this repository)
        │  step 2: src/data/aligned_dataset.py  (13 dynamic + 5 static channels,
        │            SCS-train-only normalization)
        ▼
model input tensors
```

### `data/source_builders/` — the programs

| Program | Produces / fetches |
|---|---|
| `build_v9_1deg_from_raw.py` | SCS NC (21×23, derived channels) |
| `build_gbr_1deg_from_raw.py` | GBR NC (15×12) |
| `jensen_dhw.py` | DHW→target module (imported by both builders) |
| `download_crw_5km_all.py` / `download_crw_5km_gbr.py` | NOAA CRW 5 km daily SST / SSTA / Hotspot / DHW / BAA (ERDDAP) |
| `download_oc_cci_chlorophyll.py` | ESA OC-CCI v6 monthly chlorophyll (CEDA) |
| `download_modis_par_oceancolor.py` | NASA OceanColor MODIS-Aqua PAR |
| `download_coastwatch_par.py` / `download_coastwatch_par_gbr.py` | CoastWatch daily PAR (ERDDAP, alternative) |
| `download_era5_par_windstress.py` / `download_era5_gbr.py` | ERA5 monthly SST / wind / SSR (CDS — needs free account + API key) |
| `download_ccmp_wind_stress.py` | CCMP v3.1 wind stress (RSS) |

Source URLs are hard-coded at the top of each script (e.g.
`https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.nc`,
`https://oceandata.sci.gsfc.nasa.gov/ob/getfile/`,
`https://data.ceda.ac.uk/neodc/esacci/ocean_colour/...`,
`https://cds.climate.copernicus.eu/api/v2`,
`https://data.remss.com/ccmp/v03.1`).

> **Path caveat.** The build/download scripts keep the author-project paths
> (`PROJECT_ROOT` / `DNS` / `DATA_DIR` → `.../backups/piecf_a6_nophys_20260718/
> a6_xgb_experiment/data/new_sources/`, ASCII junction `D:\piecf_dns`). Before
> re-running them elsewhere, edit those variables to point at your own raw-source
> directory. Bathymetry (ETOPO1), salinity (WOA23), coastline and cyclone
> (IBTrACS) inputs are read from `new_sources/` inside the build scripts and have
> no dedicated download script in this folder.

## Where the raw sources live (author machine, NOT shipped)

- Physical location:
  `D:\曾斌\论文\海洋环境保护\PIECF\backups\piecf_a6_nophys_20260718\a6_xgb_experiment\data\new_sources\` (~180 GB)
- ASCII junction `D:\piecf_dns` → the directory above (netCDF4's C library
  cannot open `.nc` under Chinese paths on Windows).
- Approximate sizes: `crw_5km_sst/` 84 GB, `chlorophyll/` 54 GB,
  `crw_5km_gbr/` 32 GB, `coastwatch_par/` 8 GB, `bathymetry/` 1.3 GB; plus
  smaller GBR subsets, ERA5, salinity, cyclone, coastline, turbidity.

## Key invariants (do not change when re-running)

- Channel construction (13 dynamic + 5 static) is implemented in
  `src/data/aligned_dataset.py` (`_compute_derived_channels` for
  d_hotspot_dt / hotspot_cumsum_4w / hotspot_cumsum_12w / seasonal_cos).
- Normalization statistics are computed from the **SCS training period only**
  (`experiments/08_infrastructure/compute_train_normalization.py`,
  `results/infrastructure/train_normalization_stats.json`).
- Target: `y = clip(DHW/10, 0, 1)`; severe-event threshold `y >= 0.8`.
- GCBD matching: 0.5-degree spatial buffer, ±4-week temporal window,
  4-week forecast-lead alignment, SCS test years (2016-2019) only.

## Path resolution (self-contained-friendly)

- Set `PIECF_DATA_DIR` to point at your raw dataset directory.
- Fallback: `paths.py` checks `./data` **first**, then `D:\piecf_crw_data`
  (author-machine default). A fresh clone therefore uses the shipped NetCDF
  files in `./data` automatically and never depends on an external directory.
- GCBD CSV: set `GCBD_CSV` or place `china_bco_dmo_bleaching.csv` in `data/`
  (already shipped here).

> **Windows + non-ASCII path caveat.** netCDF4's C library cannot open `.nc`
> files whose path contains non-ASCII characters (e.g. the author's checkout
> under `D:\曾斌\...`). If the repository is cloned under such a path on
> Windows, the shipped `./data/*.nc` will fail to open; set `PIECF_DATA_DIR`
> to an ASCII directory containing the NetCDF files (or copy them there).
> The author machine's `D:\piecf_crw_data` is ASCII and is used by default.
