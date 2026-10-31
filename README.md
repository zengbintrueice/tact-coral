# TACT — Thermal-stress Anchoring for Cross-basin Transfer

Code, data, frozen weights and results accompanying the manuscript
*Thermal-stress anchoring reveals transferable and basin-specific coral thermal risk*.

GitHub repository: **tact-coral**.

This repository is **self-contained**: every experiment can be re-run on a
fresh clone without downloading anything external. All numbers reported in
the manuscript and the Supplementary Information are read from the result
files shipped here (nothing hand-entered), and every figure is regenerated
by a script in `scripts/figures/`.

> **Model naming.** The manuscript uses **TSCF** (frozen contextual
> spatiotemporal model), **TSAR** (thermal-stress anchored residual = the
> gated bounded residual around the anchor) and **PersA** (the physical
> persistence anchor `clip(DHW/10, 0, 1)`). Historical/internal identifiers
> (`PIECF`, `a6`, `physics_anchored`, `Btail3`) still appear in file names,
> class names and JSON keys; they denote the same objects. Wherever this
> repository says `physics_anchored`, it means TSAR; `model_final_Btail3`
> is the frozen TSCF checkpoint.

---

## 1. Repository layout

```
├── README.md                     ← this file
├── LICENSE                       (MIT for code; data attributions inside)
├── MANIFEST.sha256               sha256 of every file in the archived release
├── EVIDENCE_LOCKS.md             MD5 of frozen weights / locked artifacts
├── requirements.txt
├── paths.py                      shared path config used by every script
├── pfv_utils.py                  pure-forecasting-verification (PFV) helpers
├── ablate_a_dh_stopgrad.py       model definitions (single-head TSCF; legacy a5/a6 ablation)
├── src/                          dataset loader + model package
│   ├── data/aligned_dataset.py   13 dynamic + 5 static channels, SCS-train-only normalization
│   └── models/piecf_model.py     ConvLSTM + temporal attention backbone
├── data/
│   ├── README.md                 raw→derived reproduction chain (see §3)
│   ├── scs_coral_bleaching_dataset_v9_1deg_raw.nc     SCS source domain (shipped)
│   ├── gbr_coral_bleaching_dataset_v9_1deg_raw.nc     GBR target domain (shipped)
│   ├── gcbd_global_bleaching_environmental.csv        GCBD global extract (shipped)
│   ├── china_bco_dmo_bleaching.csv                    GCBD via BCO-DMO (shipped)
│   ├── source_builders/          download_* + build_* programs (see §3.2–3.3)
│   └── raw_sources/              GCBD SQLite snapshot + author repo archive
├── experiments/                  entry points grouped by paper theme (01–09)
├── scripts/
│   ├── figures/                  every manuscript figure, result-file-driven
│   ├── reproduce_main.sh         one command for all main-text results
│   ├── reproduce_supplementary.sh
│   ├── reproduce_all.sh
│   └── smoke_test.py             no-GPU sanity check + evidence-lock check
├── results/
│   ├── figures/                  all manuscript figures (main + supplementary)
│   ├── model_final_Btail3/       frozen TSCF checkpoint (seed 42)
│   ├── physics_anchored/         frozen TSAR checkpoints (seeds 0–3, 42)
│   ├── transfer_coordinate/      six-direction transfer matrix, seed scan, regime controls
│   ├── residual_hierarchy/       pre-specified training-free hierarchy (H0–H3)
│   ├── conditional_residual/     matched-DHW residual-information test
│   ├── dhw_conditioned_susceptibility/   basin susceptibility at matched current DHW
│   ├── residual_mechanism_decomposition/ nested ΔR² mechanism decomposition
│   ├── mechanism_discrimination/ H1-vs-H2 evidence matrix
│   ├── counterfactual_response_surface/  frozen-model counterfactual probe
│   ├── response_surface_susceptibility_mechanism/  unified gated pipeline (main Fig. 8)
│   ├── xgb_transfer_control/     XGBoost architecture-independent control
│   ├── invariant_predictor_screen/       ICP-lite channel screen (Fig. 6e)
│   ├── gcbd_*, step9_gcbd_validation/    GCBD ecological validation (Fig. 10)
│   ├── caribbean_zero_shot/      Caribbean zero-shot predictions + protocol
│   ├── step2_persistence_v2/     persistence baseline (PersA)
│   ├── step5–step8, thermal_attribution, domain_shift, calibration, ...
│   └── <each dir>/AUDIT.md + provenance_tier1.json  producing script, hashes, QC
└── doc/
    ├── main_manuscript.md               (markdown is the tracked source)
    ├── supplementary_information.md     (markdown is the tracked source)
    ├── nc_reference.docx                (pandoc reference style, tracked)
    └── *.docx / *.pdf                   (compiled; shipped in the archive,
                                          rebuilt with pandoc — see below)
```

## 2. Installation

```bash
python -m venv venv && source venv/bin/activate    # Python 3.11
pip install -r requirements.txt
python scripts/smoke_test.py            # <1 min, CPU-only, no datasets touched
python scripts/smoke_test.py --locks    # verifies EVIDENCE_LOCKS.md MD5s
```

GPU (CUDA 12.x) is required only for retraining (`experiments/01_training/`);
every evaluation script runs on CPU.

## 3. Data

### 3.1 What ships (self-contained evaluation)

| File | Size | Role |
|---|---|---|
| `data/scs_coral_bleaching_dataset_v9_1deg_raw.nc` | 5.5 MB | SCS source domain, 21×23 1° grid, weekly 2003–2020, 13 dynamic channels + DHW target |
| `data/gbr_coral_bleaching_dataset_v9_1deg_raw.nc` | 4.8 MB | GBR zero-shot target, 15×12 1° grid, weekly 1985–2020 |
| `data/gcbd_global_bleaching_environmental.csv` | 7.2 MB | Global GCBD, 27,025 sample events 1980–2020 |
| `data/china_bco_dmo_bleaching.csv` | 3.9 MB | GCBD SCS matching set (BCO-DMO distribution) |
| `data/caribbean_coral_bleaching_dataset_v9_1deg_raw.nc` | 10.4 MB | CARIB zero-shot target domain (same build chain as SCS/GBR) |
| `data/caribbean_causal_v2.nc` | 7.6 MB | CARIB causal v2 tensors; sha256 `84138009bff596ad...` matches `results/caribbean_zero_shot/caribbean_dataset_manifest_v2.json` |

`paths.py` resolves the data directory in this order: (1) env var
`PIECF_DATA_DIR`, (2) `./data` (this repository), (3) `D:\piecf_crw_data`
(author machine). A fresh clone uses (3) automatically, so **all
evaluations run out of the box**.

### 3.2 Raw sources and how to re-download them

The two `.nc` files above are *derived* products built from public
multi-terabyte satellite archives. The complete raw archive (~180 GB) is **not
included in this repository** (size), but the author's verbatim copy is
publicly mirrored for download:

> **Raw-source mirror (Baidu netdisk):** <https://pan.baidu.com/s/1K4ar3uvUA9eFZT6gIqNBaQ?pwd=4rkn>
>
> The mirror is provided for research reproducibility of this manuscript.
> All raw sources are redistributed under the terms of their original
> providers - NOAA Coral Reef Watch and CoastWatch products are U.S.
> Government public domain; ERA5 is Copernicus C3S data (redistribution
> permitted, acknowledgement required: "Contains modified Copernicus
> Climate Change Service information"); ESA OC-CCI and NASA OceanColor
> products are open data with attribution; GCBD is CC-BY 4.0. Please cite
> the original providers (see the table below and §3.5) rather than the
> mirror; the provider archives remain the authoritative sources and may
> be more current than this snapshot.

| Raw source | Programs in `data/source_builders/` | Access point |
|---|---|---|
| NOAA Coral Reef Watch 5 km daily SST / SSTA / HotSpot / DHW / BAA (CoralTemp v3.1) | `download_crw_5km_all.py`, `download_crw_5km_gbr.py`, `download_crw_5km_caribbean.py`, `download_crw_5km_scs_2021_2024.py` | NOAA CRW ERDDAP (`https://coastwatch.pfeg.noaa.gov/erddap/`) |
| ESA OC-CCI v6 monthly chlorophyll | `download_oc_cci_chlorophyll.py` | CEDA archive |
| MODIS-Aqua PAR | `download_modis_par_oceancolor.py`, `download_modis_par_caribbean.py` | NASA OceanColor Web |
| CoastWatch daily PAR (alternative) | `download_coastwatch_par.py`, `download_coastwatch_par_gbr.py` | CoastWatch ERDDAP |
| ERA5 wind / shortwave radiation / PAR | `download_era5_gbr.py`, `download_era5_caribbean.py`, `download_era5_par_windstress.py` | Copernicus C3S (ADS/CDS) |
| CCMP wind stress | `download_ccmp_wind_stress.py` | PO.DAAC |
| Global Coral Bleaching Database | `download_gcbd_global.py` | BCO-DMO DOI 10.26008/1912/bco-dmo.773466.2; raw SQLite snapshot shipped in `data/raw_sources/` |

Each script records the exact ERDDAP/CEDA/NASA dataset ID, variables, spatial
window and temporal range in its header, so the raw fetch is fully specified.

The build/download scripts locate the raw workspace through environment
variables so they run unchanged on any machine: set `PIECF_DNS_DIR` (raw
pre-processed satellite/ERA5 workspace, default `D:\piecf_dns` on the
author machine) and `PIECF_RAW_DIR` (download target root, default
`D:\projects\PINN-OA`) before running them; evaluation experiments never
need these - only `data/` (shipped).

### 3.3 Raw → derived build chain

```
public raw sources (NOAA CRW 5 km, ESA OC-CCI, MODIS/CoastWatch PAR, ERA5)   [~180 GB]
        │  step 0: data/source_builders/download_*.py          (fetch raw files)
        ▼
raw source directory  (author machine: D:\piecf_crw_data, ~180 GB)
        │  step 1: build_v9_1deg_from_raw.py        → scs_..._v9_1deg_raw.nc
        │           build_gbr_1deg_from_raw.py      → gbr_..._v9_1deg_raw.nc
        │           build_caribbean_1deg_from_raw.py→ caribbean_..._v9_1deg_raw.nc
        │           (DHW target via jensen_dhw.py: Jensen-aware clip + 84-day
        │            rolling sum at 5 km native resolution, then 1° block-average)
        │  step 1b: build_causal_dataset_v2.py / build_causal_dataset_caribbean_v2.py
        │           → causal v2 tensors + availability audit (30/30 checks PASS)
        ▼
data/*.nc   (the two SCS/GBR files are shipped here)
        │  step 2: src/data/aligned_dataset.py
        │           13 dynamic channels + DHW coordinate, SCS-train-only
        │           normalization statistics (frozen, reused in every domain)
        ▼
model input tensors  →  experiments/*
```

### 3.4 Caribbean data

The Caribbean 1° dataset (`caribbean_coral_bleaching_dataset_v9_1deg_raw.nc`)
and its causal v2 tensors (`caribbean_causal_v2.nc`) **ship with this
repository**. The causal v2 file is hash-locked: its sha256 is recorded in
`results/caribbean_zero_shot/caribbean_dataset_manifest_v2.json`, so the
exact tensors that produced every Caribbean number in the manuscript are
verifiable. Rebuilding them from raw sources uses the `caribbean`
programs in `data/source_builders/` (§3.2–3.3).

### 3.5 GCBD licensing

GCBD (van Woesik & Kratochwill, *Scientific Data* 9, 20, 2022) is CC-BY 4.0;
the extracts above and the raw SQLite snapshot (`data/raw_sources/`) keep the
original attribution. NOAA CRW products are U.S. Government public domain.

## 4. Frozen weights and evidence locks

| Artifact | Path | Role |
|---|---|---|
| TSCF checkpoint (seed 42) | `results/model_final_Btail3/model.pt` | frozen contextual model used everywhere |
| TSAR checkpoints | `results/physics_anchored/model_seed_{0,1,2,3,42}.pt` | anchored gated residual, seed scan |
| structural-ablation predictions | `results/physics_anchored/structural_ablation/` | variants A–D of Fig. 5 |
| persistence predictions | `results/step2_persistence_v2/` | PersA baseline |
| six-direction checkpoints | `results/transfer_coordinate/weights/` | reverse-source models |

MD5 values: `EVIDENCE_LOCKS.md`. The checkpoints are the *evidence-locked
reference realisation*; all paper numbers are generated from them. Retraining
is optional (§6, `experiments/01_training/`).

## 5. Quickstart: verify the headline numbers (~10 min, CPU)

```bash
python experiments/02_scs_benchmark/scs_threshold_calibration.py
python experiments/03_gbr_transfer/gbr_zeroshot_evaluation.py
python experiments/08_infrastructure/pfv_evaluation_3level.py
```

Targets (all auto-read from the shipped result files):

| Claim | Value | Where |
|---|---|---|
| Frozen TSCF in-source (SCS test) ROC / PR / CSI@0.90 | 0.986 / 0.546 / 0.337 | `results/transfer_coordinate/transfer_matrix.csv` (diagonal) |
| PersA anchor across basins | 0.982 / 0.959 / 0.979 | `results/transfer_coordinate/transfer_matrix.csv` |
| Six-direction TSCF zero-shot | SCS→GBR 0.962; CARIB→SCS 0.981; CARIB→GBR 0.906; SCS→CARIB 0.610; GBR→CARIB 0.635; GBR→SCS 0.668 | `results/transfer_coordinate/transfer_matrix.csv` + `reverse_matrix.csv` |
| Six-direction TSAR | 0.959–0.984 in every cell, |ΔROC| ≤ 0.021 | `results/transfer_coordinate/transfer_matrix.csv` |
| Seed fragility of TSCF (CARIB→GBR, seeds 0/1/2) | 0.915 / 0.666 / 0.676; TSAR sd ≤ 0.003 | `results/transfer_coordinate/seed_scan/` |
| XGBoost CARIB ROC (FULL/THERMAL/NO-THERMAL) | 0.880 / 0.890 / 0.806 | `results/xgb_transfer_control/metrics.json` |
| GCBD severe-bleaching AUROC (cluster-aware) | 0.876 [0.808, 0.936], rate ratio 35.2 | `results/step9_gcbd_validation/` |
| Matched-DHW severe rates GBR/CARIB/SCS | 3.5% / 1.6% / 0.9% | `results/dhw_conditioned_susceptibility/summary.json` |
| Conditional residual increment (GBR / CARIB) | +0.047 / +0.023, CIs exclude zero | `results/conditional_residual/summary.json` |
| Mechanism decomposition (trajectory first) | M1−M0 ΔR² +0.274/+0.137/+0.297 | `results/residual_mechanism_decomposition/` |
| Regime controls (CARIB-source on SCS 2016/2019) | ROC 0.971 / 0.975 | `results/transfer_coordinate/regime_test/` |

## 6. Full reproduction

```bash
bash scripts/reproduce_main.sh            # main-text results
bash scripts/reproduce_supplementary.sh   # supplementary results
bash scripts/reproduce_all.sh             # both
```

`experiments/` is grouped by paper theme; each script header records
Experiment / Purpose / Output / Figure-Table / seed:

| Directory | Paper scope | Main outputs |
|---|---|---|
| `01_training/` | optional retraining (TSCF via `train_severe_classifier.py`; TSAR via `train_physics_anchored.py`; reverse-source seeds 0–3; XGBoost) | checkpoints under `results/` |
| `02_scs_benchmark/` | SCS in-source benchmark; transfer-matrix diagonal | Fig. 1–3 diagonal, SI Tables S5–S6 |
| `03_gbr_transfer/` | zero-shot transfer + threshold adaptation + domain-shift diagnosis | Figs. 2–3, SI S9 |
| `04_structural_ablation/` | PersA / ungated / gated / TSCF variants | Fig. 5, SI Table S4 |
| `05_thermal_attribution/` | thermal-only / thermal-removal / permutation / XGBoost controls | Fig. 6(d), SI S3–S5 |
| `06_calibration/` | calibration curves | SI Figs. S6–S7 |
| `07_gcbd_validation/` | GCBD independent validation | Fig. 10, SI R5 |
| `08_infrastructure/` | PFV 3-level evaluation, normalization, dual-output maps | Methods |

Newer analyses (six-direction matrix, seed scan, residual hierarchy,
mechanism chain, susceptibility, counterfactual, climatology control) record
their producing script, input hashes and QC verdicts in
`results/<dir>/AUDIT.md` + `provenance_tier1.json`.

## 7. Figure regeneration

Every figure script reads only shipped result files (no hard-coded numbers):

```bash
python scripts/figures/generate_all_figures.py        # all figures
python scripts/figures/fig2_transfer_matrix.py        # single figure
```

| Figure | Script | Source file |
|---|---|---|
| Fig. 1 framework | `fig1_framework.py` | protocol metadata |
| Fig. 2 transfer matrix | `fig2_transfer_matrix.py` | `results/transfer_coordinate/` |
| Fig. 3 transfer boundary | `fig3_transfer_boundary.py` | `results/gbr_transfer/`, `results/domain_shift/` |
| Fig. 4 anchor | `fig5_persa_anchor.py` | `results/step2_persistence_v2/` |
| Fig. 5 gated residual | `fig6_gated_residual.py` | `results/structural_ablation/` |
| Fig. 6 XGBoost (a–e) | `fig_s_x_xgb_transfer.py` | `results/xgb_transfer_control/`, `results/invariant_predictor_screen/` |
| Fig. 7 diagnostics | `fig4_crw_diagnostics.py` | `results/domain_shift/`, `results/thermal_attribution/` |
| Fig. 8 unified pipeline | `build_supplementary.py` (R13 module) | `results/response_surface_susceptibility_mechanism/` |
| Fig. 9 counterfactual | `scripts/figures/edfig_s9_spatial_maps.py` (+ counterfactual module) | `results/counterfactual_response_surface/` |
| Fig. 10 GCBD | `fig7_gcbd_validation.py` | `results/step9_gcbd_validation/` |
| SI S1–S13, R-figs | `fig_supplementary.py`, `fig_r*.py` | respective `results/` dirs |

## 8. Provenance and audits

- `results/<dir>/AUDIT.md` — what was run, on what inputs, with which seed;
  QC verdicts (PASS/FAIL) for every leakage and protocol check.
- `results/<dir>/provenance_tier1.json` — machine-readable provenance.
- `results/causal_dataset_v2/` — causal construction audit (30/30 PASS).
- `MANIFEST.sha256` — sha256 of every file in the archived release; verify with
  `sha256sum -c MANIFEST.sha256` inside the unpacked archive. A git checkout
  omits the compiled DOCX/PDF (see `.gitignore`); rebuild them with:
  `pandoc doc/main_manuscript.md -o doc/main_manuscript.docx --resource-path=doc --reference-doc doc/nc_reference.docx`
  (and likewise for `supplementary_information.md`).
- `EVIDENCE_LOCKS.md` — MD5 of frozen weights.

## 9. Known limitations of this release

1. Raw satellite archives (~180 GB) are not part of the repository; they are
   publicly mirrored for download (see §3.2), and the CRW/ERA5/OC-CCI
   download scripts fully specify the original sources.
2. The Caribbean dataset ships in full (§3.4), hash-locked against the
   recorded sha256.
3. Retraining is stochastic (see the seed-fragility analysis in the paper);
   the frozen checkpoints above are the reference realisation for all
   reported numbers.

## 10. Citation

Manuscript: *Thermal-stress anchoring reveals transferable and basin-specific
coral thermal risk* (under review).

Data: van Woesik, R. & Kratochwill, C. A global coral-bleaching database,
1980–2020. *Scientific Data* 9, 20 (2022) (CC-BY 4.0). NOAA Coral Reef Watch
CoralTemp / bleaching heat-stress product suite v3.1 (public domain).
