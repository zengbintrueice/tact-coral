# Audit — Cross-domain transfer protocol, normalization and leakage

**Date:** 2026-10-30
**Scope:** every statement in the manuscript/SI about cross-basin transfer,
source-domain normalization, the so-called "causal v2" datasets, the
train/validation/test splits and leakage control.
**No model retrained; no split, seed, threshold or matching rule changed.**

Evidence: `src/models/piecf_model.py`, `src/configs/config.py`,
`src/data/aligned_dataset.py`,
`experiments/08_infrastructure/compute_transfer_coordinate_train_reverse.py`,
`data/source_builders/build_causal_dataset_v2.py`,
`results/causal_dataset_v2/{dataset_manifest_v2.json,normalization_v2.json}`,
`results/infrastructure/train_normalization_stats.json`, and the shipped
checkpoints' `channel_norm` buffers.

---

## 1. Per-basin protocol (verified in code and manifests)

| Item | SCS | GBR | CARIB |
|---|---|---|---|
| Grid | 21 × 23 | 15 × 12 | 22 × 32 |
| Spatial bounds | 3.5–23.5°N, 100.5–122.5°E | 23.5–9.5°S, 142.5–153.5°E | 4.5–25.5°N, 90.5–58.5°W |
| Data coverage | 2003-01-05 – 2019-07-21 | 1985-04-01 – 2022-04-25 | 2003-01-05 – 2020-12-27 |
| Train | 2003–2010 (394 windows) | 2003–2010 (417) | 2003–2010 (394) |
| Validation **as executed** | 2011–2015 (261) | **2011–2015** | **2011–2015** |
| Validation written in the dataset manifest | 2011–2015 | **2014–2016** (156) | **2014–2016** (156) |
| Test | 2016–2019 (182) | 2017–2020 (209) | 2017–2020 (205) |
| Evaluation population (PFV) | 49,322 | 19,065 | 83,025 |
| Source of normalizer | SCS train 2003–2010 | **SCS train 2003–2010** | **SCS train 2003–2010** |

Window: input weeks t−23 … t (24 weeks); target y(t+4); a four-week gap
separates the last input week from the target.

---

## 2. Answers to the five focus questions

### 2.1 Do all models use SCS-train normalization?

**Yes.** `PIECFModel.__init__` always builds
`ChannelNormalizer(SCS_CHANNEL_MEAN, SCS_CHANNEL_STD)`
(`src/models/piecf_model.py:192–193`). The values are stored as buffers, so
they travel with the checkpoint and are restored on load. Verified directly in
the shipped weights: `model_final_Btail3/model.pt`,
`physics_anchored/model.pt`, and the four reverse-source checkpoints
(`weights/GBR/tscf_model.pt`, `weights/CARIB/tscf_model.pt`,
`weights/GBR/tsar_model.pt`, `weights/CARIB/tsar_model.pt`) all carry the same
`channel_norm.mean/std` (e.g. SST mean 27.97262; SST-anomaly mean −0.10284).
**No source basin uses its own normalization statistics.**

### 2.2 "SCS-style normalizer baked into every model" vs "GBR/CARIB use their own datasets"

Both are true and refer to different objects:

- **Dataset**: each source model is trained on its own basin's file
  (`scs_causal_v2.nc`, `gbr_causal_v2.nc`, `caribbean_causal_v2.nc`), built
  with the same causal rules.
- **Normalizer**: a single SCS-derived scaler is applied to all datasets and
  all models.

The two statements are consistent, but the manuscript did not spell out the
consequence: the GBR- and CARIB-sourced models are normalized with **SCS**
statistics, and SCS is one of their targets.

### 2.3 Time and space ranges

See the table in §1. The reverse-source trainer
(`compute_transfer_coordinate_train_reverse.py:44–46`) sets
`train_years=(2003,2010)` and `val_years=(2011,2015)` for **every** source and
`test_years` per source. The dataset manifests list 2014–2016 as the
GBR/CARIB validation window; the executed training overrode that with
2011–2015. Test windows are identical in both (2017–2020).

### 2.4 Are normalization parameters estimated on training data only and frozen?

**Yes, with one documentation mismatch.**

- Estimated from SCS training weeks 2003–2010 only (417 weeks).
- Stored as checkpoint buffers; no script recomputes them for validation,
  test or transfer. No scaler is fitted on any target's evaluation data.
- **Mismatch:** `results/causal_dataset_v2/normalization_v2.json` is described
  as the frozen normalizer, but it differs from the applied values in one
  channel: `sst_anomaly` mean −0.003835 (v2) vs −0.10284 (applied), std
  0.45767 vs 0.48453. The applied values are those in
  `src/configs/config.py` and `results/infrastructure/train_normalization_stats.json`,
  which match every checkpoint. **No training or evaluation script reads
  `normalization_v2.json`.**

### 2.5 Leakage assessment

| Channel / step | Rule | Verdict |
|---|---|---|
| Target y(t+4) | trailing 84-day DHW at 5 km, HotSpot ≥ 1 °C, 5 km→1° aggregation **before** thresholding | causal |
| Monthly fields | month-end availability, past-only forward fill, no interpolation | causal |
| Climatology | ERA5 monthly climatology from months with year ≤ 2010, both domains | causal |
| Derived channels | `d_hotspot_dt`, `cumsum_4w`, `cumsum_12w` strictly trailing | causal |
| Input window | t−23 … t | causal |
| Missing labels | NaN kept, never zero-filled; finite-ocean masks | causal |
| Weights / thresholds | no target-domain fitting; τ = 0.90 pinned | causal |
| **Normalization for reverse-source models** | SCS-train statistics, and SCS is a target | **not test-set leakage, but asymmetric** |

The only material issue is the last row. For a GBR- or CARIB-sourced model,
the SCS target is normalised with SCS-train statistics (2003–2010), which are
disjoint from the SCS test years (2016–2019). This is not leakage of test
labels or test features, but it means the SCS target is normalised
in-distribution while the GBR/CARIB targets are not. Cross-direction
comparisons should carry that caveat.

---

## 3. Discrepancies found (code vs documentation)

| # | Source A | Source B | Resolution |
|---|---|---|---|
| D1 | Reverse-source trainer: val 2011–2015 | Dataset manifest: GBR/CARIB val 2014–2016 | Code is authoritative; manuscript now states 2011–2015 and the SI notes the manifest value was not used |
| D2 | Applied normalizer = `config.py` / `train_normalization_stats.json` | `normalization_v2.json` (sst_anomaly differs) | Documented in `results/causal_dataset_v2/NORMALIZATION_NOTE.md`; paper text (SCS-train statistics) is correct |
| D3 | SI Note R1: "normalization is source-only" | Code: SCS statistics for every model | Corrected to "fixed to the SCS training period for every model" |
| D4 | Main text: "their own causal v2 datasets" | Internal project term | Replaced by an operational description |

---

## 4. Manuscript / SI changes made

| Location | Change |
|---|---|
| Methods, Data and protocol | Added per-basin grids, spatial bounds and source-training splits; stated that the SCS-derived normalizer is used for every basin and source model, and that no scaler is fitted on validation/test/transfer data |
| Methods, Six-direction transfer matrix | Replaced "their own causal v2 datasets" with the operational causal rules; stated explicitly that no target-domain weights, thresholds or scalers are tuned |
| SI R7 | Added the normalization asymmetry (SCS is a target for reverse-source models; SCS-train statistics are disjoint from the SCS test years but normalise that target in-distribution) and the executed split years |
| SI R1 | "normalization is source-only" → "normalization statistics are fixed to the SCS training period for every model" |
| SI S1 | Unchanged (the SCS-train-only bullet is correct) |

---

## 5. What would require re-running experiments

1. **Per-source normalizer sensitivity (recommended if a reviewer challenges
   the matrix symmetry).** Retrain the GBR-source and CARIB-source TSCF/TSAR
   models with each source basin's own 2003–2010 normalizer, then recompute
   the six-direction matrix. This directly tests whether the SCS-as-target
   normalization asymmetry affects the pair-specific pattern. Not run here;
   it requires four retrainings and a re-run of the matrix evaluation.
2. **Temporal-purge test for reverse-source models.** The existing purge
   sensitivity (SI S2) covers the SCS/GBR single-route evaluation, not the
   reverse-source models. A purge check on GBR/CARIB-source models would
   strengthen the matrix if requested.
3. **None required for leakage.** The data-construction rules are causal, the
   normalizer uses SCS training weeks only, and no target labels enter
   training. The paper's leakage claim is supported by the checks above.

---

## 6. Items requiring author confirmation

1. Whether the dataset manifest (GBR/CARIB val 2014–2016) or the executed
   script (2011–2015) reflects the intended protocol. The manuscript now
   reports the executed 2011–2015.
2. Whether `normalization_v2.json` should be regenerated to match the applied
   config values, or retained as a record of the causal-v2 dataset's own
   statistics. It is not used by any model.
3. Whether to run the per-source normalizer sensitivity experiment (§5.1).

---

## 7. Verdict

**Verified:** all models use the SCS-train normalizer; the data-construction
pipeline is causal; no target-domain weights or thresholds are fitted; the
executed splits are disjoint across train/val/test.

**Corrected in the text:** the "causal v2" terminology, the missing per-basin
split/spatial details, the "source-only" normalization phrasing, and the
undisclosed SCS-as-target normalization asymmetry.

**Not claimed:** that the protocol is fully symmetric across the six
directions. The asymmetry is now stated, and a sensitivity experiment is
listed as optional.
