# Leakage Audit — Transfer Matrix + Transferable Thermal Coordinate Stability

**Experiment:** all-to-all zero-shot transfer across SCS / GBR / CARIB for
models TSCF, TSAR (anchored residual), PersA_DHW, XGB_THERMAL, plus the
PersA_DHW–future-risk coordinate stability analysis.
**Outputs:** `results/transfer_coordinate/`
**Sources (read-only, never modified):** the frozen evidence record set
`results/physics_anchored/structural_ablation/rec_structural_ablation.npz`,
frozen checkpoints, and the causal-v2 datasets. New files are written only
under `results/transfer_coordinate/`.

---

## Audit checklist (all PASS/FAIL against code + provenance)

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| A1 | **Source-only training.** Every TSCF/TSAR model used for a direction is trained on the SOURCE basin's own 2003–2010 train only. | PASS | GBR/CARIB reverse checkpoints trained on their own train split (`create_scs_dataloaders(dataset_file=…)`); SCS uses the frozen SCS-sourced weights |
| A2 | **No target training / fine-tuning.** The target basin never appears in any model's weights or loss. | PASS | target used only for scoring; no target basin in any training set |
| A3 | **No target threshold selection.** τ = 0.90 pinned a priori from the source protocol, never selected on any target. | PASS | τ fixed = 0.90 in all scoring |
| A4 | **No target scaler fitting.** The model's channel normalizer is the fixed SCS-style mean/std constant baked into every model (`src/models/piecf_model.py:193`), applied identically to all input basins; no target-derived statistic enters any input. | PASS | hardcoded `SCS_CHANNEL_MEAN/STD`; datasets load raw features |
| A5 | **Causal target.** Targets are y(t+4); every feature at issue t is built from information at or before t (trailing cumulative HotSpot windows; four-week lead). | PASS | causal-v2 construction; trailing 4-/12-week cumsum |
| A6 | **No future feature contamination.** Features end at issue t; target t+4 never enters the feature tensor. | PASS | causal-v2 audit |
| A7 | **PersA is source-agnostic (no fitting).** PersA_DHW = clip(DHW/10,0,1) is a fixed physical mapping; its cross-basin rows are training-free and identical per target basin. | PASS | definition; no trainable parameters |
| A8 | **Same mask / population.** Metrics computed on the same ocean-masked, finite-y(t+4) populations (SCS 49,322 / GBR 19,065 / CARIB 83,025) used by the locked evaluations. | PASS | n confirmed per basin; matches evidence record set |
| A9 | **Record IDs + hashes.** Data, checkpoint and record-set hashes recorded. | PASS | `provenance_tier1.json` + reverse checkpoint md5 below |
| A10 | **Formal results untouched.** No file under `results/` outside `results/transfer_coordinate/` is modified. | PASS | all new files written under `results/transfer_coordinate/` |

---

## Provenance

- **Data files (md5):** SCS `981e0cd9…`, GBR `cd512b94…`, CARIB `8218e956…` (`provenance_tier1.json`).
- **Frozen SCS checkpoints (md5):** TSCF `7cdbca26cae03fa53be53210940e8ce1`, TSAR `ae86e017bc5ba634b0c68e5d13b4f9f0`.
- **Reverse-source checkpoints (md5), seed 42, D1/a6 protocol:**
  - `weights/GBR/tscf_model.pt` `1ed502ff…`
  - `weights/GBR/tsar_model.pt` `f51591ef…`
  - `weights/CARIB/tscf_model.pt` `ce48a7d2…`
  - `weights/CARIB/tsar_model.pt` `c4b99b2a…`
- **Record-set IDs:** SCS→GBR/CARIB reuse `rec_structural_ablation.npz` (issue-index t mapped to year); reverse rows generated from the shared test populations with the same masking.
- **Scaler provenance:** the only normalization is the hardcoded `SCS_CHANNEL_MEAN/STD` inside every model checkpoint — identical across all sources and targets (no per-basin scaler fit), consistent with the existing SCS→GBR/CARIB zero-shot design.

---

## Design notes (how leakage is prevented)

1. Each 6-direction evaluation reuses `lead = 4 wk`, `history = 24 wk`,
   `τ = 0.90`, the same severe label `y(t+4) ≥ 0.8`, and the same metric set
   (ROC/PR-CSI/POD/FAR/F1/Brier/ECE from the evidence pipeline).
2. Transfer degradation Δ = (target metric) − (source metric) for
   TSCF/TSAR/PersA, with a week-block bootstrap 95% CI on the difference
   (seed 42). Across-basin populations are disjoint, so the CI is a
   difference-of-independent week-block estimates described as a bootstrap
   CI of the degradation, not a record-level pair.
3. Coordinate stability (PersA vs y(t+4)) is computed per basin and per
   year with the same week-block scheme.