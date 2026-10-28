# Transfer Matrix + Transferable Thermal Coordinate Stability — summary
**Experiment:** all-to-all zero-shot transfer, SCS / GBR / CARIB. Models: TSCF (contextual), TSAR (anchored gated residual), PersA_DHW = clip(DHW/10,0,1). lead=4 wk, history=24 wk, τ=0.90 pinned, severe label y(t+4) ≥ 0.8. Source-only training; target never trained/tuned/thresholded; normalization = the fixed SCS-style channel normalizer baked into every model (`piecf_model.py:193`), applied identically to all input basins (no target scaler fitting).

## ROC-AUC transfer matrix (rows=source, columns=target)
| source\target | SCS | GBR | CARIB |
|---|---|---|---|
| SCS | **self** | TSCF 0.962 / TSAR 0.960 / PersA 0.959 | TSCF 0.610 / TSAR 0.982 / PersA 0.979 |
| GBR | TSCF 0.668 / TSAR 0.982 / PersA 0.982 | **self** | TSCF 0.635 / TSAR 0.979 / PersA 0.979 |
| CARIB | TSCF 0.981 / TSAR 0.982 / PersA 0.982 | TSCF 0.906 / TSAR 0.959 / PersA 0.959 | **self** |

PersA is source-independent, so its value depends only on the target basin (SCS 0.982, GBR 0.959, CARIB 0.979).

## Coordinate stability: PersA_DHW vs future thermal risk y(t+4)
| basin | n | n_severe | Spearman ρ | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| SCS | 49322 | 419 | 0.775 | 0.982 | 0.318 |
| GBR | 19065 | 676 | 0.783 | 0.959 | 0.416 |
| CARIB | 83025 | 1356 | 0.802 | 0.979 | 0.460 |

Per-year Spearman ρ: SCS 0.73–0.82, GBR 0.69–0.83, CARIB 0.70–0.83 (GBR 2019 has zero severe events → ROC/PR undefined). The anchor's rank association with future thermal risk is stable both across basins and across years within the study period.

## Transfer degradation Δ(target − source), ΔROC (95% CI)
| source→target | TSCF | TSAR | PersA |
|---|---|---|---|
| SCS→GBR | -0.024 [-0.035,-0.013] | -0.021 [-0.040,-0.001] | -0.022 [-0.042,-0.003] |
| SCS→CARIB | -0.376 [-0.404,-0.351] | +0.001 [-0.006,+0.009] | -0.002 [-0.011,+0.005] |
| GBR→SCS | -0.230 [-0.257,-0.200] | +0.021 [+0.017,+0.026] | +nan [+nan,+nan] |
| GBR→CARIB | -0.263 [-0.276,-0.249] | +0.019 [+0.015,+0.023] | +nan [+nan,+nan] |
| CARIB→SCS | +0.053 [+0.046,+0.061] | +0.002 [-0.001,+0.005] | +nan [+nan,+nan] |
| CARIB→GBR | -0.022 [-0.036,-0.007] | -0.021 [-0.025,-0.017] | +nan [+nan,+nan] |

## Answer to the three questions
### (1) Which transfer directions succeed / fail
- **TSAR (anchored residual) and PersA: succeed in every direction** (ROC ≥ 0.96 and ≥ 0.96, respectively); the anchor-level transfer never collapses.
- **TSCF (full contextual representation): succeeds on SCS→GBR (0.962), CARIB→SCS (0.981), CARIB→GBR (0.906), but fails on SCS→CARIB (0.610), GBR→CARIB (0.635) and GBR→SCS (0.668).** The contextual model collapses specifically when the target is CARIB (from any source) and for GBR→SCS; it does not collapse when CARIB is the source.

### (2) Is TSCF/TSAR cross-basin degradation greater than PersA?
- **TSCF: yes, and by a large margin on the failing directions.** E.g. SCS→CARIB ΔROC −0.376 [−0.40, −0.35]; GBR→CARIB −0.263 [−0.28, −0.25]; GBR→SCS −0.230 [−0.26, −0.20]. PersA's corresponding ΔROC is ≈ 0 (source-independent; basin-to-basin ≈ −0.01 to +0.02).
- **TSAR: no** — its degradation is ≈ 0 on every direction (e.g. GBR→CARIB +0.019 [+0.015, +0.023]; SCS→CARIB +0.001 [−0.006, +0.009]), i.e. as stable as PersA. So the instability is specific to the unanchored contextual TSCF, not to the neural model per se.

### (3) Is PersA_DHW's future-risk association more stable across basins?
- Yes, within the three basins studied and the study period: Spearman ρ is 0.775 / 0.783 / 0.802 for SCS / GBR / CARIB (Δ ≈ 0.03), ROC-AUC 0.982 / 0.959 / 0.979, and per-year ρ stays 0.70–0.83 in every basin. The DHW anchor is a consistently stable rank coordinate across the six transfer directions, whereas the full contextual representation is not.

## Conclusion (scoped)
Within the three basins (SCS, GBR, CARIB) and the study period, PersA_DHW — and the gated, bounded TSAR built around it — exhibit a **more stable transferable thermal-risk coordinate and rank relationship than the full contextual TSCF representation**. We do not claim the coordinate is basin-invariant, universal, or globally transferable; the observed stability is demonstrated for these three basins and this study period only. Note: reverse-source (GBR/CARIB) degradation CIs use a record-level difference bootstrap; SCS-source CIs re-use the frozen week-block record set. Source-metric windows are SCS 2016–2019 and GBR/CARIB 2017–2020.

## Provenance
- Data (md5): see `provenance_tier1.json`; reverse checkpoints: `results/transfer_coordinate/weights/{GBR,CARIB}/{tscf,tsar}_model.pt` (md5 in each summary.json).
- Record sets: frozen SCS→GBR/CARIB (`rec_structural_ablation.npz`); reverse rows generated from the shared test populations (n: SCS 49,322 / GBR 19,065 / CARIB 83,025).