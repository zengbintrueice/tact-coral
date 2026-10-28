# NC_REVISION_REPORT — Nature Communications submission-level revision

Date: 2026-10-08. Scope: `main_manuscript.md` + `supplementary_information.md`
(revision copies: `main_manuscript_NC_revision.md`,
`supplementary_information_NC_revision.md`).

Principles honoured: no new unverified numbers; no results altered to fit the
narrative; no invented author/DOI/repository metadata; no correlation-to-
causation upgrades; no repositioning of the method as strict
physics-constrained ML; negative results retained; evidence checked before
any document change; unverifiable items escalated to
`NC_UNRESOLVED_ISSUES.md`.

Legend: **[x]** verified against code/result files; **[t]** text-only change;
**[A]** author action required.

---

## Hard errors (cross-references and numbering)

| Item | Location before | Action | Evidence | Impact |
|---|---|---|---|---|
| Main text cited "Supplementary Information R8--R10 / R9 (x2) / R10 / R11 (x2) / R13" but Notes R8-R13 had been consolidated into Methods S14 in the SI restructure | main Results/Methods/Discussion (7 sites) | re-pointed: R8--R10 -> "Supplementary Methods S14, Supplementary Table 9"; R9 -> S14 + Table 9; R10 -> S14; R11 -> S14; R13 -> "Supplementary Table 9" | SI current structure (### Note R1-R7 only; ## S14 protocol section); grep audit | cross-references now resolve; no scientific change |
| "Supplementary Note S13" | main Results, GBR checkpoint sentence | re-pointed to "Supplementary Methods S13" (actual SI header: "### S13. GBR zero-shot transfer: protocol and verification detail") | SI grep | resolves |
| SI overview map pointed GCBD evidence to "Table 1" (now dataset/scope table); residual/mechanism to "Notes R8-R10 and R13, Table 4" | SI overview | -> "Supplementary Table 8"; -> "Methods S14, Notes R8-R9, Supplementary Table 9" | current SI table contents | readers reach the correct tables |
| Stray quote in abstract equation lead-in ("(class-weighted \"") | main Introduction, Eq. (1) lead-in | quote removed | file diff | typo only |

## Task B — TSAR / PersA / TSCF interpretation

### B1. "In the source domain C does not exceed the anchor"
- **Before**: "In the source domain C does not exceed the anchor: the gate's
  benefit is a CARIB-specific rare-event effect." (main Results, structural
  ablation paragraph)
- **Verification**: SI Table 3(c) SCS row: paired dROC -0.0009 [-0.0027,
  +0.0011]; dPR -0.0141 [-0.0503, +0.0264]; **dCSI@0.90 +0.0406 [+0.0186,
  +0.0633]**. Cross-checked against
  `results/physics_anchored/structural_ablation/structural_ablation.csv`
  (0.06337589967109816) and `.json` — **values confirmed**. **[x]**
- **After**: "In the source domain, C does not improve ROC AUC (paired
  difference -0.0009 ...) or PR AUC (...) over the anchor, but does improve
  thresholded CSI@0.90 (paired difference +0.0406 [+0.0186, +0.0633]); the
  gate's benefit is therefore metric-dependent rather than a CARIB-specific
  rare-event effect alone."
- Affects conclusions: yes (removes an overgeneralization; strengthens
  accuracy). Author review: recommended (wording).

### B2. Terminology and comparison bases
- PersA / TSAR / TSCF defined at first occurrence (Eq. 2 / Eq. 3 / Methods);
  sweep found no inconsistent variants (`PersA_DHW` retained as the
  code-repository alias at the Eq. 2 definition). **[t]**
- Comparison-basis audit: SCS/GBR/CARIB = train/source domain; all zero-shot
  targets evaluated under the frozen-protocol PFV population; baselines
  named per experiment; paired contrasts share week-block resamples
  (1,000, seed 42); "improvement" statements now each name the metric
  (ROC/PR/CSI/Brier/ECE) — see B1, C3, G4. **[x]**

### B3. Contribution framing
- Abstract and Discussion claims already scoped: no "first DHW", no "first
  ConvLSTM", no biological-mechanism claim, no all-basins-solved claim.
  Added: Discussion persistence boundary (G3). Three-basin scoping added to
  the abstract anchor-and-correct sentence. **[t]**

## Task C — TSAR output bounds, probability semantics, metric inputs

### C1. Residual formula vs code
- `ablate_a_dh_stopgrad.py`: `res = torch.tanh(self.severe_head(
  attn.detach())).squeeze(-1) * scale` -> r in (-s, s), s = 0.4;
  `gate = torch.sigmoid((y_main - tau)/temp)`;
  `y_final = y_main + gate * res`. Matches Eq. (3) in the main text. **[x]**
- PersA = clip(DHW/10, 0, 1) in [0, 1]; target = same coordinate
  (`build_v9_1deg_from_raw.py`: `y_regression = np.clip(dhw_weekly/10, 0, 1)`).
  **[x]**
- **Output bound truth**: y_main in (0,1) (sigmoid); gate in (0,1);
  res in (-0.4, 0.4) -> y_final nominally in **(-0.4, 1.4)**; predictions
  are stored **unclipped** (`p_fused.append(y_final[:, -1].cpu().numpy())`).
  Restricting the residual does NOT guarantee a valid probability.
- Action: main Methods TSAR now states the output semantics explicitly
  (C3 below). Model behaviour unchanged (no code modification).

### C2. Metric input paths
- `compute_structural_ablation_analysis.py` (loads
  `rec_structural_ablation.npz`, raw predictions):
  - ROC AUC / PR AUC: raw scores (`fast_auc`), rank-based. **[x]**
  - Brier: `mean((pred - target)^2)` on **raw** predictions. **[x]**
  - ECE: 10 equal-width bins spanning **[0, 1)** on raw predictions —
    predictions outside [0,1) fall outside all bins and are silently
    excluded from the ECE weighted average. **[x]**
  - CSI/POD/FAR/F1: binarized at `pred >= 0.90`, tau fixed a priori, never
    target-selected. **[x]**
- XGBoost evaluation path clips predictions
  (`np.clip(model.predict(...), 0, 1)` in
  `experiments/08_infrastructure/pfv_evaluation_3level.py`) — noted as an
  inter-family difference in evaluation conventions; XGB is read only
  through within-family contrasts. **[x]**

### C3. Output semantics
- Main Methods TSAR now reads: the model output is a bounded continuous
  thermal-risk score (anchor plus gated residual; nominally PersA +/- 0.4);
  not clipped in model or evaluation path; not interpreted as a calibrated
  probability; ROC/PR on raw scores; CSI/POD/FAR/F1 binarized at tau=0.90;
  Brier/ECE on raw scores with ECE bins spanning [0,1). **[t]**; code basis
  **[x]**.
- SI Table 2 caption unchanged (TSAR row unaffected).

## Task D — TSCF architecture
- Verified: `src/models/piecf_model.py` class PIECFModel contains "=== 3.
  Temporal self-attention ===" in the shared backbone (attention_proj,
  temporal_attention); `src/configs/config.py`:
  `convlstm_hidden_dims: [16, 16]  # 2-layer ConvLSTM: 16->16 channels`.
  **[x]**
- Conclusion: the frozen TSCF DOES include temporal self-attention;
  `ConvLSTM [16,16]` in SI Table 2 was an abbreviation.
- Action: SI Table 2 encoder row now reads "ConvLSTM [16,16] + temporal
  self-attention" for TSCF and "(shared backbone)" for TSAR. Main Methods
  wording was already correct. **[t]**

## Task E — SI structure, tables, captions

### E1. Empty Note R2 subsections
- "**SCS.** / **GBR.** / **CARIB.**" labels were left empty after the
  tables moved to the Tables section. Replaced by a single pointer
  paragraph to Supplementary Tables 3(b) and 7. **[t]**

### E2. Reference map (current -> target -> action)
| Position | Was | Target content | Now |
|---|---|---|---|
| SI overview, GCBD row | Table 1 | GCBD metrics + by-model | Supplementary Table 8 |
| SI overview, residual row | Notes R8-R10 and R13, Table 4 | hierarchy/mechanism protocol + records | Methods S14, Notes R8-R9, Table 9 |
| main, 7 sites | Supplementary Information R8-R13 | protocol (S14) + numeric records (Table 9) | re-pointed (see Hard errors) |
| main, GBR checkpoint sentence | Supplementary Note S13 | GBR protocol detail | Supplementary Methods S13 |

### E3. Supplementary Fig. 3 caption
- **Before**: "Removing the cumulative HotSpot channels eliminates
  severe-risk discrimination in both basins."
- **Verified** against
  `results/step6_feature_ablation/step6_ablation_results.csv` **[x]**:
  NO_CUMULATIVE: SCS ROC 0.983->0.880 (PR 0.535->0.075, CSI@0.9 0.263->0.002);
  GBR ROC 0.956->0.378 (PR 0.510->0.033, CSI@0.9 0.296->0.000);
  THERMAL_ONLY: SCS 0.881, GBR 0.680.
- **After**: caption distinguishes ranking vs thresholded discrimination and
  GBR vs SCS: GBR ranking collapses (0.956 -> 0.378); SCS ranking is
  largely retained (0.983 -> 0.880) while thresholded severe discrimination
  (CSI@0.9) is lost in both basins; thermal-only preserves SCS ranking
  (0.881) and part of GBR (0.680).
- Affects conclusions: caption accuracy only.

### E4. Table 8(b) garbled header
- **Before**: `| Model | cohort | sev | AUROC(sev-v-bg) | AUROC site-CI |
  RR(Q5/Q | )   RR site-CI | Cliff's | δ   δ site-CI | MW p |`
- **After**: `| Model | cohort | sev | AUROC (sev vs bg) | AUROC site-CI |
  RR (Q5/Q1) | RR site-CI | Cliff's δ | δ site-CI | MW p |` — RR (Q5/Q1) =
  severe-event rate ratio between the top and bottom predicted-risk
  quintiles; header cell count matched to the data rows (10 columns).
  Values untouched. **[x]**

### E5. Image paths
- All 10 SI image embeds resolve under the repository root
  (`results/figures/supplementary/Supplementary_Fig_01..10.png`);
  verified by filesystem check. Journal-platform rendering behaviour was
  NOT tested and is listed in NC_UNRESOLVED_ISSUES.md. **[x]/[A]**

## Task F — Supplementary References
- Section removed (6 name-drop entries; the SI carries no numbered
  citations to them). Datasets remain cited in the main reference list
  (GCBD, CRW) and the repository README. **[t]**

## Task G — GCBD / residual / biological boundaries
- G1: "Severe thermal risk is not severe bleaching" boundary retained in
  Note R5 and main Methods; GCBD framed as independent ecological
  validation, not biological prediction. **[t]** (verified consistent)
- G2: keyword sweep (mechanism/causal/susceptibility/independent/
  biological/generalisation) — main text already restricts to conditional
  association; the Discussion now additionally states the residual
  "contains information associated with future thermal risk conditional on
  the DHW-based anchor, not a verified biological signal". **[t]**
- G3: DHW-persistence alternative explanation added to the Discussion
  (lag-1 autocorrelation > 0.98 in all basins; no experiment isolates the
  persistence component). **[t]**
- G4: seed/robustness language audited: Table 6 reports observed ranges;
  "not averaged into a robustness claim" retained; "CI-supported" used
  only where paired CIs exclude zero. **[x]**

## Task H — key-number consistency
- See `results/qc/NC_RESULT_CONSISTENCY_AUDIT.csv` (35 rows: main value,
  SI value, source file, code path, verdict). All rows consistent except
  where marked; source files:
  - `results/transfer_coordinate/transfer_matrix.csv`,
    `reverse_matrix.csv`, `seed_scan/`
  - `results/physics_anchored/structural_ablation/structural_ablation.{csv,json}`
  - `results/xgb_transfer_control/metrics.json`
  - `results/step6_feature_ablation/step6_ablation_results.csv`
  - `results/step9_gcbd_validation/`
  - `results/seed_sensitivity/seed_metrics.csv`
- No recomputation was necessary; no result file was modified.

## Task I — abstract / discussion / conclusion
- Abstract: 148 words; every number traceable (0.25, 0.61-0.67, 0.003,
  3.6/1.6/0.9%, GBR/CARIB separation); GCBD sentence restricted to
  observed enrichment; anchor-and-correct scoped to the three-basin study.
- Discussion now answers, in order: transfer findings; why the DHW anchor
  is the right control; where the residual contributes (which metrics,
  which targets); why failure is pair-specific; seed sensitivity; GCBD
  support and limits; mechanism interpretations marked as plausible, not
  established.
- No separate Conclusion section (NC Articles do not require one); the
  Discussion closing paragraph carries the conclusion. **[t]**

## Task J — final quality checks
- Placeholders: `[Author list...]`, `[To be completed...]`,
  `[placeholder: repository DOI to be assigned]` — intentional, listed in
  NC_UNRESOLVED_ISSUES.md. No TODO/TBD/XXX. **[x]**
- Heading hierarchy continuous; equations (1)-(4) numbered and referenced;
  figure/table numbering gap-free (main 1-10; SI Figs 1-10, Tables 1-9);
  abbreviations defined at first use; rendering: pandoc -> docx (double
  spaced, TNR 12, page numbers) + SI PDF via Word. **[x]**
- Both documents rendered without errors; images verified present.

## Addendum: probability-metrics audit (2026-10-08)
- Full audit in results/qc/probability_metrics_audit/AUDIT_PROBABILITY_METRICS.md
  (code-traced computation path; per-basin output ranges and out-of-range
  masses; three-convention metric comparison; corrected paired contrasts).
- Finding: the original ECE convention silently excluded out-of-range
  predictions (asymmetrically penalizing only the TSAR family) and used two
  different implementations across figures. Fixed by a uniform convention:
  all models clipped to [0,1] before Brier/ECE; Brier against the continuous
  target; ECE vs the binary severe event with the last bin including 1.0.
- Paper numbers updated: B-A dBrier 0.126 -> 0.123; dECE 0.332 -> 0.348
  (CARIB). All qualitative conclusions survive and are strengthened.
- Figures 5 and 6 regenerated from the corrected metrics; nc_style
  reliability_curve unified (clips internally).
- ROC/PR/CSI/POD/FAR/F1 values unchanged (clipping-invariant). No
  evidence-locked value was altered.

## Addendum (N10): authors, contributions, competing interests
- Title page: Bin Zeng, Rui Wang* (corresponding) & Houpu Li; Naval
  University of Engineering, Wuhan, China. Corresponding-author email and
  ORCIDs remain the ONLY blocking author items.
- Author Contributions: convention-based draft inserted with explicit
  [confirm and amend] marker.
- Competing Interests: no-competing-interests draft inserted with
  [confirm] marker.

## Addendum (N12): author block finalized
- Correspondence: Rui Wang, 1202152003@nue.edu.cn.
- ORCID iDs inserted for all three authors (B.Z. 0009-0006-1165-2217,
  R.W. 0009-0009-5910-8415, H.L. 0009-0007-9500-7254).
- Author Contributions finalized per author rank: first author carries the
  experimental design, data pipeline, model training, analyses, figures and
  original draft; corresponding author supervised, acquired/verified
  datasets, interpreted and edited; third author verified GCBD matching and
  leakage audits, QC and figures.
- Remaining open items: Competing Interests [confirm], Zenodo deposit,
  reference [9] check.

---

## Addendum (2026-10-18)

A cross-check of the manuscript and SI against the reproduction package
(`zenodo_release/`) found 19 items; 18 were corrected. Two required re-running
analyses:

1. **`cluster_ci` bootstrap bug** in
   `experiments/08_infrastructure/compute_gcbd_year_stratification.py` — the
   site-cluster resample scored AUROC over all selected rows, letting
   any-bleaching observations enter the negative class, so the bootstrap
   disagreed with the (correct) severe-vs-background point estimate. The 2019
   stratum CI upper bound fell below its own point estimate as a result. Fixed
   and re-run; **all nine point estimates reproduced exactly**, only the CIs
   changed. Figure 10(f) regenerated.
2. **CARIB 2015-2016 cohort** — the quoted "n 476; 215 severe" is not
   reproducible from the shipped data under any cohort definition (all matched
   = 388, union = 388, anchor-defined = 264). Restated to the reproducible
   values.

Also corrected: the B-A Brier/ECE deltas (now on the unified clipped
convention), SI Table 3(a) CARIB PersA ECE, the GBR fixed-DHW severe rate
(3.6% -> 3.5%), the over-strong TSAR per-year claim, and several
cross-reference/formatting defects.

**Open item (not changed):** the `pos_weight` sweep selects its value by
SCS held-out *test* CSI (`train_severe_classifier_sweep.py`:
`best_pw = max(results, key=lambda k: results[k]["test_csi"])`), whereas
Methods states the sweep was "selected on SCS validation only". Left as
reported, pending an author decision.

Full record: `doc/CROSSCHECK_paper_vs_release.md`.
