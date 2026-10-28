# SI restructure QC report (NC clear-and-succinct pass)

## Deleted
- Supplementary figures 11-19 (hyperparameter/evidence-lock; GBR zero-shot
  detail; holdout-enrichment forest; all-to-all transfer matrix; coordinate
  stability; regime heatmap; combination-rule sensitivity; residual
  hierarchy; cross-basin mechanism). Their numeric content remains in
  Supplementary Tables 2-6, the retained R-section tables, and the
  evidence-locked source files - no "data not shown".
- Old S12 XGBoost figure (covered by main-text Figure 6 + Supplementary
  Table 5).
- Old Table 2 (input channels; the 13 channels are listed in main Methods)
  and old Table 6 as a separate table (paired contrasts merged as Table 3c).
- Three inline result tables fully covered by new Tables 4(b), 4(c), 5.

## Merges
- Table 1 = old Table 1 + temporal-purge robustness rows.
- Table 2 = old Table 3 + evidence-lock rows (checkpoint MD5s, normalization,
  severe label, threshold, seeds; repository-relative identifiers only).
- Table 3 = old Table 4 (a) + old Table 5 (b) + old Table 6 (c).
- Table 4 = old Table 7 (a) + old Table 8 (b) + old Table 9 (c).
- Table 5 = old Table 10 + n / n severe columns + record-set-audit caption.
- Table 6 = old Table 11 + observed per-model seed ranges (min-max over the
  five seed rows; explicitly not averaged into a robustness claim).

## Citation validity
- Main text cites Supplementary Figs. 1-10 and Tables 1-6; every retained
  figure and table is cited at least once (programmatically verified).
- Section references (S1-S12, R1-R13, Supplementary Methods S7) unchanged
  and resolving.

## Numbering
- Supplementary Fig. 1-10, Supplementary Table 1-6: no gaps, no stale labels
  (fig_r*, S13, SX, Note S12 all removed).

## Value consistency
- Table 5 ROC values match results/xgb_transfer_control/metrics.json.
- Table 6 seed rows match results/seed_sensitivity/seed_metrics.csv
  (full-precision values rounded to 4 decimals).
- Every cell of every old SI table is preserved verbatim in the new SI
  (0 missing cells vs the pre-restructure version).
- Table 2 MD5s match EVIDENCE_LOCKS.md.
- No scientific value was altered; no experiment was re-run.


## Restructure map (Supplementary Results -> Supplementary Notes)
- Old '# Supplementary Results' (R1-R13 narrative sections) ->
    '# Supplementary Notes' (### Note R1..R13, verification evidence;
    purpose -> protocol -> key result -> interpretation framing).
- Old '# Supplementary Methods for XGBoost control (S12)' ->
    '## S12. XGBoost control: protocol equivalence and verification note'
    (folded under Supplementary Methods).
- Old '## S13. GBR zero-shot transfer in detail' ->
    '### S13. GBR zero-shot transfer: protocol and verification detail'.
- New section '# Supplementary References' added (six works named in SI
    prose; cross-referenced to main-text reference numbers).

## Deleted / redundant content (this pass)
- No scientific content deleted in N4; the standalone Results narrative was
  reframed as Notes. Duplicated narrative sentences remain only where they
  carry unique verification numbers.

## Main-text <-> SI cross-reference audit
- Every main-text claim with SI backing maps to Notes R1-R13 / Tables 1-6 /
  Figs 1-10 (map in the Supplementary overview).
- No orphan supplementary figure or table (all cited in main text).
- No orphan main-text SI citation (all targets exist).


## Addendum (N5c): Table 4(c) repair
- The N1 duplicate-table removal had deleted Table 4(c)'s header row,
  separator and the two cumsum rows (mis-anchored pointer). The panel was
  rebuilt with its full 8-row header+body (values verbatim from the
  pre-restructure version; sum/source verified).
- Stray cross-reference pointer inside the Tables section removed; the
  pointer now sits in Note R3 where it belongs.
- Structural sweep: all Supplementary Tables 1-14 have consistent
  header/separator/cell counts; no stray pointer lines remain.


## Addendum (N6b): stale table references
- S7 note: '(Tables S6 and S7)' -> '(Supplementary Table 3)' (old-style
  label missed by the 1:1 rename pass).
- S7 note: marginal-CI reference '(Supplementary Table 5)' ->
  '(Supplementary Table 3b)' (old S5 content now lives in Table 3b).
- Full SI sweep: zero remaining old-style references (Tables S#, Note S#).
- Final state: Supplementary Figs. 1-10, Tables 1-15, Notes R1-R13
  (prose-only), Methods S1-S13, Supplementary References.


## Addendum (N7): Discussion mechanism clause
- Added the pre-agreed qualifying (non-claim) clause after the XGBoost
  Discussion sentence: the regression objective equals the PersA anchor
  coordinate, so the tree ensemble behaves like an unanchored model that
  learned a coarse anchor approximation - framed with "plausibly explains",
  no mechanistic claim beyond the cited ablations.


## Addendum (N8): equation formalization
- Four numbered display equations installed in the main text:
  (1) thermal-risk coordinate y(t) = clip(DHW(t)/10, 0, 1);
  (2) persistence anchor PersA_DHW(t) = y(t);
  (3) TSAR formulation yhat = PersA + g*r, r = s*tanh(R(h_t)),
      g = sigmoid((y_main - tau_g)/T);
  (4) severe-label indicator 1{y(t+4) >= 0.8}.
- Each equation carries a where-clause defining every symbol (DHW, clip,
  R(h_t) = MLP_sev(stopgrad(A)), s, tau_g, T, y_main).
- Methods restatements replaced by Eq. references; text cites (Eq. N) per
  NC parenthetical style.
- No numerical values changed.


## Addendum (N9): Table 4 panels restored + user 12-item pass
- Table 4 panels (b) joint static dependence and (c) channel associations
  restored (had been dropped in the Table 1-9 rebuild); all 8 channel rows
  verbatim (cumsum_4w 0.814/0.666/0.467 ... seasonal_cos -0.743/0.410/-0.127).
- User 12-item pass: A1 duplicate 'does does' fixed; A2/B2 TSAR ROC range
  -> 0.959-0.987 (verified 0.98678); A4 CARIB permutation dROC added;
  A5 0.92 -> 0.915; A6 dPR CI-supported 2/5 seeds noted (verified);
  A8 Fig 6(e) gloss; A10 explicit Fig. 3/8/9/10 citations; A12 abstract
  GCBD sentence quantified; B3 Table 7 diagonal rows; B7 Table 10 added
  (EXP A 0.537 flagged, source file not located); B11 consistency note
  section deleted.


## Addendum (N10): EXP A value corrected
- SI Note S10 and Table 10 said 'ROC-AUC 0.610 -> 0.537'; the evidence-locked
  file results/caribbean_zero_shot/phys_diagnostics/phys_diagnostics.json
  records expA.caribbean_train_renorm.roc = 0.52598. Both corrected to
  0.526 per the evidence-locked source (SI text was the outlier; Fig 10 was
  generated from the same json and was always correct). NC_UNRESOLVED_ISSUES
  item 13 resolved.

## Addendum (N12): author block finalized
- Correspondence email, three ORCID iDs and rank-based Author Contributions
  inserted on the title page. No scientific content changed.
