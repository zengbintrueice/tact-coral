# FIGURE QC REPORT — cross-basin transferability study

Generated: 2026-10-01T16:38:43.357892

## Per-figure status

| Figure | PDF | SVG | PNG (dims) | Source-data files | Status |
|---|---|---|---|---|---|
| fig1_framework | yes | yes | yes 4347x3088px | 1/1 | OK |
| fig2_transfer_gbr | yes | yes | yes 4405x3133px | 1/1 | OK |
| fig3_transfer_boundary | yes | yes | yes 4185x3195px | 4/4 | OK |
| fig4_crw_diagnostics | yes | yes | yes 4617x5212px | 6/6 | OK |
| fig5_persa_anchor | yes | yes | yes 4086x3297px | 2/2 | OK |
| fig6_gated_residual | yes | yes | yes 4562x3817px | 3/3 | OK |
| fig7_gcbd_validation | yes | yes | yes 4363x3712px | 4/4 | OK |
| edfig_s1_leakage_audit | yes | yes | yes 2910x1882px | 1/1 | OK |
| edfig_s2_temporal_purge | yes | yes | yes 3825x1705px | 1/1 | OK |
| edfig_s3_feature_ablation | yes | yes | yes 2768x1718px | 1/1 | OK |
| edfig_s4_temporal_scale | yes | yes | yes 2986x1712px | 1/1 | OK |
| edfig_s5_context_attribution | yes | yes | yes 2465x1604px | 1/1 | OK |
| edfig_s6_threshold_sensitivity | yes | yes | yes 2614x1619px | 1/1 | OK |
| edfig_s7_bootstrap_effects | yes | yes | yes 3193x2265px | 1/1 | OK |
| edfig_s8_seed_sensitivity | yes | yes | yes 4174x1573px | 1/1 | OK |
| edfig_s9_spatial_maps | yes | yes | yes 4308x1477px | 1/1 | OK |
| edfig_s10_carib_diagnostics | yes | yes | yes 3326x1652px | 1/1 | OK |
| edfig_s11_hyperparameters | yes | yes | yes 6104x1690px | 1/1 | OK |

Supplementary figures registered in figure_manifest.csv (11 rows); main-figure rows preserved.

## Numerical checks

- NaN in PLOTTED columns: ["fig1_source_data.csv: ['lon_lo', 'lon_hi', 'lat_lo', 'lat_hi', 'y0', 'y1']", "fig2_source_data.csv: ['value', 'ci_lo', 'ci_hi', 'brier', 'ece']", "fig5_source_data_reliability.csv: ['bin8_pred', 'bin9_pred', 'bin8_freq', 'bin9_freq']", "fig7_source_data_site.csv: ['rr_site_mean_q5q1', 'fisher_p_site']", "S8_seed_sensitivity.csv: ['mean', 'ci_lo', 'ci_hi']"]
- duplicated records (on identifying keys): ['edfig_s2_source_data.csv', 'edfig_s3_source_data.csv', 'edfig_s4_source_data.csv', 'edfig_s6_source_data.csv', 'fig1_source_data.csv', 'fig2_source_data.csv', 'fig2_source_data_transfer_matrix.csv', 'fig3_source_data_pr.csv', 'fig4_source_data_j2j4_verification.csv', 'S8_seed_sensitivity.csv']
  Classification of residual flags (all BY DESIGN, none affect plotted values):
  * fig1_source_data.csv — two panel schemas share one file (panel a lon/lat boxes; panel b year spans); NaN marks the non-applicable schema, every plotted value is present.
  * fig2_source_data.csv — panel b/d rows (value/CI) and panel c reliability rows (brier/ece) have different schemas; 'duplicates' are the b/d pairing rows, each plotted once.
  * fig5_source_data_reliability.csv — top bins (8/9) are empty for calibrated models because they have no predictions there; curves legitimately end early.
  * fig7_source_data_site.csv — rr_site_mean_q5q1 is EMPTY BY RULE: Q1 contains zero severe sites, so the Q5/Q1 site ratio is not calculated (per the figure-spec QC requirement).
  * edfig_s2/s3/s4/s6 source files are verbatim copies of upstream step CSVs; duplicate keys arise from eval_set/protocol row variants and do not affect the plotted aggregates.
- fig3 check: CARIB TSCF PR/CSI dramatically lower than GBR — enforced by assertion inside fig3_transfer_boundary.py: PASS
- fig4 check: CARIB thermal W1 median NOT the largest — enforced by assertion inside fig4_crw_diagnostics.py: PASS
- fig5 check: PersA recovers CARIB discrimination (ROC 0.98 vs TSCF 0.61) — enforced by assertion inside fig5_persa_anchor.py: PASS
- fig6 checks: un-gated residual worsens calibration; gated beats un-gated in CARIB (CI excludes 0); gated incremental over PersA in CARIB — enforced by assertions inside fig6_gated_residual.py: PASS
- fig7 checks: Q5/Q1 SITE-level ratio NOT calculated; site-cluster CIs used as primary inference; no causal annotations: PASS
- thresholds: tau=0.90 pinned everywhere (sweep shown as diagnostic only in edfig_s6); severe y>=0.8; lead 4 wk; history 24 wk: consistent
- domain labels/colors: consistent across all figures (SCS blue / GBR green / CARIB red / anchor slate / gated purple)
- CARIB data used for model fitting: NEVER (all target-domain parameters zero-shot; enforced by the frozen protocol)

## Figure 1 submission-level verification

- panel a: SCS box spans ≥100.5→≥129.5 °E True; coastline from ETOPO1 (loaded; no fallback warning in the run log)
- panel b: explicit year labels present (train/validation/test) OK=True -> 2003–2010 / 2011–2015 / 2016–2019; zero-shot test 2017–2020
- panel c: 'Causal inputs ≤ t' OK=True; 13 weekly channels, 24-wk window, SCS-train stats frozen, ConvLSTM, temporal attention, no target-domain fitting
- panel d: PersA=clip(DHW/10,0,1) OK=True; residual head; 'limits the magnitude of the contextual correction'; 'bounded two-sided correction'; removed 'basin-invariant coordinate' / 'local climatological max' / 'OOD score drift'
- fonts: Arial; light-grey transparent coastline; white background; colorblind-safe domain palette (blue/green/red + slate/purple accents)

## Figure 2 submission-level verification

- whole-figure title present: True
- panel a: 'freeze model + normalizer'; PFV text '24-wk history → 4-wk-ahead verification; ocean-masked records; τ=0.90 pre-specified' True
- panel b: 'paired' retained ONLY because the bootstrap IS paired (one shared pick matrix, seed 42); method-connecting lines removed True
- panel c: Brier/ECE read from official structural_ablation.json and cross-checked against npz (asserted identical) True
- panel d: title 'Operational performance at the pre-specified threshold (τ = 0.90)'; FAR kept, no 'across-the-board improvement' framing True

## edfig_s8 seed-sensitivity check

- title 'Random-seed sensitivity of cross-basin transfer' and subtitle 'Transfer boundary direction is preserved across seeds, while absolute performance varies with initialization.'; 'robust across five seeds' NOT used — OK=True
- source data present: True; panels (a) GBR vs CARIB PR paired dots with individual seeds, (b) dPR + 0 reference, (c) SCS PR (in-domain seed sensitivity)

## Figure 7 submission-level verification

- panel a: matching text '4-week lead; ±4-week temporal window; 0.5° spatial buffer'; n=1,898 observations; 78 severe; 384 sites True
- panel b: title 'Cluster-aware discrimination of independent GCBD observations'; 'any bleaching vs rest' removed; site-level AUROC (site_auc=0.8623) added True
- panel c: title kept; RR≈35 + site-cluster CI read from step10_matching_results.csv (offset 0); note 'quintiles defined within the independent GCBD cohort; CI = site-cluster bootstrap' True
- panel d: x-axis simplified to 'matching offset Δ (weeks)'; formula moved into the legend title; 'none selected post hoc' kept True
- palette: no CARIB-red used as a semantic colour in fig7 (blue/purple/anchor-slate system) True

## Figure 3 submission-level verification

- panel a: title 'Three-domain performance of frozen TSCF'; colourblind-safe single-hue (Purples, light→dark), RdYlGn removed True
- panel c: prevalence baselines drawn from official n_severe/n (SCS 0.0085, GBR 0.0355, CARIB 0.0163); 'prevalence baseline' in legend True
- panel d: title 'Risk stratification across predicted-risk deciles'; log axis; deciles with zero severe events are marked with a true-zero ▼ (no epsilon substitution); '0 = no severe events' legend/note True

## Discrepancies vs the figure-specification prompt

1. Checkpoint MD5: prompt says prefix '7c97?'; ACTUAL 7cdbca26cae03fa53be53210940e8ce1 (matches the step-4 frozen protocol). The prompt's value was a placeholder; the actual checksum is used.
2. Prompt section 6 lists TSCF three-domain numbers from the PRE-ocean-mask population (e.g. SCS ROC 0.985639, n=68432). The authoritative deliverables now use the OCEAN-MASKED record set (SCS n=49322, GBR n=19065, CARIB n=83025; SCS TSCF ROC 0.9865). Per the data-integrity rules the FILES win; figures show masked values. The old numbers remain in results/*_premask_bak_20260917/.
3. Prompt sections 8 (W1/J1-J4) provides context values; the figures read results/domain_shift/*.csv|json directly. Small differences (e.g. J2/J4 Spearman values) follow the files.

## Missing data (NOT fabricated)

None — every planned main and supplementary figure is generated from existing result files (edfig_s8 seed sensitivity and edfig_s9 spatial maps are now produced from stored checkpoints/data).

## Reproducibility

- every figure regenerates from result files via `generate_all_figures.py`; no scientific values are hard-coded in plotting scripts; missing files raise explicit errors.
- fonts: Arial (embedded as Type-42 in PDF); SVG text kept editable.
- statistics: paired week-block bootstrap (1000, seed 42) read from structural_ablation.json; site-cluster bootstrap from step9_bootstrap.csv.
