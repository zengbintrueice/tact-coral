# Conditional Residual Information Test — AUDIT

Generated: 2026-09-29
Source records: D:\曾斌\论文\海洋环境保护\PIECF\LTTS_CRW\results\physics_anchored\structural_ablation\rec_structural_ablation.npz
Source md5: 4035a19fd30fa4d14151f4c8055dfe59 (frozen rec_structural_ablation.npz)

## Provenance / definitions

- A_PersA_pred = PersA = clip(DHW/10,0,1) at issue time t
- C_anchored_pred = TSAR = PersA + gate*raw*0.4 (PhysAnchoredResidual, frozen seed-42)
- R = TSAR - PersA (the residual correction under test)
- target = future coordinate y(t+4); severe = y >= 0.8
- Analysis is purely associative diagnostics; associations are not causal.

## Protocol compliance

- [PASS] Source-only / zero-shot: uses only frozen record sets; no fitting/training
- [PASS] No training-protocol change; no target-domain training or tuning
- [PASS] No threshold selection on any target (severe >= 0.8 is the frozen definition)
- [PASS] Bootstrap: paired week-block, n=1000, seed=42 (shared resample matrix over weeks)
- [PASS] DHW-matched bins: 0.05 width, min n=40; Spearman rho bootstrap n=1000, seed=42
- [PASS] No evidence-locked result file was modified; outputs written only under results/conditional_residual/

## A/B result (severe y>=0.8, paired week-block 1000/seed42)

  Domain  A PersA ROC/PR   B TSAR ROC/PR    paired dROC [CI]              paired dPR [CI]          CI-excl0
  ------  --------------- --------------- ----------------------------- --------------------------- --------
  SCS    0.9817/0.3177     0.9808/0.3036   -0.0008 [-0.0027,+0.0011]   -0.0131 [-0.0503,+0.0264]   False
  GBR    0.9593/0.4164     0.9599/0.4877   +0.0006 [+0.0002,+0.0014]   +0.0634 [+0.0149,+0.1269]   True
  CARIB  0.9792/0.4602     0.9822/0.5479   +0.0029 [+0.0013,+0.0046]   +0.0843 [+0.0569,+0.1135]   True

## Conclusion (strict rule, per basin)

Rule: residual R contains conditional context information ONLY IF, IN THAT BASIN, BOTH
(i) the PersA+R (=TSAR) increment paired-CI (week-block, 1000, seed 42) excludes 0, AND
(ii) within matched-DHW (0.05-wide DHW bins) R shows a stable positive association with the
future severe outcome. Otherwise the residual is not proven independent of DHW.

- SCS (source): increment CI excl 0 = False (dROC -0.001, CI includes 0). Residual does
  NOT establish conditional context information (source-anchor null; no increment expected/claimed).
- GBR: increment CI excl 0 = True (dROC +0.0006, dPR +0.063, both positive CIs); matched-DHW
  rho stable = True. Residual DOES provide conditional context information.
- CARIB: increment CI excl 0 = True (dROC +0.0029, dPR +0.084, both positive CIs);
  matched-DHW rho stable = True. Residual DOES provide conditional context information.

GLOBAL: the TSAR residual provides conditional context information beyond PersA_DHW in the
two transfer target basins GBR and CARIB; not in the source SCS (by the conservative
source-anchor design, no increment).
Methodological caveat: within-DHW Spearman association is positive in ALL basins (R encodes
the `is DHW rising?` context), so the CI-supported marginal increment is the discriminating
evidence for where R adds transferable value.

Outputs: summary.json, bootstrap.csv, ab_roc_pr.csv, matched_bins.csv, AUDIT.md
Consistency: uses the frozen record set; all numbers reproduce the evidence-locked A/B values.

## Output files

- summary.json — per-basin A/B ROC/PR, paired dROC/dPR CI, matched-bin summary, verdict, conclusion
- bootstrap.csv — paired week-block dROC/dPR point + 95% CI per basin (TSAR − PersA)
- ab_roc_pr.csv — per-basin A(PersA) / B(TSAR) ROC-AUC and PR-AUC, n, n_severe
- matched_bins.csv — per-domain × 0.05-DHW-bin: n, n_severe, Spearman rho(R,y) + bootstrap CI, severe-rate by R-quantile