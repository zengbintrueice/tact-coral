# Audit — CSI@0.90 terminology and metric-type separation (Fig. 5 focus)

**Date:** 2026-10-30
**Scope:** Fig. 5 (`fig6_gated_residual`) and every place in the manuscript/SI
where CSI@0.90 is described as a ranking metric or mixed with ranking /
calibration metrics.
**No threshold or result changed; no recomputation was needed.**

---

## 1. CSI@0.90 — verified definition

Source: `experiments/08_infrastructure/compute_structural_ablation_analysis.py`
(`SEV = 0.8`, `TAU = 0.90`, `metrics_sklearn` / `metric_vec_fast`).

```
event      = 1[ y(t+4) >= 0.8 ]          # severe thermal risk
predicted  = 1[ score >= 0.90 ]          # raw score, no clipping
TP = predicted & event
FN = ~predicted & event
FP = predicted & ~event
CSI@0.90 = TP / (TP + FN + FP)
```

- **Positive class:** severe thermal risk, y(t+4) ≥ 0.8 (DHW ≥ 8 °C-weeks).
- **Negative class:** all non-severe records, i.e. background (= 0) **and**
  mild bleaching (> 0 but < 20% bleaching, < 0.8 thermal coordinate).
  It is not a "background-only" contrast.
- **Threshold:** a single pinned value, τ = 0.90, applied to the **raw** score
  (no clipping; clipping would not change the binarisation). For PersA/TSAR
  this is a threshold on the continuous risk coordinate (≈ DHW 9), not on a
  probability.
- **Missing values:** none by construction — the shared record set is
  `finite y(t+4) & ocean & all four models finite`; no imputation, no silent
  exclusion.
- **Sample sizes:** SCS 49,322 (419 severe), GBR 19,065 (676), CARIB 83,025
  (1,356).
- **What it measures:** agreement of a *thresholded decision* with the severe
  event. It is **threshold-dependent classification**, not ranking.

Ranking metrics actually used: **ROC-AUC and PR-AUC**, computed on raw scores.
Calibration metrics: **binary Brier / ECE**, only for TSCF (probability
output). Continuous error: **MSE**. These four families are now named
separately.

---

## 2. Affected locations (before → after)

| # | Location | Before | After |
|---|---|---|---|
| 1 | Main Fig. 5 caption (d) | "Paired bootstrap **dROC** for gated-PersA and gated-ungated" | "Paired bootstrap **ΔCSI@0.90** … (a threshold-dependent classification contrast, not a ranking contrast)" |
| 2 | Fig. 5 panel (b) annotation | "Ungated residual: **~10× larger MSE**" | "Ungated residual: **6–11× larger MSE**" (matches the caption's 5.8–11.4-fold) |
| 3 | Fig. 5 panel (b) y-axis | no label | "CSI@0.90 (circles) / MSE (diamonds), log scale" |
| 4 | Main Fig. 3 caption (a) | "(ROC-AUC, PR-AUC, CSI@0.9, ECE (probability))" | labels each metric's type: ranking / threshold-dependent / probability calibration |
| 5 | Main Fig. 4 caption (c) | "(ROC-AUC and CSI@0.9)" | "(ROC-AUC, ranking; CSI@0.90, threshold-dependent classification)" |
| 6 | SI S5 context-permutation sentence | "does not degrade ranking: ΔPR …, ΔCSI@0.9 …, ΔROC …" | ranking deltas (ΔPR, ΔROC) separated from "slightly raises thresholded CSI@0.9" |

Already correct before this pass (no change needed):

- Methods: "CSI/POD/FAR/F1 binarize the raw score at the pinned τ = 0.90 and
  are threshold-dependent classification metrics, not pure ranking metrics."
- Fig. 5 panel (b) title: "Structural ablation — thresholded classification vs
  continuous error".
- Fig. 5 legend: "CSI@0.9" (circles) vs "MSE (continuous)" (diamonds).
- Fig. 5 source data: explicit columns `csi_tau09`, `mse_cont`,
  `score_reliability`, `ece_probability`, `probability_calibration_valid`.
- SI Fig. 7 caption: "Paired week-block bootstrap ΔCSI@0.90 effect sizes".
- SI Table 3: separate ΔROC / ΔPR / ΔCSI columns.

---

## 3. Key finding: the Fig. 5 caption mislabelled a threshold metric as ranking

The panel labelled (d) plots **ΔCSI@0.90** (`csi_tau09` in
`structural_ablation.json`; the script's QC note records "panel d: primary
effect size switched from ΔROC to ΔCSI@0.9"). The main caption still said
"dROC", i.e. it described a threshold-dependent classification contrast as a
ranking contrast. This is the only substantive terminology error found; it was
a caption error, not a computation error.

## 4. Other figures checked

| Figure | Metrics shown | Verdict |
|---|---|---|
| Fig. 3 | ROC/PR (ranking), CSI@0.90 (threshold), ECE (probability, TSCF) | mixed panel; caption now labels each type (was ambiguous) |
| Fig. 4 | ROC/PR, CSI@0.90, score–frequency | caption now labels types |
| Fig. 5 | CSI@0.90, MSE, ΔCSI, gate diagnostics | panel titles/legend already correct; caption fixed |
| Fig. 6 (XGB) | ROC/PR only | no mixing; probability metrics explicitly excluded |
| SI Fig. 7 | ΔCSI@0.90 | correct |
| SI Table 3 | separate ROC/PR/CSI columns | correct |
| Fig. 8 | continuous risk, ΔROC/ΔPR, ΔR² | no CSI; correct |

No other caption calls CSI a ranking metric.

## 5. Implementation-error assessment

**None found.** The code computes exactly the CSI defined above. The only
discrepancy was textual (caption d). Therefore:

- No threshold changed.
- No result recomputed.
- No table value changed.

## 6. Reproduction / verification

```bash
cd LTTS_CRW/tact-coral
"$PIECF_ENV/Scripts/python.exe" scripts/figures/fig6_gated_residual.py
pdftotext results/figures/fig6_gated_residual.pdf - | grep -iE "CSI|MSE|threshold"
```

Expected: panel (b) title "thresholded classification vs continuous error";
legend "CSI@0.9" vs "MSE (continuous)"; y-axis label
"CSI@0.90 (circles) / MSE (diamonds), log scale"; annotation "6–11× larger
MSE"; panel (d) "ΔCSI@0.9 (gated − reference)".

## 7. Verdict

**Corrected:** the Fig. 5 caption's "dROC" → "ΔCSI@0.90"; the panel annotation
magnitude; the y-axis label; and the ambiguous metric lists in the Fig. 3 /
Fig. 4 captions and SI S5.

**Unchanged:** every CSI/POD/FAR/F1 value, threshold, record set and sample
size. The manuscript now consistently distinguishes ranking (ROC-AUC, PR-AUC),
threshold-dependent classification (CSI/POD/FAR/F1), calibration (Brier/ECE,
TSCF only) and continuous error (MSE).
