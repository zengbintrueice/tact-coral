# Implementation report — metric-definition correction (Brier / ECE / targets)

**Date:** 2026-10-29
**Scope:** separate continuous thermal-risk error from severe-event probability
calibration in code, result files, figures, manuscript and SI.
**No model was retrained; no split, seed, threshold, lead time or architecture
was changed.**

**Status: FIXED AND VERIFIED**, with two documented residuals (§7) and one
author-decision experiment (§8).

---

## 1. Modified files

### Code (new)
| File | Purpose |
|---|---|
| `experiments/08_infrastructure/metric_semantics.py` | canonical Task A / Task B definitions, validation, binning, ECE, score reliability |
| `experiments/08_infrastructure/recompute_calibration_metrics.py` | reads frozen predictions, writes `metrics_v2.*`, asserts agreement with the audited values |
| `tests/test_metric_semantics.py` | 12 unit tests |

### Code (edited)
| File | Change |
|---|---|
| `scripts/figures/fig3_transfer_boundary.py` | ECE now read from `metrics_v2.ece_probability`; label "ECE (probability)" |
| `scripts/figures/fig5_persa_anchor.py` | panel d relabelled score–event-frequency agreement; legend shows MSE + score reliability for PersA/TSAR, Brier + ECE for TSCF; source data split by task |
| `scripts/figures/fig6_gated_residual.py` | panel b second metric relabelled continuous MSE; QC assertion and annotations updated; source data uses `mse_cont` / `score_reliability` / `ece_probability` |
| `scripts/figures/fig_s_x_xgb_transfer.py` | XGB source data uses `mse_cont`, `score_reliability`, `brier_binary_descriptive`, `probability_calibration_valid=False`; caption/notes state that probability metrics do not apply to a continuous regressor |
| `scripts/figures/nc_style.py` | `reliability_curve` docstring marks its returned `brier` as legacy continuous MSE |
| 5 legacy metric scripts + 2 probability scripts | explicit `LEGACY METRIC NAMING` / `PROBABILITY PATH` headers |

### Results (new)
- `results/qc/calibration_target_audit/metrics_v2.json`, `.csv`, `metrics_v2_bins.json`
- `results/qc/calibration_target_audit/LEGACY_FIELDS.md`
- this report

### Results (regenerated)
- `results/figures/source_data/{fig3_source_data_metrics.csv, fig5_source_data.csv, fig5_source_data_reliability.csv, fig6_source_data_ablation.csv, SX_source_data.csv}`
- `results/figures/{fig3_transfer_boundary, fig5_persa_anchor, fig6_gated_residual, SX_xgb_transfer_control}.{png,svg,pdf}`
- `results/figures/figure_manifest.csv`, per-figure QC reports

### Manuscript / SI
- `doc/main_manuscript.md/.docx` (+ `_NC_revision` copies and the archive copies)
- `doc/supplementary_information.md/.docx/.pdf` (+ `_NC_revision` copies and the archive copies)

---

## 2. Before → after definitions

| Quantity | Before | After |
|---|---|---|
| `brier` for PersA / ungated / TSAR / XGB | continuous `mean((clip(pred)−y(t+4))²)`, called "Brier" | **`mse_cont`** (continuous risk error) + `mse_cont_raw` |
| `brier` for TSCF | same continuous MSE | **`brier_binary`** = `mean((p − 1[y≥0.8])²)`; continuous MSE kept separately as `mse_cont` |
| `ece` for PersA / ungated / TSAR / XGB | called probability calibration | **`score_reliability`** (descriptive score–frequency agreement) |
| `ece` for TSCF | probability ECE | **`ece_probability`** (unchanged numerically) |
| `logloss` | absent | **`logloss_binary`** for TSCF only |
| binning | two conventions existed | single convention: 10 equal-width bins over [0,1], last bin includes 1.0, per-bin n reported |

---

## 3. Recomputed values (source: `metrics_v2.json`)

| Domain | Model | MSE (continuous) | Brier (severe event) | ECE (probability) | Score reliability | Probability-calibration valid |
|---|---|---:|---:|---:|---:|---|
| SCS | PersA | 0.01172 | 0.02706* | — | 0.066 | No |
| SCS | Ungated | 0.13341 | 0.18436* | — | 0.413 | No |
| SCS | TSAR | 0.01162 | 0.03282* | — | 0.077 | No |
| SCS | TSCF | 0.07128 | 0.11131 | 0.315 | — | **Yes** |
| GBR | PersA | 0.02331 | 0.04983* | — | 0.074 | No |
| GBR | Ungated | 0.13421 | 0.20366* | — | 0.406 | No |
| GBR | TSAR | 0.02332 | 0.05045* | — | 0.078 | No |
| GBR | TSCF | 0.09287 | 0.14036 | 0.332 | — | **Yes** |
| CARIB | PersA | 0.01413 | 0.03291* | — | 0.066 | No |
| CARIB | Ungated | 0.13673 | 0.19929* | — | 0.414 | No |
| CARIB | TSAR | 0.01410 | 0.03503* | — | 0.077 | No |
| CARIB | TSCF | 0.09761 | 0.11925 | 0.306 | — | **Yes** |

\* descriptive only: the model is not a probability output, so this number is
not probability calibration and is not used in the manuscript.

TSCF binary log loss: 0.406 (SCS), 0.499 (GBR), 0.427 (CARIB).

XGBoost (continuous regressor; probability metrics not applicable):

| Domain | Variant | MSE (continuous) | Score reliability | (descriptive binary Brier) |
|---|---|---:|---:|---:|
| CARIB | FULL | 0.02992 | 0.0715 | 0.03649 |
| CARIB | THERMAL | 0.03031 | 0.0729 | 0.03885 |
| CARIB | NO_THERMAL | 0.04017 | 0.0229 | 0.02091 |

The reversal between continuous MSE and the descriptive binary statistic is
retained here for transparency but is **not** used as a calibration claim.

---

## 4. Manuscript and SI locations changed

| Location | Change |
|---|---|
| Methods, TSAR paragraph | mixed "probability-style metrics" sentence replaced by explicit Task A / Task B definitions, binning rule, score-reliability caveat |
| Methods, Operational metrics | metric list split by task; probability metrics restricted to TSCF |
| Results, ungated-residual paragraph | "Brier error" → "continuous MSE"; "ECE" → "score reliability"; "miscalibration" → "score–frequency mismatch"; "calibration safety" rephrased |
| Fig. 3 caption | "ECE" → "ECE (probability)" |
| Fig. 4 caption (d) | "reliability (calibration recovery)" → "score–event-frequency agreement" with task-specific metric labels |
| Fig. 5 caption (b) | "Brier (calibration) … breaks calibration" → "continuous MSE … ~10× larger error and worse score–frequency agreement" |
| Fig. 6 caption (c) + note | Brier/ECE comparison removed; state that XGBoost is a continuous regressor and probability metrics do not apply |
| SI S2 | metric list split by task |
| SI S7 | "substantially worse in calibration" → continuous MSE / score reliability |
| SI S8 | "miscalibration" → score–frequency mismatch |
| SI S12 | "relies on ROC AUC, PR AUC, Brier and ECE" → ROC/PR; continuous MSE as secondary; probability metrics explicitly undefined |
| SI S12 note | "Brier/ECE annotated" → continuous MSE annotated |
| SI S13 | "reliability (Brier/ECE)" → "probability calibration (binary Brier 0.140, ECE 0.332)" |
| SI R2 | metric names corrected; explicit statement that these are not probability-calibration errors |
| SI Table 3(a) | columns now `MSE (continuous)`, `Brier (severe event)`, `ECE (probability)`, `Score reliability`; "—" where not applicable |
| SI Table 3(b) | same column split; rebuilt from `metrics_v2.json`, not hand-entered |

---

## 5. Tests and reproduction

```bash
cd LTTS_CRW/tact-coral

# 1. unit tests (12)
"$PIECF_ENV/Scripts/python.exe" -m pytest tests/test_metric_semantics.py -q
# -> 12 passed

# 2. canonical metric recomputation (asserts agreement with the audited values)
"$PIECF_ENV/Scripts/python.exe" experiments/08_infrastructure/recompute_calibration_metrics.py
# -> integrity checks passed; 21 rows

# 3. figure regeneration
cd scripts/figures
"$PIECF_ENV/Scripts/python.exe" fig3_transfer_boundary.py
"$PIECF_ENV/Scripts/python.exe" fig5_persa_anchor.py
"$PIECF_ENV/Scripts/python.exe" fig6_gated_residual.py
"$PIECF_ENV/Scripts/python.exe" fig_s_x_xgb_transfer.py

# 4. consistency verification (tables vs metrics_v2 vs figure source data)
cd ../..
"$PIECF_ENV/Scripts/python.exe" tests/verify_metric_consistency.py
# -> 69 checks passed, 0 failed (exit 0)
```

Verification results:
- `mse_cont` reproduces every audited continuous value exactly.
- `tests/verify_metric_consistency.py`: 69 checks, 0 failures.
- TSCF `brier_binary`, `ece_probability`, `logloss_binary` match the standard
  formulas; the probability path validates [0,1] and finite values.
- Table 3(b) values match `metrics_v2.json` to the displayed precision.
- Figures 3/4/5/6 source data match `metrics_v2.json`.
- No manuscript/SI text calls a continuous MSE "Brier", and no non-probability
  ECE is called probability calibration.
- All four figure scripts run without error and re-emit their QC reports.

---

## 6. Git diff summary

See the commit accompanying this report (`git show --stat`). High level:

- 3 new code/test files, 5 legacy-metric annotations, 4 figure-script edits.
- 2 new result files (`metrics_v2.json`, `metrics_v2.csv`) + bin detail.
- 5 figure source-data files regenerated; 4 figures re-rendered.
- 4 manuscript/SI files (md) updated and re-rendered to DOCX/PDF.
- Original result files (`structural_ablation.json`, `structural_ablation_corrected.json`, `xgb_transfer_control/metrics.json`) left untouched.

---

## 7. Unresolved / residual issues

1. **Legacy `brier` fields remain in the original result files.** They are
   deliberately preserved as the evidence-locked record and are now marked
   deprecated in code headers and in `LEGACY_FIELDS.md`. The manuscript no
   longer cites them.
2. **`scs_probability_calibration.py` (isotonic) is not cited in the paper.**
   Its map collapses 99.3% of test points to ≈0 and degrades the severe-case
   Brier from 0.031 to 0.667. It remains shipped as an unreferenced diagnostic.
   No isotonic claim was added to the manuscript.

---

## 8. New experiment requiring author approval

If the authors want probability calibration for PersA / TSAR / XGBoost, a
score→probability map must be fitted **on SCS validation only**, frozen, and
applied unchanged to GBR/CARIB. The existing isotonic template is degenerate
under 0.3% prevalence, so prevalence-aware calibration (class-weighted Platt,
or a cost-sensitive threshold) would need justification and validation. This
was **not** implemented.

---

## 9. Verdict

**FIXED AND VERIFIED** for the metric-definition inconsistency: code, result
files, figure source data, figures, manuscript and SI now use one coherent
Task A / Task B vocabulary, and every reported number is traceable to
`metrics_v2.json`.

Residual items in §7 are documented limitations, not open inconsistencies.
The experiment in §8 is out of scope and requires author approval.
