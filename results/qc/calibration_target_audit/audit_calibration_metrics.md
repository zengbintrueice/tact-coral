# Audit — Brier score, ECE and target-variable consistency

**Status: audit only. No manuscript text, figure, table or result file has been
modified.** All recomputation writes only into this directory
(`results/qc/calibration_target_audit/`).

Repository: `LTTS_CRW/tact-coral` (TACT). Seed-42, four-domain record sets as
shipped.

---

## 0. Executive summary

Four findings.

1. **Two different quantities are both called "Brier".**
   - `compute_structural_ablation_analysis.py`, `xgb_transfer_control.py`,
     `compute_physics_anchored_analysis.py`, `compute_persistence_baseline.py`,
     `pfv_utils.py` and `nc_style.reliability_curve` compute
     `mean((pred − y(t+4))²)` against the **continuous** coordinate. This is a
     squared error on a continuous target, not the Brier score of a binary event.
   - `calibration_curves_4group.py` and `scs_probability_calibration.py`
     compute `mean((pred − 1[y(t+4) ≥ 0.8])²)`, the **standard binary Brier**.
   The manuscript's Tables/Figures use the first; the calibration scripts use the
   second.

2. **ECE is always computed with a binary event, but is applied to scores that are
   not probabilities.** The formula is
   `Σ_k (n_k/N)·|mean(1[y≥0.8])_k − mean(pred)_k|`. It is a legitimate calibration
   error only when `pred` is a probability of that event. Of the four structural
   variants, only **D (TSCF)** is a probability by construction (weighted-BCE
   sigmoid). **A (PersA)**, **B (ungated)** and **C (TSAR)** are continuous
   risk-coordinate estimates trained with squared error; their ECE values are
   "score-versus-event-frequency agreement", not probability calibration.

3. **One published conclusion is definition-dependent.** For the XGBoost control,
   the continuous-Brier ordering and the binary-Brier ordering are **reversed**:
   removing thermal channels *increases* continuous error but *decreases* binary
   Brier and ECE in every domain. The caption claim "removing thermal clearly
   drops" holds only under the continuous-MSE definition.

4. **The paper already states the mixed convention** (Methods: "Brier is computed
   against the continuous target coordinate; ECE treats the score as P(severe
   event)"). The problem is not concealment but an incoherent pair of definitions
   presented as one "probability-style" metric set.

No numerical value in the audit reproduces incorrectly: all published Brier/ECE
point estimates were reproduced exactly from the stored predictions
(`audit_metrics.csv`, 0 mismatches against `structural_ablation_corrected.json`).

---

## 1. Definitions in the repository

### 1.1 Targets

| Quantity | Definition | Code |
|---|---|---|
| Continuous thermal-risk coordinate | `y(t) = clip(DHW(t)/10, 0, 1)` | `data/source_builders/build_v9_1deg_from_raw.py:614`; `jensen_dhw.py:245` |
| Forecasting target | `y_lead = y(t+4)` | `src/data/aligned_dataset.py:666` |
| Valid-cell mask | `isfinite(y_lead) & ~land_mask` | `src/data/aligned_dataset.py:709` |
| Severe event | `1[y(t+4) ≥ 0.8]` (DHW ≥ 8 °C·week) | `train_severe_classifier.py:39,126`; `physics_anchored_model.py:131` |

### 1.2 Model outputs and training targets

| Model | Output definition | Range (observed) | Training target / loss | Is the output P(severe)? |
|---|---|---|---|---|
| **A PersA** | `clip(DHW(t)/10,0,1)` | [0.00, 1.00] | none (deterministic) | **No** — normalized DHW coordinate |
| **B ungated** | `PersA + tanh(R(attn.detach()))·0.4` | [0.03, 1.39] | focal-MSE on `y_lead` (continuous) + `λ_sev·MSE(·, y_lead)` on severe cells | **No** |
| **C TSAR** | `PersA + σ((y_main−0.5)/0.1)·tanh(R)·0.4` | [0.003, 1.21] | same as B, with gate | **No** |
| **D TSCF** | `σ(MLP(A))` | [0.19, 1.00] | weighted BCE on `1[y_lead≥0.8]`, pos_weight 1000 | **Yes**, by construction |
| **XGBoost** | continuous regression on `clip(DHW/10)` | [−0.01, 0.78] | squared error on `y_lead` | **No** |

Code: `experiments/04_structural_ablation/physics_anchored_model.py:68–89`;
`experiments/01_training/train_severe_classifier.py:126–130,42–49`;
`experiments/05_thermal_attribution/xgb_transfer_control.py`.

### 1.3 Record sets

Ocean-masked shared set: `finite y(t+4) & ~land_mask & all four models finite`
(`compute_structural_ablation_infer.py:134,181`).

| Domain | n | n_severe | prevalence | severe-event years |
|---|---:|---:|---:|---|
| SCS | 49,322 | 419 | 0.85% | 2016–2019 |
| GBR | 19,065 | 676 | 3.55% | 2017–2020 |
| CARIB | 83,025 | 1,356 | 1.63% | 2017–2020 |

---

## 2. Metric implementations

### 2.1 "Brier" — family 1: continuous squared error

```
Brier_cont = (1/N) Σ_i (clip(pred_i,0,1) − y_i)²      y_i ∈ [0,1] continuous
```

| File | Line | Note |
|---|---|---|
| `experiments/08_infrastructure/compute_structural_ablation_analysis.py` | 120 (point), 89 (bootstrap) | **source of paper Table 3** |
| `experiments/05_thermal_attribution/xgb_transfer_control.py` | 197 | **source of paper Table 5 / Fig. 6** |
| `experiments/08_infrastructure/compute_physics_anchored_analysis.py` | 59 | |
| `experiments/08_infrastructure/compute_persistence_baseline.py` | 210–212 | |
| `experiments/04_structural_ablation/pfv_utils.py` | 77, 82 | |
| `scripts/figures/nc_style.py` (`reliability_curve`) | 155 | **source of Fig. 4d** |

### 2.2 "Brier" — family 2: standard binary Brier

```
Brier_bin = (1/N) Σ_i (clip(pred_i,0,1) − 1[y_i ≥ 0.8])²
```

| File | Line | Note |
|---|---|---|
| `experiments/06_calibration/calibration_curves_4group.py` | 126–129 | binary target |
| `experiments/02_scs_benchmark/scs_probability_calibration.py` | 52–55, called at 194–200 with binary labels | isotonic module |

### 2.3 ECE — single formula, inconsistent applicability

```
ECE = Σ_k (n_k/N) · | mean(1[y ≥ 0.8])_k − mean(clip(pred))_k |
```
10 equal-width bins. Two binning variants exist:
- `compute_structural_ablation_analysis.py:121–128`: bins `[lo, hi)`, `pred < hi`,
  **no clipping** (original); predictions ≥ 1.0 silently excluded.
- Corrected convention (paper Tables, `nc_style.reliability_curve`,
  `structural_ablation_corrected.json`): clip to [0,1], last bin includes 1.0.

Other copies: `xgb_transfer_control.py:198–205`,
`compute_physics_anchored_analysis.py:60–67`, `compute_persistence_baseline.py:178–193`,
`scs_probability_calibration.py:58–75`, `scs_reliability_diagram.py:95–117`.

**ECE is a probability-calibration error only if `pred` is P(event).** For A, B, C
and XGBoost it is a descriptive "score-versus-event-frequency" statistic.

---

## 3. Audit table (recomputed from stored predictions)

`pred` ranges, clipping fraction, continuous MSE, standard binary Brier, ECE.
Full precision in `audit_metrics.csv`; script: `audit_recompute.py`.

| Domain | Model | n | sev | pred range | % outside [0,1] | **MSE_cont (published "Brier")** | **Brier_bin (standard)** | R² (continuous) | ρ(pred, y) | ECE (published) | ECE valid as probability calibration? |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| SCS | A PersA | 49,322 | 419 | 0.000–1.000 | 0.0% | 0.01172 | **0.02706** | 0.569 | 0.775 | 0.0665 | **No** |
| SCS | B ungated | 49,322 | 419 | 0.056–1.311 | 0.70% | 0.13341 | **0.18436** | −3.91 | 0.169 | 0.4131 | **No** |
| SCS | C TSAR | 49,322 | 419 | 0.003–1.203 | 0.31% | 0.01162 | **0.03282** | 0.572 | 0.786 | 0.0771 | **No** |
| SCS | D TSCF | 49,322 | 419 | 0.193–0.998 | 0.0% | 0.07128 | **0.11131** | −1.62 | 0.514 | 0.3150 | **Yes** |
| GBR | A PersA | 19,065 | 676 | 0.000–1.000 | 0.0% | 0.02331 | **0.04983** | 0.555 | 0.783 | 0.0736 | **No** |
| GBR | B ungated | 19,065 | 676 | 0.137–1.352 | 4.97% | 0.13421 | **0.20366** | −1.56 | 0.552 | 0.4060 | **No** |
| GBR | C TSAR | 19,065 | 676 | 0.003–1.013 | 1.51% | 0.02332 | **0.05045** | 0.555 | 0.629 | 0.0780 | **No** |
| GBR | D TSCF | 19,065 | 676 | 0.236–0.997 | 0.0% | 0.09287 | **0.14036** | −0.77 | 0.331 | 0.3317 | **Yes** |
| CARIB | A PersA | 83,025 | 1,356 | 0.000–1.000 | 0.0% | 0.01413 | **0.03291** | 0.602 | 0.802 | 0.0661 | **No** |
| CARIB | B ungated | 83,025 | 1,356 | 0.029–1.393 | 2.77% | 0.13673 | **0.19929** | −2.85 | 0.489 | 0.4141 | **No** |
| CARIB | C TSAR | 83,025 | 1,356 | 0.003–1.105 | 0.66% | 0.01410 | **0.03503** | 0.603 | 0.710 | 0.0772 | **No** |
| CARIB | D TSCF | 83,025 | 1,356 | 0.202–0.997 | 0.0% | 0.09761 | **0.11925** | −1.75 | 0.116 | 0.3065 | **Yes** |

Key reading:
- The published "Brier" is **1.3–2.9× smaller** than the standard binary Brier.
- R² is negative for B and D: squared error against the continuous coordinate is
  the wrong loss for a probability output (D) and for a residual-corrected
  coordinate (B). A model can rank well (ROC 0.986) and still have R² < 0.
- B is worst under **both** definitions (continuous 0.133–0.137 vs 0.012–0.023;
  binary 0.184–0.204 vs 0.027–0.050). The ungated-residual conclusion is robust.

### 3.1 XGBoost control (recomputed from `xgb_transfer_control/predictions.npz`)

| Domain | Model | MSE_cont (published "Brier") | Brier_bin | ECE |
|---|---|---:|---:|---:|
| SCS | FULL | 0.00404 | 0.02359 | 0.0671 |
| SCS | THERMAL | 0.00418 | 0.02306 | 0.0658 |
| SCS | NO_THERMAL | **0.01007** (worst) | **0.01891** (best) | **0.0582** (best) |
| GBR | FULL | 0.01959 | 0.03606 | 0.0619 |
| GBR | THERMAL | 0.02028 | 0.03423 | 0.0509 |
| GBR | NO_THERMAL | **0.04631** (worst) | **0.02945** (best) | **0.0290** (best) |
| CARIB | FULL | 0.02992 | 0.03649 | 0.0715 |
| CARIB | THERMAL | 0.03031 | 0.03885 | 0.0729 |
| CARIB | NO_THERMAL | **0.04017** (worst) | **0.02091** (best) | **0.0229** (best) |

**The ranking of NO_THERMAL flips between the two definitions in all three
domains.** The ranking metrics (ROC/PR) are unaffected and still favour
THERMAL/FULL.

---

## 4. Task separation (what should be reported)

**Task 1 — continuous thermal-risk estimation.** Applies to A, B, C, XGBoost.
Metric: `MSE` / `RMSE` / `MAE` against `y(t+4)`, plus `R²` and rank correlation.
This is **not** a Brier score.

**Task 2 — severe-event probability prediction.** Applies to D (TSCF) only,
unless a score→probability mapping is explicitly fitted and validated.
Metrics: binary Brier, ECE, log loss (reference: 0.406 / 0.499 / 0.427 for D in
SCS / GBR / CARIB), ROC/PR/CSI.

For A, B, C and XGBoost, the ECE-style quantity can be reported only as a
**score-reliability diagnostic** (does the score numerically track event
frequency?) with an explicit label, not as probability calibration.

---

## 5. Claim-by-claim impact

| # | Location | Current claim | Audit verdict |
|---|---|---|---|
| 1 | Methods (main, "TSAR" paragraph) | "probability-style metrics (Brier, ECE) … Brier is computed against the continuous target coordinate; ECE treats the score as P(severe event)" | **Internally incoherent.** Must be replaced by the two-task definitions (§4). |
| 2 | Table 3(a,b), Brier column | "Brier" values 0.0117–0.1367 | **Mislabel.** These are continuous MSE; standard binary Brier is 0.027–0.204. Values themselves reproduce exactly. |
| 3 | Table 3(a,b), ECE column | ECE for PersA/TSAR | **Valid as a score-reliability statistic, not as probability calibration** for A/B/C. Valid only for D. |
| 4 | Main text, "the ungated residual is unsafe" | "Brier error up ≈ 0.123 … ECE up ≈ 0.348 … systematic mid-range miscalibration" | **Direction robust** (B worst under both definitions). Wording "Brier"→"continuous squared error", "miscalibration"→"score reliability" must be fixed. |
| 5 | Main text, "gate's … calibration safety" | "calibration safety" | **Not established as probability calibration**; rephrase as range/reliability protection. |
| 6 | Fig. 4 caption (d) | "CARIB reliability (calibration recovery)" | **Overstated for PersA/TSAR** (not probabilities). Rename to score-vs-event-frequency agreement. |
| 7 | Fig. 5 caption (b) | "CSI@0.9 (ranking) versus Brier (calibration); the un-gated residual breaks calibration" | **Mislabel.** MSE vs binary Brier differ; "breaks calibration" unsupported for a non-probability score. |
| 8 | Fig. 6 caption (c) | "CARIB contrast with Brier/ECE: thermal-only ~ full, removing thermal clearly drops" | **Definition-dependent and reversed under binary Brier.** Must be restricted to continuous MSE (and relabelled) or removed. |
| 9 | Fig. 3 caption (a) | heatmap includes ECE | **Valid** — panel uses only `D_TSCF`, a probability. |
| 10 | SI R2 | "ungated residual's calibration failure persists … Brier 0.133–0.137 and ECE 0.41 … versus 0.012–0.023 and 0.07" | Same reclassification as #4; numbers correct under their definitions. |
| 11 | SI S12 (XGB) | "comparison relies on ROC AUC, PR AUC, Brier and ECE" | **Mislabel** for a continuous regressor; S12 already says it does not validate calibration (line 291) — the two sentences conflict. |
| 12 | SI S13 (GBR) | "GBR reliability (Brier/ECE): … strongly miscalibrated (ECE 0.332)" | **ECE valid** (TSCF); "Brier 0.0929" is continuous MSE, not binary Brier. |
| 13 | SI S2 | temporal purge reports "Brier and ECE" | Naming issue only. |
| 14 | `scs_probability_calibration.py` (shipped, run by `reproduce_main.sh`) | isotonic ECE 0.328→0.003, Brier 0.121→0.005 | **Not cited in the manuscript.** The map collapses 99.3% of test points to ≈0; severe-case Brier degrades 0.031→0.667. If ever reported, must carry this caveat. |

### Conclusions that require no change
- TSCF is a probability and is poorly calibrated despite strong ranking
  (ECE 0.315 / 0.332 / 0.307; binary Brier 0.111 / 0.140 / 0.119). **Verified.**
- The ungated residual has ~10× larger continuous squared error and much worse
  score-frequency agreement than PersA/TSAR in all domains. **Verified under both
  definitions.**
- XGBoost ranking: NO_THERMAL < THERMAL/FULL in ROC/PR. **Verified and unaffected.**

### Conclusions that do not currently stand as stated
- "PersA/TSAR are calibrated" (in the probability sense). **Not established** —
  they are not probabilities and no score→probability map was fitted for them.
- "Removing thermal worsens XGBoost calibration-type scores." **False under the
  standard binary Brier**; true only for continuous MSE.
- "The gate provides calibration safety." **Not established** as probability
  calibration; established only as limiting score deviation from the anchor.

### Requires a new experiment (if probability calibration of anchored models is
desired)
Fit a monotone score→probability map (isotonic or Platt) on **SCS validation**
only, freeze it, apply unchanged to GBR/CARIB, then compute binary Brier/ECE.
The existing `scs_probability_calibration.py` is a template but its isotonic fit
is degenerate under 0.3% prevalence (severe-case Brier 0.031→0.667), so a
prevalence-aware calibration (e.g., Platt with class weights, or a cost-sensitive
threshold) would need to be justified and validated. Until then, no probability
claim should be made for A/B/C.

---

## 6. Recommended corrections (NOT YET APPLIED)

### 6.1 Code (proposed, not executed)

1. In `compute_structural_ablation_analysis.py`, `xgb_transfer_control.py`,
   `compute_physics_anchored_analysis.py`, `compute_persistence_baseline.py`,
   `pfv_utils.py`, `nc_style.reliability_curve`:
   - rename the existing field `brier` → `mse_cont`;
   - add `brier_bin = mean((clip(pred) − 1[y≥0.8])²)` and
     `logloss_bin` only for outputs flagged `is_probability=True`;
   - rename `ece` → `ece_score_reliability` for non-probability outputs, keep
     `ece` for TSCF;
   - emit a `probability_valid` flag per model.
2. Keep `calibration_curves_4group.py` and `scs_probability_calibration.py`
   unchanged (they already use the binary Brier).
3. Regenerate `structural_ablation_corrected.json`, the figure source data
   (`fig5_source_data*.csv`, `fig6_source_data*.csv`, `SX_source_data.csv`,
   `fig3_source_data_metrics.csv`) and Figures 3/4/5/6.

### 6.2 Text (proposed, not executed)

- Methods: replace the mixed sentence with the two-task definitions (§4).
- Table 3: split the Brier column into `MSE (risk coordinate)` for all models and
  `Brier (severe event)` for TSCF; keep ECE but label it "ECE (probability)" only
  for TSCF, "score reliability" otherwise.
- Main text: "Brier error" → "continuous squared error"; "miscalibration" →
  "score–frequency mismatch"; delete "calibration safety" or restate.
- Figure captions 4d, 5b, 6c as in §5.
- SI R2, S12, S13, S2 as in §5.
- Do **not** change any model, training protocol, split or threshold.

### 6.3 What must not be done
- Do not relabel the continuous MSE as "Brier" without qualification.
- Do not convert PersA/TSAR scores to probabilities without a fitted, validated
  mapping.
- Do not keep the XGB "removing thermal clearly drops" claim under a
  Brier/calibration label; it is an MSE statement.

---

## 7. Reproduction

```bash
cd LTTS_CRW/tact-coral
# audit table (read-only; writes only into results/qc/calibration_target_audit/)
"$PIECF_ENV/Scripts/python.exe" results/qc/calibration_target_audit/audit_recompute.py
```

Outputs: `audit_metrics.csv`, `audit_metrics.json`.

Everything in §3 was recomputed from
`results/physics_anchored/structural_ablation/rec_structural_ablation.npz`
(MD5 `4035a19fd30fa4d14151f4c8055dfe59`) and
`results/xgb_transfer_control/predictions.npz`; no synthetic values.

---

## 8. Bottom line

**Verified:** TSCF is a probability and is genuinely miscalibrated; the ungated
residual has far larger continuous error and far worse score–frequency agreement;
XGBoost ranking conclusions are unaffected.

**Not established:** any probability-calibration claim for PersA, TSAR or the
ungated residual; the "calibration safety" role of the gate; the XGB
"removing thermal drops Brier/ECE" statement under the standard binary Brier.

**Needs new experiments:** a validated, frozen score→probability mapping for the
anchored family, if probability calibration is to be claimed for it.
