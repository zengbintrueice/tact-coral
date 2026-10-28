# AUDIT_PROBABILITY_METRICS

Date: 2026-10-08. Scope: TSAR-family output bounds and the Brier/ECE
convention used in the structural-ablation comparison (main Fig. 5/6, main
text Results, SI Table 3(b), SI Note R2).

## 1. Code-traced computation path (2.1)

| Step | File:lines | Fact |
|---|---|---|
| Backbone probability | `ablate_a_dh_stopgrad.py` (StopGradDualHeadPIECFModel.forward) | y_main = sigmoid(head(attn)) ∈ (0, 1) |
| Gate | same | g = sigmoid((y_main − 0.5)/0.1) ∈ (0, 1) |
| Bounded residual | same | res = 0.4 · tanh(sev_head(attn.detach())) ∈ (−0.4, 0.4); input detached |
| Final output | same | **y_final = y_main + gate · res — no sigmoid, no clip** |
| Nominal range | derived | y_final ∈ (−0.4, 1.4); restricting the residual does NOT bound the output to [0, 1] |
| Storage | inference loop | `p_fused.append(y_final[:, -1].cpu().numpy())` — raw, unclipped |
| Label | dataset | continuous coordinate y(t+4) = clip(DHW(t+4)/10, 0, 1) ∈ [0, 1] (not binary) |
| Loss / val / test tensors | `stopgrad_loss`, evaluate | all use y_final (last position); train loss = focal MSE (main) + λ_sev·MSE restricted to severe pixels (sev head); no clipping anywhere in the loss |

Conclusion: the paper's Eq. (3) matches the code exactly, but a bounded
residual does not make y_final a probability.

## 2. Output audit (2.1/2.2) — per basin, test split

| Domain | Model | n | min | max | mean | >1 frac | <0 frac |
|---|---|---|---|---|---|---|---|
| SCS | A_PersA | 49,322 | 0.000 | 1.000 | — | 0.00000 | 0.00000 |
| SCS | B_ungated | 49,322 | 0.056 | 1.311 | — | 0.00695 | 0.00000 |
| SCS | C_anchored | 49,322 | 0.003 | 1.203 | — | 0.00308 | 0.00000 |
| SCS | D_TSCF | 49,322 | 0.193 | 0.998 | — | 0.00000 | 0.00000 |
| GBR | A_PersA | 19,065 | 0.000 | 1.000 | — | 0.00000 | 0.00000 |
| GBR | B_ungated | 19,065 | 0.137 | 1.352 | — | 0.04972 | 0.00000 |
| GBR | C_anchored | 19,065 | 0.003 | 1.013 | — | 0.01505 | 0.00000 |
| GBR | D_TSCF | 19,065 | 0.236 | 0.997 | — | 0.00000 | 0.00000 |
| CARIB | A_PersA | 83,025 | 0.000 | 1.000 | — | 0.00000 | 0.00000 |
| CARIB | B_ungated | 83,025 | 0.029 | 1.393 | — | 0.02765 | 0.00000 |
| CARIB | C_anchored | 83,025 | 0.003 | 1.105 | — | 0.00660 | 0.00000 |
| CARIB | D_TSCF | 83,025 | 0.202 | 0.997 | — | 0.00000 | 0.00000 |

(Means/stds in `probability_metrics_audit.json`.) n identical across models
within each domain (paired design) — no sample-count unfairness.

## 3. Metric-input audit (2.2)

`compute_structural_ablation_analysis.py` (point estimates, sklearn):
- ROC AUC / PR AUC: raw scores vs binary severe target (y(t+4) ≥ 0.8);
  rank-based -> invariant to clipping. **[x]**
- CSI/POD/FAR/F1: `pred >= 0.90` on raw scores; τ fixed a priori. Invariant
  to clipping (clip preserves the ≥0.9 set). **[x]**
- Brier: `mean((pred − target)²)` on **raw** predictions against the
  **continuous** target coordinate. **[x]**
- ECE: 10 equal-width bins spanning **[0, 1)** with `pred < hi` —
  predictions ≥ 1.0 (and < 0) fall in no bin and are **silently excluded
  with zero penalty**; also PersA == 1.0 predictions were excluded.
  **Asymmetry confirmed**: only B/C are affected (0.3–5% of their
  predictions); A/D are untouched. This understates B's calibration error.
- Second implementation existed in `nc_style.reliability_curve` (fig 5d):
  out-of-range predictions clamped into the last bin with raw bin centers —
  a DIFFERENT convention from the ablation script. Both now unified.
- XGBoost path clips predictions before metrics
  (`pfv_evaluation_3level.py`) — third convention. Documented.

## 4. Chosen fix (2.3)

**Uniform clipping**: predictions of every model are clipped to [0, 1]
before Brier/ECE (the target coordinate lives in [0, 1], so clipping is the
natural range enforcement and applies identically to all models).
- ROC/PR/CSI/POD/FAR/F1 are mathematically unchanged by clipping.
- Brier stays computed against the continuous target coordinate (matches
  the training objective); ECE keeps the binary-severe reference but with
  the last bin including 1.0 (no silent exclusion).
- The TSAR output is therefore documented as a **bounded continuous
  thermal-risk score**, not a calibrated probability; the Brier/ECE
  comparison remains valid as a probability-style evaluation after the
  uniform range enforcement.
- No isotonic/probability mapping was fitted (the paper makes no
  probability-calibration claim for TSAR; adding one would require a
  validation-fitted mapping and is unnecessary for the claims made).

## 5. Old vs new numbers (2.4)

Brier (vs continuous target; old = raw, new = uniform-clip):

| Domain | Model | old | new |
|---|---|---|---|
| SCS | A / B / C / D | 0.01172 / 0.13389 / 0.01169 / 0.07128 | 0.01172 / 0.13341 / 0.01162 / 0.07128 |
| GBR | A / B / C / D | 0.02331 / 0.14208 / 0.02334 / 0.09287 | 0.02331 / 0.13421 / 0.02332 / 0.09287 |
| CARIB | A / B / C / D | 0.01413 / 0.14041 / 0.01412 / 0.09761 | 0.01413 / 0.13673 / 0.01410 / 0.09761 |

ECE (P(sev) reference; old = raw with exclusion, new = uniform-clip, last
bin inclusive):

| Domain | Model | old | new |
|---|---|---|---|
| SCS | A / B / C / D | 0.066 / 0.409 / 0.075 / 0.315 | 0.066 / 0.413 / 0.077 / 0.315 |
| GBR | A / B / C / D | 0.068 / 0.375 / 0.072 / 0.332 | 0.074 / 0.406 / 0.078 / 0.332 |
| CARIB | A / B / C / D | 0.064 / 0.396 / 0.075 / 0.307 | 0.066 / 0.414 / 0.077 / 0.307 |

Paired contrasts (new convention): B−A dBrier +0.122 (SCS) / +0.111 (GBR) /
+0.123 (CARIB); B−A dECE +0.347 / +0.332 / +0.348; C−B dBrier −0.122 /
−0.111 / −0.123 and dECE −0.336 / −0.328 / −0.337.

## 6. Do the paper's claims survive?

- "The ungated residual breaks calibration" — **YES, strengthened**: under
  the fair convention B's ECE rises (its previously excluded overconfident
  errors are now counted).
- "The gate restores calibration safety" — **YES**: C's ECE returns to the
  anchor level (0.077–0.078 vs B's 0.41); C−B dECE ≈ −0.33 in all basins.
- "Clipping alone does not rescue calibration" — **YES, by construction**:
  under the uniform convention every model is clipped, and B still fails
  (mid-range miscalibration).
- "Brier error up ≈ 0.126" → updated to **≈ 0.123**; "ECE up ≈ 0.332" →
  **≈ 0.348** (main text updated).
- ROC/PR/CSI/POD/FAR/F1 values: unchanged.

## 7. Files changed

- SI Table 3(b): Brier/ECE columns updated (12 rows); the "Ungated (clipped)"
  row merged into "Ungated" (identical under the uniform convention); caption
  states the convention.
- SI Note R2: rewritten under the unified convention.
- Main Methods (TSAR): output-semantics + metric-convention sentence updated.
- Main Results: two Brier/ECE deltas updated.
- `scripts/figures/nc_style.py` (reliability_curve clips internally),
  `scripts/figures/fig6_gated_residual.py` (reads corrected JSON),
  fig5/fig6 regenerated.
- Corrected metrics: `results/qc/probability_metrics_audit/
  structural_ablation_corrected.json`; audit JSON:
  `probability_metrics_audit.json`. Original result files untouched.
