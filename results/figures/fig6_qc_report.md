# Figure 6 — final scientific / layout QC report

Generated: 2026-10-29T21:06:41.678913

## Sources (no hard-coded results)
- structural_ablation.json (point metrics + paired week-block
  bootstrap 1000, seed 42): panels b and d
- residual_diagnostics.csv (scale=0.4 frozen): panel c
- The only computed-on-the-fly quantity is the anchor-level MSE
  reference line in panel b (mean PersA MSE over the three domains
  = 0.0164).

## Embedded pre-save scientific assertions (all PASS)
- un-gated residual has MSE > 3× PersA in every
  domain
- gated − ungated CARIB ΔCSI@0.9 CI excludes 0
- gated − PersA CARIB ΔPR point estimate > 0
- no 'unique necessity' claim anywhere

## Changes applied (per specification)
- Panel a: time symbols unified — ŷ_final(t+4)=PersA(t)+g(t)·tanh(R(h_t))·s,
  g(t)=σ((y_main(t)−τ)/T); gate wording now 'the gate bounds the
  contextual correction, limiting its magnitude under domain shift'.
- Panel b: model names 'PersA / ungated residual / gated residual /
  frozen TSCF' (no A/B/C/D codes); annotation 'Ungated residual:
  ~10× larger MSE'.
- Panel c: annotation 'Gate values remain small across domains,
  constraining the residual correction.'; gate median, 97.5th
  percentile and mean |y−PersA| retained.
- Panel d: primary effect size switched from ΔROC to ΔCSI@0.9;
  comparisons labelled 'gated − PersA' and 'gated − ungated' (no C−A/C−B
  codes); zero-effect reference line drawn and labelled.
- Panels a–c data unchanged; four-panel structure, white background,
  sans-serif, colour-blind-safe palette retained.
