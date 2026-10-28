# Figure 5 — final scientific QC report

Generated: 2026-10-01T18:50:10.581540

All plotted values are read from:
- results/physics_anchored/structural_ablation/structural_ablation.json  (per-model point metrics + paired week-block bootstrap CIs)
- results/physics_anchored/structural_ablation/rec_structural_ablation.npz  (per-pixel predictions for the reliability curves in panel d)
Nothing is hard-coded.

## Bootstrap configuration (from the frozen protocol, read from the
result JSON)

- method: ocean-masked shared: finite y(t+4) & ~land_mask & all 4 models finite
- n_boot: 1000, seed: 42, threshold: 0.9, scale: 0.4

## Panel-by-panel verification

| Panel | Domain | Model | n | n_severe | ROC | PR | CSI@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| b/c | SCS | A_PersA | 49322 | 419 | 0.9817 | 0.3177 | 0.1571 | 0.01172 | 0.0658 |
| b/c | SCS | C_anchored | 49322 | 419 | 0.9808 | 0.3036 | 0.1977 | 0.01169 | 0.0754 |
| b/c | SCS | D_TSCF | 49322 | 419 | 0.9865 | 0.5463 | 0.3374 | 0.07128 | 0.3150 |
| b/c | GBR | A_PersA | 19065 | 676 | 0.9593 | 0.4164 | 0.2494 | 0.02331 | 0.0676 |
| b/c | GBR | C_anchored | 19065 | 676 | 0.9599 | 0.4877 | 0.2541 | 0.02334 | 0.0717 |
| b/c | GBR | D_TSCF | 19065 | 676 | 0.9624 | 0.5543 | 0.3648 | 0.09287 | 0.3317 |
| b/c | CARIB | A_PersA | 83025 | 1356 | 0.9792 | 0.4602 | 0.2807 | 0.01413 | 0.0640 |
| b/c | CARIB | C_anchored | 83025 | 1356 | 0.9822 | 0.5479 | 0.3041 | 0.01412 | 0.0750 |
| b/c | CARIB | D_TSCF | 83025 | 1356 | 0.6101 | 0.0321 | 0.0262 | 0.09761 | 0.3065 |

Panels b and c plot the point estimates above with the bootstrap 95% CIs
stored in the same JSON ('bootstrap' dict per model); panel c additionally
plots B_ungated (not shown here for brevity; read from the same source).
Panel d recomputes the 10-bin reliability, Brier and ECE from the stored
CARIB predictions (rec_structural_ablation.npz) via the SAME binning
convention as the official evaluation; values match the JSON Brier/ECE in
the table above.

## Wording / layout changes applied (per specification)
- Panel a scientific-units QC (final): schematic relabelled 'schematic
  12-week HotSpot history (illustrative — not observations)'; the phrase
  'daily HotSpot' removed — the strip shows 12 WEEK positions representing
  the 84-day accumulation window, and the flow reads 'accumulate HotSpot
  values ≥ 1 °C / 12-week accumulation — 12 week positions = 84 days; each
  qualifying week contributes its HotSpot value'. Unit logic is therefore
  consistent: the 12 qualifying WEEKLY contributions sum directly to DHW
  in °C·weeks (DHW = 7.4, PersA = 0.74 for the illustrative values,
  computed in code); no implication that daily °C values are summed.
  Threshold label: 'HotSpot accumulation threshold = 1 °C'. The repeated
  'clip(·/10, 0, 1)' side-label was removed — the formal formula
  PersA(t)=clip(DHW(t)/10,0,1) is the only mapping shown.
- Panel c: display labels compressed to 'PersA | ungated | gated | TSCF'
  for spacing; the model labels stored in fig5_source_data.csv are
  UNCHANGED (display-only change).
- Panels b/d and ALL plotted values, CIs, Brier/ECE unchanged; no new
  experiments. The only non-file numbers are the clearly-labelled
  illustrative schematic values in panel a (synthetic, computed in code).