# Seed-sensitivity QC (5-seed D1 severe-classifier)

NO retraining; diagnostics from saved checkpoints + prediction behaviour. Per-epoch training loss/val and gradient norms were NOT persisted, so this report separates observation-based evidence from inference and distinguishes genuine seed variation from a likely training-degeneration fingerprint.

| seed | best_val_tau | SCS CSI@0.9 | weight NaN/Inf | trainable L2 | pred mean / std | frac>0.9 | severe gap | ρ(pred,target) | SCS PR |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.925 | 0.3649 | 0/0 | 3.37e+02 | 0.2134/0.1284 | 0.0070 | 0.6084 | 0.352 | 0.5663 |
| 1 | 0.875 | 0.1593 | 0/0 | 3.42e+02 | 0.2525/0.1096 | 0.0060 | 0.3963 | 0.205 | 0.2353 |
| 2 | 0.575 | 0.0000 | 0/0 | 2.94e+02 | 0.4857/0.0555 | 0.0000 | 0.0631 | 0.702 | 0.0280 |
| 3 | 0.925 | 0.2980 | 0/0 | 3.47e+02 | 0.0418/0.1234 | 0.0086 | 0.6225 | 0.342 | 0.4284 |
| 42 | 0.925 | 0.3374 | 0/0 | 3.24e+02 | 0.3235/0.1139 | 0.0113 | 0.5518 | 0.514 | 0.5463 |

## Seed 2 (catastrophic: CSI=0 everywhere, very low PR)
- checkpoint: no NaN/Inf; trainable L2 2.94e+02 in line with other seeds.
- best_val_tau = 0.575 (others 0.875–0.925): the val-selected operating point collapsed — in-domain severe scores are barely separated.
- SCS prediction behaviour: mean 0.4857, std 0.0555 (near-constant), fraction >0.90 = 0.0000, severe gap 0.0631.
- GBR ROC 0.376 (inverted) and CARIB ROC 0.460 with PR ≈0.01–0.03: consistent with a near-constant low-variance output whose small residual ordering aligns weakly with SCS labels but anti-aligns with GBR — a degenerate 'flat-prior' solution under pos_weight=1000.
- Verdict: likely a training-degeneration seed (failed to escape a flat minimum), NOT a data or leakage artefact.

## Seed 3 (GBR drop)
- GBR ROC 0.842 / PR 0.277 / CSI 0.078 vs seeds 0/42 (0.910/0.509/0.298 and 0.962/0.554/0.365) — a real transfer-specific spread.
- SCS in-domain intact (ROC 0.953 / PR 0.428): the drop is transfer-specific (less transferable representation), not an in-domain collapse.

## Conclusion
Seeds 0 and 42 cluster at high performance; seed 1 is mildly weaker (SCS PR 0.235); seed 3 shows a transfer-specific GBR drop; seed 2 is a degenerate run (near-constant severe scores, collapsed val-tau). This is genuine random-seed sensitivity of a heavily imbalanced training setup (pos_weight=1000), not an evaluation bug. The transfer-boundary DIRECTION (GBR >> CARIB in PR-AUC) is preserved for all non-degenerate seeds.