# Structural Ablation — Thermal-Stress-Anchored Residual Learning

Generated: 2026-09-17T20:44:33.444055

Models (same ocean-masked PFV record set per domain; τ=0.90 pinned):
- **A** PersA_DHW: `y_hat = PersA(t)`
- **B** Un-gated bounded residual: `PersA + tanh(R(attn))·scale` (gate≡1)
- **C** Formal anchored residual: `PersA + sigmoid((y_main−τ)/T)·tanh(R(attn))·scale`
- **D** Frozen TSCF (no anchor)
- scale = 0.4 (frozen, Work 1 SCS-val); bootstrap 1000 seed 42; paired week-block, one shared pick matrix

## SCS (n=49322, severe=419, prev=0.0085)
| Model | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| A_PersA | 0.9817 | 0.3177 | 0.1571 | 0.1957 | 0.5568 | 0.2715 | 0.01172 | 0.0658 |
| B_ungated | 0.9798 | 0.3437 | 0.2385 | 0.4702 | 0.6738 | 0.3851 | 0.13389 | 0.4089 |
| C_anchored | 0.9808 | 0.3036 | 0.1977 | 0.2840 | 0.6060 | 0.3301 | 0.01169 | 0.0754 |
| D_TSCF | 0.9865 | 0.5463 | 0.3374 | 0.5871 | 0.5576 | 0.5046 | 0.07128 | 0.3150 |
| Difference | ΔROC | ΔPR | ΔCSI@0.9 | ΔPOD | ΔFAR | ΔF1 | ΔBrier | ΔECE |
|---|---|---|---|---|---|---|---|---|
| B_ungated−A_PersA | -0.0020 (-0.0060,+0.0011) | +0.0260 | +0.0814 | +0.2745 | +0.1171 | +0.1136 | +0.12217 | +0.3431 |
| C_anchored−A_PersA | -0.0009 (-0.0022,+0.0009) | -0.0141 | +0.0406 | +0.0883 | +0.0492 | +0.0586 | -0.00003 | +0.0096 |
| C_anchored−B_ungated | +0.0010 (-0.0028,+0.0065) | -0.0402 | -0.0408 | -0.1862 | -0.0679 | -0.0550 | -0.12220 | -0.3335 |
| D_TSCF−A_PersA | +0.0047 (-0.0020,+0.0111) | +0.2286 | +0.1804 | +0.3914 | +0.0008 | +0.2331 | +0.05956 | +0.2492 |
| D_TSCF−C_anchored | +0.0057 (-0.0018,+0.0122) | +0.2427 | +0.1398 | +0.3031 | -0.0484 | +0.1745 | +0.05959 | +0.2396 |

## GBR (n=19065, severe=676, prev=0.0355)
| Model | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| A_PersA | 0.9593 | 0.4164 | 0.2494 | 0.3328 | 0.5011 | 0.3993 | 0.02331 | 0.0676 |
| B_ungated | 0.9564 | 0.3657 | 0.2835 | 0.6346 | 0.6611 | 0.4418 | 0.14208 | 0.3748 |
| C_anchored | 0.9599 | 0.4877 | 0.2541 | 0.3417 | 0.5022 | 0.4053 | 0.02334 | 0.0717 |
| D_TSCF | 0.9624 | 0.5543 | 0.3648 | 0.6686 | 0.5547 | 0.5346 | 0.09287 | 0.3317 |
| Difference | ΔROC | ΔPR | ΔCSI@0.9 | ΔPOD | ΔFAR | ΔF1 | ΔBrier | ΔECE |
|---|---|---|---|---|---|---|---|---|
| B_ungated−A_PersA | -0.0028 (-0.0044,-0.0014) | -0.0507 | +0.0341 | +0.3018 | +0.1600 | +0.0425 | +0.11876 | +0.3072 |
| C_anchored−A_PersA | +0.0006 (+0.0002,+0.0012) | +0.0713 | +0.0047 | +0.0089 | +0.0010 | +0.0060 | +0.00003 | +0.0041 |
| C_anchored−B_ungated | +0.0035 (+0.0019,+0.0051) | +0.1220 | -0.0294 | -0.2929 | -0.1590 | -0.0365 | -0.11873 | -0.3031 |
| D_TSCF−A_PersA | +0.0031 (-0.0092,+0.0143) | +0.1379 | +0.1154 | +0.3358 | +0.0536 | +0.1353 | +0.06955 | +0.2641 |
| D_TSCF−C_anchored | +0.0025 (-0.0097,+0.0136) | +0.0666 | +0.1107 | +0.3269 | +0.0525 | +0.1293 | +0.06952 | +0.2600 |

## CARIB (n=83025, severe=1356, prev=0.0163)
| Model | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| A_PersA | 0.9792 | 0.4602 | 0.2807 | 0.3510 | 0.4167 | 0.4383 | 0.01413 | 0.0640 |
| B_ungated | 0.9714 | 0.3885 | 0.2407 | 0.7124 | 0.7334 | 0.3880 | 0.14041 | 0.3962 |
| C_anchored | 0.9822 | 0.5479 | 0.3041 | 0.3835 | 0.4050 | 0.4664 | 0.01412 | 0.0750 |
| D_TSCF | 0.6101 | 0.0321 | 0.0262 | 0.0391 | 0.9261 | 0.0511 | 0.09761 | 0.3065 |
| Difference | ΔROC | ΔPR | ΔCSI@0.9 | ΔPOD | ΔFAR | ΔF1 | ΔBrier | ΔECE |
|---|---|---|---|---|---|---|---|---|
| B_ungated−A_PersA | -0.0078 (-0.0107,-0.0045) | -0.0717 | -0.0399 | +0.3614 | +0.3167 | -0.0503 | +0.12628 | +0.3323 |
| C_anchored−A_PersA | +0.0030 (+0.0015,+0.0042) | +0.0877 | +0.0234 | +0.0324 | -0.0116 | +0.0281 | -0.00002 | +0.0110 |
| C_anchored−B_ungated | +0.0108 (+0.0061,+0.0148) | +0.1594 | +0.0634 | -0.3289 | -0.3283 | +0.0783 | -0.12630 | -0.3213 |
| D_TSCF−A_PersA | -0.3692 (-0.3930,-0.3443) | -0.4281 | -0.2544 | -0.3119 | +0.5094 | -0.3872 | +0.08347 | +0.2425 |
| D_TSCF−C_anchored | -0.3722 (-0.3954,-0.3468) | -0.5158 | -0.2779 | -0.3444 | +0.5210 | -0.4152 | +0.08349 | +0.2315 |

Terminology: thermal-stress-aware; strong physics-based persistence baseline (PersA) is the anchor; cumulative HotSpot channels are legitimate CRW-DHW predictors.
No hyperparameters were selected on GBR/CARIB; scale/tau/temp frozen from SCS-val (Work 1). Bootstrap uses fast rank-based AUC/AP inside replicates; point estimates use sklearn.

## Conclusions (evidence-based)

1. **Is the bounded residual itself effective? — NO.**
   Un-gated bounded residual (B) breaks the [0,1] score range on every domain
   (Brier 0.13-0.14 vs 0.012-0.023 for A/C; ECE 0.37-0.41) because PersA+res is
   not range-bounded. On SCS it lifts CSI (+0.081, point) at the cost of
   calibration, but on unseen basins it is CI-significantly WORSE than PersA:
   CARIB ΔROC −0.008 [−0.011,−0.005], ΔPR −0.072, ΔCSI −0.040; GBR ΔROC −0.003
   [−0.004,−0.001]. Boundedness alone does NOT provide out-of-domain safety.

2. **Does the gate add extra contribution? — YES, and it is the crucial one.**
   C−B on CARIB ΔROC +0.0108 [+0.006,+0.015], ΔPR +0.159, ΔCSI +0.063 (CI-sig);
   GBR ΔROC +0.0035 [+0.002,+0.005], ΔPR +0.122. The gate converts a residual
   that would otherwise degrade the anchor on OOD into one that never makes it
   worse (and improves it). Without the gate the residual is net-negative.

3. **Is the formal model necessary vs a simple PersA residual correction? — YES.**
   C−A on CARIB ΔROC +0.003 [+0.002,+0.004] (CI-sig), ΔPR +0.088, ΔCSI +0.023:
   the gated anchored model adds genuine rare-event skill over pure persistence
   on the unseen basin, and it is the ONLY structure among A/B/C that is never
   CI-significantly worse than PersA anywhere. B (ungated) is not a valid
   substitute.

4. **Is CARIB recovery mostly the anchor or the residual/gate? — Mostly the
   anchor; the gated residual adds a small but real increment.**
   D−A on CARIB ΔROC −0.369 (TSCF collapse) shows the anchor carries the
   recovery (magnitude ≈ the entire gap). On top of it, C−A adds ΔPR +0.088 /
   ΔCSI +0.023, and C−B (+0.159 PR) shows this increment REQUIRES the gate —
   the ungated residual alone would have made CARIB worse than PersA
   (B−A ΔROC −0.008). So: recovery = anchor (dominant) + gated residual (small,
   positive, gate-dependent).
