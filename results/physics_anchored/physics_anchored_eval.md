# Physics-anchored residual head — three-domain PFV evaluation (OCEAN-MASKED)

Generated: 2026-09-17T11:09:29.370051

Record set per domain: finite y(t+4) & ocean cell (aligned with the training label_mask; land cells with finite y are excluded). PersA_DHW = y_regression at issue time t (the physics anchor).

| Domain | Method | n | sev | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SCS | PhysicsAnchored | 49322 | 419 | 0.9808 | 0.3036 | 0.1977 | 0.2840 | 0.6060 | 0.3301 | 0.01169 | 0.0754 |
| SCS | PersA_DHW | 49322 | 419 | 0.9817 | 0.3177 | 0.1571 | 0.1957 | 0.5568 | 0.2715 | 0.01172 | 0.0658 |
| SCS | TSCF_frozen | 49322 | 419 | 0.9865 | 0.5463 | 0.3374 | 0.5871 | 0.5576 | 0.5046 | 0.07128 | 0.3150 |
| GBR | PhysicsAnchored | 19065 | 676 | 0.9599 | 0.4877 | 0.2541 | 0.3417 | 0.5022 | 0.4053 | 0.02334 | 0.0717 |
| GBR | PersA_DHW | 19065 | 676 | 0.9593 | 0.4164 | 0.2494 | 0.3328 | 0.5011 | 0.3993 | 0.02331 | 0.0676 |
| GBR | TSCF_frozen | 19065 | 676 | 0.9624 | 0.5543 | 0.3648 | 0.6686 | 0.5547 | 0.5346 | 0.09287 | 0.3317 |
| CARIB | PhysicsAnchored | 83025 | 1356 | 0.9822 | 0.5479 | 0.3041 | 0.3835 | 0.4050 | 0.4664 | 0.01412 | 0.0750 |
| CARIB | PersA_DHW | 83025 | 1356 | 0.9792 | 0.4602 | 0.2807 | 0.3510 | 0.4167 | 0.4383 | 0.01413 | 0.0640 |
| CARIB | TSCF_frozen | 83025 | 1356 | 0.6101 | 0.0321 | 0.0262 | 0.0391 | 0.9261 | 0.0511 | 0.09761 | 0.3065 |

Terminology: thermal-stress-aware; strong physics-based persistence baseline (PersA) is the anchor; cumulative HotSpot channels are legitimate CRW-DHW predictors.
