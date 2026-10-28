# Step 9 Audit (A1-A10)

Overall: PASS

- **Ccohort** PASS — GCBD matched cohort equals pinned counts n=1898/any=530/sev=78/bg=1368 (n=1898/any=530/sev=78/bg=1368)
- **Cguard** PASS — Step-4/5/6/7/8 files not overwritten (hashes identical) ((86 files))
- **A1** PASS — GCBD never entered model fitting (frozen checkpoint MD5 verified; training pipeline train_severe_classifier.py reads only causal v2 nc)
- **A2** PASS — GCBD never determined a threshold (no tau/binaries selected on GCBD; quantile bins use predicted-score quantiles only) (bins = score quantiles; no outcome-based boundary search)
- **A3** PASS — GCBD never determined normalization (SCS train 2003-2010 stats via SCSPIECFDataset frozen config)
- **A4** PASS — GCBD observations temporally matched only after model prediction (forecast_map computed before matching; code order enforced)
- **A5** PASS — No future target information enters model inputs (trailing 24-wk window <= issue t; forecast targets t+4; verbatim Steps 2-8 function)
- **A6** PASS — Frozen Step-4 checkpoint used (no retraining, no weight modification) (MD5 7cdbca26cae03fa53be53210940e8ce1)
- **A7** PASS — Checkpoint hash recorded in step9_protocol.json
- **A8** PASS — No biological probability calibration fitted (no isotonic/Platt/probability transform in this script)
- **A9** PASS — Cluster/event dependence assessed (Site_ID-level bootstrap vs naive; 384 sites, max cluster 24)
- **A10** PASS — Step-4-8 outputs remain unchanged (86 files hash-verified)

Consistency checks: 12/12 PASS
OUTCOME: A
