# Step 8 Audit (A1-A17)

Overall: PASS

- **A1** PASS — No future target enters input (sliding window strictly <= issue t; verify t+4) (predict_sliding_temporal uses X_full[t-23..t]; PFV identical to Steps 2-7)
- **A2** PASS — All cumulative windows are causal (trailing spot verification PASS both domains)
- **A3** PASS — C1/C4 definitions exactly match Step 7 (compute_cumsum_hs1c verbatim; stats loaded frozen from step7_feature_manifest.json)
- **A4** PASS — Environmental features exactly match Step-6 manifest (assert vs step6_feature_manifest.json groups A/D passed at startup)
- **A5** PASS — No new feature engineering introduced (channels = dataset 13 minus replaced cumsums, plus C1/C4 from Step 7)
- **A6** PASS — SCS normalization only uses 2003-2010 training data (SCS_CHANNEL_MEAN/STD frozen + step7 C stats (train windows only))
- **A7** PASS — GBR never used for fitting (training script fits SCS only)
- **A8** PASS — GBR never used for normalization (norm vectors = SCS-train stats)
- **A9** PASS — GBR does not determine tau (tau=0.90 pinned; sweep diagnostic only)
- **A10** PASS — All models use identical test masks (same (t,i,j) population from Step-2 npz; defensive year check PASS)
- **A11** PASS — All nested comparisons use identical populations (single preds_by_model dict per domain; paired bootstrap same picks)
- **A12** PASS — Threshold sweep is diagnostic only (formal tau=0.90 never modified; sweep written to separate CSV)
- **A13** PASS — No Step-4/5/6/7 outputs overwritten (58 files hash-verified)
- **A14** PASS — Model hashes are recorded (checkpoint_md5 in step8_protocol.json)
- **A15** PASS — Random seeds are recorded (seed 42 (train/boot), 42 (perm), 7 (spot-check))
- **A16** PASS — Bootstrap seed = 42 (N_BOOT=1000)
- **A17** PASS — Permutation preserves thermal variables and labels (assert array_equal on non-env channels inside every replicate; target untouched)

Consistency checks: 23/23 PASS
OUTCOME: E
