# Step 5 — Leakage / Protocol Audit (temporal-purge robustness)

Generated: 2026-09-17 18:35:27

Step 5 is supplementary robustness evidence; the formal Step-4 protocol remains the primary result. The two protocols are NOT merged.

| ID | Requirement | Verdict | Evidence |
|---|---|---|---|
| A1 | Frozen checkpoint unchanged | **PASS** | checkpoint MD5 7cdbca26cae03fa53be53210940e8ce1 equals frozen reference 7cdbca26cae03fa53be53210940e8ce1 (verified); model not even loaded in Step 5 |
| A2 | SCS normalization unchanged | **PASS** | normalization baked inside the checkpoint (frozen Step-4 state); no recomputation anywhere in Step 5 |
| A3 | Causal-v2 features unchanged | **PASS** | source arrays read from persistence_v2 npz (SHA256-locked); datasets opened read-only only for the weekly date record; dataset SHA256s locked in Step-4 protocol.json |
| A4 | No future target enters X | **PASS** | predictions are the stored Step-2 arrays (PFV, target only as label at t+4); no new features computed |
| A5 | Derived features remain trailing | **PASS** | no feature recomputation in Step 5; Step-2 arrays carry the trailing derived channels (verified in causal v2 build, max_err <= 6.7e-05) |
| A6 | Monthly variables remain causal | **PASS** | no recomputation; causal month-end hold inherited from causal v2 datasets |
| A7 | Climatology remains year<=2010 | **PASS** | causal v2 ERA5 climatology cutoff; datasets unmodified |
| A8 | GBR remains zero-shot | **PASS** | no GBR fitting of any kind; GBR predictions are the frozen Step-2 inference outputs |
| A9 | GBR cannot affect model weights | **PASS** | no model loaded, no gradient step; arrays read-only |
| A10 | GBR cannot affect normalization | **PASS** | no normalization computed or modified |
| A11 | GBR cannot affect tau | **PASS** | tau ∈ {0.8, 0.9} pinned a priori (identical to Step 4); no sweep-based selection performed |
| A12 | GCBD is not used | **PASS** | no GCBD file read in the entire Step-5 chain |
| A13 | Standard and purged masks derived only from temporal sample structure | **PASS** | purged mask = f(issue week t, weekly date record, boundary b); no prediction, target, or performance information enters the mask |
| A14 | TSCF, persistence, and cumulative-only use identical masks | **PASS** | single shared boolean mask per protocol applied to all three methods (C8) |
| A15 | Purging changes sample inclusion only; no model parameters or feature values change | **PASS** | no model loaded; predictions bitwise identical on retained samples (C6); Step-4 file hashes unchanged (C109) |

**OVERALL: PASS**

## Consistency checks

| ID | Check | Verdict | Detail |
|---|---|---|---|
| C1 | all standard samples have issue week in nominal test years | **PASS** | years 2017–2020 |
| C2 | purged mask keeps iff t-(H-1) >= b (no sample history crosses b) | **PASS** | removed 2139/19065 samples (11.2%) |
| C3 | target week t+4 exists for every sample (PFV verification at t+4) | **PASS** | max t+4 = 1865 < n_dates=1935 |
| C4 | input history ends at issue week t (X_full[t-23:t+1], predict_sliding_temporal; labels never enter X) | **PASS** | structural, verified in compute_persistence_baseline.py:128-132 / collect_records:298-304 |
| C5 | purging changes sample inclusion only (subset of standard set) | **PASS** | purged arrays are index subsets of the standard arrays (verified in C6) |
| C6 | TSCF predictions on retained samples bitwise identical to standard | **PASS** |  |
| C7 | all method predictions & targets identical to Step-4 arrays | **PASS** |  |
| C8 | standard & purged masks identical across the three methods | **PASS** | single shared mask per protocol; finite patterns identical across methods |
| C9 | recomputed standard metrics match frozen Step-4 results exactly | **PASS** | all equal |
| C109 | formal Step-4 files not overwritten (hashes identical before/after) | **PASS** | 6 files verified |
