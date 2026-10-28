# Step 7 — Audit (temporal-scale decomposition)

Generated: 2026-09-17 18:58:32

Cumulative HotSpot channels are legitimate thermal-stress predictors (NOAA CRW DHW definition); nothing here implies a shortcut or leakage.

| ID | Requirement | Verdict | Evidence |
|---|---|---|---|
| A1 | No future HotSpot values enter any cumulative window | **PASS** | trailing construction (t-W+1..t]; spot-verified against explicit per-window recompute in both domains |
| A2 | All windows end exactly at issue time t | **PASS** | cs[t] = sum over (t-W+1..t] by construction; spot checks confirm |
| A3 | No centered rolling windows | **PASS** | only trailing windows implemented |
| A4 | No future interpolation | **PASS** | no interpolation anywhere in the C_W chain |
| A5 | Normalization uses SCS 2003-2010 only | **PASS** | non-cumulative: frozen SCS-train stats; C_W: train-window stats stored in step7_feature_manifest.json (mean/std/min/max/n_valid) |
| A6 | GBR is never used for model fitting | **PASS** | all 8 models trained on SCS only; GBR inference no-grad eval |
| A7 | GBR is never used for normalization | **PASS** | SCS-train stats applied to GBR inputs as-is (zero-shot) |
| A8 | GBR test data does not determine tau | **PASS** | tau ∈ {0.8, 0.9} pinned a priori; sweep diagnostic only |
| A9 | No target y(t+4) enters the input | **PASS** | PFV: target only as label; inputs are trailing channels |
| A10 | All models use identical masks | **PASS** | single shared Step-4 common (t,i,j) population applied to every model |
| A11 | All models use identical test population | **PASS** | n=68432/502 (SCS), n=20295/729 (GBR) for every model; per-model finiteness verified |
| A12 | Cumulative windows use identical HotSpot >= 1 degC definition | **PASS** | one shared compute_cumsum_hs1c function for all W and both domains |
| A13 | Step-4/5/6 files are not modified | **PASS** | 32 files hash-verified before/after |
| A14 | Every model checkpoint and output has a reproducible hash | **PASS** | checkpoint MD5s in step7_protocol.json; output SHA256s recorded |

**OVERALL: PASS**
