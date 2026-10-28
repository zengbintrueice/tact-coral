# Record-set audit — XGB_THERMAL vs formal TSCF (CARIB zero-shot 2017–2020)

Audited: Both models evaluated on the **same (t, i, j) evaluation record IDs**.

| Quantity | XGB_THERMAL | formal TSCF | intersection |
|---|---|---|---|
| n            | 83025 | 83025 | 83025 |
| n_severe     | 1356  | 1356  | 1356  |
| XGB-only     | 0     | —     | —     |
| TSCF-only    | —     | 0     | —     |
| ID hash (sha256) | c29b34f4aec2cef7… | c29b34f4aec2cef7… | identical |
| **Exact match** | **YES** | — | — |

## 10-point check (all PASS)

1. **Ocean mask** — identical: both keep only finite y(t+4) & ocean cells
   (~land_mask). XGB: loader `label_mask` (finite y_lead & ~land_mask);
   TSCF: ocean-masked shared PFV record set.
2. **PFV record set** — identical construction: issue t = window end,
   windows from the same CARIB grid, both evaluated on the shared population.
3. **Lead = 4 weeks** — both y(target at issue + 4).
4. **Target = y(t+4)** — XGB `forecast_targets` = y_full[s+T-1+4]; TSCF y(t+4).
5. **Severe label** — both y >= 0.8.
6. **Spatial/temporal filtering** — both: ocean cells only; issue-year in
   2017–2020.
7. **Missing-value filtering** — XGB requires finite y(t+4) only; TSCF also
   requires finite y(t) (the PersA anchor). In the CARIB test data there are
   NO pixels with finite y(t+4) but NaN y(t), so the two filters coincide
   (otherwise XGB-only would be > 0 — it is 0).
8. **2017–2020 evaluation period** — identical.
9. **Sample count** — identical, 83025 (205 test windows).
10. **Feature history cutoff / no t+1..t+4 leakage** — XGB features =
    windowed input `ds.X_dynamic` [N, T, H, W, C], last step = issue t (window
    end); target y(t+4) never enters the feature vector. XGB reads the raw
    caribbean file whose sst_anomaly uses the causal (year<=2010) climatology;
    hotspot/cumsum are trailing; monthly channels use month-end hold — no
    future information in features. (TSCF causal-v2 is leak-free by the Step-4
    A1–A12 audit.)

## Resolution

Because the two protocols' record sets coincide exactly on CARIB test
(n=83025, severe=1356), the XGB_THERMAL CARIB numbers in
results/xgb_transfer_control/metrics.json are directly comparable to the formal
TSCF CARIB numbers (e.g. physics_anchored/structural_ablation), without any
record-set mismatch.
