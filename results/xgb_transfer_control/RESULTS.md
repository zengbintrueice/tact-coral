# XGBoost transfer-control experiment (SCS -> GBR / CARIB)

Generated: 2026-09-24T20:36:23.783198  ·  tau=0.9 pinned  ·  lead 4 wk  ·  history 24 wk

Same XGBoost regressor, same pipeline as the existing baseline; the three variants differ ONLY in feature selection. Each is trained on SCS-train (2003-2010, train-only scaler) and evaluated zero-shot (GBR/CARIB 2017-2020, no target fitting/tuning/threshold).

Models / feature dim (24-wk flattened history + 5 static):
- **XGB_FULL**: all 13 dynamic channels (317 features)
- **XGB_THERMAL**: thermal-only [ssta, hotspot, d_hotspot_dt, cumsum_4w, cumsum_12w] (125)
- **XGB_NO_THERMAL**: all non-thermal dynamic [sst, chl, par, seasonal_sin, wind, ssr, sst_anomaly, seasonal_cos] (197)

| Model | Domain | ROC | PR | CSI@0.90 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| XGB_FULL | SCS | 0.9943 | 0.6360 | 0.0000 | 0.0000 | nan | 0.0000 | 0.00404 | 0.0671 |
| XGB_FULL | GBR | 0.9581 | 0.4330 | 0.0000 | 0.0000 | nan | 0.0000 | 0.01959 | 0.0619 |
| XGB_FULL | CARIB | 0.8798 | 0.0867 | 0.0000 | 0.0000 | nan | 0.0000 | 0.02992 | 0.0715 |
| XGB_THERMAL | SCS | 0.9955 | 0.6801 | 0.0000 | 0.0000 | nan | 0.0000 | 0.00418 | 0.0658 |
| XGB_THERMAL | GBR | 0.9617 | 0.4810 | 0.0000 | 0.0000 | nan | 0.0000 | 0.02028 | 0.0509 |
| XGB_THERMAL | CARIB | 0.8900 | 0.0921 | 0.0000 | 0.0000 | nan | 0.0000 | 0.03031 | 0.0729 |
| XGB_NO_THERMAL | SCS | 0.9723 | 0.2118 | 0.0000 | 0.0000 | nan | 0.0000 | 0.01007 | 0.0582 |
| XGB_NO_THERMAL | GBR | 0.9182 | 0.2998 | 0.0000 | 0.0000 | nan | 0.0000 | 0.04631 | 0.0290 |
| XGB_NO_THERMAL | CARIB | 0.8064 | 0.0486 | 0.0000 | 0.0000 | nan | 0.0000 | 0.04017 | 0.0229 |

## Important caveat on the operational metrics (tau = 0.90 pinned)
XGBoost is a REGRESSOR on y = clip(DHW/10); its predictions are mean-reverting and never reach 0.90 on any domain (max predicted score observed 0.72-0.83, frac > 0.90 = 0). With tau pinned a priori at 0.90, the severe-operational metrics (CSI/POD/FAR/F1) are therefore identically 0 / undefined for all variants — this is a scale property of the regression baseline, NOT a statement about skill. The transfer comparison must rest on ROC-AUC and PR-AUC (ranking) and Brier/ECE (calibration), which are scale-invariant or scale-aware and are reported above.

## Key comparison (CARIB transfer & the role of thermal features)
- **SCS** ROC/PR: FULL 0.994/0.636 · THERMAL 0.995/0.680 · NO_THERMAL 0.972/0.212
- **GBR** ROC/PR: FULL 0.958/0.433 · THERMAL 0.962/0.481 · NO_THERMAL 0.918/0.300
- **CARIB** ROC/PR: FULL 0.880/0.087 · THERMAL 0.890/0.092 · NO_THERMAL 0.806/0.049

### Is CARIB transfer preserved by the thermal features?
- Thermal variant CARIB ROC-AUC = 0.890 vs no-thermal = 0.806 -> thermal dominates the no-thermal variant on CARIB discrimination.
- Reference FULL CARIB ROC = 0.880.
- The thermal channel subset is sufficient to reproduce most transfer skill, and removing thermal features collapses it — CARIB transfer is carried by the thermal-stress history.

## Leakage audit
- ✅ L1_no_target_training: train_on_scs body references ONLY SCS_FILE (GBR/CARIB files never used for fitting)
- ✅ L2_no_target_tuning: hyperparameters are the fixed baseline XGB_PARAMS (module constant); no GBR/CARIB-driven hyperparameter search (no tuning loop over target)
- ✅ L3_threshold_pinned: tau=0.9 is a single module constant set a priori; no threshold-selection sweep over any domain's data exists in the code
- ✅ L4_train_only_scaler: StandardScaler fit on SCS train subset only (fit_transform inside train_on_scs); domains only .transform()
- ✅ L5_feature_reuse: feature construction & label & ocean mask reused verbatim from the existing XGBoost pipeline (no redesign)
- ✅ L6_target_not_touched: target y(t+4) read only at evaluation time in forecast_targets; never an input feature (windowed history <= t)
- ✅ L7_repro: code_md5 + data SHA256s recorded in config.json (git is absent in this tree; hash-based version)

Verdict: PASS — target domains entered neither training, tuning, nor threshold selection.
