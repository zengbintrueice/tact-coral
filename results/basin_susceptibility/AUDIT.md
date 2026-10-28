# Basin-Specific Susceptibility / Thermal-Response Heterogeneity — AUDIT

Generated: 2026-09-30T13:48:11.098396
Source: D:\曾斌\论文\海洋环境保护\PIECF\LTTS_CRW\results\physics_anchored\structural_ablation\rec_structural_ablation.npz
Source md5: 4035a19fd30fa4d14151f4c8055dfe59

## Provenance

- Exposure: PersA = clip(DHW/10,0,1) at ISSUE time t (issue-time information only)
- target y = y(t+4); severe = y >= 0.8 (frozen definition)
- R = TSAR - PersA (frozen seed-42), used ONLY in Part D as the object tested
- Common DHW support: 0.1-wide bins where ALL basins have n >= 100 => bins [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = [0.00, 1.00)

## Protocol compliance

- [PASS] Frozen records only; NO retraining / target tuning / threshold optimisation
- [PASS] Identical bins and protocol across SCS/GBR/CARIB; comparison restricted to common support
- [PASS] Bootstrap: paired week-block, n=1000, seed 42, weeks resampled WITHIN basin
- [PASS] Part B logistic/rank-logistic fitted by fixed IRLS (no regularisation, no tuning)
- [PASS] TSAR training outputs NOT used as explanatory variables (Part D tests fixed R only)
- [NOTE] No GPU kernel invoked: all quantities are closed-form statistics of frozen predictions (no model inference); piecf_env used

## Strict conclusion

- logistic DHWxGBR = -2.1098 CI [-3.6600,-0.2697]
- logistic DHWxCARIB = -0.2744 CI [-1.5768,+1.1275]
- basin-specific susceptibility supported = True
- If either interaction CI excludes 0, heterogeneity is supported at the pooled common-support level; if BOTH include 0, report 'not proven'.
- All associations associative/predictive; no causal interpretation.

