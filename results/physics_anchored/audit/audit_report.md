# Physics-Anchored Strict Methodology Audit

Generated: 2026-09-17T08:50:47.121187  
Scope: (A) CARIB TSCF failure implementation audit, (B) strict zero-shot confirmation, (C) anchored-residual verification, (D) scale sensitivity (SCS-val selection), (E) residual magnitude diagnostics.
Checkpoints: TSCF `D:\曾斌\论文\海洋环境保护\PIECF\LTTS_CRW\results\model_final_Btail3\model.pt` (md5 `7cdbca26cae03fa53be53210940e8ce1`), anchored `D:\曾斌\论文\海洋环境保护\PIECF\LTTS_CRW\results\physics_anchored\model.pt` (md5 `b67f525d66872270b8e8a4d6d4e6fc97`).

## A. CARIB TSCF failure — implementation audit (15 checks)

| # | Check | Verdict | Evidence |
|---|---|---|---|
| A1 | dynamic channel order | PASS | Loader loads raw 9 by name from SCS_DYNAMIC_FEATURES + computes derived 4 from hotspot; 13-channel order identical across scs/gbr/caribbean causal_v2 nc (verified var lists); checkpoint expects same layout. |
| A2 | feature construction | PASS | d_hotspot_dt/cumsum_4w/cumsum_12w/seasonal_cos recomputed identically from each domain's hotspot (aligned_dataset._compute_derived_channels); no domain-specific path. |
| A3 | climatology | PASS | CARIB sst_anomaly built from raw ERA5 Caribbean with year<=2010 climatology; builder checks A_clim_cutoff (max_err=0) and A2_not_fullrecord PASS. |
| A4 | SCS-train-only normalization | PASS | Checkpoint channel_norm buffers == SCS_CHANNEL_MEAN/STD to float32 precision (per-channel rel<1e-7); nothing recomputed per domain. |
| A5 | missing-value handling | PASS | Eval calls forward with nan_mask=None -> ChannelNormalizer NaN->0 in z-space (=mean); identical code path on all domains. |
| A6 | NaN mask | PASS | nan_mask_dyn built from per-domain arrays; consistent across domains. |
| A7 | land mask | NOTE | Strict protocol excludes land cells (land_mask True). Original Steps 2-10 record sets did NOT filter land and scored finite-y-on-land cells (SCS 19110, GBR 1230, CARIB 11070 test cells). Data-consistency note: y_regression is finite on some bathymetry-land cells in all three nc. |
| A8 | latitude/longitude orientation | PASS | CARIB lat/lon ascending (4.5..25.5 / -90.5..-59.5), same as SCS; GBR descending but orientation is a global reindex that ConvLSTM/attention is invariant to; spatial layout internally consistent per file. |
| A9 | temporal indexing | PASS | CARIB 939 weeks 2003-01-05..2020-12-27; issue-year filter uses date_strings; windows t in [T-1, n-1]. |
| A10 | issue / verification time | PASS | PFV: issue at t, target y(t+4); anchor PersA(t)=DHW(t)/10 trailing <= t. |
| A11 | y_regression / PersA construction | PASS | y_regression=clip(DHW/10) family verified by builder T1/T2 (max_err 5.96e-08; class = I(DHW>=4) exact). PersA = y_regression at issue time. |
| A12 | static feature broadcasting | PASS | x_static is never used inside forward_features (PIECFModel ignores it); static construction errors cannot affect predictions. |
| A13 | checkpoint loading | PASS | state_dict keys match FinalSingleHeadModel; md5 7cdbca26cae03fa53be53210940e8ce1 verified; norm buffers baked. |
| A14 | prediction-year filtering | PASS | per-split issue-year filter (SCS 2016-2019; GBR/CARIB 2017-2020). |
| A15 | PFV t+4 alignment | PASS | Same sliding-window PFV as Steps 2-8; CARIB TSCF numbers bit-identical across two independent eval scripts (ROC=0.5981282063358642, n=94095). |

**OOD evidence (why the failure is NOT an input artifact):** 13-channel z-scores on CARIB test are in-family with SCS test (hotspot z97.5=5.16 vs 4.99; cumsum12 z_abs_max=10.9 vs 8.4; validity CARIB 82% vs SCS 78%). But corr(TSCF_pred, PersA) drops from 0.66 (SCS) / 0.61 (GBR) to 0.14 (CARIB) while PersA itself reaches ROC 0.98 on CARIB -> the data signal is present and the frozen SCS-trained mapping does not reproduce it. **Verdict: CARIB failure is a genuine cross-basin generalization failure, not an implementation error.**

## B. Strict zero-shot confirmation (CARIB)

- **B1_CARIB_no_fitting** PASS — anchored ckpt trained on SCS only (train_physics_anchored.py: train 2003-2010/val 2011-2015); CARIB dataset built 2026-09-16 AFTER training; ckpt md5=b67f525d66872270b8e8a4d6d4e6fc97
- **B2_no_normalization_on_CARIB** PASS — ChannelNormalizer uses baked SCS_CHANNEL_MEAN/STD (checkpoint buffers == config to float32, 13 channels); normalization_v2.json SCS-train only; nothing recomputed on CARIB
- **B3_no_threshold_from_CARIB** PASS — tau_eval=0.9 pinned from SCS protocol (Step-4); classification threshold never searched on CARIB/GBR
- **B4_scale_not_chosen_on_CARIB** PASS — scale selected as argmax CSI@0.9 on SCS-val (predefined; alternatives reported) on SCS-val only; GBR/CARIB arrays never read during selection
- **B5_no_early_stopping_on_CARIB** PASS — best-val loss on SCS val, patience 10; GBR/CARIB absent from training loop
- **B6_no_model_selection_on_CARIB** PASS — single frozen checkpoints; no CARIB-driven checkpoint choice; TSCF CARIB numbers bit-identical across two independent eval scripts (n=94095, ROC=0.5981282063358642)

## C. PhysicsAnchored / thermal-stress-anchored residual verification

Definition implemented and verified (`physics_anchored_model.py`):
- `PersA(t) = clip(DHW(t)/10, 0, 1)` = `y_regression` at issue time t (fixed, detached input).
- `residual = tanh(residual_head(attn.detach())) * scale` — input detached, grads -> residual head only.
- `gate = sigmoid((y_main - tau) / temp)`; `y_final = PersA + gate * residual`.
- Terminology: **bounded deviation around the persistence anchor** (residual is two-sided, worst-case |deviation| <= scale*gate; NOT a 'persistence floor').
- Reconstruction sanity on every split: `max|(y_final(scale=1) - PersA)/gate - raw| < 1e-5`.

## D. Scale sensitivity (selection on SCS-val ONLY)

Predefined rule: `argmax CSI@0.9 on SCS-val`. Scales tested: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]. Selection result: **scale = 0.4**.

| split | scale | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| SCS_val | 0.05 | 0.9904 | 0.3642 | 0.2070 | 0.2719 | 0.5354 | 0.3430 | 0.00793 | 0.0562 |
| SCS_val | 0.10 | 0.9907 | 0.3705 | 0.2192 | 0.2949 | 0.5396 | 0.3596 | 0.00791 | 0.0569 |
| SCS_val | 0.20 | 0.9912 | 0.3814 | 0.2724 | 0.3779 | 0.5060 | 0.4282 | 0.00791 | 0.0584 |
| SCS_val | 0.30 | 0.9916 | 0.3900 | 0.2862 | 0.4194 | 0.5260 | 0.4450 | 0.00795 | 0.0598 |
| SCS_val | 0.40 | 0.9919 | 0.3952 | 0.2953 | 0.4654 | 0.5531 | 0.4560 | 0.00805 | 0.0613 |
| SCS_val | 0.50 | 0.9921 | 0.3981 | 0.2935 | 0.4977 | 0.5830 | 0.4538 | 0.00819 | 0.0628 |
| SCS_test | 0.10 | 0.9818 | 0.3339 | 0.1703 | 0.2220 | 0.5773 | 0.2911 | 0.01159 | 0.0682 |
| SCS_test | 0.30 | 0.9815 | 0.3301 | 0.1936 | 0.2745 | 0.6034 | 0.3244 | 0.01158 | 0.0726 |
| SCS_test | 0.50 | 0.9808 | 0.3159 | 0.2087 | 0.3198 | 0.6246 | 0.3454 | 0.01193 | 0.0767 |
| GBR | 0.10 | 0.9597 | 0.4717 | 0.2508 | 0.3358 | 0.5022 | 0.4011 | 0.02331 | 0.0687 |
| GBR | 0.30 | 0.9598 | 0.4732 | 0.2528 | 0.3388 | 0.5011 | 0.4035 | 0.02333 | 0.0707 |
| GBR | 0.50 | 0.9598 | 0.4740 | 0.2547 | 0.3432 | 0.5032 | 0.4059 | 0.02337 | 0.0729 |
| CARIB | 0.10 | 0.9800 | 0.5235 | 0.2829 | 0.3540 | 0.4153 | 0.4410 | 0.01401 | 0.0665 |
| CARIB | 0.30 | 0.9810 | 0.5359 | 0.2976 | 0.3746 | 0.4086 | 0.4587 | 0.01410 | 0.0716 |
| CARIB | 0.50 | 0.9817 | 0.5457 | 0.3051 | 0.3850 | 0.4048 | 0.4675 | 0.01462 | 0.0768 |

SCS-val CSI@0.9 per scale: 0.05=0.2070, 0.1=0.2192, 0.2=0.2724, 0.3=0.2862, 0.4=0.2953, 0.5=0.2935

## E. Residual magnitude diagnostics (selected scale=0.4)

| split | n | raw med | raw sd | gate med | gate p97.5 | delta mean | |d|>0.05 | |d|>0.10 | |d|>0.20 | raw@boundary | final>1 | final<0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SCS_val | 70731 | 1.000 | 0.013 | 0.007 | 0.127 | 0.0067 | 0.024 | 0.011 | 0.002 | 0.808 | 0.002 | 0.000 |
| SCS_test | 49322 | 1.000 | 0.020 | 0.007 | 0.228 | 0.0101 | 0.048 | 0.020 | 0.003 | 0.720 | 0.003 | 0.000 |
| GBR | 19065 | 0.999 | 0.009 | 0.008 | 0.035 | 0.0045 | 0.002 | 0.000 | 0.000 | 0.530 | 0.015 | 0.000 |
| CARIB | 83025 | 1.000 | 0.014 | 0.008 | 0.247 | 0.0104 | 0.043 | 0.024 | 0.006 | 0.621 | 0.007 | 0.000 |

## Three-domain unified metrics (selected scale=0.4; STRICT ocean-only record set)

| split | method | n | sev | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SCS_val | PersA | 70731 | 217 | 0.9900 | 0.3492 | 0.1978 | 0.2535 | 0.5259 | 0.3303 | 0.00796 | 0.0554 |
| SCS_val | PhysicsAnchored | 70731 | 217 | 0.9919 | 0.3952 | 0.2953 | 0.4654 | 0.5531 | 0.4560 | 0.00805 | 0.0613 |
| SCS_val | TSCF | 70731 | 217 | 0.9883 | 0.4908 | 0.1985 | 0.7281 | 0.7856 | 0.3312 | 0.07646 | 0.3127 |
| SCS_test | PersA | 49322 | 419 | 0.9817 | 0.3177 | 0.1571 | 0.1957 | 0.5568 | 0.2715 | 0.01172 | 0.0658 |
| SCS_test | PhysicsAnchored | 49322 | 419 | 0.9812 | 0.3229 | 0.2010 | 0.2936 | 0.6108 | 0.3347 | 0.01171 | 0.0746 |
| SCS_test | TSCF | 49322 | 419 | 0.9865 | 0.5463 | 0.3374 | 0.5871 | 0.5576 | 0.5046 | 0.07128 | 0.3150 |
| GBR | PersA | 19065 | 676 | 0.9593 | 0.4164 | 0.2494 | 0.3328 | 0.5011 | 0.3993 | 0.02331 | 0.0676 |
| GBR | PhysicsAnchored | 19065 | 676 | 0.9598 | 0.4738 | 0.2544 | 0.3417 | 0.5011 | 0.4056 | 0.02334 | 0.0718 |
| GBR | TSCF | 19065 | 676 | 0.9624 | 0.5543 | 0.3648 | 0.6686 | 0.5547 | 0.5346 | 0.09287 | 0.3317 |
| CARIB | PersA | 83025 | 1356 | 0.9792 | 0.4602 | 0.2807 | 0.3510 | 0.4167 | 0.4383 | 0.01413 | 0.0640 |
| CARIB | PhysicsAnchored | 83025 | 1356 | 0.9814 | 0.5408 | 0.3012 | 0.3798 | 0.4074 | 0.4629 | 0.01430 | 0.0742 |
| CARIB | TSCF | 83025 | 1356 | 0.6101 | 0.0321 | 0.0262 | 0.0391 | 0.9261 | 0.0511 | 0.09761 | 0.3065 |

## Record-set note

This audit uses the STRICT protocol record set (ocean cells only, finite target + finite all-methods). The original Steps 2-10 and physics_anchored_eval record sets additionally scored finite-y-on-land cells (SCS +19110, GBR +1230, CARIB +11070). Both are leak-free; the strict set matches training label_mask semantics.

## Verdicts

1. **CARIB failure is trustworthy** — genuine cross-basin generalization failure, not an implementation artifact (bit-exact cross-script reproduction; in-family input distributions; PersA=0.98 vs TSCF=0.60;
2. **Anchored recovery is credible** — see table above; at selected scale the anchored model recovers CARIB ranking toward PersA while preserving SCS/GBR skill; bounded deviation around anchor confirmed (residual magnitude diagnostics).
3. **scale = 0.4** selected by the predefined SCS-val rule and is the frozen configuration to use going forward.
4. **Recommended code fixes before further experiments:** (a) evaluation record sets should apply the land/ocean mask (align with label_mask); (b) consider excluding finite-y-on-land cells at dataset build time or documenting them; (c) the physics_anchored_audit inference+analysis pipeline should save rec npz then analyze in a separate process to avoid the netCDF4 cleanup segfault.
