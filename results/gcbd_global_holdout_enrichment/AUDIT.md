# AUDIT — GCBD global site windows + 2011-2015 temporal holdout

Generated: 2026-10-18 19:22:32
Frozen TSCF md5 `7cdbca26cae03fa53be53210940e8ce1`; frozen TSAR md5 `ae86e017bc5ba634b0c68e5d13b4f9f0` (both verified).

## Data-coverage note (GBR)

The global GCBD download contains no GBR-box observations after 2014, so a
GBR zero-shot enrichment window is not feasible with the available data;
this is a survey-coverage gap, not a model result.

## Protocol

- Matching verbatim (obs − 4 wk issue, ±4 wk window, 0.5° buffer,
 issue-year filter per window); identical record indices reused across
 TSCF/PersA/TSAR within each window (like-for-like cohorts per window).
- Part A = SCS validation window 2011-2015 (training 2003-2010; primary
 test 2016-2019) → clean temporal holdout.
- Part B = GBR / CARIB zero-shot windows (2017-2020) matched against the
 global GCBD download; SCS-trained frozen models only.
- Analytics identical to step9/R5: AUROC(sev≥20% vs background), top-
quintile RR, Cliff's δ, MW p; site-cluster bootstrap 95% CI (1,000,
 seed 42) primary; naive-observation CI reported alongside.
- Pre-specified floor: a window is interpreted only with ≥10 severe
 observations; otherwise **not established**.
- No training, no target fitting, no threshold selection; ranking-
enrichment interpretation only.

## Results

- **SCS_val2011_2015 / TSCF**: n=1993, severe=38; AUROC 0.317 [0.172, 0.495] (site-cluster); RR 0.51 (0.09, 2.38); δ -0.367 (-0.66, -0.01); interpretable
- **SCS_val2011_2015 / PersA**: n=1429, severe=38; AUROC 0.620 [0.476, 0.754] (site-cluster); RR 2.12 (0.52, 13.45); δ 0.239 (-0.05, 0.51); interpretable
- **SCS_val2011_2015 / TSAR**: n=1429, severe=38; AUROC 0.614 [0.460, 0.759] (site-cluster); RR 3.42 (0.42, 10.81); δ 0.228 (-0.08, 0.52); interpretable
- **CARIB_2015_2016 / TSCF**: n=388, severe=188; AUROC 0.773 [0.611, 0.889] (site-cluster); RR 1.23 (0.92, 1.94); δ 0.546 (0.22, 0.78); interpretable
- **CARIB_2015_2016 / PersA**: n=264, severe=119; AUROC 0.915 [0.806, 0.977] (site-cluster); RR 5.43 (2.83, 13.95); δ 0.831 (0.61, 0.95); interpretable
- **CARIB_2015_2016 / TSAR**: n=264, severe=119; AUROC 0.914 [0.808, 0.976] (site-cluster); RR 5.14 (2.69, 42.9); δ 0.828 (0.62, 0.95); interpretable

## Interpretation guard

Enrichment = ranking association between the thermal-risk coordinate and
independent field observations; it is not biological prediction and no
calibration or threshold was derived from GCBD.

