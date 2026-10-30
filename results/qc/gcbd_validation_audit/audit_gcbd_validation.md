# Audit — GCBD external validation: statistics, sample sets and attribution

**Date:** 2026-10-30
**Scope:** every GCBD statistic quoted in the Abstract, Results, Discussion,
Fig. 10 caption and SI Note R5; the three sample definitions; the bootstrap
unit; and the separation of (i) DHW-coordinate association, (ii) model
increment and (iii) biological-mechanism claims.
**No model retrained; no split, seed, threshold or matching rule changed.**

Primary source of truth: `results/step9_gcbd_validation/step9_gcbd_results.csv`
(one row per matched GCBD observation, TSCF score only), recomputed by
`results/qc/gcbd_validation_audit/audit_gcbd_recompute.py`.

---

## 1. Sample sets (must not be conflated)

| Sample | Definition | TSCF | PersA | TSAR | severe |
|---|---|---:|---:|---:|---:|
| **Full matched cohort** | GCBD observations matched to SCS test-period forecasts, 2016–2019, ±4 wk, 0.5° buffer; valid Percent_Bleaching | **1,898** | — | — | 78 |
| **Common cohort** | full cohort ∩ {TSCF, PersA, TSAR all finite} | 1,444 | 1,444 | 1,444 | 28 |
| **Anchor-missing subset** | current DHW absent at issue time → PersA/TSAR undefined | 454 | 0 | 0 | 50 |
| SCS temporal holdout 2011–2015 | same matching in the validation window | 1,993 | 1,429 | 1,429 | 38 |
| CARIB 2015–2016 window | global GCBD, zero-shot | 388 | 264 | 264 | 188 / 119 (anchor-defined) |

Composition of the full cohort: 530 any-bleaching, 78 severe
(Percent_Bleaching ≥ 20%), 1,368 background (= 0), **384 sites**, maximum
24 observations per site.

The 1,898 vs 1,444 gap is **not** a data-cleaning choice: TSCF is defined on
every matched observation, whereas PersA = clip(DHW/10) and TSAR =
PersA + gate·residual require a finite current-DHW anchor. The 454 excluded
observations are missing at the anchor stage, not at model evaluation.

---

## 2. Statistic-by-statistic verification

All values below were recomputed from the raw record file and reproduced the
published values exactly.

| Statistic | Value | Source / computation |
|---|---|---|
| n matched | 1,898 | row count |
| any-bleaching | 530 | `pb > 0` |
| severe | 78 | `pb ≥ 20` |
| background | 1,368 | `pb == 0` |
| sites / max per site | 384 / 24 | `Site_ID` |
| AUROC severe vs background | **0.876331** | ROC on rows with `severe | background` |
| AUROC severe vs all others | 0.847689 | ROC on all rows |
| AUROC any-bleaching vs others | 0.530445 | ROC on all rows |
| Mann–Whitney p (severe vs background) | 2.151e−29 | one-sided |
| Cliff's δ | 0.7527 | rank identity |
| Quintile severe rates | 0.53 / 1.05 / 0.53 / 0.00 / 18.52 % | outcome-free TSCF-score quintiles |
| Rate ratio Q5/Q1 | **35.185** | 18.52/0.53 |
| RR naive bootstrap | mean 41.990 | 1,000 observation resamples, seed 42 |
| RR site-cluster CI | **[8.951, 48.312]** | 646 valid replicates |
| AUROC site-cluster CI | **[0.807632, 0.935942]** | 1,000 replicates |
| AUROC naive CI | [0.829634, 0.921825] | 1,000 replicates |
| Site-level AUROC | 0.8623 | site-aggregated score vs severe-site |
| Site permutation p | 0.00049975 | 2,000 permutations (floor 1/2001) |
| Site Q5 vs Q1 severe rate | 38.96% vs 0% | site-risk quintiles |
| ±4-week offsets AUROC | 0.8145 / 0.8121 / 0.8763 / 0.8756 / 0.8881 | offsets −2…+2 wk |
| Severity thresholds ≥10/20/30% | 0.8003 / 0.8763 / 0.9392 | severe-versus-background |
| Per-year AUROC 2016/17/18/19 | 0.9409 / 0.9510 / 0.1909 / 0.8749 | year strata |
| Leave-one-year-out | 0.8143 / 0.8851 / 0.8979 / 0.8917 | excluding 2016/17/18/19 |
| Within-year RR 2016 / 2019 | 22.8 / 1.29 | review2 quintiles; 2019 numerator and denominator each 2 severe |
| 2011–2015 holdout | TSCF 0.3166 [0.172, 0.495]; PersA 0.6195 [0.476, 0.754]; TSAR 0.6141 | n = 1,993, 38 severe |
| CARIB 2015–2016 | TSCF 0.7728 [0.611, 0.889] (n = 388); PersA 0.9153 [0.806, 0.977], TSAR 0.9139 (n = 264, 119 severe) | global GCBD |

### Common-cohort comparison (the decisive attribution)

| Model | n | severe | AUROC | site-cluster CI | RR (Q5/Q1) | Cliff's δ |
|---|---:|---:|---:|---|---:|---:|
| TSCF | 1,444 | 28 | **0.8240** | [0.672, 0.954] | 10.49 | 0.648 |
| PersA | 1,444 | 28 | **0.8281** | [0.704, 0.937] | 8.25 | 0.656 |
| TSAR | 1,444 | 28 | **0.8139** | [0.674, 0.940] | undefined | 0.628 |

Source: `results/review2_revision/gcbd_common_cohort.json`.
PersA/TSAR per-observation scores are not shipped; the aggregates above are
verified against the shipped JSON. Full independent recomputation requires
re-running the inference (no retraining):

```bash
cd LTTS_CRW/tact-coral
"$PIECF_ENV/Scripts/python.exe" experiments/08_infrastructure/compute_review2_gcbd_common_cohort_20260928.py
```

---

## 3. Confidence intervals: method, unit and limits

- **Method:** percentile bootstrap over **site clusters** (`Site_ID`),
  1,000 resamples, seed 42, sites drawn with replacement, each site's
  observations kept together.
- **Unit:** the matched GCBD observation, clustered by site (384 clusters,
  up to 24 observations per site). The naive observation bootstrap is also
  reported (AUROC [0.830, 0.922]) and is narrower; the site-cluster interval
  is the primary inference.
- **Rate-ratio interval is conditional:** it uses the 646/1,000 replicates in
  which the bottom quintile contains at least one severe observation; the
  AUROC interval uses all 1,000. This is now stated in SI R5.
- **What the interval does *not* cover:** within-site correlation is handled;
  **year-level correlation across sites is not**. The year-stratified and
  leave-one-year-out analyses are sensitivity checks, not a formal
  temporal-cluster interval. No claim of independent and identically
  distributed observations is made.
- The 2018 stratum (two severe) is inverted (AUROC 0.191) and is reported as
  observed; it is not used to support the main claim.

---

## 4. The three claims, separated

| Claim | Status | Evidence |
|---|---|---|
| **DHW thermal-risk coordinate is associated with severe bleaching** | **Supported** | PersA alone: AUROC 0.828 common cohort; 0.915 in the CARIB 2015–2016 window; 0.620 in the 2011–2015 holdout |
| **The neural model adds ecological information beyond the DHW coordinate** | **Not demonstrated** | Common cohort: TSCF 0.824 vs PersA 0.828 (TSCF slightly lower); RR 10.49 vs 8.25; δ 0.648 vs 0.656 |
| **A biological mechanism (susceptibility/causality)** | **Not tested** | GCBD is a bleaching outcome; no mechanistic experiment; the residual is a model-internal quantity |

The full-cohort TSCF values (0.876, RR 35.2, n = 78 severe) are **coverage
effects**: TSCF is defined on 454 observations (50 severe) where the anchor
is missing. The paper now states this in the Abstract, Results, Fig. 10
caption and SI R5.

---

## 5. XGBoost: what the control does and does not show

- XGBoost is a **continuous clip(DHW/10, 0, 1) regressor**; it is read only
  through within-family contrasts (FULL / THERMAL / NO-THERMAL) on ROC/PR.
- It supports: thermal predictors carry a transferable **ranking** signal
  (CARIB ROC 0.880 / 0.890 / 0.806).
- It does **not** test: conditional residual risk, matched-DHW susceptibility,
  ecological enrichment, biological mechanism, or causal feature effects.
- It does **not** provide probability calibration (the audit of 2026-10-29
  covers this separately).
- The Abstract's previous clause "XGBoost independently confirms this" was
  removed because its antecedent was the matched-DHW residual separation,
  which XGBoost did not test.

---

## 6. Before → after claim table

| Location | Before | After | Action |
|---|---|---|---|
| Abstract | "…XGBoost independently confirms this." | removed | **deleted** (over-attribution) |
| Abstract | "Independent GCBD observations (n = 1,898) show severe-bleaching enrichment (AUROC 0.876, 95% CI 0.808–0.936; top-quintile rate ratio 35.2)." | "…(n = 1,898; 2016–2019) … by the frozen score (AUROC 0.876 versus background; top-quintile rate ratio 35.2), but the DHW anchor alone matches it on the common cohort (0.81–0.83)." | **narrowed + comparator + sample caveat** |
| Results, first GCBD sentence | "The frozen thermal-risk score… cluster-aware AUROC 0.876 (95% CI 0.808–0.936)" | "The frozen **TSCF** score… (…; 1,000 site-cluster resamples, seed 42, 384 sites, up to 24 observations per site)" | **model named + CI unit** |
| Results, after the 2016-concentration caveat | — | "These full-cohort values are TSCF-specific; the like-for-like comparison below shows that the shared DHW coordinate accounts for the enrichment, not a model-specific ecological advantage." | **added forward attribution** |
| Results, common-cohort paragraph | already present | unchanged | **retained** |
| Discussion, XGBoost paragraph | "…not calibrated event probabilities." | "+ It is a cross-architecture ranking check; it does not test conditional residual risk, ecological mechanism or causal feature effects." | **scope limited** |
| Fig. 10 caption (b) | "Cluster-aware discrimination of severe observations" | "…by the frozen TSCF score (severe versus background…)" | **model + comparator** |
| Fig. 10 caption (c) | "…across predicted-risk quintiles" | "…across **TSCF** predicted-risk quintiles" | **model named** |
| SI R5, results header | "site-cluster bootstrap 95% CI" | "+ 1,000 resamples of the 384 site clusters, seed 42, sites with replacement, observations kept together" | **unit stated** |
| SI R5, after RR sentence | — | "+ RR interval conditions on 646/1,000 non-degenerate replicates; AUROC uses all 1,000; no year-level clustering correction" | **CI limits stated** |

**Retained without change:** all point estimates and intervals; the
common-cohort comparison; the 2016/2019 concentration caveat; the 2011–2015
inversion; the GBR coverage gap; the 2018 inverted stratum.

**Deleted:** only the abstract clause "XGBoost independently confirms this".

**Not changed:** model, training, splits, seeds, thresholds, matching rule.

---

## 7. Reproduction

```bash
cd LTTS_CRW/tact-coral
"$PIECF_ENV/Scripts/python.exe" results/qc/gcbd_validation_audit/audit_gcbd_recompute.py
```

Outputs `audit_gcbd_recompute.json`: counts, AUROC/CI, RR/CI, naive CI,
site-cluster valid-replicate counts. The script reproduces the published
AUROC 0.876331, CI [0.807632, 0.935942], RR 35.185 and CI [8.950738, 48.311898]
exactly.

---

## 8. Verdict

**Verified:** every GCBD statistic quoted in the paper reproduces from the raw
record file. The sample definitions and the 1,898 → 1,444 reduction are
correctly explained by the missing current-DHW anchor.

**Corrected:** the paper no longer attributes the full-cohort enrichment to a
model-specific ecological advantage; the Abstract, Results, Discussion,
Fig. 10 caption and SI R5 now state the common-cohort comparison and the
bootstrap limits.

**Still not established:** any incremental ecological value of the neural
model over the DHW coordinate, any biological-mechanism interpretation, and
any temporal-cluster-adjusted interval. These would require new experiments
(temporal-cluster bootstrap; independent reef systems), not new wording.
