> **Point-in-time snapshot.** This audit records the state of the Supplementary
> Information on **2026-10-09**, before the 2026-10-18 cross-check against the
> reproduction package. The SI has since changed in the following places, so
> the row quotes below no longer match the current text:
>
> - GCBD year-stratified CIs corrected (a `cluster_ci` bootstrap bug had mixed
>   any-bleaching observations into the negative class): 2016 `[0.807, 0.980]`
>   -> `[0.823, 0.996]`; 2017 `[0.880, 0.954]` -> `[0.918, 0.977]`;
>   2019 `[0.631, 0.819]` -> `[0.775, 0.948]`; 2018 `[0.133, 0.265]`
>   -> `[0.118, 0.264]`; leave-one-year-out also updated.
> - The R7 sentence "every TSAR cell stays >= 0.96 in all sixteen direction-year
>   combinations" was replaced by a scoped statement (27 of 33 defined cells).
> - The SI overview no longer cites "Notes R8-R9" (no such notes exist).
> - SI Table 3(a) CARIB PersA ECE `0.0640` -> `0.0661`; Table 10 fixed-DHW GBR
>   severe rate `3.6%` -> `3.5%`.
> - CARIB 2015-2016 window cohort restated as 388 window-matched / 188 severe
>   (anchor defined on 264, 119 severe).
>
> Current values: `doc/CROSSCHECK_paper_vs_release.md` (section 5, execution
> record) and the SI itself.

# SI number <-> source-file traceability audit

Generated 2026-10-08. Scope: every numeric token in the SI narrative (embedded table rows were audited cell-by-cell in earlier passes and are excluded here). Each token was checked against the raw text of its evidence family's result files (full-precision and rounded forms).

Tokens checked: 218 | matched: 218 | unmatched: 0

## MANUAL_REVIEW_REQUIRED (unmatched tokens)

| Line | Section | Token | Context |
|---|---|---|---|
| (none) | | | |

## Matched mapping (full)

| Line | Section | Token | Context | Evidence family |
|---|---|---|---|---|
| 63 | S2. Temporal-purge sensitivity | 0.975 | ...example, GBR TSCF ROC AUC is 0.975 under the purged pr | S2/R1-purge |
| 64 | S2. Temporal-purge sensitivity | 0.962 | ...0.962 under the standard | S2/R1-purge |
| 64 | S2. Temporal-purge sensitivity | 0.976 | ...standard protocol, while PersA remains near 0.976 and | S2/R1-purge |
| 65 | S2. Temporal-purge sensitivity | 0.959 | ...0.959, respectively. | S2/R1-purge |
| 100 | S5. Context attribution | 0.11 | ...Status: positive in SCS (ΔPR AUC ≈ +0.11, CI [0.036, 0.191]); | S5-context |
| 100 | S5. Context attribution | 0.036 | ...tatus: positive in SCS (ΔPR AUC ≈ +0.11, CI [0.036, 0.191]); null in G | S5-context |
| 100 | S5. Context attribution | 0.191 | ...positive in SCS (ΔPR AUC ≈ +0.11, CI [0.036, 0.191]); null in GBR | S5-context |
| 101 | S5. Context attribution | 0.011 | ...(ΔPR AUC ≈ +0.011 [−0.008, +0.034]); | S5-context |
| 101 | S5. Context attribution | 0.008 | ...(ΔPR AUC ≈ +0.011 [−0.008, +0.034]); and **nu | S5-context |
| 101 | S5. Context attribution | 0.034 | ...(ΔPR AUC ≈ +0.011 [−0.008, +0.034]); and **null-to-ne | S5-context |
| 102 | S5. Context attribution | 83,025 | ...For the frozen TSCF on CARIB (n = 83,025, severe = 1,356), | S5-context |
| 104 | S5. Context attribution | 0.003 | ...degrade ranking: ΔPR AUC −0.003 [−0.008, +0.001] (i | S5-context |
| 104 | S5. Context attribution | 0.008 | ...degrade ranking: ΔPR AUC −0.003 [−0.008, +0.001] (includes | S5-context |
| 104 | S5. Context attribution | 0.001 | ...degrade ranking: ΔPR AUC −0.003 [−0.008, +0.001] (includes zero), Δ | S5-context |
| 105 | S5. Context attribution | 0.010 | ...+0.010 [+0.002, +0.016], a | S5-context |
| 105 | S5. Context attribution | 0.002 | ...+0.010 [+0.002, +0.016], and ΔROC | S5-context |
| 105 | S5. Context attribution | 0.016 | ...+0.010 [+0.002, +0.016], and ΔROC AUC −0.0 | S5-context |
| 105 | S5. Context attribution | 0.053 | ...+0.010 [+0.002, +0.016], and ΔROC AUC −0.053 [−0.087, −0.017] (a | S5-context |
| 105 | S5. Context attribution | 0.087 | ....010 [+0.002, +0.016], and ΔROC AUC −0.053 [−0.087, −0.017] (a small R | S5-context |
| 105 | S5. Context attribution | 0.017 | ....002, +0.016], and ΔROC AUC −0.053 [−0.087, −0.017] (a small ROC | S5-context |
| 143 | S7. Bootstrap effect sizes | 0.003 | ...+0.003 ROC AUC, +0.088 PR | S2/R1-purge |
| 143 | S7. Bootstrap effect sizes | 0.088 | ...+0.003 ROC AUC, +0.088 PR AUC and +0.023 C | S2/R1-purge |
| 143 | S7. Bootstrap effect sizes | 0.023 | ...+0.003 ROC AUC, +0.088 PR AUC and +0.023 CSI@0.90. The ungat | S2/R1-purge |
| 163 | S7. Bootstrap effect sizes | 83,025 | ...grid-cell** level: the 83,025 CARIB records are | R2/S7-ablation |
| 168 | S7. Bootstrap effect sizes | 0.0030 | ...spatial clustering: ΔROC +0.0030 [0.0023, 0.0037], | S2/R1-purge |
| 168 | S7. Bootstrap effect sizes | 0.0023 | ...spatial clustering: ΔROC +0.0030 [0.0023, 0.0037], ΔPR +0.0 | S2/R1-purge |
| 168 | S7. Bootstrap effect sizes | 0.0037 | ...spatial clustering: ΔROC +0.0030 [0.0023, 0.0037], ΔPR +0.0877 | S2/R1-purge |
| 168 | S7. Bootstrap effect sizes | 0.0877 | ...stering: ΔROC +0.0030 [0.0023, 0.0037], ΔPR +0.0877 | S2/R1-purge |
| 169 | S7. Bootstrap effect sizes | 0.0734 | ...[0.0734, 0.1015], ΔCSI@0.9 | S2/R1-purge |
| 169 | S7. Bootstrap effect sizes | 0.1015 | ...[0.0734, 0.1015], ΔCSI@0.90 +0.023 | S2/R1-purge |
| 169 | S7. Bootstrap effect sizes | 0.0234 | ...[0.0734, 0.1015], ΔCSI@0.90 +0.0234 [0.0172, 0.0292], | S2/R1-purge |
| 169 | S7. Bootstrap effect sizes | 0.0172 | ...[0.0734, 0.1015], ΔCSI@0.90 +0.0234 [0.0172, 0.0292], all excl | S2/R1-purge |
| 169 | S7. Bootstrap effect sizes | 0.0292 | ...[0.0734, 0.1015], ΔCSI@0.90 +0.0234 [0.0172, 0.0292], all excluding ze | S2/R1-purge |
| 177 | S8. Seed sensitivity | 0.026 | ...xample, GBR PR AUC ranges from approximately 0.026 to 0.554, while | S2/R1-purge |
| 177 | S8. Seed sensitivity | 0.554 | ...BR PR AUC ranges from approximately 0.026 to 0.554, while | S2/R1-purge |
| 178 | S8. Seed sensitivity | 0.014 | ...CARIB PR AUC ranges from approximately 0.014 to 0.039 in that au | S2/R1-purge |
| 178 | S8. Seed sensitivity | 0.039 | ...IB PR AUC ranges from approximately 0.014 to 0.039 in that audit; | S2/R1-purge |
| 180 | S8. Seed sensitivity | 0.01 | ...= PR_GBR − PR_CARIB > 0 in 5/5 seeds, range +0.01 to +0.52) and by | S2/R1-purge |
| 180 | S8. Seed sensitivity | 0.52 | ...− PR_CARIB > 0 in 5/5 seeds, range +0.01 to +0.52) and by | S2/R1-purge |
| 186 | S8. Seed sensitivity | 0.979 | ...CARIB, C reaches ROC AUC 0.979–0.982 and PR AUC 0. | S2/R1-purge |
| 186 | S8. Seed sensitivity | 0.982 | ...CARIB, C reaches ROC AUC 0.979–0.982 and PR AUC 0.475–0. | S2/R1-purge |
| 186 | S8. Seed sensitivity | 0.475 | ...IB, C reaches ROC AUC 0.979–0.982 and PR AUC 0.475–0.548 across | S2/R1-purge |
| 186 | S8. Seed sensitivity | 0.548 | ...reaches ROC AUC 0.979–0.982 and PR AUC 0.475–0.548 across | S2/R1-purge |
| 187 | S8. Seed sensitivity | 0.610 | ...well above the frozen contextual model (ROC 0.610) and, in every | S2/R1-purge |
| 188 | S8. Seed sensitivity | 0.979 | ...seed, at or near the PersA anchor (ROC 0.979, PR 0.460; seed 3 R | S2/R1-purge |
| 188 | S8. Seed sensitivity | 0.460 | ..., at or near the PersA anchor (ROC 0.979, PR 0.460; seed 3 ROC | S2/R1-purge |
| 189 | S8. Seed sensitivity | 0.9788 | ...0.9788 is marginally belo | S2/R1-purge |
| 189 | S8. Seed sensitivity | 0.9792 | ...0.9788 is marginally below the anchor 0.9792, so "at or near" i | S2/R1-purge |
| 192 | S8. Seed sensitivity | 0.0030 | ...of 5 seeds (the formal seed-42 estimate is +0.0030 [0.0013, | S2/R1-purge |
| 192 | S8. Seed sensitivity | 0.0013 | ...eds (the formal seed-42 estimate is +0.0030 [0.0013, | S2/R1-purge |
| 193 | S8. Seed sensitivity | 0.0046 | ...0.0046]); one seed shows | S2/R1-purge |
| 194 | S8. Seed sensitivity | 0.0004 | ...≈ −0.0004). The direction of | S2/R1-purge |
| 225 | S10. CARIB-specific diagnostics | 83,025 | ...(ocean-masked, n = 83,025): whether the fail | S10-carib-diag |
| 233 | S10. CARIB-specific diagnostics | 0.610 | ...recover CARIB discrimination (ROC-AUC 0.610 → 0.526), indicatin | S10-carib-diag |
| 233 | S10. CARIB-specific diagnostics | 0.526 | ...ecover CARIB discrimination (ROC-AUC 0.610 → 0.526), indicating | S10-carib-diag (rounded) |
| 281 | S12. XGBoost control: protocol equivalence | 0.0867 | ...The corresponding CARIB PR AUC values are 0.0867, 0.0921 and 0.0486 | R4-xgb |
| 281 | S12. XGBoost control: protocol equivalence | 0.0921 | ...orresponding CARIB PR AUC values are 0.0867, 0.0921 and 0.0486. | R4-xgb |
| 301 | S12. XGBoost control: protocol equivalence | 83,025 | ...never enters features). The audit found n = 83,025 and | R4-xgb |
| 302 | S12. XGBoost control: protocol equivalence | 83,025 | ...n_severe = 1,356 for both, with intersection 83,025, zero | R4-xgb |
| 321 | S13. GBR zero-shot transfer: protocol and  | 0.962 | ...(TSCF ROC AUC 0.962, PR AUC 0.554, CSI@ | S13-gbr-detail |
| 321 | S13. GBR zero-shot transfer: protocol and  | 0.554 | ...(TSCF ROC AUC 0.962, PR AUC 0.554, CSI@0.90 0.365). ( | S13-gbr-detail |
| 321 | S13. GBR zero-shot transfer: protocol and  | 0.365 | ...(TSCF ROC AUC 0.962, PR AUC 0.554, CSI@0.90 0.365). (c) GBR reliabili | S13-gbr-detail |
| 322 | S13. GBR zero-shot transfer: protocol and  | 0.332 | ...textual score is strongly miscalibrated (ECE 0.332) | S13-gbr-detail |
| 324 | S13. GBR zero-shot transfer: protocol and  | 0.365 | ...tau=0.90 (CSI 0.365, POD 0.669, FAR 0.5 | S13-gbr-detail |
| 324 | S13. GBR zero-shot transfer: protocol and  | 0.669 | ...tau=0.90 (CSI 0.365, POD 0.669, FAR 0.555, F1 0.53 | S13-gbr-detail |
| 324 | S13. GBR zero-shot transfer: protocol and  | 0.555 | ...tau=0.90 (CSI 0.365, POD 0.669, FAR 0.555, F1 0.535). Ranking | S13-gbr-detail |
| 324 | S13. GBR zero-shot transfer: protocol and  | 0.535 | ...au=0.90 (CSI 0.365, POD 0.669, FAR 0.555, F1 0.535). Ranking transfers | S13-gbr-detail |
| 354 | Note R2. Structural ablation (verification | 0.7 | ...es across all three domains even though only 0.7–5% of | R2/S7-ablation |
| 364 | Marginal shift | 0.402 | ...-   instantaneous-thermal: SCS→SCS test 0.402, SCS→GBR 0.202, | S2/R1-purge |
| 364 | Marginal shift | 0.202 | ...taneous-thermal: SCS→SCS test 0.402, SCS→GBR 0.202, | S2/R1-purge |
| 365 | Marginal shift | 0.291 | ...SCS→CARIB 0.291; | S2/R1-purge |
| 366 | Marginal shift | 0.710 | ...-   cumulative-thermal: SCS→GBR 0.710, SCS→CARIB 0.497; | R2/S7-ablation |
| 366 | Marginal shift | 0.497 | ...cumulative-thermal: SCS→GBR 0.710, SCS→CARIB 0.497; | S2/R1-purge |
| 367 | Marginal shift | 0.006 | ...-   context/calendar: SCS→SCS 0.006, SCS→GBR 0.043, SCS | S2/R1-purge |
| 367 | Marginal shift | 0.043 | ...-   context/calendar: SCS→SCS 0.006, SCS→GBR 0.043, SCS→CARIB 0.218. | S2/R1-purge |
| 377 | Joint static dependence | 0.867 | ...us distances from SCS training are: SCS test 0.867, GBR test 4.722, CA | R3-domain-shift |
| 377 | Joint static dependence | 4.722 | ...m SCS training are: SCS test 0.867, GBR test 4.722, CARIB test 2.300 ( | R3-domain-shift |
| 377 | Joint static dependence | 2.300 | ...: SCS test 0.867, GBR test 4.722, CARIB test 2.300 (full three-column | R3-domain-shift |
| 401 | Note R4. XGBoost control (verification evi | 0.890 | ...oving thermal predictors lowers ROC AUC from 0.890 to 0.806 and PR AUC | R4-xgb |
| 401 | Note R4. XGBoost control (verification evi | 0.806 | ...rmal predictors lowers ROC AUC from 0.890 to 0.806 and PR AUC from 0.0 | R4-xgb |
| 401 | Note R4. XGBoost control (verification evi | 0.092 | ...ROC AUC from 0.890 to 0.806 and PR AUC from 0.092 to 0.049, whereas T | R4-xgb |
| 401 | Note R4. XGBoost control (verification evi | 0.049 | ...from 0.890 to 0.806 and PR AUC from 0.092 to 0.049, whereas THERMAL an | R4-xgb |
| 445 | Note R5. GCBD validation (verification evi | 22.8 | ...ns in 2016 and 2019; within-year rate ratios 22.8 and 1.29). | R5-gcbd |
| 445 | Note R5. GCBD validation (verification evi | 1.29 | ...6 and 2019; within-year rate ratios 22.8 and 1.29). | R5-gcbd |
| 449 | Note R5. GCBD validation (verification evi | 22.8 | ...22.8; the top quintile ho | R5-gcbd |
| 451 | Note R5. GCBD validation (verification evi | 3.2 | ...lowest (3.2% vs 2.5%, rate-ratio | R5-gcbd |
| 451 | Note R5. GCBD validation (verification evi | 2.5 | ...lowest (3.2% vs 2.5%, rate-ratio 1.29) an | R5-gcbd |
| 451 | Note R5. GCBD validation (verification evi | 1.29 | ...lowest (3.2% vs 2.5%, rate-ratio 1.29) and 2017/2018 conta | R5-gcbd |
| 453 | Note R5. GCBD validation (verification evi | 33.1 | ...high-stress years gives 33.1). The headline metri | R5-gcbd |
| 453 | Note R5. GCBD validation (verification evi | 0.876 | ...ars gives 33.1). The headline metrics (AUROC 0.876, | R5-gcbd |
| 454 | Note R5. GCBD validation (verification evi | 35.2 | ...rate-ratio 35.2) therefore largely c | R5-gcbd |
| 473 | Note R5. GCBD validation (verification evi | 18.5 | ...18.5% (Q5). Rate ratio (Q | R5-gcbd |
| 473 | Note R5. GCBD validation (verification evi | 35.2 | ...18.5% (Q5). Rate ratio (Q5/Q1) = 35.2 (naive bootstrap mea | R5-gcbd |
| 473 | Note R5. GCBD validation (verification evi | 42.0 | ...e ratio (Q5/Q1) = 35.2 (naive bootstrap mean 42.0; | R5-gcbd |
| 474 | Note R5. GCBD validation (verification evi | 9.0 | ...site-cluster 95% CI 9.0–48.3). | R5-gcbd |
| 474 | Note R5. GCBD validation (verification evi | 48.3 | ...site-cluster 95% CI 9.0–48.3). | R5-gcbd |
| 495 | GCBD enrichment by model (PersA / TSAR / T | 0.81 | ...dependent severe bleaching comparably (AUROC 0.81–0.83; | R5-gcbd |
| 495 | GCBD enrichment by model (PersA / TSAR / T | 0.83 | ...dent severe bleaching comparably (AUROC 0.81–0.83; | R5-gcbd |
| 496 | GCBD enrichment by model (PersA / TSAR / T | 0.63 | ...Cliff's δ 0.63–0.66; MW p < 0.01; r | R5-gcbd |
| 496 | GCBD enrichment by model (PersA / TSAR / T | 0.66 | ...Cliff's δ 0.63–0.66; MW p < 0.01; rate-r | R5-gcbd |
| 496 | GCBD enrichment by model (PersA / TSAR / T | 0.01 | ...Cliff's δ 0.63–0.66; MW p < 0.01; rate-ratio CI lower | R5-gcbd |
| 499 | GCBD enrichment by model (PersA / TSAR / T | 0.876 | ...ion. TSCF's larger full-cohort values (AUROC 0.876, RR 35.2, | R5-gcbd |
| 499 | GCBD enrichment by model (PersA / TSAR / T | 35.2 | ...s larger full-cohort values (AUROC 0.876, RR 35.2, | R5-gcbd |
| 510 | GCBD enrichment by model (PersA / TSAR / T | 0.81 | ...0.81–0.89 across the five | R5-gcbd |
| 510 | GCBD enrichment by model (PersA / TSAR / T | 0.89 | ...0.81–0.89 across the five pred | R5-gcbd |
| 512 | GCBD enrichment by model (PersA / TSAR / T | 0.86 | ...0% in the lowest (site AUROC 0.86, Cliff's δ 0.72, Man | R5-gcbd |
| 512 | GCBD enrichment by model (PersA / TSAR / T | 0.72 | ...0% in the lowest (site AUROC 0.86, Cliff's δ 0.72, Mann–Whitney | R5-gcbd |
| 513 | GCBD enrichment by model (PersA / TSAR / T | 0.0005 | ..., and a site-preserving permutation with p = 0.0005 at | R5-gcbd |
| 520 | GCBD enrichment by model (PersA / TSAR / T | 0.941 | ...Per-year AUROC is 0.941 [0.807, 0.980] (201 | R5-gcbd |
| 520 | GCBD enrichment by model (PersA / TSAR / T | 0.807 | ...Per-year AUROC is 0.941 [0.807, 0.980] (2016), 0.9 | R5-gcbd |
| 520 | GCBD enrichment by model (PersA / TSAR / T | 0.980 | ...Per-year AUROC is 0.941 [0.807, 0.980] (2016), 0.951 [0.8 | R5-gcbd |
| 520 | GCBD enrichment by model (PersA / TSAR / T | 0.951 | ...r-year AUROC is 0.941 [0.807, 0.980] (2016), 0.951 [0.880, 0.954] | R5-gcbd |
| 520 | GCBD enrichment by model (PersA / TSAR / T | 0.880 | ...AUROC is 0.941 [0.807, 0.980] (2016), 0.951 [0.880, 0.954] | R5-gcbd |
| 520 | GCBD enrichment by model (PersA / TSAR / T | 0.954 | ...s 0.941 [0.807, 0.980] (2016), 0.951 [0.880, 0.954] | R5-gcbd |
| 521 | GCBD enrichment by model (PersA / TSAR / T | 0.875 | ...(2017; two severe events), 0.875 [0.631, 0.819] (201 | R5-gcbd |
| 521 | GCBD enrichment by model (PersA / TSAR / T | 0.631 | ...(2017; two severe events), 0.875 [0.631, 0.819] (2019) and | R5-gcbd |
| 521 | GCBD enrichment by model (PersA / TSAR / T | 0.819 | ...(2017; two severe events), 0.875 [0.631, 0.819] (2019) and 0.191 | R5-gcbd |
| 521 | GCBD enrichment by model (PersA / TSAR / T | 0.191 | ...ere events), 0.875 [0.631, 0.819] (2019) and 0.191 | R5-gcbd |
| 522 | GCBD enrichment by model (PersA / TSAR / T | 0.133 | ...[0.133, 0.265] (2018; two | R5-gcbd |
| 522 | GCBD enrichment by model (PersA / TSAR / T | 0.265 | ...[0.133, 0.265] (2018; two severe | R5-gcbd |
| 524 | GCBD enrichment by model (PersA / TSAR / T | 0.81 | ...0.81–0.90 under every sin | R5-gcbd |
| 524 | GCBD enrichment by model (PersA / TSAR / T | 0.814 | ...0.81–0.90 under every single-year exclusion: 0.814 [0.688, 0.861] | R5-gcbd |
| 524 | GCBD enrichment by model (PersA / TSAR / T | 0.688 | ...90 under every single-year exclusion: 0.814 [0.688, 0.861] | R5-gcbd |
| 524 | GCBD enrichment by model (PersA / TSAR / T | 0.861 | ...r every single-year exclusion: 0.814 [0.688, 0.861] | R5-gcbd |
| 525 | GCBD enrichment by model (PersA / TSAR / T | 0.892 | ...excluding 2016 (42 severe retained), 0.892 [0.780, 0.962] excl | R5-gcbd |
| 525 | GCBD enrichment by model (PersA / TSAR / T | 0.780 | ...excluding 2016 (42 severe retained), 0.892 [0.780, 0.962] excluding | R5-gcbd |
| 525 | GCBD enrichment by model (PersA / TSAR / T | 0.962 | ...ing 2016 (42 severe retained), 0.892 [0.780, 0.962] excluding | R5-gcbd |
| 526 | GCBD enrichment by model (PersA / TSAR / T | 0.885 | ...2019, and 0.885/0.898 excluding 201 | R5-gcbd |
| 526 | GCBD enrichment by model (PersA / TSAR / T | 0.898 | ...2019, and 0.885/0.898 excluding 2017/2018 | R5-gcbd |
| 529 | GCBD enrichment by model (PersA / TSAR / T | 22.8 | ...phenomenon (22.8 in 2016; undefined i | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.620 | ...sical anchor PersA retains enrichment (AUROC 0.620 [0.476, 0.754], sit | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.476 | ...nchor PersA retains enrichment (AUROC 0.620 [0.476, 0.754], site-clust | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.754 | ...ersA retains enrichment (AUROC 0.620 [0.476, 0.754], site-cluster CI e | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 2.1 | ...6, 0.754], site-cluster CI excluding 0.5; RR 2.1; δ +0.24; TSAR 0.614) | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.24 | ...], site-cluster CI excluding 0.5; RR 2.1; δ +0.24; TSAR 0.614) but the | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.614 | ...ster CI excluding 0.5; RR 2.1; δ +0.24; TSAR 0.614) but the frozen con | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.317 | ...e frozen contextual TSCF does **not** (AUROC 0.317 [0.172, 0.495] — in | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.172 | ...n contextual TSCF does **not** (AUROC 0.317 [0.172, 0.495] — inverted; | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.495 | ...xtual TSCF does **not** (AUROC 0.317 [0.172, 0.495] — inverted; RR 0.5 | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.51 | ...* (AUROC 0.317 [0.172, 0.495] — inverted; RR 0.51; δ −0.37; n = 1,993 | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.37 | ...0.317 [0.172, 0.495] — inverted; RR 0.51; δ −0.37; n = 1,993 matched, | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.915 | ...endent Caribbean field observations at AUROC 0.915 [0.806, 0.977] (RR | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.806 | ...Caribbean field observations at AUROC 0.915 [0.806, 0.977] (RR 5.4; δ | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.977 | ...an field observations at AUROC 0.915 [0.806, 0.977] (RR 5.4; δ +0.83; | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 5.4 | ...servations at AUROC 0.915 [0.806, 0.977] (RR 5.4; δ +0.83; TSAR 0.914) | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.83 | ...ns at AUROC 0.915 [0.806, 0.977] (RR 5.4; δ +0.83; TSAR 0.914) and the | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.914 | ...0.915 [0.806, 0.977] (RR 5.4; δ +0.83; TSAR 0.914) and the contextual | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.773 | ...0.83; TSAR 0.914) and the contextual TSCF at 0.773 [0.611, 0.889]. (3) | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.611 | ...SAR 0.914) and the contextual TSCF at 0.773 [0.611, 0.889]. (3) *GBR c | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.889 | ...14) and the contextual TSCF at 0.773 [0.611, 0.889]. (3) *GBR coverage | R5-gcbd |
| 533 | GCBD enrichment by model (PersA / TSAR / T | 0.876 | ...ntation's is not; the TSCF enrichment claim (0.876) is therefore scope | R5-gcbd |
| 551 | Note R6. Within-basin spatial holdout (SCS | 3.5 | ...latitude: the southern band (rows 0–10, 3.5–13.5°N, which contain | R6-spatial |
| 551 | Note R6. Within-basin spatial holdout (SCS | 13.5 | ...latitude: the southern band (rows 0–10, 3.5–13.5°N, which contains mo | R6-spatial |
| 562 | Note R6. Within-basin spatial holdout (SCS | 0.993 | ...the DHW anchor (held-out ROC 0.993 for both; PR 0.519 | R6-spatial |
| 562 | Note R6. Within-basin spatial holdout (SCS | 0.519 | ...DHW anchor (held-out ROC 0.993 for both; PR 0.519 vs 0.420). The | R6-spatial |
| 562 | Note R6. Within-basin spatial holdout (SCS | 0.420 | ...or (held-out ROC 0.993 for both; PR 0.519 vs 0.420). The | R6-spatial |
| 581 | Note R7. All-to-all transfer matrix and th | 49,322 | ...SCS 49,322 / GBR 19,065 / CAR | R7-transfer |
| 581 | Note R7. All-to-all transfer matrix and th | 19,065 | ...SCS 49,322 / GBR 19,065 / CARIB 83,025). P | R7-transfer |
| 581 | Note R7. All-to-all transfer matrix and th | 83,025 | ...SCS 49,322 / GBR 19,065 / CARIB 83,025). PersA is source- | R7-transfer |
| 588 | Note R7. All-to-all transfer matrix and th | 0.376 | ...TSCF −0.376 [−0.40, −0.35] vs P | R7-transfer |
| 588 | Note R7. All-to-all transfer matrix and th | 0.40 | ...TSCF −0.376 [−0.40, −0.35] vs PersA −0. | R7-transfer |
| 588 | Note R7. All-to-all transfer matrix and th | 0.35 | ...TSCF −0.376 [−0.40, −0.35] vs PersA −0.002 [−0 | R7-transfer |
| 588 | Note R7. All-to-all transfer matrix and th | 0.002 | ...TSCF −0.376 [−0.40, −0.35] vs PersA −0.002 [−0.011, +0.005] vs | R7-transfer |
| 588 | Note R7. All-to-all transfer matrix and th | 0.011 | ...TSCF −0.376 [−0.40, −0.35] vs PersA −0.002 [−0.011, +0.005] vs TSAR +0 | R7-transfer |
| 588 | Note R7. All-to-all transfer matrix and th | 0.005 | ...376 [−0.40, −0.35] vs PersA −0.002 [−0.011, +0.005] vs TSAR +0.001 | R7-transfer |
| 588 | Note R7. All-to-all transfer matrix and th | 0.001 | ...5] vs PersA −0.002 [−0.011, +0.005] vs TSAR +0.001 | R7-transfer |
| 589 | Note R7. All-to-all transfer matrix and th | 0.006 | ...[−0.006, +0.009]; GBR→CARIB | R7-transfer |
| 589 | Note R7. All-to-all transfer matrix and th | 0.009 | ...[−0.006, +0.009]; GBR→CARIB TSCF −0 | R7-transfer |
| 589 | Note R7. All-to-all transfer matrix and th | 0.263 | ...[−0.006, +0.009]; GBR→CARIB TSCF −0.263 [−0.28, −0.25]; GBR | R7-transfer |
| 589 | Note R7. All-to-all transfer matrix and th | 0.28 | ...[−0.006, +0.009]; GBR→CARIB TSCF −0.263 [−0.28, −0.25]; GBR→SCS TSC | R7-transfer |
| 589 | Note R7. All-to-all transfer matrix and th | 0.25 | ...006, +0.009]; GBR→CARIB TSCF −0.263 [−0.28, −0.25]; GBR→SCS TSCF | R7-transfer |
| 590 | Note R7. All-to-all transfer matrix and th | 0.230 | ...−0.230 [−0.26, −0.20]; CAR | R7-transfer |
| 590 | Note R7. All-to-all transfer matrix and th | 0.26 | ...−0.230 [−0.26, −0.20]; CARIB→SCS a | R7-transfer |
| 590 | Note R7. All-to-all transfer matrix and th | 0.20 | ...−0.230 [−0.26, −0.20]; CARIB→SCS and CARI | R7-transfer |
| 591 | Note R7. All-to-all transfer matrix and th | 0.02 | ...(+0.05, −0.02). PersA and TSAR sho | R7-transfer |
| 594 | Note R7. All-to-all transfer matrix and th | 0.775 | ...k target is stable across basins (Spearman ρ 0.775 / 0.783 / | R7-transfer |
| 594 | Note R7. All-to-all transfer matrix and th | 0.783 | ...is stable across basins (Spearman ρ 0.775 / 0.783 / | R7-transfer |
| 595 | Note R7. All-to-all transfer matrix and th | 0.802 | ...0.802 and ROC-AUC 0.982 / | R7-transfer |
| 595 | Note R7. All-to-all transfer matrix and th | 0.982 | ...0.802 and ROC-AUC 0.982 / 0.959 / 0.979 for | R7-transfer |
| 595 | Note R7. All-to-all transfer matrix and th | 0.959 | ...0.802 and ROC-AUC 0.982 / 0.959 / 0.979 for SCS / G | R7-transfer |
| 595 | Note R7. All-to-all transfer matrix and th | 0.979 | ...0.802 and ROC-AUC 0.982 / 0.959 / 0.979 for SCS / GBR / CAR | R7-transfer |
| 596 | Note R7. All-to-all transfer matrix and th | 0.70 | ...years within each basin (per-year ρ 0.70–0.83; GBR 2019 has z | R7-transfer |
| 596 | Note R7. All-to-all transfer matrix and th | 0.83 | ...years within each basin (per-year ρ 0.70–0.83; GBR 2019 has zero s | R7-transfer |
| 613 | Note R7. All-to-all transfer matrix and th | 0.016 | ...prevalence 0.016 versus 0.035 in GBR | R7-transfer |
| 613 | Note R7. All-to-all transfer matrix and th | 0.035 | ...prevalence 0.016 versus 0.035 in GBR), so the col | R7-transfer |
| 616 | Note R7. All-to-all transfer matrix and th | 0.0163 | ...re prevalence by record subsampling (R_low = 0.0163, the | R7-transfer |
| 617 | Note R7. All-to-all transfer matrix and th | 0.0355 | ...SCS/GBR severe records downsampled; R_high = 0.0355, | R7-transfer |
| 620 | Note R7. All-to-all transfer matrix and th | 0.962 | ...der both matched regimes: at R_low, TSCF ROC 0.962 | R7-transfer |
| 621 | Note R7. All-to-all transfer matrix and th | 0.957 | ...[0.957, 0.968] (GBR) versu | R7-transfer |
| 621 | Note R7. All-to-all transfer matrix and th | 0.968 | ...[0.957, 0.968] (GBR) versus 0.610 | R7-transfer |
| 621 | Note R7. All-to-all transfer matrix and th | 0.610 | ...[0.957, 0.968] (GBR) versus 0.610 (CARIB), with the P | R7-transfer |
| 622 | Note R7. All-to-all transfer matrix and th | 0.406 | ...persisting (0.406 versus 0.032); at R | R7-transfer |
| 622 | Note R7. All-to-all transfer matrix and th | 0.032 | ...persisting (0.406 versus 0.032); at R_high, TSCF R | R7-transfer |
| 622 | Note R7. All-to-all transfer matrix and th | 0.962 | ...ng (0.406 versus 0.032); at R_high, TSCF ROC 0.962 (GBR) | R7-transfer |
| 623 | Note R7. All-to-all transfer matrix and th | 0.610 | ...versus 0.610 [0.596, 0.624] (CAR | R7-transfer |
| 623 | Note R7. All-to-all transfer matrix and th | 0.596 | ...versus 0.610 [0.596, 0.624] (CARIB), wh | R7-transfer |
| 623 | Note R7. All-to-all transfer matrix and th | 0.624 | ...versus 0.610 [0.596, 0.624] (CARIB), while the | R7-transfer |
| 624 | Note R7. All-to-all transfer matrix and th | 0.581 | ...prevalence (CARIB 0.581 [0.567, 0.595] vers | R7-transfer |
| 624 | Note R7. All-to-all transfer matrix and th | 0.567 | ...prevalence (CARIB 0.581 [0.567, 0.595] versus GBR | R7-transfer (rounded) |
| 624 | Note R7. All-to-all transfer matrix and th | 0.595 | ...prevalence (CARIB 0.581 [0.567, 0.595] versus GBR 0.554). | R7-transfer |
| 624 | Note R7. All-to-all transfer matrix and th | 0.554 | ...lence (CARIB 0.581 [0.567, 0.595] versus GBR 0.554). Because | R7-transfer |
| 637 | Note R7. All-to-all transfer matrix and th | 0.022 | ...mass-bleaching years 2016 (severe prevalence 0.022) and 2019 (0.0062). | R7-transfer |
| 637 | Note R7. All-to-all transfer matrix and th | 0.0062 | ...ars 2016 (severe prevalence 0.022) and 2019 (0.0062). | R7-transfer |
| 639 | Note R7. All-to-all transfer matrix and th | 0.971 | ...ROC 0.971 (2016) and 0.975 (2 | R7-transfer |
| 639 | Note R7. All-to-all transfer matrix and th | 0.975 | ...ROC 0.971 (2016) and 0.975 (2019); CARIB TSAR | R7-transfer |
| 639 | Note R7. All-to-all transfer matrix and th | 0.964 | ...OC 0.971 (2016) and 0.975 (2019); CARIB TSAR 0.964 and 0.984, so the | R7-transfer |
| 639 | Note R7. All-to-all transfer matrix and th | 0.984 | ...2016) and 0.975 (2019); CARIB TSAR 0.964 and 0.984, so the | R7-transfer |
| 642 | Note R7. All-to-all transfer matrix and th | 0.725 | ...years (GBR TSCF ROC 0.725 and 0.485), which c | R7-transfer |
| 642 | Note R7. All-to-all transfer matrix and th | 0.485 | ...years (GBR TSCF ROC 0.725 and 0.485), which confirms th | R7-transfer |
| 644 | Note R7. All-to-all transfer matrix and th | 0.96 | ...cell stays ≥ 0.96 in all sixteen direc | R7-transfer |
| 650 | Note R7. All-to-all transfer matrix and th | 0.25 | ...ion: the same source-target pair spans up to 0.25 ROC AUC (CARIB-GBR: | R7-transfer |
| 650 | Note R7. All-to-all transfer matrix and th | 0.915 | ...et pair spans up to 0.25 ROC AUC (CARIB-GBR: 0.915/0.666/0.676) and th | R7-transfer |
| 650 | Note R7. All-to-all transfer matrix and th | 0.666 | ...r spans up to 0.25 ROC AUC (CARIB-GBR: 0.915/0.666/0.676) and the seed | R7-transfer |
| 650 | Note R7. All-to-all transfer matrix and th | 0.676 | ...s up to 0.25 ROC AUC (CARIB-GBR: 0.915/0.666/0.676) and the seed-42 pa | R7-transfer |
| 650 | Note R7. All-to-all transfer matrix and th | 0.959 | ...fully reproduce; TSAR is seed-invariant (ROC 0.959–0.987, per-cell sd | R7-transfer |
| 650 | Note R7. All-to-all transfer matrix and th | 0.987 | ...reproduce; TSAR is seed-invariant (ROC 0.959–0.987, per-cell sd <= 0.0 | R7-transfer |
| 650 | Note R7. All-to-all transfer matrix and th | 0.003 | ...d-invariant (ROC 0.959–0.987, per-cell sd <= 0.003, |dROC| <= 0.05 in | R7-transfer |
| 800 | Supplementary Tables | 83,025 | ...lations are exactly equivalent on CARIB (n = 83,025); zero target-doma | R2/S7-ablation |

## Notes

- Protocol constants (seeds, lead/history, thresholds, splits, channel
  counts) are whitelisted and skipped from file matching.
- Embedded table rows were verified cell-by-cell against source files
  in earlier passes (see SI_QC_report.md).
- Matching is textual (a rounded or full-precision form of the number
  occurs in the family file); the per-column semantic assignment of
  primary results is documented in NC_RESULT_CONSISTENCY_AUDIT.csv.
