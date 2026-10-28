# Which GCBD enrichment values are canonical

Two generations of the same statistic exist in this release.

| Quantity | `step9_summary.md` (older) | canonical (paper) | canonical source file |
|---|---|---|---|
| severe RR (Q5/Q1), quintile | 35.185 | 35.185 | `results/review2_revision/gcbd_model_wise_enrichment.json` |
| severe RR site-cluster 95% CI | [8.845, 44.400] | **[8.951, 48.312]** | same |
| severe RR naive bootstrap mean | 40.478 | 41.990 | same |
| AUROC severe-vs-background site-cluster CI | [0.785, 0.902] (year-strat file) | **[0.808, 0.936]** | same |

**The manuscript and Supplementary Table 8 use the `review2_revision/` values**
(site-cluster CI [8.95, 48.3], AUROC 0.876 [0.808, 0.936]). The numbers in this directory's `step9_summary.md` come from the same
resampling draws (identical seed, 1,000 replicates, identical AUROC
confidence interval and identical valid-replicate counts). The **only**
difference is how the top-quintile rate ratio is bootstrapped:

- `step9_bootstrap.csv` holds the full-sample quintile labels **fixed**
  and slices them per replicate (`labels[idx]`), giving RR CI
  [8.845, 44.400] and naive mean 40.478;
- `review2_revision/gcbd_model_wise_enrichment.json` (and
  `step10_gcbd_matching_audit/step10_matching_results.csv`, offset 0)
  **re-estimates** `pd.qcut(..., 5)` inside every replicate, giving RR CI
  [8.951, 48.312] and naive mean 41.990.

The manuscript and Figure 7/10 use the review2/step10 convention for RR
and either source for AUROC (they are identical).
`results/gcbd_year_stratification/summary.json` is a separate analysis
with its own bootstrap draws; see its own protocol block.
