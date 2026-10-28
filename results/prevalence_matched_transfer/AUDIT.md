# AUDIT - Prevalence-matched directional comparison

- Prevalence confound removed by subsampling to a common severe prevalence:
  R_low p*=0.01633 (CARIB; SCS/GBR severe downsampled) and
  R_high p*=0.03546 (GBR; SCS/CARIB background downsampled, all severe kept).
- 500 subsample replicates (seed 42); percentile CIs; frozen seed-42 models,
  reverse checkpoints md5-tracked in transfer_coordinate/weights.
- ROC-AUC is prevalence-invariant at population level, so the informative
  matched comparison is PR-AUC (prevalence-dependent) plus the composition-
  robustness check on ROC.
- Associative conditional-risk interpretation only; no retraining, no target
  fitting, no threshold optimisation.
