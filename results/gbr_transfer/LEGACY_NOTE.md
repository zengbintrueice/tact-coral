# Legacy artifact in this directory

`gbr_zeroshot_evaluation.json` is a **legacy a6-era** result and does NOT
correspond to any number reported in the manuscript.

- It records `"SCS-trained a6"`, `n_pixels = 20,295`, `n_severe = 729`,
  `severe_ratio = 0.0359`, CSI@0.9 = 0.3479.
- The manuscript's GBR zero-shot values (ROC 0.962, PR 0.554, CSI@0.90 0.365,
  POD 0.669, FAR 0.555, F1 0.535, n = 19,065, 676 severe) come from the
  structural-ablation evaluation of the frozen D1 checkpoint:
  `results/physics_anchored/structural_ablation/structural_ablation.json`
  (domain `GBR`, model `D_TSCF`).

The file is kept only because `experiments/03_gbr_transfer/gbr_zeroshot_evaluation.py`
writes to this location; re-running that script regenerates it from the frozen
checkpoint. **Do not use it as a source for the manuscript's GBR numbers.**
