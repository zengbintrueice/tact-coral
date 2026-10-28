# Caribbean Zero-Shot Evaluation — Frozen TSCF

Generated: 2026-09-17T10:05:44.796897

Grid 22x32, n_dates=939, test years 2017-2020, PFV t+4 verification.

| Method | n | severe | prev | ROC | PR | CSI@0.9 | POD@0.9 | FAR@0.9 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | 83025 | 1356 | 0.0163 | 0.6101 | 0.0321 | 0.0262 | 0.0391 | 0.9261 | 0.09761 | 0.3065 |
| PersistenceA_DHW | 83025 | 1356 | 0.0163 | 0.9792 | 0.4602 | 0.2807 | 0.3510 | 0.4167 | 0.01413 | 0.0661 |
| PersistenceB_cumsum12 | 83025 | 1356 | 0.0163 | 0.8910 | 0.0942 | 0.0991 | 0.4889 | 0.8894 | 0.09608 | 0.2123 |

TSCF minus baselines (common record set):

- **TSCF_minus_PersistenceA_DHW**: ΔROC=-0.3692, ΔPR=-0.4281, ΔCSI@0.9=-0.2544
- **TSCF_minus_PersistenceB_cumsum12**: ΔROC=-0.2809, ΔPR=-0.0622, ΔCSI@0.9=-0.0729

Terminology: thermal-stress-aware model; strong physics-based persistence baselines (never 'naive'). Cumulative HotSpot channels are legitimate predictors (NOAA CRW DHW).
Normalization is SCS-train-only and frozen; nothing recomputed on Caribbean.
