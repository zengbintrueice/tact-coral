# Physics-Anchored Residual Head — Audit Verdicts (2026-09-17)

Scope: four questions from the user, answered with evidence. Companion:
`audit/audit_report.md` (2026-09-17T08:50, A/B/C/D/E groups) and the
ocean-mask fix + process split implemented 2026-09-17 09:5x–10:0x.

Frozen config now: **scale = 0.4** (SCS-val argmax CSI@0.9; the original
train/eval used 0.3 — see §3/§4). Checkpoints: TSCF `model_final_Btail3/
model.pt` md5 `7cdbca26cae03fa53be53210940e8ce1`; anchored `physics_anchored/
model.pt` md5 `b67f525d66872270b8e8a4d6d4e6fc97` (trained 0.3) → re-trained
at 0.4 (2026-09-17; 0.3 copy kept as `model_s03.pt`).

---

## 1. CARIB failure — TRUSTWORTHY ✅

Evidence:
- **Bit-exact reproduction across independent scripts**: TSCF CARIB
  ROC=0.5981282063358642 (n=94095, pre-mask) identical in two independent eval
  scripts; ocean-masked population gives 0.6101 (n=83025) in both
  `compute_caribbean_zero_shot.py` and `compute_physics_anchored_*`.
- **Harness validated**: same verbatim inference reproduces official SCS
  (0.9856/0.5271) and GBR (0.9600/0.5217) — the eval path is not the problem.
- **Inputs in-family**: 13-channel z-scores on CARIB test are within SCS-test
  distributions (hotspot z97.5 5.16 vs 4.99; validity 82% vs 78%) — NOT an
  input artifact (audit A "OOD evidence").
- **Signal present, model fails**: PersA_DHW (current DHW) reaches ROC 0.98 on
  CARIB while TSCF≈0.61; corr(TSCF_pred, PersA) collapses 0.66 (SCS) / 0.61
  (GBR) → 0.14 (CARIB). The data signal is strong; the frozen SCS-trained
  mapping does not reproduce it.
- **Physical sanity**: CARIB DHW reproduces documented events (2005: 1202 sev
  px, 2010: 1595, 2017: 313, 2019: 460, 2020: 1057).
- Robust to the record-population fix (ocean-masked vs not).

Verdict: genuine cross-basin generalization failure, NOT an implementation or
data artifact. Report as a negative result bounding generalization claims.

## 2. Anchored recovery — CREDIBLE ✅ (with one precision)

Evidence:
- **Strict zero-shot**: anchored ckpt trained on SCS 2003-2010 only (val
  2011-2015); CARIB dataset built 2026-09-16 AFTER training; no CARIB
  normalization/threshold/model selection (audit B1–B6).
- **Recovery magnitude** (scale=0.4, ocean-masked): CARIB anchored
  ROC 0.9822 / PR 0.5479 / CSI@0.9 0.3041 vs PersA 0.9792/0.4602/0.2807 vs
  TSCF 0.6101/0.0321/0.0262.
- **vs TSCF: overwhelmingly significant** — 95% CI ROC [0.9755,0.9869] vs
  [0.5864,0.6343] (disjoint).
- **vs PersA: NOT statistically significant** — CIs overlap: PR
  [0.40,0.64] vs [0.34,0.58]; CSI [0.21,0.37] vs [0.20,0.35].
  → Honest phrasing: the anchored model RECOVERS the strong physics-based
  persistence floor on an unseen basin (where frozen TSCF collapses); its
  point-estimate advantage over pure persistence is positive but not
  CI-supported at this sample size. Do NOT claim "beats persistence".
- **Mechanism**: residual magnitude diagnostics show gate median ≈0.007 →
  bounded deviation |y_final − PersA| ≤ scale·gate is tiny; the anchor
  dominates BY CONSTRUCTION (basin-invariant DHW). The recovery is thus
  structural: the model cannot be worse than PersA on CARIB, and it never is.
- **No leakage**: anchor = DHW(t)/10 available at t; target y(t+4) loss-only.

## 3. scale = 0.3 as a frozen configuration — NOT as-is ⚠️

- The predefined SCS-val rule (audit D-group, `argmax CSI@0.9 on SCS-val`)
  selects **scale = 0.4** (CSI 0.2953) over 0.3 (0.2862).
- The original training CFG and eval used **scale = 0.3**, which contradicts
  that committed selection rule.
- The scientific conclusion is ROBUST to scale (CARIB recovery holds at
  every scale tested 0.1–0.5: ROC ≈0.98, PR 0.52–0.55, CSI 0.28–0.31), so the
  choice of 0.3 vs 0.4 does not change any conclusion — but the FROZEN
  configuration must be single-valued and match the training scale.
- **Resolution applied**: re-trained at scale=0.4 (best val 0.498 < 0.3's
  0.530); `summary.json` now records cfg.scale=0.4 + model_md5; the eval
  infer script reads scale from `summary.json` (single source of truth).
  `model_s03.pt` keeps the 0.3 checkpoint for reference.

## 4. Code issues requiring fix + re-run — status

| # | Issue | Status |
|---|---|---|
| a | Eval record sets did NOT apply the land/ocean mask (scored finite-y-on-land: SCS 19110, GBR 1230, CARIB 11070 test cells) → misaligned with training `label_mask` | ✅ FIXED + re-run (all three eval scripts; ocean-masked) |
| b | Dataset build keeps finite-y-on-land targets | ✅ FIXED in `build_caribbean_1deg_from_raw.py` + `build_causal_dataset_caribbean_v2.py` (mask + record count); raw+v2 rebuilt, 14/14 checks PASS |
| c | Single-process infer+analysis segfaults at exit (netCDF4/mmap, exit 139) | ✅ FIXED — eval split into `compute_physics_anchored_infer.py` → `records.npz` → `compute_physics_anchored_analysis.py`; clean exit |
| d | scale config mismatch: trained 0.3 vs selected 0.4 | ✅ FIXED — re-trained at 0.4, infer reads scale from summary.json; three-domain eval re-run |
| e | Leftover `_test.{csv,json,npz}` debris in results/physics_anchored | 🧹 hygiene — delete |
| f | **Official Step-2/4 deliverables still report the UN-MASKED population** (SCS 0.9856/0.527, GBR 0.9600/0.522 vs masked 0.9865/0.546, 0.9624/0.554) | ✅ RESOLVED 2026-09-17 — `compute_persistence_baseline.py` collect_records now applies the ocean mask (finite y(t+4) & ~land_mask); Step-2 re-run (SCS n 68432→49322, GBR 20295→19065) and Step-4 re-run (AUDIT 12/12 PASS) on the masked population. Full downstream chain re-run for consistency: Step-3 ✅, Step-5 ✅ (PASS), Step-6 ✅ (Outcome A), Step-7 ✅ (Outcome A), Step-8 ✅ (Outcome E; NOTE: on the masked population GBR permutation ΔPR CI now includes 0 and GBR ENV gain C1-scale now False — previously True), Step-9 ✅ (Outcome A; GCBD numbers unchanged — land cells were never GCBD sites), Step-10 ✅ (ROBUST). Pre-mask deliverables preserved in `results/step2_persistence_v2_premask_bak_20260917/` and `results/step4_causal_zero_shot_premask_bak_20260917/`. New official Step-4 masked numbers: SCS TSCF n=49322/sev=419 ROC 0.9865 PR 0.5463 CSI@0.9 0.3374; GBR n=19065/sev=676 ROC 0.9624 PR 0.5543 CSI@0.9 0.3648. |
| g | `physics_anchored_eval.csv` is frequently locked by an open spreadsheet → CSV writes now fall back to `.new` (never kills the run) | ✅ FIXED |

## Bottom line
1. CARIB failure: **trustworthy**. 2. Anchored recovery: **credible** — as
"recovers the persistence floor on unseen basins", NOT "beats persistence"
(CI overlap). 3. scale: use **0.4** (SCS-val rule), now re-trained + frozen;
0.3 not acceptable as the frozen config. 4. All identified code issues are
fixed and the full downstream chain (Steps 2-10) re-run on the ocean-masked
population; pre-mask deliverables backed up. No outstanding fix required.
