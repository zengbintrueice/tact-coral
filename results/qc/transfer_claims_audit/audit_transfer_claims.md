# Audit — transfer-matrix claims: seeds, geometry, difficulty and anchoring

**Date:** 2026-10-30
**Scope:** every unconditional claim about the six-direction transfer matrix
(geographic distance, target difficulty, seed sensitivity, anchoring
stability) in the Abstract, Results, Discussion, Fig. 2 caption and SI R7.
**No model retrained; no new experiment started.**

---

## 1. Evidence available

| Source | Seeds | Cells | What it supports |
|---|---|---|---|
| `results/transfer_coordinate/` seed-42 matrix | 1 | 6 directed × 2 models | the reference realisation |
| `results/transfer_coordinate/seed_scan/` | 0, 1, 2 | reverse-source rows (GBR/CARIB sources) | reverse-row seed variability |
| `results/seed_sensitivity/seed_metrics.csv` | 0, 1, 2, 3, 42 | SCS-source rows (3 targets) | forward-row seed variability |
| `results/physics_anchored/seed_sensitivity_tsar.csv` | 0, 1, 2, 3, 42 | TSAR, 3 targets | TSAR seed stability |
| `results/step5...`, `prevalence_matched_transfer/`, `regime_test` | seed 42 | — | prevalence and year-stratum controls |

Repeated experiments: **five seeds** for the SCS-source rows and TSAR; **three
seeds** for the reverse-source rows. This is enough to characterise variability,
but not to estimate a population distribution of seed outcomes.

---

## 2. Verified numbers

| Quantity | Value | Source |
|---|---|---|
| TSCF ROC range, CARIB→GBR, seeds 0/1/2 | 0.915 / 0.666 / 0.676; **range 0.249** | seed_scan |
| TSCF ROC range, GBR→CARIB | 0.765 / 0.694 / 0.894; range 0.200 | seed_scan |
| TSCF ROC range, GBR→SCS | 0.850 / 0.649 / 0.735; range 0.201 | seed_scan |
| TSCF ROC range, CARIB→SCS | 0.975 / 0.900 / 0.833; range 0.142 | seed_scan |
| **TSCF ROC range, SCS→GBR, 5 seeds** | 0.376–0.962; **range 0.587** | seed_metrics |
| TSCF ROC range, SCS→CARIB, 5 seeds | 0.460–0.728; range 0.268 | seed_metrics |
| TSCF ROC range, SCS→SCS, 5 seeds | 0.816–0.988; range 0.172 | seed_metrics |
| TSAR ROC range, 3 reverse seeds | 0.0003–0.0048; max sd 0.0020 | seed_scan |
| TSAR ROC range, 5 seeds | 0.0035–0.0060; max sd **0.0021** | tsar seed CSV |
| TSAR \|ΔROC\|, 6 directed cells × 3 seeds | max **0.0245** | seed_scan |
| GBR-vs-CARIB boundary | PR 5/5, CSI 5/5, ROC 4/5 (seed 2 flips: 0.376 < 0.460) | seed_transfer_boundary |

**Inconsistency found:** the Abstract and Results used "up to 0.25 ROC AUC"
as the headline seed range, but that is the maximum of the *reverse-source
three-seed* scan. The paper's own five-seed forward scan reaches **0.59**
(SCS→GBR, 0.376–0.962), which the Results later reports as "GBR 0.38–0.96".
The headline number understated the observed variability by more than 2×.

---

## 3. Claim-by-claim assessment

| # | Original sentence | Evidence level | Verdict | Revised |
|---|---|---|---|---|
| 1 | "…no geographic-distance or target-difficulty account reproduces this structure." | 3 basins, 1 seed; no formal test of all distance/difficulty accounts | **Overreach** (universal exclusion) | "Within these three basins, geographic distance and target difficulty do not order the matrix in a way that accounts for this pattern; because the pair-specific structure does not reproduce across seeds, we treat it as a reference realisation rather than a general law." |
| 2 | "The instability is a property of the unanchored contextual representation, and is not a property of deep networks or of transfer per se…" | one architecture, anchoring contrast | **Partially supported**; the "not deep networks" half generalises beyond the data | "Within this architecture and protocol, the instability is associated with the unanchored contextual representation rather than with transfer itself … We do not claim that this extends to other architectures." |
| 3 | "…the same source→target pair spans up to 0.25 ROC AUC across seeds" | 3 reverse seeds (0.25) vs 5 forward seeds (0.59) | **Numerically wrong as a global statement** | "…up to 0.25 ROC AUC across seeds **among the reverse-source pairs** … and the **five-seed forward scan spans up to 0.59** (SCS→GBR, 0.376–0.962)." |
| 4 | Abstract: "…seed-fragile (up to 0.25 ROC AUC) and collapses in three of six directions" | 0.25 is reverse-only; the 3/6 collapse is seed-42-specific | **Understated variability + unqualified single-seed structure** | "…seed-fragile (ROC AUC range up to 0.59 across seeds) and, **at seed 42**, collapses in three of six directions." |
| 5 | "Collapse is a property of the learned source--target pair…" | mechanism-flavoured; based on one matrix | **Overreach** | "**In this matrix**, collapse is pair-specific and has no analogue among the anchored formulations." |
| 6 | "The failure is not explained by marginal distance, static correlation or autocorrelation" | three specific diagnostics, 3 basins | **Broad exclusion** | "**None of** marginal distance, static correlation or autocorrelation **orders the transfer outcome in these three basins**; the channel-to-future-target mapping is the most discriminating of the diagnostics examined." |
| 7 | "TSAR … per-cell sd ≤ 0.003, \|ΔROC\| ≤ 0.05 in 6/6 cells" | 3 seeds (reverse) and 5 seeds (forward) | **Supported** (max sd 0.0021; max \|ΔROC\| 0.0245) | retained |
| 8 | "anchoring … substantially reduces seed-to-seed variation" | TSCF sd up to 0.215 vs TSAR sd ≤ 0.0021 | **Supported** | retained |
| 9 | "GBR-vs-CARIB boundary preserved by PR (5/5) and CSI (5/5), ROC (4/5)" | 5 seeds | **Supported** | retained |
| 10 | "the matrix rejects both [readings]" (Discussion) | already removed in the current draft | — | no action |

### Evidence levels used

- **Observation at one seed:** the seed-42 matrix, the per-year and
  prevalence-matched controls (these remove specific confounds at seed 42 but
  do not establish seed generality).
- **Seed-stable finding:** TSAR stability (sd ≤ 0.003 across both scans);
  GBR-vs-CARIB boundary in expectation (PR/CSI 5/5).
- **Speculative interpretation:** any statement about *why* a pair collapses
  (geography, difficulty, "property of the learned pair"), and any claim that
  a class of explanations is excluded.

---

## 4. Contradictions resolved

| Location | Before | After |
|---|---|---|
| Abstract vs Results | 0.25 vs "GBR 0.38–0.96" (0.59) | both now report 0.59 for the five-seed forward scan and 0.25 for reverse pairs |
| Results vs seed paragraph | "collapses in three of six" stated as a fact, then "seed-fragile in every direction" | the 3/6 collapse is now explicitly "at seed 42" |
| Discussion vs seed paragraph | "Collapse is a property of the learned pair" | "In this matrix, collapse is pair-specific" |
| SI R7 | "the collapse is an unanchored-context property, not a neural-model property" | "in this architecture and protocol … rather than with the neural architecture itself" |

---

## 5. Edits applied

1. **Abstract** — seed range 0.25 → 0.59; "collapses" qualified with "at seed 42".
2. **Results, matrix paragraph** — geographic/difficulty exclusion downgraded
   to an ordering statement + reference-realisation framing.
3. **Results, anchoring paragraph** — "not a property of deep networks"
   replaced by an architecture-scoped statement.
4. **Results, seed paragraph** — 0.25 scoped to reverse pairs; 0.59 added for
   the five-seed forward scan.
5. **Discussion** — "property of the learned pair" → "in this matrix,
   pair-specific"; "not explained by" → "none of … orders the outcome".
6. **SI R7** — "unanchored-context property, not a neural-model property" →
   architecture-scoped association; seed range 0.25 → 0.25 reverse / 0.59
   forward; "robust law" → "robust finding".

**Unchanged:** all point estimates, intervals, seed counts, thresholds and
sample sizes.

---

## 6. What would need new experiments

None for the revised wording. If the authors want a stronger general claim,
the following would be required (not started):

1. **More basins** (≥ 5) to test whether the pair-specific pattern recurs.
2. **More seeds** (≥ 10) to estimate the seed distribution rather than a range.
3. **A formal distance/difficulty model** fitted across basins, rather than a
   verbal exclusion.

Until then, the paper correctly states an observation at three basins and
five (forward) / three (reverse) seeds, and rests its robust claim on the
seed-invariant anchoring result.

---

## 7. Verdict

**Verified:** the seed ranges, TSAR stability, and boundary counts.
**Corrected:** the 0.25 headline (understated), the single-seed structure
presented as general, and three exclusion/mechanism sentences.
**Retained:** the anchoring stability result and the GBR-vs-CARIB boundary
finding, both supported across seeds.
