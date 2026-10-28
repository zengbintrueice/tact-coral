# Manuscript <-> SI consistency audit (NC submission-level pass)

Date: 2026-10-08. Scope: main_manuscript.md + supplementary_information.md.
No evidence-locked numerical result was modified; no experiment was re-run.

## Checklist

| Item | Status |
|---|---|
| Main-figure numbering 1-10, placement map matches Figures 1-10 | PASS |
| Main Figure 8 = conditional response / residual susceptibility; Figure 9 = counterfactual model-response geometry (non-causal caveat retained); Figure 10 = GCBD | PASS |
| Supplementary figure numbering 1-10, no gaps | PASS |
| Supplementary table numbering 1-9, no gaps | PASS |
| Every main-text SI citation resolves (Figs [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; Tables [1, 2, 3, 4, 5, 6]; Notes R['3', '4', '5', '6', '7', '8', '9', '10', '11', '13'] via 'Supplementary Information R#') | PASS |
| Every supplementary figure/table cited in main text or marked standalone verification evidence (Tables 7-9 are SI-internal verification records, cited from Notes R2/R5/R7-R9 and the overview map) | PASS |
| No stale S/R references (Tables S#, Note S#, Note R8-R13, (SX), fig_r*, fig2_transfer_gbr all removed) | PASS |
| 'Supplementary Note S13' re-pointed to 'Supplementary Methods S13' | PASS |
| No duplicate table content (old T7 merged into Table 3b; T9/T15 deleted, data retained in repository source files; main text quotes their key numbers) | PASS |
| No Results narrative embedded in SI (R1-R7 = verification notes; R8-R13 protocol content consolidated into Methods S14; results conclusions kept in the main text) | PASS |
| Supplementary Discussion trimmed to a single three-statement verification paragraph | PASS |
| Table 6 narrowed to 5 columns (per-basin ROC/PR/CSI combined) | PASS |
| Fig 6(e) renamed 'Cross-basin rank-association stability screen'; 'Invariant Causal Prediction'/'INVARIANT' removed from main text, figure script and regenerated figure (no causal-invariance claim) | PASS |
| Main wording: 'deterministic' -> 'substantially reduces seed-to-seed variation' (2 sites); GCBD abstract sentence restricted to observed enrichment; anchor-and-correct claim scoped to the three-basin study | PASS |
| Evidence-locked numerical values unchanged (spot checks: 0.610/0.962/0.959/0.876/83,025/seed 42/tau 0.90/0.975 present in both documents) | PASS |
| Figure 6(e) regenerated from the same associations.csv source data (no value change; terminology only) | PASS |

## Counts

- Main text: figures 10, tables 0, abstract ~148 words, main text ~3,930 words
  (Introduction -> Discussion, excluding legends/references/Methods).
- SI: figures 10, tables 9, Notes R1-R7 (verification), Methods S1-S14,
  Supplementary References (6 entries), Discussion (1 paragraph).

## MANUAL_REVIEW_REQUIRED

- Author list, affiliations, corresponding author (placeholder in place).
- Author Contributions and Competing Interests (placeholders in place).
- Acknowledgements (not included; optional).
- Reference [9] journal/author details should be re-verified against the
  original COMSNETS proceedings record.
- Continuous line numbers: add in Word (Layout -> Line Numbers -> Continuous)
  before PDF export if required by the submission portal.
