# Seed Scan — Six-Direction Transfer Matrix Robustness (seeds 0/1/2)

Generated: 2026-10-02 10:49:14  
Provenance: GBR/CARIB sources retrained per seed (0/1/2, identical D1/a6 protocol);
seed-42 checkpoints untouched (`weights/{GBR,CARIB}`); forward SCS rows are not
re-scanned here (their five-seed audit is published in Supplementary Fig. S8).
Evaluation identical to `reverse_matrix.csv` (PFV t+4, frozen tau=0.90).
Note: the `collapse_tag`/`offdiag` columns of the intermediate CSV belong to a
superseded aggregation; verdicts below are recomputed from ROC/dROC directly.

## TSCF — ROC per (source → target) × seed

| source → target | seed 0 | seed 1 | seed 2 | mean | sd | range |
|---|---|---|---|---|---|---|
| CARIB → CARIB | 0.9153 | 0.9258 | 0.9281 | 0.9231 | 0.0068 | 0.0128 |
| CARIB → GBR | 0.9151 | 0.6664 | 0.6760 | 0.7525 | 0.1409 | 0.2487 |
| CARIB → SCS | 0.9746 | 0.9002 | 0.8327 | 0.9025 | 0.0710 | 0.1419 |
| GBR → CARIB | 0.7650 | 0.6941 | 0.8942 | 0.7844 | 0.1015 | 0.2001 |
| GBR → GBR | 0.8601 | 0.7904 | 0.9221 | 0.8575 | 0.0659 | 0.1317 |
| GBR → SCS | 0.8504 | 0.6493 | 0.7354 | 0.7450 | 0.1009 | 0.2011 |

## TSAR — ROC per (source → target) × seed

| source → target | seed 0 | seed 1 | seed 2 | mean | sd | range |
|---|---|---|---|---|---|---|
| CARIB → CARIB | 0.9803 | 0.9803 | 0.9836 | 0.9814 | 0.0019 | 0.0033 |
| CARIB → GBR | 0.9595 | 0.9591 | 0.9592 | 0.9593 | 0.0002 | 0.0004 |
| CARIB → SCS | 0.9818 | 0.9816 | 0.9811 | 0.9815 | 0.0004 | 0.0007 |
| GBR → CARIB | 0.9787 | 0.9791 | 0.9778 | 0.9785 | 0.0007 | 0.0013 |
| GBR → GBR | 0.9651 | 0.9603 | 0.9618 | 0.9624 | 0.0025 | 0.0048 |
| GBR → SCS | 0.9814 | 0.9817 | 0.9817 | 0.9816 | 0.0002 | 0.0003 |

## Pre-specified verdicts (COLLAPSE_THR = dROC ≤ −0.10)

- **GBR source, TSCF** — expectation: collapses to every target; holds 2/6 (seed, target) cells.
    - seed 0: SCS -0.010  CARIB -0.095
    - seed 1: SCS -0.141  CARIB -0.096
    - seed 2: SCS -0.187  CARIB -0.028
- **CARIB source, TSCF** — expectation: transfers to every target; holds 4/6 (seed, target) cells.
    - seed 0: SCS +0.059  GBR -0.000
    - seed 1: SCS -0.026  GBR -0.260
    - seed 2: SCS -0.095  GBR -0.252
- **GBR source, TSAR** — expectation: |dROC| ≤ 0.05 to every target; holds 6/6 (seed, target) cells.
    - seed 0: SCS +0.016  CARIB +0.014
    - seed 1: SCS +0.021  CARIB +0.019
    - seed 2: SCS +0.020  CARIB +0.016
- **CARIB source, TSAR** — expectation: |dROC| ≤ 0.05 to every target; holds 6/6 (seed, target) cells.
    - seed 0: SCS +0.002  GBR -0.021
    - seed 1: SCS +0.001  GBR -0.021
    - seed 2: SCS -0.003  GBR -0.025

## Interpretation (seed-fragility framing)

- **TSCF zero-shot transfer is seed-fragile in every direction.** The same
 source→target pair spans up to 0.25 ROC across seeds (CARIB→GBR: 0.915,
 0.666, 0.676); the seed-42 pair-specific pattern (CARIB-as-source succeeds,
 GBR-as-source collapses) does not reproduce: at seeds 1–2 CARIB→GBR collapses
 (dROC −0.26) and GBR→SCS collapses in seeds 1–2 (dROC −0.14/−0.19) while
 seed 0 does not. The pre-specified GBR/'collapse-everywhere' and
 CARIB/'transfer-everywhere' patterns each hold only in a subset of seeds.
- **The anchored formulation removes seed fragility.** TSAR ROC is 0.959–0.984
 in every cell with per-cell sd ≤ 0.002 and |dROC| ≤ 0.05 in 6/6 cells —
 the gated-residual design makes transfer outcome deterministic rather than
 a seed lottery.
- **Manuscript consequence**: the robust law is not 'which pair collapses' but
 'the unanchored contextual representation transfers unpredictably (across
 basins AND seeds), while anchoring around the physical coordinate stabilises
 transfer'. The seed-42 pair-specific matrix remains the evidence-locked
 reference realisation; the fragility statistics above bound its reliability.

