# Self-containment verification

**Date:** 2026-10-31
**Method:** `git checkout-index -a -f --prefix=<clean temp dir>` — a checkout of
the tracked files only, with **no `.git`**, **no untracked files**, **no
`doc/`** and **no `results/qc/`**. All commands were run from that directory.

## Commands and results (git checkout, no `doc/`)

| Step | Command | Result |
|---|---|---|
| Evidence locks | `python scripts/smoke_test.py --locks` | `locks: ALL OK` |
| Metric consistency | `python tests/verify_metric_consistency.py` | 33 passed, 0 failed, 2 skipped |
| Unit tests | `python -m pytest tests/ -q` | 12 passed |
| Metric recomputation | `python experiments/08_infrastructure/recompute_calibration_metrics.py` | 21 rows, integrity checks passed |
| Figure 3 | `python scripts/figures/fig3_transfer_boundary.py` | PNG/SVG/PDF regenerated |
| Figure 5 | `python scripts/figures/fig6_gated_residual.py` | PNG/SVG/PDF + QC report regenerated |

The two skipped checks are the manuscript/SI **text** comparisons, which need
`doc/`. The script reports them explicitly (`SKIP ... (doc/ absent)`); they are
not counted as passing. With `doc/` present (in the archive) the same command
runs all **69** checks.

Checked-out files: **721**. Every `S.require(...)` data dependency of the
figure scripts exists **and is git-tracked** (0 missing, 0 untracked).

All entry points referenced by `scripts/reproduce_main.sh` and
`scripts/reproduce_supplementary.sh` exist.

## Hash verification

`MANIFEST.sha256` contains 761 entries. In the git checkout:

- **720 entries present and byte-identical** (0 mismatches);
- **41 entries absent by design**: 17 under `doc/` and 24 under `results/qc/`.

## Deliberate exclusions

| Item | Why absent from git | Where it ships |
|---|---|---|
| `doc/` (manuscript/SI markdown, compiled DOCX/PDF, pandoc template) | the manuscript is submitted separately; the git tree carries code, data and results | Zenodo archive (`tact-coral.zip`) |
| `results/qc/` | internal QC audits and intermediate tables | Zenodo archive only |

Runtime metric inputs were moved out of `results/qc/` into the tracked
`results/metrics/` (`metrics_v2.json/.csv/_bins.json`,
`structural_ablation_corrected.json`), so the figure and consistency pipelines
work from a git clone.

## Known limitation (documented, not a defect of this check)

`data/source_builders/*.py` (12 scripts) hard-code author-machine paths
(`D:\piecf_dns`, `D:\piecf_crw_data`, …) for the raw → derived dataset build
chain. That chain additionally requires the ~180 GB raw satellite archive,
which is not in the repository. It does not affect the shipped datasets,
evaluation, figures or tests.
