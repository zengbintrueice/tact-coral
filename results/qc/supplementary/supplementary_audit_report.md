# Supplementary Information — final build audit report

Generated: 2026-10-07T08:11:49.807953
Overall status: **PASS**

Authoritative numerical source: `results/figures/SOURCE_DATA_SUPPLEMENTARY.md` and the machine-readable source-data files it embeds. No value was invented, typed or approximated.

## Figures built

### GLOBAL
- sources: 
- n: None | threshold: None | CI: None | seed: None
- consistency checks: 1/1 PASS

### Supplementary Fig 01
- sources: results/figures/source_data/edfig_s1_source_data.csv
- n: 12 | threshold: n/a | CI: n/a | seed: n/a
- consistency checks: 2/2 PASS

### Supplementary Fig 02
- sources: results/figures/source_data/edfig_s2_source_data.csv
- n: SCS 49322/43089; GBR 19065/16926 | threshold: tau=0.90 | CI: none (point estimates) | seed: n/a
- consistency checks: 36/36 PASS

### Supplementary Fig 03
- sources: results/figures/source_data/edfig_s3_source_data.csv
- n: 49322 / 19065 | threshold: tau=0.90 | CI: none | seed: 42 (frozen training)
- consistency checks: 8/8 PASS

### Supplementary Fig 04
- sources: results/figures/source_data/edfig_s4_source_data.csv
- n: 49322 / 19065 | threshold: tau=0.90 | CI: none | seed: 42 (frozen training)
- consistency checks: 16/16 PASS

### Supplementary Fig 05
- sources: results/figures/source_data/edfig_s5_source_data.csv; results/review2_revision/carib_env_permutation.json
- n: permutation replicates | threshold: tau=0.90 | CI: 95% CI | seed: 42 (permutation)
- consistency checks: 4/4 PASS

### Supplementary Fig 06
- sources: results/step4_causal_zero_shot/causal_zero_shot_threshold_sweep_diagnostic.csv; (edfig_s6_source_data.csv is a truncated copy)
- n: 49322/19065 (SCS/GBR; CARIB has no formal sweep) | threshold: tau=0.9 pre-specified | CI: none | seed: n/a
- consistency checks: 2/2 PASS

### Supplementary Fig 07
- sources: results/figures/source_data/edfig_s7_source_data.csv
- n: 49322/19065/83025 | threshold: tau=0.90 | CI: paired week-block bootstrap 95% (n_boot=1000, seed=42) | seed: 42
- consistency checks: 12/12 PASS

### Supplementary Fig 08
- sources: results/figures/source_data/S8_seed_sensitivity.csv; results/seed_sensitivity/seed_transfer_boundary.csv
- n: 5 seeds x (49322 SCS / 19065 GBR / 83025 CARIB) | threshold: tau=0.90 | CI: t-based descriptive 95% | seed: 0,1,2,3,42 (formal seed 42 pinned a priori)
- consistency checks: 8/8 PASS

### Supplementary Fig 09
- sources: results/figures/source_data/edfig_s9_source_data.npz
- n: per-map grid (21x23 / 15x12 / 22x32) | threshold: severe 0.8 | CI: none | seed: n/a
- consistency checks: 7/7 PASS

### Supplementary Fig 10
- sources: results/figures/source_data/edfig_s10_source_data.json
- n: 83025 | threshold: tau=0.9 | CI: none | seed: n/a
- consistency checks: 3/3 PASS

## Duplicate detection

- No duplicate source signatures among the built figures (exactly one seed-sensitivity figure).

## Terminology guards

Verified absent from all titles/annotations: 'physics-informed', 'naive baseline', PersA as 'floor', thermal anchor as 'basin-invariant', universal-transfer claims, biological bleaching-prediction claims, internal experiment IDs in titles.

## Superseded outputs

The previous per-item extended-data figures (edfig_s1..s11 under results/figures/) are superseded by this rebuilt set. The former duplicate seed-sensitivity rendering (edfig_s8) is retired; the rebuilt set contains exactly one seed-sensitivity figure (Supplementary Fig. 8).
