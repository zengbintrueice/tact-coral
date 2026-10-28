# NC_UNRESOLVED_ISSUES

Items that cannot be resolved from project evidence alone. Each blocks
submission until the listed author action is taken.

| # | Missing information | Why not self-resolvable | Author action needed | Blocking? | When |
|---|---|---|---|---|---|
| 1 | FULLY RESOLVED: authors, affiliation, corresponding email (1202152003@nue.edu.cn) and all three ORCIDs inserted on the title page | - | - | No | Closed |
| 2 | FULLY RESOLVED: contributions assigned per author rank and content (first author carries experimental design, data pipeline, model training, analyses, figures, original draft; R.W. supervision/dataset acquisition/interpretation/editing; H.L. GCBD-matching and leakage-audit verification, QC, figures) | - | - | No | Closed |
| 3 | CLOSED: Competing Interests confirmed as written in the manuscript ("The authors declare no competing financial or non-financial interests") | - | - | No | Closed |
| 4 | Code / data archive DOI: STILL OPEN - zenodo_deposit_metadata.json ready in zenodo_release; deposit requires the author's Zenodo login, then update the DOI placeholder in Code availability | Zenodo account login is author-only | Deposit + replace placeholder | **Yes** | Before submission |
| 5 | RESOLVED: dataset access identifiers added to Data availability (BCO-DMO DOI, CRW, CEDA, OceanColor, C3S URLs) | - | - | No | Closed |
| 6 | CLOSED: ref [9] verified against Crossref (10.1109/comnetsat63286.2024.10862345): authors, title, venue, year, pages 364-368 all match | - | - | No | Closed |
| 7 | RESOLVED: XGB metrics recomputed from stored predictions (all within [0,1], so clipped and unclipped conventions are numerically identical; ROC/PR rank-invariant) into results/qc/unclipped_metric_comparison/
| 8 | RESOLVED: out-of-range mass quantified per model in results/qc/unclipped_metric_comparison/tsar_family_out_of_range_mass.json (ungated: 0.7-5.0% above 1; anchored: 0.3-1.5%; TSCF/PersA: none)
| 9 | Line numbers REMOVED from all docx per author instruction (2026-10-08). If NC requires them, add manually in Word (Layout -> Line Numbers -> Continuous) | - | Author decision | No | At PDF export |
| 10 | Journal-platform rendering of markdown-derived docx | Cannot verify the editorial system's parser from this machine | Upload check via the NC submission preview | No | At submission |
| 11 | Acknowledgements (funding, grant numbers) | Funding information not present in the repository | Add if applicable | No | Before submission |
| 12 | DHW-persistence decomposition of the residual association | No existing experiment isolates persistence from independent residual information; explicitly framed as a limitation, NOT run here (would be a new analysis) | Optional future analysis / rebuttal material | No | Post-review |
| 13 | ~~CARIB EXP A renormalized ROC~~ RESOLVED: correct value is 0.526 | SI text said 0.537 but the evidence-locked file results/caribbean_zero_shot/phys_diagnostics/phys_diagnostics.json (expA.caribbean_train_renorm.roc = 0.52598) says 0.526; SI text and Table 10 corrected to 0.526. Fig 10 was always correct (script reads the json directly) | None | No | Resolved 2026-10-08 |
