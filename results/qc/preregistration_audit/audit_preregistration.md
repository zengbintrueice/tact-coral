# Audit — "pre-registered" claims

**Scope:** main manuscript, Supplementary Information, release README, protocol
JSONs, and the repository history of `tact-coral`.

**Verdict:** no evidence of formal pre-registration. The term has been replaced
by **"pre-specified"** in all submission files, with an explicit definition in
Methods. One item requires author confirmation (§5).

---

## 1. Search performed

| Location | Result |
|---|---|
| `doc/main_manuscript.md` | 3 occurrences before change (Results residual paragraph, Methods residual hierarchy, Fig. 8 caption) |
| `doc/supplementary_information.md` | 1 occurrence (S14 Definitions) |
| `README.md` | 1 occurrence (repository layout) |
| `doc/main_manuscript_NC_revision.md`, `doc/supplementary_information_NC_revision.md` | stale duplicates, same 4 occurrences; now synced |
| `results/**/protocol.json`, `summary.json` | protocol records exist but carry only analysis-run `"generated"` timestamps (e.g. `2026-09-29T20:08:31`), not a pre-registration record |
| `doc/archive*/` (parent project) | historical manuscript snapshots containing the term; not part of the submission |
| git history (`git log -S "pre-registered"`) | term present across commits; no commit adds a registration record |

**Registration evidence searched and not found:** OSF, AsPredicted,
ClinicalTrials, Registered Report, registration number/ID, public timestamped
protocol file, or any external link.

## 2. Evidence assessment

| Claim level | Required evidence | Available |
|---|---|---|
| "pre-registered" | A verifiable registration filed before data analysis (registry record, ID, timestamped public protocol) | **None** |
| "pre-specified" | A design fixed before evaluation on the test records; internal documentation sufficient | **Not provable from the repository alone** — requires author confirmation |

The analyses are documented in protocol files whose timestamps postdate the
analysis runs. They describe a fixed design but are not a pre-registration.

## 3. Changes applied

| File | Location | Before | After |
|---|---|---|---|
| `main_manuscript.md` | Results, residual paragraph | "First, a pre-registered training-free hierarchy" | "First, a **pre-specified** training-free hierarchy" |
| `main_manuscript.md` | Methods, residual hierarchy (i) | "(i) A pre-registered training-free hierarchy" | "(i) A **pre-specified** training-free hierarchy" |
| `main_manuscript.md` | Figure 8 caption | "the pre-registered matched-DHW, conditional-residual and mechanism-decomposition analyses" | "the **pre-specified** ... analyses" |
| `main_manuscript.md` | Methods, residual hierarchy paragraph | — | added: "Throughout, 'pre-specified' denotes that the analysis design was fixed before evaluation on the frozen records; the study was not publicly registered." |
| `supplementary_information.md` | S14 Definitions | "The pre-registered training-free hierarchy" | "The **pre-specified** training-free hierarchy" |
| `README.md` | repository layout | "pre-registered training-free hierarchy (H0–H3)" | "**pre-specified** training-free hierarchy (H0–H3)" |

Post-change scan: **0** occurrences of `pre-registered` / `preregistered` in
main manuscript, SI, their `_NC_revision` copies, or README.

## 4. Not changed

- **Archived manuscript versions** under the parent project's `doc/archive*`:
  historical snapshots, not part of the submitted files. Rewriting them would
  falsify the record.
- **Git history**: the term appears in earlier commits. History is not rewritten;
  the current commit is the submission state.

## 5. Requires author confirmation

1. **Was the residual hierarchy (H0–H3), the matched-DHW binning, and the
   mechanism-decomposition design fixed *before* evaluation on the frozen test
   records?**
   - If yes: "pre-specified" is accurate as written.
   - If no: replace with "predefined" or "a fixed ... hierarchy" and drop the
     timing claim.
2. **Will a public registration be filed before submission?** If an OSF (or
   similar) registration is created, "pre-registered" may be restored with the
   link and date.
3. **Is the sentence "the study was not publicly registered" acceptable in the
   Methods?** It is factually accurate based on the repository, but the authors
   may prefer a shorter disclaimer or none (in which case "pre-specified" still
   needs item 1 to hold).

## 6. Reproduction

```bash
cd LTTS_CRW/tact-coral
grep -rniE "pre-?registered|preregist" doc/ README.md
grep -n "pre-specified" doc/main_manuscript.md doc/supplementary_information.md
```
