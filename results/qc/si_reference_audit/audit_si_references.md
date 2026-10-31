# Audit — supplementary-material references (main text ↔ SI)

**Scope:** `doc/main_manuscript.md` and `doc/supplementary_information.md`
(current submission files).

**Result:** all 46 supplementary references resolve to an existing SI item;
0 unresolved. Six pre-existing defects were repaired (format, structure,
orphan). No citation was deleted to hide missing content.

---

## 1. Actual SI inventory

| Series | Items | Location |
|---|---|---|
| Supplementary Methods | **S1–S14** | `## S1.` … `## S14.` |
| Supplementary Notes | **R1–R7** | `### Note R1.` … `### Note R7.` |
| Supplementary Tables | **1–10** | `**Supplementary Table N \| …**` |
| Supplementary Figures | **1–10** | `**Supplementary Fig. N \| …**` |

The `S` and `R` prefixes are internal series labels of this SI (S = Methods
section, R = verification Note). They are kept unchanged; no plausible-looking
new numbers were invented.

## 2. Defects found and repaired

| # | Defect | Evidence | Repair |
|---|---|---|---|
| 1 | 16 citations used the generic form "Supplementary Information R#/S#" | e.g. "Supplementary Information R7"; SI heading is "Note R7" | Replaced with **"Supplementary Note R#"** and **"Supplementary Methods S#"** (R4 ×2, R5 ×6, R6 ×1, R7 ×6, S5 ×1, S7 ×1) |
| 2 | `S13` was an H3 subsection nested under `S12` | `### S13.` under `## S12.` | Promoted to **`## S13.`** (separate Methods section) |
| 3 | Stale label "**Supplementary Note 12**" inside S12 (no Note 12 exists; it also said "figure embedded below" with no figure) | SI line ~311 | Relabelled **"Main-text Figure 6."** and removed the false embedding statement |
| 4 | `S14` appeared **after** the Notes (Methods S1–S13 → Notes → S14) | section order | Moved S14 before `# Supplementary Notes`; Methods now S1–S14 contiguously |
| 5 | **Supplementary Table 7** (six-direction matrix) cited nowhere | no hit in main or SI | Added citation at main line 47: "Full matrices (Supplementary Table 7) …" |
| 6 | Dangling lead-in "The latest control gives:" with no table beneath it (S12) | SI line ~279 | Removed |

## 3. Reference mapping (current state)

Main-text line numbers refer to `doc/main_manuscript.md`.

| SI target | Type | Occurrences | Main-text lines | Status |
|---|---|---:|---|---|
| Supplementary Methods S5 | Methods | 1 | 83 | OK |
| Supplementary Methods S7 | Methods | 2 | 75, 144 | OK |
| Supplementary Methods S13 | Methods | 1 | 47 | OK |
| Supplementary Methods S14 | Methods | 6 | 97, 121, 148, 150 | OK |
| Supplementary Note R4 | Note | 2 | 85, 152 | OK |
| Supplementary Note R5 | Note | 6 | 99, 103, 105, 121, 132 | OK |
| Supplementary Note R6 | Note | 1 | 121 | OK |
| Supplementary Note R7 | Note | 6 | 43, 47, 49, 121, 146 | OK |
| Supplementary Table 1 | Table | 1 | 39 | OK |
| Supplementary Table 2 | Table | 1 | 140 | OK |
| Supplementary Table 3 | Table | 1 | 75 | OK |
| Supplementary Table 4 | Table | 1 | 91 | OK |
| Supplementary Table 5 | Table | 2 | 85, 152 | OK |
| Supplementary Table 6 | Table | 1 | 49 | OK |
| Supplementary Table 7 | Table | 1 | 47 | OK (newly cited) |
| Supplementary Table 9 | Table | 4 | 97, 148 | OK |
| Supplementary Table 10 | Table | 1 | 81 | OK |
| Supplementary Fig. 1 | Figure | 1 | 39 | OK |
| Supplementary Fig. 2 | Figure | 1 | 39 | OK |
| Supplementary Fig. 5 | Figure | 1 | 83 | OK |
| Supplementary Fig. 6 | Figure | 1 | 142 | OK |
| Supplementary Fig. 7 | Figure | 1 | 75 | OK |
| Supplementary Fig. 8 | Figure | 1 | 75 | OK |
| Supplementary Fig. 9 | Figure | 1 | 231 | OK |
| Supplementary Fig. 10 | Figure | 1 | 47 | OK |

Items not cited by the main text but cited inside the SI (acceptable):
Supplementary Methods S1–S4, S6, S8–S12; Notes R1–R3; Tables 8;
Figs 3–4.

## 4. Invalid or missing references

**None after repair.** Before repair, the substantive issues were:

- `S13` existed but as the wrong heading level (still resolvable; structural).
- "Supplementary Note 12" did not exist (dangling internal label).
- Supplementary Table 7 existed but was orphaned (no citation).
- 16 citations used a non-standard series name but pointed to real targets
  (format, not missing content).

No content was missing from the SI, and no reference was deleted.

## 5. Main-text self-containedness

Each key conclusion is stated with its numbers in the main text; the SI
supplies audits, full matrices and sensitivity details.

| Key conclusion | Main-text support | SI support (supplementary) |
|---|---|---|
| Six-direction transfer matrix; collapse is pair-specific at seed 42 | Results §2, Fig. 2 (ROC values, ΔROC CIs) | Note R7, Table 7 |
| TSCF transfer is seed-fragile; anchoring reduces seed variation | Results §2 (ranges, sd), Fig. 2d | Note R7, Table 6 |
| PersA is source-independent and stable | Results §2–3 (ROC values) | Note R7, Table 7 |
| Structural ablation A/B/C/D | Results §4, Fig. 5, Table 3 references | Table 3, Methods S7 |
| Conditional residual beyond DHW | Results §6, Fig. 8 (ΔROC/ΔPR CIs) | Methods S14, Table 9 |
| GCBD external validation and its limits | Results §7, Fig. 10 (AUROC, CI, RR, n) | Note R5, Table 8 |
| XGBoost architecture control | Results §5, Fig. 6 (ROC/PR) | Table 5, Note R4 |
| Thermal attribution / domain shift | Results §5, Fig. 7 | Table 4, Notes R3 |

- No key conclusion is supported **only** by the SI.
- Internal terms used in the main text (TSCF, TSAR, PersA, H0–H3, MH1/MH2,
  CSI@0.90, ECE) are defined in the main text; a scan for internal project
  jargon (`causal v2`, `PFV`, `D1/a6`, `a6`, `Btail3`, `PIECF`, "pure
  forecasting") returned **no hits**.
- The main text cites the SI 46 times over ~250 lines. Most citations are to
  audits and sensitivity analyses; the primary estimates and CIs are in the
  main text. This is heavy but not a transfer of the core argument.

## 6. Author confirmation

None blocking. Optional: if the journal copy-editor prefers
"Supplementary Note 1" without the internal `R` prefix, the SI headings and all
citations would need a joint renumbering. This audit deliberately preserves the
existing `S`/`R` numbering rather than guessing new labels.

## 7. Reproduction

```bash
cd LTTS_CRW/tact-coral
grep -oE "Supplementary (Note R[0-9]+|Methods S[0-9]+|Table [0-9]+|Fig\. [0-9]+)" doc/main_manuscript.md | sort | uniq -c
grep -nE "^## S[0-9]+\.|^### Note R[0-9]+\.|^\*\*Supplementary (Table|Fig\.)" doc/supplementary_information.md
```
