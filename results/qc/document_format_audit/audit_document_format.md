# Audit — manuscript heading/paragraph structure fix

**Files:** `doc/main_manuscript.md`, `doc/main_manuscript.docx`
**Pre-fix originals:** `LTTS_CRW/_pre_formatfix_backup_20261029/` (md, docx and both
`_NC_revision` copies; not overwritten).

---

## 1. Root cause

The markdown had `# References` (line 163) and `# Figure Legends` (line 205)
**immediately after a non-empty paragraph, with no blank line in between**.
Pandoc's default `blank_before_header` rule treats such a `#` as literal text,
so the heading was merged into the preceding paragraph.

## 2. Original DOCX locations (before fix)

| Paragraph | Style | Text (tail) |
|---:|---|---|
| 73 | `First Paragraph` | "… DOI: [INSERT VERIFIED PUBLIC REPOSITORY URL AND DOI]. **# References**" |
| 113 | `Compact` | "… https://doi.org/10.1038/s41467-025-67506-w **# Figure Legends**" |

Both matched the reported symptom exactly.

## 3. Fix applied

Two blank lines inserted in the markdown:

- before `# References`
- before `# Figure Legends`

No other character, reference, DOI or caption text was edited.

## 4. Post-fix DOCX structure

| Paragraph | Style | Text | Neighbour |
|---:|---|---|---|
| 74 | **Heading 1** | `References` | previous: Code-availability sentence (DOI placeholder intact) |
| 75…113 | `Compact` (numbered list, numId 1001) | reference entries 1–40 | first: "Oliver, E. C. J. et al. …" |
| 115 | **Heading 1** | `Figure Legends` | previous: reference 40 (Eakin et al., DOI intact) |
| 116… | `First Paragraph` | Figure 1–10 captions | first: "Figure 1 \| Study design …" |

**Heading 1 style:** Times New Roman, 20 pt, dark blue (`0F4761`), space before
18 pt, space after 4 pt, `keep_with_next = True`, **no page-break-before**.
No explicit page breaks (`w:type="page"`) exist anywhere in the body, so the
fix did not introduce or remove a page break.

## 5. Content-integrity check

Text-level diff between the pre-fix and post-fix DOCX:

- **Added paragraphs: 2** (`References`, `Figure Legends`)
- **Removed text: 0**
- Every DOI, reference entry, caption and body sentence is unchanged.
- Reference numbering starts a fresh numbered list (`numId 1001`) after the
  heading.

## 6. Other Markdown-leakage scan

| Check | Result |
|---|---|
| Headings without a preceding blank line, main md | 2 (both fixed) → **0** |
| Headings without a preceding blank line, SI md | **0** |
| Literal inline `#` inside a paragraph (e.g. "text # Heading") | **0** |
| SI structure unaffected | Yes |

## 7. Remaining author item (not changed)

The Code availability paragraph still contains an explicit insertion marker:

> "… replace this sentence with the permanent public repository URL and DOI:
> **[INSERT VERIFIED PUBLIC REPOSITORY URL AND DOI]**."

This is a deliberate author placeholder, not a formatting artefact. It must be
replaced with the real repository URL/DOI before submission. It is preserved
unchanged by this fix.

## 8. Diff report

### Markdown (only two lines added)

```diff
 The complete self-contained reproduction package … URL and DOI: [INSERT VERIFIED PUBLIC REPOSITORY URL AND DOI].
+
 # References
@@
 40.  Eakin, C. M. et al. … https://doi.org/10.1038/s41467-025-67506-w
+
 # Figure Legends
```

### DOCX

| | Before | After |
|---|---|---|
| Paragraphs containing `# References` | 1 (para 73, glued) | 0 |
| Paragraphs containing `# Figure Legends` | 1 (para 113, glued) | 0 |
| Standalone "References" heading | — | para 74, Heading 1 |
| Standalone "Figure Legends" heading | — | para 115, Heading 1 |
| Removed text | — | none |
| Added text | — | "References", "Figure Legends" |

## 9. Reproduction

```bash
cd LTTS_CRW/tact-coral/doc
pandoc main_manuscript.md -o main_manuscript.docx \
  --resource-path=. --reference-doc nc_reference.docx
```

Verification:

```python
from docx import Document
d = Document("main_manuscript.docx")
assert not any("# References" in p.text or "# Figure Legends" in p.text
               for p in d.paragraphs)
assert any(p.text.strip() == "References" and p.style.name == "Heading 1"
           for p in d.paragraphs)
assert any(p.text.strip() == "Figure Legends" and p.style.name == "Heading 1"
           for p in d.paragraphs)
```
