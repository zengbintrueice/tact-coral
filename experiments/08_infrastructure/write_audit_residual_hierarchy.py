# write_audit_residual_hierarchy.py — AUDIT.md generator for the residual
# hierarchy test (reads summary.json / outputs; no hand-entered numbers).

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "residual_hierarchy"

S = json.load(open(RES / "summary.json", encoding="utf-8"))
deltas = pd.read_csv(RES / "paired_deltas.csv")
expl = pd.read_csv(RES / "residual_explanation.csv")

REC = RES.parents[1] / "results" / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
h = hashlib.md5()
with open(REC, "rb") as f:
    for c in iter(lambda: f.read(1 << 20), b""):
        h.update(c)

L = []
L.append("# AUDIT — Residual Hierarchy / Incremental Information Test\n")
L.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ")
L.append(f"Frozen record set: `physics_anchored/structural_ablation/"
         f"rec_structural_ablation.npz` (md5 `{h.hexdigest()}`, verified before run)\n")

L.append("## Fixed protocol (adherence)\n")
L.append("- PersA = clip(DHW/10,0,1) from the frozen record set; "
         "R = TSAR − PersA (TSAR = seed-42 C_anchored, scale 0.4). No model was "
         "retrained; no target fitting, target tuning or threshold optimisation "
         "was performed anywhere in this test.")
L.append("- Identical frozen records per basin (single complete-case mask over "
         "all hierarchy variables); identical issue-week grouped 5-fold CV for "
         "the residual-explanation models.")
L.append("- All uncertainty: paired week-block bootstrap (1,000, seed 42); the "
         "SAME resampled weeks are used across all hierarchy levels, so adjacent "
         f"level deltas are paired.")
L.append("- Interpretation restricted to predictive / representation level; no "
         "causal claims.\n")

L.append("## Pre-specified hierarchy construction\n")
L.append("The hierarchy scores are TRAINING-FREE equal-weight means of "
         "within-basin percentile ranks (no coefficient is fitted to the target, "
         "so no target information enters score construction):")
L.append("")
L.append("| Level | Components |")
L.append("|---|---|")
L.append("| H0 | PersA |")
L.append("| H1 | PersA + {dDHW_4w, cumsum_4w, cumsum_12w} (thermal trajectory/persistence) |")
L.append("| H2 | H1 + {chl, par, wind} (nonthermal context) |")
L.append("| H3 | H2 + R (the frozen TSAR residual) |")
L.append("")
L.append("The frozen TSAR score itself is reported alongside for the "
         "consistency check H3 vs TSAR. Because H1–H3 are rank means, Brier and "
         "log-loss are **diagnostic within-family** comparisons, not calibrated "
         "probabilities; log-loss uses the pre-fixed clipping "
         f"{S['protocol']['logloss_clip']}.\n")

L.append("## Complete-case handling\n")
L.append("| Basin | records | complete-case | dropped | severe |")
L.append("|---|---|---|---|---|")
for b, d in S["basins"].items():
    L.append(f"| {b} | {d['n_total']} | {d['n_complete_case']} | "
             f"{d['n_dropped']} | {d['n_severe']} |")
L.append("")

L.append("## Pre-specified analysis choices\n")
L.append("- Residual-explanation models: fixed OLS on standardized predictors; "
         "out-of-sample R² via deterministic week-grouped 5-fold CV; the paired "
         "bootstrap of the out-of-sample ΔR² recomputes the same grouped-fold "
         "split (weeks enumerated in sorted order, folded i mod 5) within each "
         "replicate — fixed rule, no tuning.")
L.append("- Verdict rule (pre-specified): the residual's additional information "
         "is declared *established* for a basin iff the H3−H2 95% CI excludes 0 "
         "in the positive direction for ROC-AUC or PR-AUC; otherwise reported "
         "as **not established**.\n")

L.append("## Deviations\n")
L.append("- None from the requested protocol.\n")

L.append("## Verdict summary (pre-specified rule)\n")
L.append("| Basin | H3−H2 ΔROC [CI] | H3−H2 ΔPR [CI] | residual extra info |")
L.append("|---|---|---|---|")
for b, v in S["verdicts"].items():
    r, p = v["H3_over_H2_roc"], v["H3_over_H2_pr"]
    L.append(f"| {b} | {r['point']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
             f"| {p['point']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] "
             f"| {'**established**' if v['residual_extra_info_established'] else 'not established'} |")
L.append("")

L.append("## Interpretation guard\n")
L.append("Hierarchy levels share all lower-level components by construction; "
         "the equal-weight rank mean is one pre-specified combination rule. A "
         "failure of H3 to beat H2 does not prove the residual is a "
         "re-combination of the listed variables under other combination rules; "
         "a success shows only that, under this fixed rule, the residual adds "
         "discrimination beyond the explicit candidate set. All statements are "
         "predictive/representation-level.\n")

with open(RES / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"AUDIT written -> {RES / 'AUDIT.md'}")
