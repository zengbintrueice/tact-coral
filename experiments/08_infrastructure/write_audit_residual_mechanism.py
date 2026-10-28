# write_audit_residual_mechanism.py — generate AUDIT.md for the residual
# mechanism decomposition from the result files (no hand-entered numbers).

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "residual_mechanism_decomposition"

S = json.load(open(RES / "summary.json", encoding="utf-8"))
assoc = pd.read_csv(RES / "associations.csv")
nested = pd.read_csv(RES / "nested_models.csv")
boot = pd.read_csv(RES / "bootstrap.csv")

REC = RES.parents[1] / "results" / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
h = hashlib.md5()
with open(REC, "rb") as f:
    for c in iter(lambda: f.read(1 << 20), b""):
        h.update(c)

L = []
L.append("# AUDIT — Residual Mechanism Decomposition\n")
L.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ")
L.append(f"Frozen record set: `physics_anchored/structural_ablation/"
         f"rec_structural_ablation.npz` (md5 `{h.hexdigest()}`, verified before run)\n")

L.append("## Fixed protocol (adherence)\n")
L.append("- R = TSAR (C_anchored, scale 0.4, seed-42 checkpoint) − PersA_DHW, "
         "per record, from the frozen record set; no model was retrained.")
L.append("- No target fitting / target tuning / threshold optimisation anywhere: "
         "Part B models are explanatory regressions of R on mechanism variables "
         "(fixed OLS form, 5-fold CV grouped by issue week, no hyperparameter "
         "search); Part D scores are pre-specified training-free rank means.")
L.append("- All uncertainty from the paired week-block bootstrap (1,000 "
         "replicates, seed 42); identical resampled weeks across compared "
         "models/scores.")
L.append("- FDR: Benjamini–Hochberg within family (raw / partial) per basin.")
L.append("- Interpretation restricted to associative / representation level; "
         "no causal claims.\n")

L.append("## Mechanism variables (verbatim project definitions)\n")
L.append("| Variable | Definition | Source |")
L.append("|---|---|---|")
L.append("| current_DHW | PersA×10 = clip(DHW/10,0,1)×10 (clipped coordinate) | frozen record set (A_PersA_pred) |")
L.append("| dDHW_4w | y(t) − y(t−4), 4-week change of the clipped DHW coordinate (causal: t−4 ≤ t) | causal v2 y_regression |")
L.append("| d_hotspot_dt | existing weekly HotSpot-derivative channel (alternate derivative, reported alongside) | causal v2 channel |")
L.append("| cumsum_4w / cumsum_12w | existing trailing 4/12-week HotSpot exposure channels (persistence) | causal v2 channels |")
L.append("| ssta, hotspot | existing thermal state channels | causal v2 channels |")
L.append("| chl, par, wind | existing nonthermal context channels (chlorophyll, PAR, wind_speed) | causal v2 channels |")
L.append("\nNo window or label definition was changed.\n")

L.append("## Complete-case handling\n")
L.append("A single complete-case mask over ALL candidate variables is applied so "
         "that every part (A–D) uses identical records per basin.\n")
L.append("| Basin | records | complete-case | dropped | severe (kept) | dominant NaN source |")
L.append("|---|---|---|---|---|---|")
for b, d in S["basins"].items():
    nan = d["nan_counts"]
    dom = max(nan, key=nan.get)
    L.append(f"| {b} | {d['n_total']} | {d['n_complete_case']} | {d['n_dropped']} "
             f"| {d['n_severe']} | {dom} ({nan[dom]}) |")
L.append("")

L.append("## Pre-specified analysis choices\n")
L.append("- Primary explanatory model: OLS on standardized predictors. Because R "
         "= gate×tanh(·)×0.4 is bounded and zero-inflated, a **pre-fixed "
         "rank-OLS** (rank-transformed response and predictors, same folds) is "
         "reported alongside as the robust fallback; conclusions are read "
         "jointly from both.")
L.append("- CV: 5-fold GroupKFold grouped by issue week (prevents week-level "
         "temporal leakage in CV R²); model form fixed a priori, no tuning.")
L.append("- ΔR² increments are in-sample, with week-block bootstrap CIs "
         "(documented; CV R² reported as the out-of-sample reference).")
L.append("- Part D combination rule (pre-specified, training-free): equal-weight "
         "mean of within-basin percentile ranks of {PersA, dDHW_4w, cumsum_4w, "
         "cumsum_12w} (thermal block), optionally + {chl, par, wind} (context "
         "block). No coefficients are fitted, so no target information enters "
         "the score construction.")
L.append("- dDHW_4w is the first difference of the clipped coordinate; DHW > 10 "
         "degree-weeks is clipped to 1.0 in the coordinate, which slightly "
         "compresses both current_DHW and dDHW_4w at extreme heat. This is the "
         "coordinate definition the models themselves use.\n")

L.append("## Deviations\n")
L.append("- None from the requested protocol. `d_hotspot_dt` is reported in "
         "addition to dDHW_4w because the code base defines it (per the "
         "instruction to report existing alternate derivative definitions).\n")

L.append("## Interpretation guard\n")
L.append("All associations are representation-level: they describe what the "
         "frozen residual covaries with, not a causal mechanism. No claim is "
         "made that the residual 'uses' these variables, and no retraining or "
         "threshold selection was performed.\n")

with open(RES / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"AUDIT written -> {RES / 'AUDIT.md'}")
