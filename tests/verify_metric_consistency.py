# -*- coding: utf-8 -*-
"""End-to-end metric consistency check (manuscript/SI vs metrics_v2 vs figures).

Run from the repository root:
    python tests/verify_metric_consistency.py
Exit code 0 = all checks passed.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc"
M2 = json.load(open(ROOT / "results/qc/calibration_target_audit/metrics_v2.json",
                    encoding="utf-8"))
MV = {(r["domain"], r["model"]): r for r in M2 if r["source"] == "structural_ablation"}
XM = {(r["domain"], r["model"]): r for r in M2 if r["source"] == "xgb_transfer_control"}

ok, bad = 0, []

# ---- 1. SI Table 3(b) vs metrics_v2 --------------------------------------
si = (DOC / "supplementary_information.md").read_text(encoding="utf-8")
block = si[si.index("**(b) Structural-ablation point estimates"):si.index("**(c) Paired")]
name = {"PersA": "A_PersA", "Ungated residual": "B_ungated",
        "Anchored gated residual": "C_anchored", "Frozen TSCF": "D_TSCF"}
for line in block.splitlines():
    if not line.startswith("| ") or "Domain" in line or "---" in line:
        continue
    c = [x.strip() for x in line.strip("|").split("|")]
    dom, lab = c[0], c[1]
    v = MV[(dom, name[lab])]
    checks = [("mse", float(c[5]), v["mse_cont"], 5e-5)]
    if v["probability_calibration_valid"]:
        checks += [("brier", float(c[6]), v["brier_binary"], 5e-5),
                   ("ece", float(c[7]), v["ece_probability"], 5e-4)]
    else:
        checks.append(("srel", float(c[8]), v["score_reliability"], 5e-4))
    for tag, got, want, tol in checks:
        if abs(got - want) <= tol:
            ok += 1
        else:
            bad.append(f"Table3b {dom}/{lab} {tag}: {got} vs {want}")

# ---- 2. figure source data vs metrics_v2 --------------------------------
f6 = pd.read_csv(ROOT / "results/figures/source_data/fig6_source_data_ablation.csv")
f6n = {"PersA": "A_PersA", "ungated residual": "B_ungated",
       "gated residual": "C_anchored", "frozen TSCF": "D_TSCF"}
for _, r in f6.iterrows():
    v = MV[(r["domain"], f6n[r["model"]])]
    ok, bad = (ok + 1, bad) if abs(r["mse_cont"] - v["mse_cont"]) <= 5e-6 else \
        (ok, bad + [f"fig6 {r['domain']}/{r['model']} mse"])

f5 = pd.read_csv(ROOT / "results/figures/source_data/fig5_source_data.csv")
f5n = {"PersA": "A_PersA", "un-gated\nresidual": "B_ungated",
       "gated\nresidual": "C_anchored", "TSCF": "D_TSCF"}
for _, r in f5[f5.metric == "mse_cont"].iterrows():
    v = MV[(r["domain"], f5n[r["model"]])]
    ok, bad = (ok + 1, bad) if abs(r["value"] - v["mse_cont"]) <= 5e-6 else \
        (ok, bad + [f"fig5 {r['domain']}/{r['model']} mse"])

sx = pd.read_csv(ROOT / "results/figures/source_data/SX_source_data.csv")
for _, r in sx[sx.panel == "a/b"].iterrows():
    v = XM[(r["domain"], r["model"])]
    ok, bad = (ok + 1, bad) if abs(r["mse_cont"] - v["mse_cont"]) <= 5e-6 else \
        (ok, bad + [f"SX {r['domain']}/{r['model']} mse"])

# ---- 3. banned phrasing --------------------------------------------------
mm = (DOC / "main_manuscript.md").read_text(encoding="utf-8")
banned = ["Brier is computed against the continuous target",
          "probability-style metrics",
          "the un-gated residual breaks calibration",
          "calibration safety",
          "CARIB reliability (calibration recovery)"]
for s in banned:
    if s.lower() in mm.lower() or s.lower() in si.lower():
        bad.append(f"banned phrasing present: {s}")
    else:
        ok += 1

# every "Brier" mention in the main text must be qualified
for m in re.finditer(r"[^.\n]*Brier[^.\n]*", mm):
    seg = m.group(0)
    low = seg.lower()
    if any(k in low for k in ("severe", "binary", "not a brier")):
        ok += 1
    else:
        bad.append(f"unqualified Brier: {seg[:90]}")

print(f"metric consistency: {ok} passed, {len(bad)} failed")
for b in bad:
    print("  FAIL", b)
sys.exit(1 if bad else 0)
