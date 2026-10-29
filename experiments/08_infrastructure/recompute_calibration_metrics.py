# -*- coding: utf-8 -*-
"""Recompute all calibration/error metrics under the canonical semantics.

Reads the stored frozen predictions (no retraining) and writes a NEW,
explicitly-typed result file.  The original result files are left untouched.

  results/qc/calibration_target_audit/metrics_v2.json
  results/qc/calibration_target_audit/metrics_v2.csv
  results/qc/calibration_target_audit/metrics_v2_bins.json

Run:
    python experiments/08_infrastructure/recompute_calibration_metrics.py
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metric_semantics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/qc/calibration_target_audit"
SAB_NPZ = ROOT / "results/physics_anchored/structural_ablation/rec_structural_ablation.npz"
XGB_NPZ = ROOT / "results/xgb_transfer_control/predictions.npz"

DOMAINS = ("SCS", "GBR", "CARIB")
SAB_MODELS = ("A_PersA", "B_ungated", "C_anchored", "D_TSCF")
XGB_MODELS = ("XGB_FULL", "XGB_THERMAL", "XGB_NO_THERMAL")

rows, bins_all = [], {}
z = np.load(SAB_NPZ, allow_pickle=True)
for dom in DOMAINS:
    y = z[f"{dom}_target"].astype(float)
    for model in SAB_MODELS:
        pred = z[f"{dom}_{model}_pred"].astype(float)
        m = M.finite_mask(pred, y)
        s = M.summarize(pred[m], y[m], model, dom)
        s["source"] = "structural_ablation"
        rows.append(s)
        bins_all[f"{dom}/{model}"] = s.pop("reliability_bins")

x = np.load(XGB_NPZ, allow_pickle=True)
for dom in DOMAINS:
    for model in XGB_MODELS:
        kp, kt = f"{model}_{dom}_pred", f"{model}_{dom}_target"
        if kp not in x:
            continue
        pred = x[kp].astype(float)
        y = x[kt].astype(float)
        m = M.finite_mask(pred, y)
        s = M.summarize(pred[m], y[m], model, dom)
        s["source"] = "xgb_transfer_control"
        rows.append(s)
        bins_all[f"{dom}/{model}"] = s.pop("reliability_bins")

# ---- integrity checks against the audited values ----
ref = json.load(open(ROOT / "results/qc/probability_metrics_audit/"
                     "structural_ablation_corrected.json", encoding="utf-8"))
for r in rows:
    if r["source"] != "structural_ablation":
        continue
    d, m = r["domain"], r["model"]
    assert abs(r["mse_cont"] - ref[d][m]["brier_cont_target"]) < 5e-6, (d, m, "mse")
    if M.MODEL_SPEC[m]["is_probability"]:
        # the reference JSON stores ECE rounded to 4 dp
        assert abs(r["ece_probability"] - ref[d][m]["ece_binary_sev"]) < 1e-4, (d, m, "ece")
    else:
        # the reference JSON stores ECE rounded to 4 dp
        assert abs(r["score_reliability"] - ref[d][m]["ece_binary_sev"]) < 1e-4, (d, m, "srel")

OUT.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(OUT / "metrics_v2.json", "w", encoding="utf-8"), indent=1)
json.dump(bins_all, open(OUT / "metrics_v2_bins.json", "w", encoding="utf-8"), indent=1)

fields = []
for r in rows:
    for k in r:
        if k not in fields:
            fields.append(k)
with open(OUT / "metrics_v2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"{'domain':6s} {'model':17s} {'task':14s} {'MSEcont':>8s} {'BrierBin':>9s} "
      f"{'ECEprob':>8s} {'ScoreRel':>9s}")
for r in rows:
    print(f"{r['domain']:6s} {r['model']:17s} {r['task']:14s} "
          f"{r['mse_cont']:8.5f} "
          f"{r.get('brier_binary', r.get('brier_binary_descriptive', float('nan'))):9.5f} "
          f"{r.get('ece_probability', float('nan')):8.4f} "
          f"{r.get('score_reliability', float('nan')):9.4f}")
print(f"\n{len(rows)} rows -> {OUT/'metrics_v2.json'}")
print("integrity checks passed")
