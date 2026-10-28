#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute the structural-ablation metrics under the CORRECTED
probability-metric convention (2026-10-08 audit).

Corrected convention (why):
  The original compute_structural_ablation_analysis.py computed Brier on raw
  predictions against the continuous target, and ECE over 10 equal-width bins
  spanning [0, 1) with `pred < hi` -- so predictions >= 1.0 (and < 0) fell in
  no bin and were silently excluded with zero penalty. That exclusion applied
  asymmetrically: only B_ungated (0.7-5% of predictions) and C_anchored
  (0.3-1.5%) have out-of-range outputs; A_PersA / D_TSCF never do. PersA == 1.0
  predictions were excluded by the same boundary. Additionally,
  nc_style.reliability_curve (fig 5d) used a different convention (clamp bin
  index, include out-of-range). The paper now uses the corrected convention
  below for every model identically.

Corrected convention:
  1. ALL model predictions are clipped to [0, 1] before Brier/ECE
     (the target coordinate lives in [0, 1]; ROC/PR/CSI/POD/FAR/F1 are
     clipping-invariant and unchanged).
  2. Brier = mean((clip(pred) - target)^2) against the CONTINUOUS target
     coordinate (matches the training objective).
  3. ECE = sum_b (n_b/N) * |mean(sev_b) - mean(pred_b)| with
     sev = target >= 0.8 and 10 equal-width bins, the LAST bin including 1.0
     (no silent exclusion).

Output: structural_ablation_corrected.json (used by fig6 and the SI/main
text numbers). Run from anywhere:
    python recompute_corrected_metrics.py
"""
import io, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
NPZ = HERE.parents[1] / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"

MODELS = ["A_PersA", "B_ungated", "C_anchored", "D_TSCF"]
SEV, NBINS = 0.8, 10


def ece_bin(pred, target, nbins=NBINS):
    """pred must be within [0, 1]. Last bin includes 1.0."""
    sev = (target >= SEV).astype(float)
    n = len(pred)
    e = 0.0
    for b in range(nbins):
        lo, hi = b / nbins, (b + 1) / nbins
        msk = (pred >= lo) & ((pred <= hi) if b == nbins - 1 else (pred < hi))
        if msk.sum() == 0:
            continue
        e += (msk.sum() / n) * abs(sev[msk].mean() - pred[msk].mean())
    return float(e)


def main():
    z = np.load(NPZ)
    corrected = {}
    for dom in ("SCS", "GBR", "CARIB"):
        target = z[f"{dom}_target"]
        # unified convention: ALL models clipped to [0, 1] before Brier/ECE
        P = {m: np.clip(z[f"{dom}_{m}_pred"], 0.0, 1.0) for m in MODELS}
        corrected[dom] = {}
        for m, p in P.items():
            assert p.min() >= 0 and p.max() <= 1, (dom, m)
            sev = (target >= SEV)
            corrected[dom][m] = {
                "n": int(len(p)), "n_severe": int(sev.sum()),
                "roc_auc": round(float(roc_auc_score(sev, p)), 4),
                "pr_auc": round(float(average_precision_score(sev, p)), 4),
                "brier_cont_target": round(float(np.mean((p - target) ** 2)), 5),
                "ece_binary_sev": round(ece_bin(p, target), 4)}
    out = HERE / "structural_ablation_corrected.json"
    out.write_text(json.dumps(corrected, indent=1), encoding="utf-8")
    print("[saved]", out)
    for dom in corrected:
        for m, v in corrected[dom].items():
            print(f"{dom:6s} {m:12s} ROC {v['roc_auc']:.4f} PR {v['pr_auc']:.4f} "
                  f"Brier {v['brier_cont_target']:.5f} ECE {v['ece_binary_sev']:.4f}")


if __name__ == "__main__":
    main()
