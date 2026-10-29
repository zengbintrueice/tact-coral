# -*- coding: utf-8 -*-
"""Audit (READ-ONLY): recompute Brier/ECE under every plausible definition.

Does NOT modify any result file. Writes only into this audit directory.

For each domain x model:
  cont_MSE_raw   = mean((pred - y)^2)                    (current "Brier", unclipped)
  cont_MSE_clip  = mean((clip(pred,0,1) - y)^2)          (current "Brier", clipped)
  bin_Brier      = mean((clip(pred,0,1) - 1[y>=0.8])^2)  (standard binary Brier)
  ECE_cur        = 10 bins over [0,1), pred<hi, UNCLIPPED (original code)
  ECE_corr       = 10 bins, pred clipped, last bin includes 1.0
  R2_cont, spearman(pred, y)
  logloss_bin    ONLY for genuinely probabilistic outputs (D), as a reference.
"""
import io, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NPZ = ROOT / "results/physics_anchored/structural_ablation/rec_structural_ablation.npz"
SEV = 0.8

z = np.load(NPZ, allow_pickle=True)


def ece(pred, y, clip: bool, last_incl: bool):
    p = np.clip(pred, 0, 1) if clip else np.asarray(pred, float)
    sev = (y >= SEV).astype(float)
    n = len(p)
    total = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (p >= lo) & (p <= hi) if (last_incl and b == 9) else (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(sev[m].mean() - p[m].mean())
    return total


rows = []
for dom in ("SCS", "GBR", "CARIB"):
    y = z[f"{dom}_target"].astype(float)
    sev = (y >= SEV)
    for model in ("A_PersA", "B_ungated", "C_anchored", "D_TSCF"):
        pred = z[f"{dom}_{model}_pred"].astype(float)
        pc = np.clip(pred, 0, 1)
        ss_res = np.sum((pc - y) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        row = dict(
            domain=dom, model=model, n=int(len(y)), n_severe=int(sev.sum()),
            prevalence=float(sev.mean()),
            pred_min=float(pred.min()), pred_max=float(pred.max()),
            frac_out_of_unit=float(np.mean((pred < 0) | (pred > 1))),
            cont_MSE_raw=float(np.mean((pred - y) ** 2)),
            cont_MSE_clip=float(np.mean((pc - y) ** 2)),
            cont_R2=float(1 - ss_res / ss_tot),
            spearman_pred_y=float(spearmanr(pred, y).statistic),
            roc_sev=float(roc_auc_score(sev, pred)),
            bin_Brier=float(np.mean((pc - sev.astype(float)) ** 2)),
            ECE_current=float(ece(pred, y, clip=False, last_incl=False)),
            ECE_corrected=float(ece(pc, y, clip=True, last_incl=True)),
        )
        # binary log loss only where the output is a genuine probability
        if model == "D_TSCF":
            q = np.clip(pred, 1e-6, 1 - 1e-6)
            row["logloss_bin"] = float(-np.mean(
                sev * np.log(q) + (~sev) * np.log(1 - q)))
        rows.append(row)

import csv
fields = []
for r in rows:
    for k in r:
        if k not in fields:
            fields.append(k)
with open(OUT / "audit_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"{'domain':6s} {'model':11s} {'n':>7s} {'sev':>5s} "
      f"{'MSEraw':>8s} {'MSEclip':>8s} {'binBrier':>9s} {'ECEcur':>7s} {'ECEcorr':>8s}")
for r in rows:
    print(f"{r['domain']:6s} {r['model']:11s} {r['n']:7d} {r['n_severe']:5d} "
          f"{r['cont_MSE_raw']:8.5f} {r['cont_MSE_clip']:8.5f} {r['bin_Brier']:9.5f} "
          f"{r['ECE_current']:7.4f} {r['ECE_corrected']:8.4f}")

json.dump(rows, open(OUT / "audit_metrics.json", "w", encoding="utf-8"), indent=1)
print("\nwrote", OUT / "audit_metrics.csv")
