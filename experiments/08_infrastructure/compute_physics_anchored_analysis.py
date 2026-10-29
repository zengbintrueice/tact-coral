#!/usr/bin/env python3
# LEGACY METRIC NAMING — see experiments/08_infrastructure/metric_semantics.py.
# The `brier` field written by this script is mean((pred - y_continuous)**2),
# i.e. a CONTINUOUS MSE, not the binary Brier score. It is kept only for
# backward compatibility. Canonical fields (mse_cont, score_reliability,
# brier_binary, ece_probability) live in
# results/qc/calibration_target_audit/metrics_v2.json. Do not use `brier`
# for new tables, figures or manuscript text.
# -*- coding: utf-8 -*-
"""PHASE 2 — analysis of the physics-anchored residual head record sets.

Reads results/physics_anchored/physics_anchored_records.npz (produced by
compute_physics_anchored_infer.py in a separate process — this script never
opens netCDF, so the netCDF4/mmap cleanup segfault cannot occur here).

Computes per-domain per-method metrics (ROC/PR/CSI@0.9/POD/FAR/F1/Brier/ECE)
and paired week-block bootstrap 95% CIs (1000, seed 42), writes:
  physics_anchored_eval.json / .csv / .md

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_physics_anchored_analysis.py
"""
import io, sys, os, json, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
import numpy as np

LTTS_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = LTTS_ROOT / "results" / "physics_anchored"

SEV = 0.8
TAU = 0.90
N_BOOT = 1000
BOOT_SEED = 42
METHODS = ["PhysicsAnchored", "PersA_DHW", "TSCF_frozen"]


def classification_counts(pred, target):
    pb = pred >= TAU
    ab = target >= SEV
    hits = int(np.sum(pb & ab))
    miss = int(np.sum(~pb & ab))
    fa = int(np.sum(pb & ~ab))
    csi = hits / (hits + miss + fa) if (hits + miss + fa) > 0 else np.nan
    pod = hits / (hits + miss) if (hits + miss) > 0 else np.nan
    far = fa / (hits + fa) if (hits + fa) > 0 else np.nan
    f1 = 2 * hits / (2 * hits + fa + miss) if (2 * hits + fa + miss) > 0 else np.nan
    return csi, pod, far, f1


def metrics(pred, target):
    from sklearn.metrics import roc_auc_score, average_precision_score
    sev = (target >= SEV)
    out = {"n": int(len(pred)), "n_severe": int(sev.sum()),
           "prevalence": float(sev.mean())}
    if len(np.unique(sev)) < 2:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan})
    else:
        out["roc_auc"] = float(roc_auc_score(sev, pred))
        out["pr_auc"] = float(average_precision_score(sev, pred))
    csi, pod, far, f1 = classification_counts(pred, target)
    out.update({"csi_tau09": csi, "pod_tau09": pod, "far_tau09": far, "f1_tau09": f1})
    out["brier"] = float(np.mean((pred - target) ** 2))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        msk = (pred >= lo) & (pred < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / len(pred)) * abs(sev[msk].mean() - pred[msk].mean())
    out["ece"] = float(ece)
    return out


def boot_ci(pred, target, t_idx, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    weeks = np.unique(t_idx)
    picks = weeks[rng.integers(0, len(weeks), size=(n_boot, len(weeks)))]
    keys = list(metrics(pred[:1], target[:1]))
    boot = {k: [] for k in keys}
    for b in range(n_boot):
        m = np.isin(t_idx, picks[b])
        if m.sum() < 10:
            continue
        r = metrics(pred[m], target[m])
        for k in keys:
            boot[k].append(r[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in boot.items()}


def _py(x):
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    npz_path = OUT_DIR / "physics_anchored_records.npz"
    meta_path = OUT_DIR / "physics_anchored_infer_meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} missing — run compute_physics_anchored_infer.py first")
    z = np.load(npz_path)
    meta = json.load(open(meta_path, encoding="utf-8"))
    domains = list(meta["domains"].keys())

    results = {"protocol": meta["protocol"],
               "record_set": meta["record_set"],
               "tau_pinned": TAU, "severe": SEV,
               "bootstrap": {"type": "paired week-block", "n": N_BOOT, "seed": BOOT_SEED},
               "generated_utc": datetime.now().isoformat(),
               "domains": {}}
    rows = []
    md = ["# Physics-anchored residual head — three-domain PFV evaluation (OCEAN-MASKED)", "",
          f"Generated: {results['generated_utc']}", "",
          "Record set per domain: finite y(t+4) & ocean cell (aligned with the training "
          "label_mask; land cells with finite y are excluded). "
          "PersA_DHW = y_regression at issue time t (the physics anchor).",
          "", "| Domain | Method | n | sev | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    for name in domains:
        dom_meta = meta["domains"][name]
        target = z[f"{name}_target"]
        dom_res = {"n": int(len(target)),
                   "n_severe": int((target >= SEV).sum()),
                   "prevalence": float((target >= SEV).mean()),
                   "finite_y_on_land_excluded": int(dom_meta["finite_y_on_land_excluded"])}
        for m in METHODS:
            pred = z[f"{name}_{m}_pred"]
            r = metrics(pred, target)
            r["bootstrap"] = boot_ci(pred, target, z[f"{name}_{m}_t"])
            dom_res[m] = r
            print(f"  [{name} {m}] n={r['n']} sev={r['n_severe']} "
                  f"ROC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} CSI@0.9={r['csi_tau09']:.4f} "
                  f"Brier={r['brier']:.5f} ECE={r['ece']:.4f}", flush=True)
            rows.append({"domain": name, "method": m,
                         **{k: v for k, v in r.items() if k != "bootstrap"}})
            md.append(f"| {name} | {m} | {r['n']} | {r['n_severe']} | "
                      f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['csi_tau09']:.4f} | "
                      f"{r['pod_tau09']:.4f} | {r['far_tau09']:.4f} | {r['f1_tau09']:.4f} | "
                      f"{r['brier']:.5f} | {r['ece']:.4f} |")
        results["domains"][name] = dom_res

    md += ["", "Terminology: thermal-stress-aware; strong physics-based persistence baseline "
              "(PersA) is the anchor; cumulative HotSpot channels are legitimate CRW-DHW "
              "predictors.", ""]

    (OUT_DIR / "physics_anchored_eval.md").write_text("\n".join(md), encoding="utf-8")
    with open(OUT_DIR / "physics_anchored_eval.json", "w", encoding="utf-8") as f:
        json.dump(_py(results), f, indent=2, ensure_ascii=False)
    import pandas as pd
    try:
        pd.DataFrame(rows).to_csv(OUT_DIR / "physics_anchored_eval.csv", index=False)
        print("[saved] physics_anchored_eval.csv", flush=True)
    except PermissionError:
        # CSV often open in Excel/spreadsheet — never let a lock kill the run
        alt = OUT_DIR / "physics_anchored_eval.csv.new"
        pd.DataFrame(rows).to_csv(alt, index=False)
        print(f"[warn] physics_anchored_eval.csv is LOCKED (open in a spreadsheet) — "
              f"wrote {alt.name}; close the file and replace manually", flush=True)
    print(f"[saved] {OUT_DIR / 'physics_anchored_eval.json'}", flush=True)
    print(f"[saved] {OUT_DIR / 'physics_anchored_eval.md'}", flush=True)
    print("PHASE 2 DONE", flush=True)


if __name__ == "__main__":
    main()
