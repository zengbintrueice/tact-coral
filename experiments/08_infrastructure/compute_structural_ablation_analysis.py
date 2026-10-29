#!/usr/bin/env python3
# LEGACY METRIC NAMING — see experiments/08_infrastructure/metric_semantics.py.
# The `brier` field written by this script is mean((pred - y_continuous)**2),
# i.e. a CONTINUOUS MSE, not the binary Brier score. It is kept only for
# backward compatibility. Canonical fields (mse_cont, score_reliability,
# brier_binary, ece_probability) live in
# results/qc/calibration_target_audit/metrics_v2.json. Do not use `brier`
# for new tables, figures or manuscript text.
# -*- coding: utf-8 -*-
"""PHASE 2 — structural ablation analysis (reads rec_structural_ablation.npz).

Metrics per model per domain: ROC-AUC, PR-AUC, CSI@0.9, POD, FAR, F1, Brier, ECE.

NOTE (2026-10-08 audit): this script's ECE bins span [0, 1) with
`pred < hi`, so predictions >= 1.0 (B_ungated 0.7-5%, C_anchored 0.3-1.5%)
and PersA == 1.0 are silently excluded, and Brier is computed against the
CONTINUOUS target. The paper and SI now use the CORRECTED uniform-clip
convention instead (all models clipped to [0, 1]; ECE last bin includes
1.0) -- see results/qc/probability_metrics_audit/{recompute_corrected_metrics.py,
structural_ablation_corrected.json, AUDIT_PROBABILITY_METRICS.md}. This
script is preserved unchanged as the original evidence-locked record.
Paired week-block bootstrap (1000, seed 42), ONE shared pick matrix for all four
models. For each bootstrap replicate the 8-metric vector is computed once per
model on the SAME resampled weeks; differences (B−A, C−A, C−B, D−A, D−C) are
then vector subtractions of paired replicates. AUC/AP use fast rank-based numpy
(ties: average ranks) inside the bootstrap; point estimates use sklearn so they
match the other deliverables exactly.

Outputs -> results/physics_anchored/structural_ablation/:
  structural_ablation.json / .csv / .md

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_structural_ablation_analysis.py
"""
import io, sys, json, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
# this script does NOT import ablate_a_dh_stopgrad, so wrapping stdout is safe
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import rankdata

LTTS_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = LTTS_ROOT / "results" / "physics_anchored" / "structural_ablation"

SEV = 0.8
TAU = 0.90
N_BOOT = 1000
BOOT_SEED = 42
MODELS = ["A_PersA", "B_ungated", "C_anchored", "D_TSCF"]
DIFFS = [("B_ungated", "A_PersA"), ("C_anchored", "A_PersA"),
         ("C_anchored", "B_ungated"), ("D_TSCF", "A_PersA"),
         ("D_TSCF", "C_anchored")]
METRIC_KEYS = ["roc_auc", "pr_auc", "csi_tau09", "pod_tau09", "far_tau09",
               "f1_tau09", "brier", "ece"]


def fast_auc(y, score):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[y.astype(bool)].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def fast_ap(y, score):
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    sy = y[order]
    tp = np.cumsum(sy)
    prec = tp / np.arange(1, len(y) + 1)
    return float(prec[sy.astype(bool)].mean())


def metric_vec_fast(pred, target):
    """8-metric vector with fast AUC/AP (bootstrap loop)."""
    sev = (target >= SEV)
    n = len(pred)
    out = np.empty(8)
    out[0] = fast_auc(sev, pred)
    out[1] = fast_ap(sev, pred)
    pb = pred >= TAU
    ab = sev
    hits = int(np.sum(pb & ab)); miss = int(np.sum(~pb & ab)); fa = int(np.sum(pb & ~ab))
    den = hits + miss + fa
    out[2] = hits / den if den > 0 else np.nan
    out[3] = hits / (hits + miss) if (hits + miss) > 0 else np.nan
    out[4] = fa / (hits + fa) if (hits + fa) > 0 else np.nan
    out[5] = 2 * hits / (2 * hits + fa + miss) if (2 * hits + fa + miss) > 0 else np.nan
    out[6] = float(np.mean((pred - target) ** 2))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        msk = (pred >= lo) & (pred < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / n) * abs(sev[msk].mean() - pred[msk].mean())
    out[7] = ece
    return out


def metrics_sklearn(pred, target):
    """Point metrics with sklearn (match other deliverables exactly)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    sev = (target >= SEV)
    out = {"n": int(len(pred)), "n_severe": int(sev.sum()),
           "prevalence": float(sev.mean())}
    if len(np.unique(sev)) < 2:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan})
    else:
        out["roc_auc"] = float(roc_auc_score(sev, pred))
        out["pr_auc"] = float(average_precision_score(sev, pred))
    pb = pred >= TAU
    ab = sev
    hits = int(np.sum(pb & ab)); miss = int(np.sum(~pb & ab)); fa = int(np.sum(pb & ~ab))
    out["csi_tau09"] = hits / (hits + miss + fa) if (hits + miss + fa) > 0 else np.nan
    out["pod_tau09"] = hits / (hits + miss) if (hits + miss) > 0 else np.nan
    out["far_tau09"] = fa / (hits + fa) if (hits + fa) > 0 else np.nan
    out["f1_tau09"] = 2 * hits / (2 * hits + fa + miss) if (2 * hits + fa + miss) > 0 else np.nan
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


def _py(x):
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    npz_path = OUT_DIR / "rec_structural_ablation.npz"
    meta = json.load(open(OUT_DIR / "infer_meta.json", encoding="utf-8"))
    z = np.load(npz_path)
    domains = list(meta["domains"].keys())

    results = {"protocol": meta,
               "generated_utc": datetime.now().isoformat(),
               "domains": {}}
    md = ["# Structural Ablation — Thermal-Stress-Anchored Residual Learning", "",
          f"Generated: {results['generated_utc']}", "",
          "Models (same ocean-masked PFV record set per domain; τ=0.90 pinned):",
          "- **A** PersA_DHW: `y_hat = PersA(t)`",
          "- **B** Un-gated bounded residual: `PersA + tanh(R(attn))·scale` (gate≡1)",
          "- **C** Formal anchored residual: `PersA + sigmoid((y_main−τ)/T)·tanh(R(attn))·scale`",
          "- **D** Frozen TSCF (no anchor)",
          f"- scale = {meta['frozen_config']['scale']} (frozen, Work 1 SCS-val); "
          f"bootstrap {N_BOOT} seed {BOOT_SEED}; paired week-block, one shared pick matrix", ""]

    rows = []
    for name in domains:
        dom_meta = meta["domains"][name]
        target = z[f"{name}_target"]
        print("=" * 60, flush=True)
        print(f"Domain {name}: n={dom_meta['n']} sev={dom_meta['n_severe']}", flush=True)
        dom = {"n": int(dom_meta["n"]), "n_severe": int(dom_meta["n_severe"]),
               "prevalence": float(dom_meta["prevalence"])}
        md.append(f"## {name} (n={dom_meta['n']}, severe={dom_meta['n_severe']}, "
                  f"prev={dom_meta['prevalence']:.4f})")
        md.append("| Model | ROC | PR | CSI@0.9 | POD | FAR | F1 | Brier | ECE |")
        md.append("|---|---|---|---|---|---|---|---|---|")

        t_idx = z[f"{name}_A_PersA_t"]
        weeks = np.unique(t_idx)
        rng = np.random.default_rng(BOOT_SEED)
        picks = weeks[rng.integers(0, len(weeks), size=(N_BOOT, len(weeks)))]
        preds = {m: z[f"{name}_{m}_pred"] for m in MODELS}

        # point metrics (sklearn)
        dom["models"] = {}
        for m in MODELS:
            r = metrics_sklearn(preds[m], target)
            dom["models"][m] = r
            print(f"  {m:<12} ROC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} "
                  f"CSI={r['csi_tau09']:.4f} Brier={r['brier']:.5f} ECE={r['ece']:.4f}", flush=True)
            rows.append({"domain": name, "model": m, **{k: r[k] for k in METRIC_KEYS}})
            md.append(f"| {m} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['csi_tau09']:.4f} | "
                      f"{r['pod_tau09']:.4f} | {r['far_tau09']:.4f} | {r['f1_tau09']:.4f} | "
                      f"{r['brier']:.5f} | {r['ece']:.4f} |")

        # bootstrap: one metric vector per model per replicate (paired)
        M = np.full((N_BOOT, len(MODELS), len(METRIC_KEYS)), np.nan)
        for b in range(N_BOOT):
            msk = np.isin(t_idx, picks[b])
            if msk.sum() < 10:
                continue
            for mi, m in enumerate(MODELS):
                M[b, mi] = metric_vec_fast(preds[m][msk], target[msk])

        # per-model bootstrap CIs
        for mi, m in enumerate(MODELS):
            dom["models"][m]["bootstrap"] = {}
            for ki, k in enumerate(METRIC_KEYS):
                v = M[:, mi, ki]
                v = v[np.isfinite(v)]
                dom["models"][m]["bootstrap"][k] = (
                    [np.nan, np.nan] if len(v) == 0 else
                    [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

        # differences: paired vector subtraction
        md.append("| Difference | ΔROC | ΔPR | ΔCSI@0.9 | ΔPOD | ΔFAR | ΔF1 | ΔBrier | ΔECE |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        dom["diffs"] = {}
        for xname, yname in DIFFS:
            xi, yi = MODELS.index(xname), MODELS.index(yname)
            d = {}
            for ki, k in enumerate(METRIC_KEYS):
                point = float(metrics_sklearn(preds[xname], target)[k]
                              - metrics_sklearn(preds[yname], target)[k])
                dv = M[:, xi, ki] - M[:, yi, ki]
                dv = dv[np.isfinite(dv)]
                ci = ([np.nan, np.nan] if len(dv) == 0 else
                      [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))])
                d[k] = {"point": point, "ci": ci}
            dom["diffs"][f"{xname}−{yname}"] = d
            label = f"{xname}−{yname}"
            print(f"  {label:<22} ΔROC={d['roc_auc']['point']:+.4f} "
                  f"[{d['roc_auc']['ci'][0]:+.4f},{d['roc_auc']['ci'][1]:+.4f}] "
                  f"ΔPR={d['pr_auc']['point']:+.4f} "
                  f"ΔCSI={d['csi_tau09']['point']:+.4f}", flush=True)
            md.append(f"| {label} | {d['roc_auc']['point']:+.4f} "
                      f"({d['roc_auc']['ci'][0]:+.4f},{d['roc_auc']['ci'][1]:+.4f}) | "
                      f"{d['pr_auc']['point']:+.4f} | {d['csi_tau09']['point']:+.4f} | "
                      f"{d['pod_tau09']['point']:+.4f} | {d['far_tau09']['point']:+.4f} | "
                      f"{d['f1_tau09']['point']:+.4f} | {d['brier']['point']:+.5f} | "
                      f"{d['ece']['point']:+.4f} |")
            rows.append({"domain": name, "model": label,
                         **{k: d[k]["point"] for k in METRIC_KEYS}})
        results["domains"][name] = dom
        md.append("")

    md += ["Terminology: thermal-stress-aware; strong physics-based persistence baseline "
           "(PersA) is the anchor; cumulative HotSpot channels are legitimate CRW-DHW predictors.",
           "No hyperparameters were selected on GBR/CARIB; scale/tau/temp frozen from SCS-val "
           "(Work 1). Bootstrap uses fast rank-based AUC/AP inside replicates; point estimates "
           "use sklearn.", ""]
    (OUT_DIR / "structural_ablation.md").write_text("\n".join(md), encoding="utf-8")
    with open(OUT_DIR / "structural_ablation.json", "w", encoding="utf-8") as f:
        json.dump(_py(results), f, indent=2, ensure_ascii=False)
    import pandas as pd
    try:
        pd.DataFrame(rows).to_csv(OUT_DIR / "structural_ablation.csv", index=False)
        print("[saved] structural_ablation.csv", flush=True)
    except PermissionError:
        alt = OUT_DIR / "structural_ablation.csv.new"
        pd.DataFrame(rows).to_csv(alt, index=False)
        print(f"[warn] csv locked -> wrote {alt.name}", flush=True)
    print("[saved] structural_ablation.json", flush=True)
    print("[saved] structural_ablation.md", flush=True)
    print("PHASE 2 DONE", flush=True)


if __name__ == "__main__":
    main()
