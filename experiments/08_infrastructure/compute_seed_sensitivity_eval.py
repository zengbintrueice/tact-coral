#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary Fig. S8 — multi-seed robustness of the formal TSCF.

Re-evaluates the FORMAL TSCF pipeline trained independently with
seeds = [0, 1, 2, 3, 42] on three domains:

  SCS formal test 2016-2019, GBR zero-shot 2017-2020, CARIB zero-shot 2017-2020.

Protocol frozen (identical to the formal deliverables):
  - causal v2, lead 4 wk, history 24 wk, severe y>=0.8, tau=0.90 pinned
  - SCS-train-only normalization (baked in checkpoints)
  - ocean-masked PFV record set (finite y(t+4) & ~land_mask & finite prediction)
  - reported under BOTH the standard protocol and the temporal purge
    (issue-time purge 23 wk == verification-time 27 wk; KEEP iff t-(H-1) >= b,
     b = first test-period week) — the purge is a population filter and does
    not interact with the seed.

No model/feature/split/hyperparameter/normalization changes; only the random
seed varies. Seed 42 is the formal configuration; NO seed is selected based on
results (all seeds reported).

Inputs : results/seed_sensitivity/seed_{N}/model.pt
         (trained by experiments/01_training/train_severe_classifier.py
          --seed N --out-dir results/seed_sensitivity/seed_{N}  — the FORMAL
          D1 pipeline: FinalSingleHeadModel, weighted BCE pos_weight=1000)
Outputs: results/seed_sensitivity/seed_metrics.csv
         results/seed_sensitivity/seed_transfer_boundary.csv
         results/seed_sensitivity/seed_sensitivity_summary.json

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_seed_sensitivity_eval.py
"""
import io, sys, os, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
# do NOT pre-wrap stdout — ablate_a_dh_stopgrad re-wraps it on import

import numpy as np
import pandas as pd
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

SEEDS = [0, 1, 2, 3, 42]
LEAD = 4
HISTORY = 24
SEV = 0.8
TAU = 0.90
OUT_DIR = RESULTS_DIR / "seed_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    {"name": "SCS", "file": "scs_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2011, 2015),
                   test_years=(2016, 2019)), "test_years": (2016, 2019)},
    {"name": "GBR", "file": "gbr_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2014, 2016),
                   test_years=(2017, 2020)), "test_years": (2017, 2020)},
    {"name": "CARIB", "file": "caribbean_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2014, 2016),
                   test_years=(2017, 2020)), "test_years": (2017, 2020)},
]


def restore(path):
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def predict_sliding(model, ds, device):
    """Verbatim PFV sliding-window inference (formal TSCF convention)."""
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n_dates, C = ds.n_dates, ds.C_dyn
    fm = np.full((n_dates, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    bs = 16
    with torch.no_grad():
        for b in range(0, n_dates - T + 1, bs):
            be = min(b + bs, n_dates - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1:te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            for k in range(be - b):
                fm[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return fm


def collect(ds, fm, test_years, purged=False):
    """Ocean-masked PFV record set; optional temporal purge
    (KEEP iff t-(H-1) >= b, b = first test-period week)."""
    T = ds.history_weeks
    n = ds.n_dates
    y = ds.y_regression_full
    ocean = ~ds.land_mask
    b_idx = None
    if purged:
        b_str = next(s for s in ds.date_strings
                     if int(s[:4]) == test_years[0])
        b_idx = ds.date_strings.index(b_str)
    pred, tgt, tks = [], [], []
    for t in range(T - 1, n):
        yy = int(ds.date_strings[t][:4])
        if not (test_years[0] <= yy <= test_years[1]):
            continue
        if purged and (t - (T - 1) < b_idx):
            continue
        tt = t + LEAD
        if tt >= n:
            continue
        v = ~np.isnan(y[tt]) & ocean
        if v.sum() == 0:
            continue
        v &= ~np.isnan(fm[t])
        if v.sum() == 0:
            continue
        pred.append(fm[t][v])
        tgt.append(y[tt][v])
        tks.append(np.full(int(v.sum()), t, np.int32))
    return (np.concatenate(pred) if pred else np.array([]),
            np.concatenate(tgt) if tgt else np.array([]),
            np.concatenate(tks) if tks else np.array([], dtype=np.int32))


def metrics(pred, target):
    from sklearn.metrics import roc_auc_score, average_precision_score
    sev = (target >= SEV)
    out = {"n": int(len(pred)), "n_severe": int(sev.sum())}
    if len(np.unique(sev)) < 2:
        out.update({"ROC": np.nan, "PR": np.nan})
    else:
        out["ROC"] = float(roc_auc_score(sev, pred))
        out["PR"] = float(average_precision_score(sev, pred))
    pb = pred >= TAU
    hits = int(np.sum(pb & sev)); miss = int(np.sum(~pb & sev))
    fa = int(np.sum(pb & ~sev))
    out["CSI"] = hits / (hits + miss + fa) if (hits + miss + fa) > 0 else np.nan
    out["POD"] = hits / (hits + miss) if (hits + miss) > 0 else np.nan
    out["FAR"] = fa / (hits + fa) if (hits + fa) > 0 else np.nan
    out["F1"] = 2 * hits / (2 * hits + fa + miss) if (2 * hits + fa + miss) > 0 else np.nan
    out["Brier"] = float(np.mean((pred - target) ** 2))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        msk = (pred >= lo) & (pred < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / len(pred)) * abs(sev[msk].mean() - pred[msk].mean())
    out["ECE"] = float(ece)
    return out


def md5(p):
    import hashlib
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    rows, boundary, meta = [], [], {"seeds": {}, "generated_utc":
                                    datetime.now().isoformat()}
    datasets = {}
    for dom in DOMAINS:
        ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                             lead_weeks=LEAD, dataset_file=dom["file"],
                             year_based_split=True, **dom["split"])
        datasets[dom["name"]] = ds

    ok = 0
    for seed in SEEDS:
        ckpt = OUT_DIR / f"seed_{seed}" / "model.pt"
        if not ckpt.exists():
            print(f"[MISSING] seed {seed}: {ckpt}", flush=True)
            continue
        model = restore(ckpt).to(device)
        meta["seeds"][str(seed)] = {"checkpoint_md5": md5(ckpt), "domains": {}}
        for dom in DOMAINS:
            name = dom["name"]
            ds = datasets[name]
            fm = predict_sliding(model, ds, device)
            for prot, purge in [("standard", False), ("purged", True)]:
                p, t, _ = collect(ds, fm, dom["test_years"], purged=purge)
                r = metrics(p, t)
                rows.append({"seed": seed, "domain": name, "protocol": prot, **r})
                if prot == "standard":
                    meta["seeds"][str(seed)]["domains"][name] = r
            print(f"  seed {seed} {name}: "
                  f"std ROC={metrics(*collect(ds, fm, dom['test_years'])[:2])['ROC']:.4f}",
                  flush=True)
        ok += 1
        del model
        torch.cuda.empty_cache()

    # formal checkpoint reference (seed-42 pipeline reproduction check)
    formal = restore(A6_WEIGHT).to(device)
    ref = {}
    for dom in DOMAINS:
        name = dom["name"]
        ds = datasets[name]
        fm = predict_sliding(formal, ds, device)
        p, t, _ = collect(ds, fm, dom["test_years"], purged=False)
        ref[name] = metrics(p, t)
        print(f"  [formal ckpt ref] {name}: ROC={ref[name]['ROC']:.4f} "
              f"PR={ref[name]['PR']:.4f}", flush=True)
    meta["formal_checkpoint_reference"] = {
        "md5": md5(A6_WEIGHT), "domains": ref}
    del formal
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "seed_metrics.csv", index=False)
    print(f"[saved] seed_metrics.csv ({len(df)} rows)", flush=True)

    std = df[df.protocol == "standard"]
    for seed in SEEDS:
        g = std[(std.seed == seed) & (std.domain == "GBR")]
        c = std[(std.seed == seed) & (std.domain == "CARIB")]
        if len(g) and len(c):
            boundary.append({
                "seed": seed,
                "PR_GBR": float(g.PR.iloc[0]), "PR_CARIB": float(c.PR.iloc[0]),
                "dPR": float(g.PR.iloc[0] - c.PR.iloc[0]),
                "CSI_GBR": float(g.CSI.iloc[0]), "CSI_CARIB": float(c.CSI.iloc[0]),
                "dCSI": float(g.CSI.iloc[0] - c.CSI.iloc[0]),
            })
    pd.DataFrame(boundary).to_csv(OUT_DIR / "seed_transfer_boundary.csv", index=False)
    meta["n_seeds_ok"] = ok
    meta["n_seeds_expected"] = len(SEEDS)
    with open(OUT_DIR / "seed_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[saved] seed_transfer_boundary.csv", flush=True)
    print(f"[saved] seed_sensitivity_summary.json", flush=True)

    # ------------------------------------------------------------ QC summary --
    print("\n" + "=" * 60)
    print("SEED SENSITIVITY QC SUMMARY")
    print("=" * 60)
    print(f"Seeds completed: {ok}/{len(SEEDS)}  "
          f"({'ALL OK' if ok == len(SEEDS) else 'INCOMPLETE — CHECK FAILURES'})")
    std_all = df[df.protocol == "standard"]
    for d in ["SCS", "GBR", "CARIB"]:
        sub = std_all[std_all.domain == d]
        if len(sub):
            print(f"  {d:5s} ROC {sub.ROC.min():.4f}–{sub.ROC.max():.4f}  "
                  f"PR {sub.PR.min():.4f}–{sub.PR.max():.4f}  "
                  f"CSI {sub.CSI.min():.4f}–{sub.CSI.max():.4f}  (n={int(sub.n.iloc[0])})")
    if boundary:
        bd = pd.DataFrame(boundary)
        print(f"  transfer boundary ΔPR  GBR−CARIB: "
              f"{bd.dPR.min():.4f}–{bd.dPR.max():.4f} (all >0: "
              f"{bool((bd.dPR > 0).all())})")
        print(f"  transfer boundary ΔCSI GBR−CARIB: "
              f"{bd.dCSI.min():.4f}–{bd.dCSI.max():.4f} (all >0: "
              f"{bool((bd.dCSI > 0).all())})")
    s42 = std_all[(std_all.seed == 42)]
    if len(s42):
        diffs = {d: abs(float(s42[s42.domain == d].ROC.iloc[0])
                        - ref[d]["ROC"]) for d in ref}
        print(f"  seed-42 rerun vs formal checkpoint |ΔROC|: "
              + ", ".join(f"{d}={v:.4f}" for d, v in diffs.items()))
    print("Terminology: thermal-stress-aware; no seed selected on results.")
    print("DONE")


if __name__ == "__main__":
    main()
