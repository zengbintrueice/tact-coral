#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed-sensitivity evaluation of the anchored residual (TSAR, model C).

For each seed in {0,1,2,3,42}: load results/physics_anchored/model_seed_{N}.pt
(TSAR, scale=0.4), evaluate on the shared ocean-masked PFV record sets
(SCS/GBR/CARIB) and compute paired week-block bootstrap (1000, seed 42)
difference CIs for the key contrasts:

  C−A (anchored residual − PersA)      — does the CARIB gain hold across seeds?
  C−B (anchored residual − ungated)    — does the gating advantage hold?
  C−D (anchored residual − frozen TSCF)

PersA (A) and frozen TSCF (D) are seed-independent; the ungated residual (B)
is the seed-42 model (structural_ablation/model_B.pt). Outputs:
results/physics_anchored/seed_sensitivity_tsar.csv + .json.

No retraining here.
"""
import io, sys, json, warnings, importlib
from pathlib import Path

warnings.filterwarnings("ignore")
# do NOT pre-wrap stdout — ablate re-wraps it
import numpy as np
import pandas as pd
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))
from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel
from structural_ablation_models import UnGatedAnchoredResidualModel

SEEDS = [0, 1, 2, 3, 42]
LEAD = 4
SEV = 0.8
TAU = 0.90
N_BOOT = 1000
BOOT_SEED = 42
SCALE = 0.4
TAU_C, TEMP_C = 0.5, 0.1
OUT = RESULTS_DIR / "physics_anchored"
B_CKPT = OUT / "structural_ablation" / "model_B.pt"
A6 = RESULTS_DIR / "model_final_Btail3" / "model.pt"

DOMAINS = [
    {"name": "SCS", "file": "scs_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019)),
     "test_years": (2016, 2019)},
    {"name": "GBR", "file": "gbr_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020)),
     "test_years": (2017, 2020)},
    {"name": "CARIB", "file": "caribbean_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020)),
     "test_years": (2017, 2020)},
]


def restore_tscf(path):
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def restore_tsar(path):
    m = PhysicsAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def restore_ungated(path):
    m = UnGatedAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def predict_tsar(m, ds, device):
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n, C = ds.n_dates, ds.C_dyn
    y_full = ds.y_regression_full
    fm = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    with torch.no_grad():
        for b in range(0, n - T + 1, 16):
            be = min(b + 16, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1:te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            anch = torch.FloatTensor(y_full[b + T - 1:be + T - 1].copy()).to(device)
            out = m(xd, xs, lm, anch, scale=SCALE, tau=TAU_C, temp=TEMP_C)
            for k in range(be - b):
                fm[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return fm


def predict_plain(m, ds, device):
    """PFV inference for a model whose forward takes no anchor arg (frozen TSCF)."""
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n, C = ds.n_dates, ds.C_dyn
    fm = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    with torch.no_grad():
        for b in range(0, n - T + 1, 16):
            be = min(b + 16, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1:te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            out = m(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            for k in range(be - b):
                fm[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return fm


def predict_ungated(m, ds, device):
    """PFV inference for the un-gated anchored residual (B): needs pers_anchor."""
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n, C = ds.n_dates, ds.C_dyn
    y_full = ds.y_regression_full
    fm = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    with torch.no_grad():
        for b in range(0, n - T + 1, 16):
            be = min(b + 16, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1:te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            anch = torch.FloatTensor(y_full[b + T - 1:be + T - 1].copy()).to(device)
            out = m(xd, xs, lm, anch, scale=SCALE)
            for k in range(be - b):
                fm[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return fm


def collect(ds, maps, test_years):
    T = ds.history_weeks
    n = ds.n_dates
    y = ds.y_regression_full
    ocean = ~ds.land_mask
    rec = {k: [] for k in maps}
    rec["target"] = []
    rec["t"] = []
    for t in range(T - 1, n):
        yy = int(ds.date_strings[t][:4])
        if not (test_years[0] <= yy <= test_years[1]):
            continue
        li = t + LEAD
        if li >= n:
            continue
        v = ~np.isnan(y[li]) & ocean
        for k in maps:
            v &= ~np.isnan(maps[k][t])
        if v.sum() == 0:
            continue
        ii, jj = np.where(v)
        for k in maps:
            rec[k].append(maps[k][t][v])
        rec["target"].append(y[li][v])
        rec["t"].append(np.full(len(ii), t, np.int32))
    return {k: np.concatenate(v) if v else np.array([]) for k, v in rec.items()}


def metrics(pred, target):
    from sklearn.metrics import roc_auc_score, average_precision_score
    sev = target >= SEV
    if len(np.unique(sev)) < 2:
        roc = pr = np.nan
    else:
        roc = roc_auc_score(sev, pred)
        pr = average_precision_score(sev, pred)
    pb = pred >= TAU
    h = int(np.sum(pb & sev)); ms = int(np.sum(~pb & sev)); fa = int(np.sum(pb & ~sev))
    csi = h / (h + ms + fa) if (h + ms + fa) > 0 else np.nan
    brier = float(np.mean((pred - target) ** 2))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        msk = (pred >= lo) & (pred < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / len(pred)) * abs(sev[msk].mean() - pred[msk].mean())
    return dict(roc_auc=float(roc), pr_auc=float(pr), csi_tau09=float(csi),
                brier=float(brier), ece=float(ece))


def fast_metric_vec(pred, target):
    """Rank-based ROC/PR (fast for the bootstrap loop); CIs only."""
    from scipy.stats import rankdata
    sev = (target >= SEV)
    n = len(pred)
    n_pos = int(sev.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        roc = np.nan
    else:
        r = rankdata(pred)
        roc = float((r[sev.astype(bool)].sum() - n_pos * (n_pos + 1) / 2.0)
                    / (n_pos * n_neg))
    if n_pos == 0:
        pr = np.nan
    else:
        order = np.argsort(-pred, kind="mergesort")
        sy = sev[order]
        tp = np.cumsum(sy)
        prec = tp / np.arange(1, n + 1)
        pr = float(prec[sy.astype(bool)].mean())
    pb = pred >= TAU
    h = int(np.sum(pb & sev)); ms = int(np.sum(~pb & sev)); fa = int(np.sum(pb & ~sev))
    csi = h / (h + ms + fa) if (h + ms + fa) > 0 else np.nan
    return dict(roc_auc=float(roc), pr_auc=float(pr), csi_tau09=float(csi))


def pair_ci(x, y, tgt, week, n_boot=N_BOOT, seed=BOOT_SEED, keys=("roc_auc", "pr_auc", "csi_tau09")):
    rng = np.random.default_rng(seed)
    weeks = np.unique(week)
    picks = weeks[rng.integers(0, len(weeks), size=(n_boot, len(weeks)))]
    out = {}
    for k in keys:
        dv = []
        for b in range(n_boot):
            m = np.isin(week, picks[b])
            if m.sum() < 10:
                continue
            dv.append(fast_metric_vec(x[m], tgt[m])[k]
                      - fast_metric_vec(y[m], tgt[m])[k])
        dv = np.array([v for v in dv if np.isfinite(v)])
        out[k] = ([] if len(dv) == 0 else
                  [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))])
    return out


def main():
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    mD = restore_tscf(A6).to(device)
    mB = restore_ungated(B_CKPT).to(device)
    mC = {s: restore_tsar(OUT / f"model_seed_{s}.pt").to(device) for s in SEEDS}

    results = {"seeds": SEEDS, "domains": {}}
    csv_path = OUT / "seed_sensitivity_tsar.csv"
    all_rows = []
    for dom in DOMAINS:
        name = dom["name"]
        ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test",
                             random_split=False, lead_weeks=LEAD,
                             dataset_file=dom["file"],
                             year_based_split=True, **dom["split"])
        y = ds.y_regression_full
        a_map = y.astype(np.float32)                      # PersA
        d_map = predict_plain(mD, ds, device)
        b_map = predict_ungated(mB, ds, device)           # ungated (seed 42)
        c_maps = {s: predict_tsar(mC[s], ds, device) for s in SEEDS}
        rec = collect(ds, {"A": a_map, "B": b_map, "D": d_map, **{f"C{s}": c_maps[s] for s in SEEDS}},
                      dom["test_years"])
        tgt, week = rec["target"], rec["t"].astype(int)
        dom_res = {"n": int(len(tgt)), "n_severe": int((tgt >= SEV).sum())}
        rows_dom = []
        for s in SEEDS:
            dom_res[f"C{s}"] = metrics(rec[f"C{s}"], tgt)
            dom_res[f"C{s}_minus_A_ci"] = pair_ci(rec[f"C{s}"], rec["A"], tgt, week)
            dom_res[f"C{s}_minus_B_ci"] = pair_ci(rec[f"C{s}"], rec["B"], tgt, week)
            rows_dom.append(dict(domain=name, seed=s, n=dom_res["n"],
                                 C_roc=dom_res[f"C{s}"]["roc_auc"],
                                 C_pr=dom_res[f"C{s}"]["pr_auc"],
                                 C_csi=dom_res[f"C{s}"]["csi_tau09"],
                                 CA_droc_lo=dom_res[f"C{s}_minus_A_ci"]["roc_auc"][0],
                                 CA_droc_hi=dom_res[f"C{s}_minus_A_ci"]["roc_auc"][1],
                                 CA_dpr_lo=dom_res[f"C{s}_minus_A_ci"]["pr_auc"][0],
                                 CA_dpr_hi=dom_res[f"C{s}_minus_A_ci"]["pr_auc"][1],
                                 CA_dcsi_lo=dom_res[f"C{s}_minus_A_ci"]["csi_tau09"][0],
                                 CA_dcsi_hi=dom_res[f"C{s}_minus_A_ci"]["csi_tau09"][1],
                                 CB_droc_lo=dom_res[f"C{s}_minus_B_ci"]["roc_auc"][0],
                                 CB_droc_hi=dom_res[f"C{s}_minus_B_ci"]["roc_auc"][1]))
            print(f"[{name} seed {s}] C ROC={dom_res[f'C{s}']['roc_auc']:.3f} "
                  f"PR={dom_res[f'C{s}']['pr_auc']:.3f} "
                  f"C-A dROC CI={dom_res[f'C{s}_minus_A_ci']['roc_auc']}", flush=True)
        dom_res["A"] = metrics(rec["A"], tgt)
        dom_res["B"] = metrics(rec["B"], tgt)
        dom_res["D"] = metrics(rec["D"], tgt)
        results["domains"][name] = dom_res
        all_rows += rows_dom
        # INCREMENTAL save (robust to the netCDF/mmap segfault at process exit)
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        with open(OUT / "seed_sensitivity_tsar.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[saved interim] {name} -> {csv_path.name}", flush=True)
        # release domain dataset + tensors before the next domain
        del ds, a_map, d_map, b_map, c_maps, rec, tgt, week
        gc.collect()
        torch.cuda.empty_cache()
    print("[saved] seed_sensitivity_tsar.csv/.json (final)", flush=True)
    print("TSAR SEED AUDIT DONE", flush=True)


if __name__ == "__main__":
    main()