#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE 1 — seed-sensitivity inference (data + models; opens netCDF).

Loads each results/seed_sensitivity/seed_{s}/model.pt, runs sliding-window PFV
inference, collects the ocean-masked records, and saves everything to
seed_qc_records.npz. NEVER computes metrics here (that phase must not open
netCDF to avoid the shutdown segfault). Analysis: seed_sensitivity_analysis.py.
"""
import io, sys, json, warnings, importlib
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))
from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

SEED_DIR = RESULTS_DIR / "seed_sensitivity"
LEAD, SEV = 4, 0.8
SEEDS = [0, 1, 2, 3, 42]
# per seed, which domains to diagnose (SCS for all; GBR/CARIB for the problems)
DOMAIN_PLAN = {"2": ["SCS", "GBR", "CARIB"], "3": ["GBR"]}

FILE = {"SCS": "scs_causal_v2.nc", "GBR": "gbr_causal_v2.nc", "CARIB": "caribbean_causal_v2.nc"}
YEARS = {"SCS": (2016, 2019), "GBR": (2017, 2020), "CARIB": (2017, 2020)}


def restore(seed, device):
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(SEED_DIR / f"seed_{seed}" / "model.pt",
                                 map_location="cpu", weights_only=False))
    m.eval()
    return m.to(device)


def infer(seed, dom, model, device):
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=LEAD, dataset_file=FILE[dom],
                         year_based_split=True, train_years=(2003, 2010),
                         val_years=(2014, 2016), test_years=YEARS[dom])
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n, C = ds.n_dates, ds.C_dyn
    y = ds.y_regression_full
    ocean = ~ds.land_mask
    dates = np.array([s[:4] for s in ds.date_strings], dtype=str)  # 'YYYY'
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
            out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            for k in range(be - b):
                fm[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    # collect ocean-masked record per test-window (pred at issue t, target t+4)
    p, t = [], []
    y0, y1 = YEARS[dom]
    for ti in range(T - 1, n):
        if not (y0 <= int(dates[ti]) <= y1):
            continue
        tt = ti + LEAD
        if tt >= n:
            continue
        v = ~np.isnan(y[tt]) & ocean & ~np.isnan(fm[ti])
        if v.sum() == 0:
            continue
        p.append(fm[ti][v]); t.append(y[tt][v])
    pred = np.concatenate(p) if p else np.array([], np.float32)
    tgt = np.concatenate(t) if t else np.array([], np.float32)
    del ds, X, fm
    return pred, tgt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    npz, meta = {}, {"seed_domains": {}}
    for seed in SEEDS:
        model = restore(seed, device)
        doms = ["SCS"] + DOMAIN_PLAN.get(str(seed), [])
        doms = list(dict.fromkeys(doms))
        for dom in doms:
            pred, tgt = infer(seed, dom, model, device)
            npz[f"{seed}_{dom}_pred"] = pred
            npz[f"{seed}_{dom}_target"] = tgt
            null = int(np.isnan(pred).sum())
            print(f"  seed {seed} {dom}: n={len(pred)} NaN_pred={null} "
                  f"range=[{np.min(pred):.4f},{np.max(pred):.4f}]", flush=True)
        meta["seed_domains"][str(seed)] = doms
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(SEED_DIR / "seed_qc_records.npz", **npz)
    with open(SEED_DIR / "seed_qc_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {SEED_DIR / 'seed_qc_records.npz'}", flush=True)
    print("PHASE 1 DONE", flush=True)


if __name__ == "__main__":
    main()