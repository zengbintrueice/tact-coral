#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE 1 — structural ablation inference (models A/B/C/D on SCS/GBR/CARIB).

  A PersA_DHW              = y_regression at issue time t (no model)
  B Un-gated bounded res.  = results/physics_anchored/structural_ablation/model_B.pt
  C Formal anchored res.   = results/physics_anchored/model.pt  (scale=0.4 frozen)
  D Frozen TSCF            = results/model_final_Btail3/model.pt

Same ocean-masked PFV record set per domain (v = finite y(t+4) & ~land_mask &
all four models finite at t). Saves per-domain arrays to
results/physics_anchored/structural_ablation/rec_structural_ablation.npz
(+ infer_meta.json), then exits. Analysis in a separate process.

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_structural_ablation_infer.py
"""
import io, sys, os, json, time, warnings, importlib
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from structural_ablation_models import UnGatedAnchoredResidualModel
from physics_anchored_model import PhysicsAnchoredResidualModel

LEAD = 4
SEV = 0.8
# Frozen config (Work 1): scale = 0.4 selected on SCS-val; tau/temp for C's gate
SCALE = 0.4
TAU_C = 0.5
TEMP_C = 0.1
OUT_DIR = RESULTS_DIR / "physics_anchored" / "structural_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_C = RESULTS_DIR / "physics_anchored" / "model.pt"
CKPT_B = OUT_DIR / "model_B.pt"

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
MODELS = ["A_PersA", "B_ungated", "C_anchored", "D_TSCF"]


def restore_frozen():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(A6_WEIGHT, map_location="cpu", weights_only=False))
    m.eval()
    return m


def restore_anchored(path):
    m = PhysicsAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def restore_ungated(path):
    m = UnGatedAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def predict_anchored_like(model, ds, device, mode):
    """PFV inference for C (gated) and B (ungated) — both take pers_anchor."""
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n_dates, C = ds.n_dates, ds.C_dyn
    y_full = ds.y_regression_full
    forecast = np.full((n_dates, H, W), np.nan, np.float32)
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
            anch = torch.FloatTensor(y_full[b + T - 1:be + T - 1].copy()).to(device)
            if mode == "C":
                out = model(xd, xs, lm, anch, scale=SCALE, tau=TAU_C, temp=TEMP_C)
            else:
                out = model(xd, xs, lm, anch, scale=SCALE)
            for k in range(be - b):
                forecast[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return forecast


def predict_plain(model, ds, device):
    """PFV inference for D (frozen TSCF, no anchor arg)."""
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n_dates, C = ds.n_dates, ds.C_dyn
    forecast = np.full((n_dates, H, W), np.nan, np.float32)
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
                forecast[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return forecast


def collect_shared(ds, maps, test_years):
    """Ocean-masked SHARED record set: finite y(t+4) & ocean & all models finite."""
    T = ds.history_weeks
    n = ds.n_dates
    y = ds.y_regression_full
    ocean = ~ds.land_mask
    rec = {m: {"pred": [], "t": [], "i": [], "j": []} for m in maps}
    rec["target"] = []
    for t in range(T - 1, n):
        yy = int(ds.date_strings[t][:4])
        if not (test_years[0] <= yy <= test_years[1]):
            continue
        tt = t + LEAD
        if tt >= n:
            continue
        v = ~np.isnan(y[tt]) & ocean
        if v.sum() == 0:
            continue
        vc = v.copy()
        for m in maps:
            vc &= ~np.isnan(maps[m][t])
        if vc.sum() == 0:
            continue
        ii, jj = np.where(vc)
        for m in maps:
            rec[m]["pred"].append(maps[m][t][vc])
            rec[m]["t"].append(np.full(len(ii), t, np.int32))
            rec[m]["i"].append(ii.astype(np.int16))
            rec[m]["j"].append(jj.astype(np.int16))
        rec["target"].append(y[tt][vc])
    rec["target"] = np.concatenate(rec["target"]) if rec["target"] else np.array([])
    for m in maps:
        rec[m] = {k: np.concatenate(v) if v else np.array([], dtype=np.float32)
                  for k, v in rec[m].items()}
    return rec


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Frozen config: scale={SCALE}, tau(C)={TAU_C}, temp(C)={TEMP_C}", flush=True)
    mD = restore_frozen().to(device)
    mC = restore_anchored(CKPT_C).to(device)
    mB = restore_ungated(CKPT_B).to(device)

    npz = {}
    meta = {"frozen_config": {"scale": SCALE, "tau": TAU_C, "temp": TEMP_C,
                              "threshold": 0.90, "seed": 42, "n_boot": 1000},
            "record_set": "ocean-masked shared: finite y(t+4) & ~land_mask & all 4 models finite",
            "models": {"A_PersA": "PersA(t)=y_regression(t)",
                       "B_ungated": "PersA + tanh(R(attn))*0.4",
                       "C_anchored": "PersA + sigmoid((y_main-0.5)/0.1)*tanh(R(attn))*0.4",
                       "D_TSCF": "frozen single-head TSCF"},
            "domains": {}}
    for dom in DOMAINS:
        name = dom["name"]
        print("=" * 64, flush=True)
        print(f"Domain {name}: {dom['file']} (test {dom['test_years'][0]}-{dom['test_years'][1]})",
              flush=True)
        ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                             lead_weeks=LEAD, dataset_file=dom["file"],
                             year_based_split=True, **dom["split"])
        t0 = time.time()
        maps = {
            "A_PersA": ds.y_regression_full.astype(np.float32),
            "B_ungated": predict_anchored_like(mB, ds, device, "B"),
            "C_anchored": predict_anchored_like(mC, ds, device, "C"),
            "D_TSCF": predict_plain(mD, ds, device),
        }
        print(f"  inference {time.time()-t0:.1f}s", flush=True)
        rec = collect_shared(ds, maps, dom["test_years"])
        target = rec["target"]
        for m in MODELS:
            npz[f"{name}_{m}_pred"] = rec[m]["pred"]
            npz[f"{name}_{m}_t"] = rec[m]["t"]
        npz[f"{name}_target"] = target
        npz[f"{name}_i"] = rec[MODELS[0]]["i"]
        npz[f"{name}_j"] = rec[MODELS[0]]["j"]
        meta["domains"][name] = {
            "n": int(len(target)),
            "n_severe": int((target >= SEV).sum()),
            "prevalence": float((target >= SEV).mean()),
            "grid": [int(ds.n_lat), int(ds.n_lon)],
            "test_years": list(dom["test_years"]),
        }
        print(f"  shared record n={meta['domains'][name]['n']} "
              f"sev={meta['domains'][name]['n_severe']}", flush=True)
        del ds
        torch.cuda.empty_cache()

    npz_path = OUT_DIR / "rec_structural_ablation.npz"
    np.savez_compressed(npz_path, **npz)
    with open(OUT_DIR / "infer_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[saved] {npz_path}", flush=True)
    print("PHASE 1 DONE", flush=True)


if __name__ == "__main__":
    main()
