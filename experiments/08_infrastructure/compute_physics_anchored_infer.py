#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE 1 — inference for the physics-anchored residual head (three domains).

Separate process from the analysis so netCDF4/mmap cleanup cannot segfault the
metrics/bootstrap step (previously a single process exited with code 139).

  * loads the frozen TSCF + physics-anchored checkpoints,
  * runs sliding-window PFV inference on SCS / GBR / Caribbean causal v2,
  * builds the shared record set per domain with the OCEAN MASK applied
    (v = finite y(t+4) & ~land_mask, aligned with training label_mask),
  * saves per-domain prediction arrays to results/physics_anchored/
    physics_anchored_records.npz (+ counts json) and exits.

No sklearn, no metrics, no bootstrap here. Phase 2:
  compute_physics_anchored_analysis.py (reads the npz).

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_physics_anchored_infer.py
"""
import io, sys, os, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

# NOTE: do NOT wrap sys.stdout here — importing ablate_a_dh_stopgrad re-wraps it.
warnings.filterwarnings("ignore")

import numpy as np
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel

LEAD = 4
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}
OUT_DIR = RESULTS_DIR / "physics_anchored"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen configuration source of truth: results/physics_anchored/summary.json
# (training CFG, incl. scale selected on SCS-val). Eval scale must match the
# scale the residual head was TRAINED at.
try:
    _sum = json.load(open(OUT_DIR / "summary.json", encoding="utf-8"))
    _cfg = _sum.get("cfg", {})
    MODEL_CFG["scale"] = float(_cfg.get("scale", MODEL_CFG["scale"]))
    print(f"[frozen cfg] scale={MODEL_CFG['scale']} "
          f"(from summary.json, trained ckpt md5={_sum.get('model_md5', '?')})", flush=True)
except Exception as e:
    print(f"[warn] summary.json unreadable ({type(e).__name__}); scale={MODEL_CFG['scale']}", flush=True)

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
METHODS = ["PhysicsAnchored", "PersA_DHW", "TSCF_frozen"]


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


def predict_sliding_anchored(model, ds, device):
    """PFV inference -> (forecast, anchor). forecast[t] = y_hat(t+lead)."""
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n_dates, C = ds.n_dates, ds.C_dyn
    y_full = ds.y_regression_full
    forecast = np.full((n_dates, H, W), np.nan, np.float32)
    anchor = y_full.astype(np.float32).copy()          # PersA(t) = y_regression[t]
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
            out = model(xd, xs, lm, anch, scale=MODEL_CFG["scale"],
                        tau=MODEL_CFG["tau"], temp=MODEL_CFG["temp"])
            for k in range(be - b):
                forecast[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return forecast, anchor


def predict_sliding_plain(model, ds, device):
    """PFV inference for models without an anchor arg (TSCF frozen)."""
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
            out = model(xd, xs, lm, scale=MODEL_CFG["scale"],
                        tau=MODEL_CFG["tau"], temp=MODEL_CFG["temp"])
            for k in range(be - b):
                forecast[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return forecast


def collect(ds, maps, test_years):
    """PFV record set, OCEAN-MASKED (aligned with training label_mask:
    v = finite y(t+4) & ~land_mask). Returns per-method arrays + counts of
    pixels excluded because they are land (finite-y-on-land) or method-NaN."""
    T = ds.history_weeks
    n = ds.n_dates
    y = ds.y_regression_full
    ocean = ~ds.land_mask                       # [H,W]
    out = {m: {"pred": [], "t": [], "i": [], "j": []} for m in maps}
    out["target"] = []
    n_land_fin = 0      # pixels excluded solely because they are land
    for t in range(T - 1, n):
        yy = int(ds.date_strings[t][:4])
        if not (test_years[0] <= yy <= test_years[1]):
            continue
        tt = t + LEAD
        if tt >= n:
            continue
        v = ~np.isnan(y[tt]) & ocean             # finite target AND ocean cell
        if v.sum() == 0:
            continue
        vc = v.copy()
        for m in maps:
            vc &= ~np.isnan(maps[m][t])
        if vc.sum() == 0:
            continue
        ii, jj = np.where(vc)
        for m in maps:
            out[m]["pred"].append(maps[m][t][vc])
            out[m]["t"].append(np.full(len(ii), t, np.int32))
            out[m]["i"].append(ii.astype(np.int16))
            out[m]["j"].append(jj.astype(np.int16))
        out["target"].append(y[tt][vc])
        n_land_fin += int((~np.isnan(y[tt]) & ~ocean).sum())
    out["target"] = np.concatenate(out["target"]) if out["target"] else np.array([])
    for m in maps:
        out[m] = {k: np.concatenate(v) if v else np.array([], dtype=np.float32)
                  for k, v in out[m].items()}
    return out, n_land_fin


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print("Loading checkpoints...", flush=True)
    anchored = restore_anchored(OUT_DIR / "model.pt").to(device)
    frozen = restore_frozen().to(device)

    npz = {}
    meta = {"protocol": "PFV t+4; issue-year filter; causal v2; SCS-train-only norm; "
                        "OCEAN-MASKED record set (aligned with label_mask)",
            "record_set": "finite y(t+4) & ~land_mask (ocean cells only)",
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
        m_anch, anch_map = predict_sliding_anchored(anchored, ds, device)
        m_frozen = predict_sliding_plain(frozen, ds, device)
        maps = {"PhysicsAnchored": m_anch, "PersA_DHW": anch_map, "TSCF_frozen": m_frozen}
        print(f"  inference {time.time()-t0:.1f}s", flush=True)
        rec, n_land = collect(ds, maps, dom["test_years"])
        target = rec["target"]
        for m in METHODS:
            npz[f"{name}_{m}_pred"] = rec[m]["pred"]
            npz[f"{name}_{m}_t"] = rec[m]["t"]
        npz[f"{name}_target"] = target
        npz[f"{name}_i"] = rec[METHODS[0]]["i"]
        npz[f"{name}_j"] = rec[METHODS[0]]["j"]
        meta["domains"][name] = {
            "n": int(len(target)),
            "n_severe": int((target >= 0.8).sum()),
            "prevalence": float((target >= 0.8).mean()),
            "finite_y_on_land_excluded": int(n_land),
            "grid": [int(ds.n_lat), int(ds.n_lon)],
            "test_years": list(dom["test_years"]),
        }
        print(f"  record n={meta['domains'][name]['n']} "
              f"sev={meta['domains'][name]['n_severe']} "
              f"(excluded finite-y-on-land={n_land})", flush=True)
        # free per-domain dataset memory before next domain
        del ds
        torch.cuda.empty_cache()

    npz_path = OUT_DIR / "physics_anchored_records.npz"
    np.savez_compressed(npz_path, **npz)
    meta_path = OUT_DIR / "physics_anchored_infer_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[saved] {npz_path}", flush=True)
    print(f"[saved] {meta_path}", flush=True)
    print("PHASE 1 DONE", flush=True)


if __name__ == "__main__":
    main()
