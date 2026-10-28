# -*- coding: utf-8 -*-
"""Audit A evidence: per-channel OOD diagnostics for CARIB vs SCS-test vs GBR.

Uses the FROZEN checkpoint's baked ChannelNormalizer stats (== SCS_CHANNEL_MEAN/STD
to float32 precision). Computes, per domain, per 13-channel:
  - validity fraction over the full record
  - z-score percentiles (2.5/50/97.5) on test-period ocean+valid-target cells
  - z-score |max| (worst-case normalization excursion)
Also: correlation of frozen-TSCF prediction with PersA per domain (from the
physics_anchored npz, which shares one record set per domain).
"""
import io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
from src.data.aligned_dataset import SCSPIECFDataset
import torch

DATA_DIR = os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
LEAD = 4
DOMAINS = [
    {"name": "SCS", "file": "scs_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019))},
    {"name": "GBR", "file": "gbr_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))},
    {"name": "CARIB", "file": "caribbean_causal_v2.nc",
     "split": dict(train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020))},
]
CHANNELS = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
            "wind_speed", "ssr", "sst_anomaly",
            "d_hotspot_dt", "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]

ckpt = torch.load(str(LTTS_ROOT / "results" / "model_final_Btail3" / "model.pt"),
                  map_location="cpu", weights_only=False)
MEAN = ckpt["channel_norm.mean"].view(-1).numpy()
STD = ckpt["channel_norm.std"].view(-1).numpy()

npz = np.load(str(LTTS_ROOT / "results" / "physics_anchored" / "physics_anchored_eval_predictions.npz"))

print(f"{'domain':6s} {'chan':18s} {'valid%':>7s} {'z_p2.5':>8s} {'z_p50':>8s} {'z_p97.5':>8s} {'|z|max':>8s}")
summary = {}
for dom in DOMAINS:
    name = dom["name"]
    ds = SCSPIECFDataset(data_dir=DATA_DIR, mode="test", random_split=False,
                         lead_weeks=LEAD, dataset_file=dom["file"],
                         year_based_split=True, **dom["split"])
    ty = dom["split"]["test_years"]
    X = ds.X_dynamic_full                       # [n_dates, H, W, 13]
    y = ds.y_regression_full                    # [n_dates, H, W]
    ocean = ~ds.land_mask
    yrs = np.array([int(s[:4]) for s in ds.date_strings])
    test_t = np.where((yrs >= ty[0]) & (yrs <= ty[1]))[0]
    # target-valid at t+LEAD
    tv = np.zeros(X.shape[:3], dtype=bool)
    for t in test_t:
        tt = t + LEAD
        if tt < X.shape[0]:
            tv[t] = ocean & ~np.isnan(y[tt])
    tv2 = tv[:, :, :, None]
    z = (X - MEAN[None, None, None, :]) / STD[None, None, None, :]
    z[~np.isfinite(z)] = np.nan
    valid_frac = 1.0 - np.isnan(X).mean(axis=(0, 1, 2))
    rows = []
    for c, ch in enumerate(CHANNELS):
        zc = z[:, :, :, c][tv]
        p = np.nanpercentile(zc, [2.5, 50, 97.5]) if zc.size else [np.nan]*3
        mx = np.nanmax(np.abs(zc)) if np.isfinite(zc).any() else np.nan
        print(f"{name:6s} {ch:18s} {100*valid_frac[c]:7.1f} {p[0]:8.2f} {p[1]:8.2f} {p[2]:8.2f} {mx:8.2f}")
        rows.append({"channel": ch, "valid_frac": round(float(valid_frac[c]), 4),
                     "z_p2_5": round(float(p[0]), 3), "z_p50": round(float(p[1]), 3),
                     "z_p97_5": round(float(p[2]), 3), "z_abs_max": round(float(mx), 3)})
    summary[name] = {"test_cells": int(tv.sum()), "channels": rows}
    # TSCF vs PersA correlation (shared record set)
    tscf = npz[f"{name}_TSCF_frozen_pred"]
    pers = npz[f"{name}_PersA_DHW_pred"]
    r = np.corrcoef(tscf, pers)[0, 1]
    print(f"   {name}: corr(TSCF_pred, PersA) = {r:.4f}  n={len(tscf)}")
    summary[name]["corr_tscf_persa"] = round(float(r), 4)
    del ds

print("\nDONE")
