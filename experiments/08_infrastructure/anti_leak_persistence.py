#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test C — Persistence baseline (anti-leakage).

Persistence forecast: yhat(t+4) = y(t), where y = clip(DHW/10,0,1) is the
current-week thermal-stress value (window-end input step). This is the
naive "tomorrow = today" baseline the learned model must beat.

CSI computed on SCS test (2016-2019) and GBR test (2017-2020, re-sliced),
at thresholds 0.6/0.8/0.9, severe = y(t+4) >= 0.8.
"""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import DATA_DIR
from src.data.aligned_dataset import SCSPIECFDataset

SEV = 0.8

def persistence_csi(ds):
    T = ds.history_weeks
    yfull = ds.y_regression_full
    pers, tgt, m = [], [], []
    for i, s in enumerate(ds.window_indices):
        li = s + T - 1
        ti = s + T - 1 + 4
        if ti < ds.n_dates:
            pers.append(yfull[li])
            tgt.append(yfull[ti])
            m.append(np.isfinite(yfull[li]) & np.isfinite(yfull[ti]) & ds.label_mask[i])
    p = np.concatenate(pers).ravel(); t = np.concatenate(tgt).ravel(); mm = np.concatenate(m).ravel() > 0.5
    res = {}
    for tau in [0.6, 0.8, 0.9]:
        tp = np.sum((p>=tau)&(t>=SEV)&mm); fp = np.sum((p>=tau)&(t<SEV)&mm); fn = np.sum((p<tau)&(t>=SEV)&mm)
        res[f"tau{tau}"] = {"csi": float(tp/(tp+fp+fn+1e-9)), "pod": float(tp/(tp+fn+1e-9)), "far": float(fp/(tp+fp+1e-9))}
    return res, int(np.sum((t>=SEV)&mm)), int(mm.sum())

for label, mode, yrs in [("SCS test", "test", (2016,2019)),
                         ("GBR test", "test", (2017,2020))]:
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode=mode, random_split=False,
                         year_based_split=True, train_years=(2003,2010),
                         val_years=(2014,2016), test_years=yrs, lead_weeks=4)
    res, n_sev, n = persistence_csi(ds)
    print(f"\n=== Persistence baseline — {label} (n={n:,}, severe={n_sev}) ===")
    for k, v in res.items():
        print(f"  {k}: CSI={v['csi']:.4f} POD={v['pod']:.4f} FAR={v['far']:.4f}")
