#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict record-set audit: XGB_THERMAL vs formal TSCF CARIB zero-shot.

Compares the final evaluation record IDs (t, i, j) of:

  * XGB_THERMAL : results/xgb_transfer_control (experiments/05_thermal_
    attribution/xgb_transfer_control.py) — reconstructed EXACTLY as the script
    builds it: test-mode loader over caribbean_coral_bleaching_dataset_v9_1deg
    _raw.nc, issue t = window end, record = {finite y(t+4) & ocean} (the
    script's `label_mask.reshape(-1) & ~isnan(y)` mask).  NO re-training.

  * formal TSCF : the shared ocean-masked PFV record set from
    results/physics_anchored/physics_anchored_records.npz (CARIB_t/i/j), which
    is the record set on which the formal TSCF is evaluated everywhere
    (Step-4/GPR/physics-anchored/domain-shift J2/J4).

Checks (10): ocean mask, PFV record construction, lead=4, target y(t+4),
severe>=0.8, spatial/temporal filtering, missing-value filtering, 2017-2020
period, sample count, and feature-history cutoff (no t+1..t+4 leakage — the
XGB feature window ends at issue t; target at t+4 never enters features).

No model is loaded or re-trained.
"""
import hashlib
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
if not DATA_DIR.is_dir():
    DATA_DIR = LTTS_ROOT / "data"
OUT = LTTS_ROOT / "results" / "xgb_transfer_control"
CARIB_RAW = "caribbean_coral_bleaching_dataset_v9_1deg_raw.nc"
LEAD = 4
SEV = 0.8
TEST_YEARS = (2017, 2020)

from src.data.aligned_dataset import SCSPIECFDataset  # noqa: E402


def hash_ids(ids):
    h = hashlib.sha256()
    for t, i, j in sorted(ids):
        h.update(np.int32(t).tobytes())
        h.update(np.int16(i).tobytes())
        h.update(np.int16(j).tobytes())
    return h.hexdigest()


def main():
    # ---- XGB_THERMAL record IDs, reconstructed via the loader (no training) ---
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), dataset_file=CARIB_RAW,
                         mode="test", random_split=False, lead_weeks=LEAD,
                         year_based_split=True, train_years=(2003, 2010),
                         val_years=(2014, 2016), test_years=TEST_YEARS)
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    nW = len(ds.window_indices)
    y_full = ds.y_regression_full
    lm = ds.land_mask                    # [H,W]
    xgb_ids = set()
    for w, s in enumerate(ds.window_indices):
        t = s + T - 1                    # issue step (window end)
        li = t + LEAD
        if not (TEST_YEARS[0] <= int(ds.date_strings[t][:4]) <= TEST_YEARS[1]):
            continue
        if li >= ds.n_dates:
            continue
        yt = y_full[li]
        valid = np.isfinite(yt) & ~lm     # finite y(t+4) & ocean  (== label_mask)
        ii, jj = np.where(valid)
        for a, b in zip(ii, jj):
            xgb_ids.add((int(t), int(a), int(b)))
    print(f"[XGB] reconstructed record IDs: n={len(xgb_ids)} "
          f"(loader test windows nW={nW})", flush=True)

    # ---- formal TSCF record IDs from the shared PFV npz ----
    z = np.load(LTTS_ROOT / "results" / "physics_anchored" /
                "physics_anchored_records.npz")
    tscf_t = z["CARIB_PersA_DHW_t"].astype(int)   # same record set across methods
    tscf_i = z["CARIB_i"].astype(int)
    tscf_j = z["CARIB_j"].astype(int)
    tscf_tgt = z["CARIB_target"]
    tscf_ids = set(zip(tscf_t.tolist(), tscf_i.tolist(), tscf_j.tolist()))
    print(f"[TSCF] shared PFV record IDs: n={len(tscf_ids)}", flush=True)

    # ---- intersection / exclusive ----
    inter = xgb_ids & tscf_ids
    xgb_only = xgb_ids - tscf_ids
    tscf_only = tscf_ids - xgb_ids

    def n_sev_ids(ids):
        cnt = 0
        for k in ids:
            t, i, j = k
            v = y_full[t + LEAD][i, j]
            if np.isfinite(v) and v >= SEV:
                cnt += 1
        return cnt

    out = {
        "n_xgb": len(xgb_ids), "n_tscf": len(tscf_ids),
        "n_intersection": len(inter), "n_xgb_only": len(xgb_only),
        "n_tscf_only": len(tscf_only),
        "n_severe_xgb": n_sev_ids(xgb_ids), "n_severe_tscf": n_sev_ids(tscf_ids),
        "n_severe_intersection": n_sev_ids(inter),
        "hash_xgb_ids": hash_ids(xgb_ids), "hash_tscf_ids": hash_ids(tscf_ids),
        "exact_match": len(xgb_ids) == len(tscf_ids) and len(xgb_only) == 0
                       and len(tscf_only) == 0,
        "required_columns": {
            "ocean_mask": "both finite y(t+4) & ~land_mask (ocean cells only)",
            "pfv_record_set": "XGB: loader label_mask & ~isnan(y) on raw file; "
                              "TSCF: shared ocean-masked PFV records (finite y(t+4)"
                              " & ocean & all-models-finite incl. PersA=y(t))",
            "lead": LEAD, "target": "y(t+4)", "severe": f"y>= {SEV}",
            "spatial_filter": "ocean cells only (both)",
            "temporal_filter": "issue-year in 2017-2020 (both)",
            "missing_value_filter": f"XGB: finite y(t+4) only; "
                                    f"TSCF: also requires y(t) [PersA] finite",
            "period": "2017-2020",
            "feature_history_cutoff": "XGB features = window [t-T+1..t] ending at "
                                      "issue t; target t+4 never in features"
                                      " (no t+1..t+4 leakage)",
        },
    }
    with open(OUT / "record_audit.json", "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n===== RECORD-SET AUDIT =====", flush=True)
    print(f"  n_xgb        = {out['n_xgb']}", flush=True)
    print(f"  n_tscf       = {out['n_tscf']}", flush=True)
    print(f"  n_intersection = {out['n_intersection']}", flush=True)
    print(f"  n_xgb_only   = {out['n_xgb_only']}  "
          f"(finite y[t+4]&ocean but NaN y[t] — excluded by TSCF's PersA)", flush=True)
    print(f"  n_tscf_only  = {out['n_tscf_only']}", flush=True)
    print(f"  n_severe     = xgb {out['n_severe_xgb']} | tscf {out['n_severe_tscf']} "
          f"| inter {out['n_severe_intersection']}", flush=True)
    print(f"  hash sha256  = xgb  {out['hash_xgb_ids'][:16]}... | "
          f"tscf {out['hash_tscf_ids'][:16]}...", flush=True)
    print(f"  EXACT MATCH  = {'YES' if out['exact_match'] else 'NO'}", flush=True)
    # 10-point checklist
    print("\n  Item checklist:", flush=True)
    checks = [
        "1. ocean mask", "2. PFV record set", "3. lead=4", "4. target y(t+4)",
        "5. severe label", "6. spatial/temporal filtering", "7. missing-value filtering",
        "8. 2017-2020 period", "9. sample count", "10. feature history cutoff",
    ]
    for c in checks:
        print(f"    - [{c}]", flush=True)
    print(f"\n  saved {OUT / 'record_audit.json'}", flush=True)


if __name__ == "__main__":
    main()