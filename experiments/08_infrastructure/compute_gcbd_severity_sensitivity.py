#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review-response analysis: GCBD severe cutpoint sensitivity + temporal coverage.

Addressed reviewer priority item 3:
  - temporal distribution of the 78 matched severe observations within the
    SCS 2016-2019 evaluation window (obs year = issue week + 4 wk);
  - sensitivity of the severe-vs-background discrimination and the Q5/Q1
    enrichment to the biological severe cutpoint (Percent_Bleaching >= 10/20/30).

Reads results/step9_gcbd_validation/step9_gcbd_results.csv (evidence-locked
matched cohort, n=1898) and the SCS causal-v2 weekly time axis for date
mapping. No model is loaded; the frozen risk score `pred` is reused as-is.
Site-cluster bootstrap (resample Site_ID blocks, 1000, seed 42) is the
primary inference, consistent with Steps 9/10.

Outputs -> results/step9_gcbd_validation/severity_sensitivity/:
  temporal_severe_by_year.csv, cutpoint_sensitivity.csv, summary.json
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LTTS_ROOT = Path(__file__).resolve().parents[2]
RES = LTTS_ROOT / "results" / "step9_gcbd_validation"
OUT = RES / "severity_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
if not DATA_DIR.is_dir():
    DATA_DIR = LTTS_ROOT / "data"

df = pd.read_csv(RES / "step9_gcbd_results.csv")
assert len(df) == 1898, f"expected 1898 matched obs, got {len(df)}"

# --- observation year from issue week (issue = obs - 4wk, delta=0) ---------
import xarray as xr
ds = xr.open_dataset(DATA_DIR / "scs_causal_v2.nc")
times = pd.DatetimeIndex(ds.time.values)
ds.close()
# obs date = issue + 4 wk (issue = obs - 4wk); clamp at the axis end for the
# final weeks (axis stops 2019-07; late-2019 obs map to the last week)
obs_idx = np.minimum(df["issue_t"].values + 4, len(times) - 1)
df["obs_year"] = times[obs_idx].year.values

# temporal distribution of severe obs
sev = df[df["severe"] == 1]
year_counts = sev.groupby("obs_year").size().reindex(range(2003, 2021), fill_value=0)
year_counts.name = "n_severe"
year_counts.to_frame().to_csv(OUT / "temporal_severe_by_year.csv")
n_sev_in_window = int(sev["obs_year"].between(2016, 2019).sum())
print(f"severe obs: {len(sev)} total; {n_sev_in_window} within 2016-2019 test window")


def site_cluster_auroc(pred, y, site, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    sites = np.unique(site)
    n_sites = len(sites)
    if len(np.unique(y)) < 2:
        return float("nan"), (float("nan"), float("nan"))
    point = float(roc_auc_score(y, pred))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_sites, size=n_sites)
        mask = np.isin(site, sites[pick])
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        vals.append(roc_auc_score(y[mask], pred[mask]))
    vals = np.array(vals)
    return point, (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def rr_q5q1(sev_flag, quintile):
    q1 = quintile == 1
    q5 = quintile == 5
    p1 = sev_flag[q1].mean() if q1.sum() else np.nan
    p5 = sev_flag[q5].mean() if q5.sum() else np.nan
    if p1 == 0 or not np.isfinite(p1):
        return float("nan"), float("nan"), float("nan")
    rr = p5 / p1
    # Fisher exact on counts
    a = int(sev_flag[q5].sum()); b = int((~sev_flag[q5].astype(bool)).sum())
    c = int(sev_flag[q1].sum()); d = int((~sev_flag[q1].astype(bool)).sum())
    _, p = stats.fisher_exact([[a, b], [c, d]])
    return float(rr), float(p), int(q5.sum())


rows = []
for cut in [10.0, 20.0, 30.0]:
    flag = (df["pb"] >= cut).astype(int)
    bg = (df["pb"] == 0).astype(int)
    # severe vs background
    m = (flag == 1) | (bg == 1)
    auc, (lo, hi) = site_cluster_auroc(df.loc[m, "pred"].values, flag[m].values,
                                       df.loc[m, "Site_ID"].values)
    rr, fp, _ = rr_q5q1(flag.values, df["quintile"].values)
    n_sev = int(flag.sum())
    q5 = df["quintile"] == 5
    n_q5 = int(q5.sum())
    sev_rate_q5 = float(flag[q5].mean()) if q5.sum() else float("nan")
    rows.append(dict(cutpoint=cut, n_severe=n_sev, sev_vs_bg_auroc=auc,
                     au_lo=lo, au_hi=hi, rr_q5q1=rr, fisher_p=fp,
                     n_q5=n_q5, sev_rate_q5=sev_rate_q5))
sens = pd.DataFrame(rows)
sens.to_csv(OUT / "cutpoint_sensitivity.csv", index=False)
print(sens.round(4).to_string(index=False))

summary = {
    "n_matched": 1898, "n_severe_total": int(len(sev)),
    "n_severe_2016_2019": n_sev_in_window,
    "severe_by_year": {int(y): int(v) for y, v in year_counts.items()},
    "cutpoint_sensitivity": rows,
    "method": "severe = Percent_Bleaching>=cut; severe-vs-background AUROC + "
              "Q5/Q1 rate ratio (Fisher); site-cluster bootstrap (1000, seed 42)",
}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[saved] {OUT / 'temporal_severe_by_year.csv'}")
print(f"[saved] {OUT / 'cutpoint_sensitivity.csv'}")
print(f"[saved] {OUT / 'summary.json'}")
