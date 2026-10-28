# compute_review2_gcbd_by_year_rr.py
# R2-M2 follow-up: per-year / per-event rate-ratio breakdown of the GCBD
# severe-bleaching enrichment (frozen step9/R5 cohort, TSCF).
#
# For each issue-year Y (2016-2019) we take that year's matched observations,
# form predicted-risk quintiles from the TSCF score quantiles OF THAT YEAR
# (outcome-free boundaries), and compute the severe-rate ratio
# RR = P(severe|Q5) / P(severe|Q1). Where Q1 or Q5 contains zero severe
# observations the RR is undefined and the rates are still reported. We also
# report the pooled high-stress-year (2016, 2019) vs low-stress-year
# (2017, 2018) contrast. Ties to the two-year concentration note in §8/R5.
#
# Run (piecf_env):
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_review2_gcbd_by_year_rr.py

import io, sys, json, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = RESULTS_DIR / "review2_revision"
CKPT_TSCF = LTTS_ROOT / "results" / "model_final_Btail3" / "model.pt"
DATASET_FILE = "scs_causal_v2.nc"
GCBD_PATH = LTTS_ROOT / "data" / "china_bco_dmo_bleaching.csv"
SCS_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                 val_years=(2011, 2015), test_years=(2016, 2019))
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}
SEV = 20.0


def restore_tscf():
    import ablate_a_dh_stopgrad as mod
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(CKPT_TSCF, map_location="cpu", weights_only=False))
    m.eval()
    return m


def predict_sliding(model, ds, device):
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    fc = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    bs = 16
    with torch.no_grad():
        for b in range(0, n - T + 1, bs):
            be = min(b + bs, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, **MODEL_CFG)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc


def _rows_ok(rows):
    return rows


if __name__ == "__main__":
    print("[GCBD by-year RR] load + infer ...", flush=True)
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=DATASET_FILE, **SCS_SPLIT)
    m = restore_tscf().to(DEVICE)
    fmap = predict_sliding(m, ds, DEVICE)
    del m
    torch.cuda.empty_cache()
    dnc = xr.open_dataset(Path(str(DATA_DIR)) / DATASET_FILE)
    lats = dnc['lat'].values.astype(float); lons = dnc['lon'].values.astype(float)
    times = pd.DatetimeIndex(dnc['time'].values); dnc.close()
    valid_mask = (times.year >= 2016) & (times.year <= 2019)

    df = pd.read_csv(GCBD_PATH, low_memory=False)
    df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')

    recs = []
    for _, row in df.iterrows():
        m_ = None
        try:
            lat = float(row['Latitude_Degrees']); lon = float(row['Longitude_Degrees'])
            year = int(row['Date_Year']); month = int(row['Date_Month'])
            day = int(row.get('Date_Day', 15))
            if not (1 <= month <= 12):
                continue
            day = min(max(day, 1), 28)
            date = pd.Timestamp(year, month, day)
            lat_idx = int(np.argmin(np.abs(lats - lat))); lon_idx = int(np.argmin(np.abs(lons - lon)))
            if abs(lats[lat_idx] - lat) > 0.5 or abs(lons[lon_idx] - lon) > 0.5:
                continue
            target_issue = date - pd.Timedelta(weeks=4)
            tdiff = np.abs(times - target_issue).values.astype(np.float64)
            in_win = (np.abs(times - target_issue) <= pd.Timedelta(weeks=4)) & valid_mask
            if not in_win.any():
                continue
            idx = int(np.argmin(np.where(in_win, tdiff, np.inf)))
            v = fmap[idx, lat_idx, lon_idx]
            pb = row['Percent_Bleaching']
            if not np.isfinite(v) or pd.isna(pb) or not (0 <= pb <= 100):
                continue
            recs.append({"pred": float(v), "pb": float(pb),
                         "issue_year": int(times[idx].year),
                         "obs_year": int(year)})
        except Exception:
            continue
    co = pd.DataFrame(recs)
    print(f"  matched cohort n={len(co)} severe={int((co['pb']>=SEV).sum())}", flush=True)
    # sanity: should reproduce step9 counts
    print("  obs-year severe counts:", co[co['pb']>=SEV]['obs_year'].value_counts().sort_index().to_dict(), flush=True)

    def quint_rr(sub, label):
        n = len(sub)
        sev = (sub['pb'] >= SEV).values.astype(int)
        if n < 10 or sev.sum() == 0:
            return {"group": label, "n": int(n), "n_severe": int(sev.sum()),
                    "q1_severe_rate": float("nan"), "q5_severe_rate": float("nan"),
                    "rr": float("nan"), "rr_status": "undefined (no severe observation)"}
        q = pd.qcut(sub['pred'].values, 5, labels=False, duplicates="drop")
        q = np.asarray(q)
        top = q == np.nanmax(q); bot = q == np.nanmin(q)
        p_top = float(sev[top].mean()); p_bot = float(sev[bot].mean())
        n_top = int(sev[top].sum()); n_bot = int(sev[bot].sum())
        rr = (p_top / p_bot) if (p_bot and p_bot > 0) else float("nan")
        status = "ok" if (np.isfinite(rr) and n_bot > 0 and n_top > 0) else \
                 ("undefined (zero severe in a boundary quintile)" if not np.isfinite(rr) else "ok")
        return {"group": label, "n": int(n), "n_severe": int(sev.sum()),
                "n_severe_Q5": int(n_top), "n_severe_Q1": int(n_bot),
                "q1_severe_rate": float(p_bot), "q5_severe_rate": float(p_top),
                "rr": float(rr) if np.isfinite(rr) else None, "rr_status": status}

    rows = []
    for y in [2016, 2017, 2018, 2019]:
        rows.append(quint_rr(co[co['issue_year'] == y], f"year {y}"))
    hi = co[co['issue_year'].isin([2016, 2019])]
    lo = co[co['issue_year'].isin([2017, 2018])]
    rows.append(quint_rr(hi, "2016+2019 (high-stress years)"))
    rows.append(quint_rr(lo, "2017+2018 (low-stress years)"))
    rows.append(quint_rr(co, "all years (R5 cohort)"))

    rr_df = pd.DataFrame(rows)
    print("\n== quintile severe RR by year ==")
    print(rr_df.to_string(index=False), flush=True)
    rr_df.to_csv(OUT / "gcbd_quintile_by_year.csv", index=False)
    with open(OUT / "gcbd_quintile_by_year.json", "w", encoding="utf-8") as f:
        json.dump({"cohort": {"n": int(len(co)), "severe": int(co['pb'].ge(SEV).sum())},
                   "by_year": rows}, f, indent=2, ensure_ascii=False, default=str)
    print("DONE ->", OUT / "gcbd_quintile_by_year.csv")