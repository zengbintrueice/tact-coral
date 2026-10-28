# compute_gcbd_year_stratification.py
#
# Weakness-fix #1A: target-year severity stratification of the GCBD
# enrichment. Limitation: the 78 severe observations concentrate in two
# high-stress years (36 in 2016, 38 in 2019). Question: is the enrichment
# an artefact of pooling those two anomalous years, or does it hold within
# each heavy year separately (and under leave-one-year-out)?
#
# PROTOCOL (frozen, verbatim step9 R5):
#   - frozen TSCF (D1, md5-checked), scs_causal_v2.nc, obs-4wk matching,
#     0.5 deg buffer, +/-4 wk window, issue years 2016-2019, cohort pinned
#     n=1898 / any=530 / severe=78 / background=1368 (fail-fast)
#   - AUROC = severe vs background (primary), site-cluster bootstrap
#     (1000, seed 42) per stratum
#   - strata: 2016, 2017, 2018, 2019 (separately), and leave-one-year-out
#   - top-quintile RR within the two heavy strata (2016, 2019)
#
# Outputs -> results/gcbd_year_stratification/

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import torch
from sklearn.metrics import roc_auc_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
from paths import DATA_DIR, RESULTS_DIR, A6_WEIGHT
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "gcbd_year_stratification"
OUT.mkdir(parents=True, exist_ok=True)

TSCF_MD5 = "7cdbca26cae03fa53be53210940e8ce1"
DATASET_FILE = "scs_causal_v2.nc"
GCBD_PATH = LTTS_ROOT / "data" / "china_bco_dmo_bleaching.csv"
EXPECTED = {"n": 1898, "any": 530, "severe": 78, "background": 1368}
N_BOOT = 1000
BOOT_SEED = 42
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def predict_sliding(model, ds, device):
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    fc = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    with torch.no_grad():
        for b in range(0, n - T + 1, 16):
            be = min(b + 16, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, **MODEL_CFG)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc


print("=" * 66)
print(f"GCBD target-year stratification — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(A6_WEIGHT) == TSCF_MD5, "frozen TSCF md5 mismatch"
print("frozen TSCF OK")

import ablate_a_dh_stopgrad as tr
ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                     lead_weeks=4, dataset_file=DATASET_FILE,
                     year_based_split=True, train_years=(2003, 2010),
                     val_years=(2011, 2015), test_years=(2016, 2019))
model = tr.FinalSingleHeadModel(dynamic_channels=13)
model.load_state_dict(torch.load(A6_WEIGHT, map_location="cpu", weights_only=False))
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
fmap = predict_sliding(model, ds, "cuda" if torch.cuda.is_available() else "cpu")
del model, ds
torch.cuda.empty_cache()

dnc = xr.open_dataset(Path(str(DATA_DIR)) / DATASET_FILE)
lats = dnc['lat'].values.astype(float); lons = dnc['lon'].values.astype(float)
times = pd.DatetimeIndex(dnc['time'].values); dnc.close()
vmask = (times.year >= 2016) & (times.year <= 2019)

df = pd.read_csv(GCBD_PATH, low_memory=False)
df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')

recs = []
for _, row in df.iterrows():
    try:
        lat = float(row['Latitude_Degrees']); lon = float(row['Longitude_Degrees'])
        date = pd.Timestamp(int(row['Date_Year']), int(row['Date_Month']),
                            min(max(int(row.get('Date_Day', 15)), 1), 28))
    except Exception:
        continue
    li = int(np.argmin(np.abs(lats - lat))); lj = int(np.argmin(np.abs(lons - lon)))
    if abs(lats[li] - lat) > 0.5 or abs(lons[lj] - lon) > 0.5:
        continue
    tgt_issue = date - pd.Timedelta(weeks=4)
    win = (np.abs(times - tgt_issue) <= pd.Timedelta(weeks=4)) & vmask
    if not win.any():
        continue
    diff = np.where(win, np.abs(times - tgt_issue).values.astype(np.float64), np.inf)
    idx = int(np.argmin(diff))
    v = fmap[idx, li, lj]
    if not np.isfinite(v):
        continue
    pb = row['Percent_Bleaching']
    if pd.isna(pb) or not (0 <= pb <= 100):
        continue
    recs.append({"pred": float(v), "pb": float(pb), "year": date.year,
                 "site": str(row.get('Site_ID')) if pd.notna(row.get('Site_ID'))
                         else f"__s{len(recs)}"})
co = pd.DataFrame(recs)
n_any, n_sev = int((co.pb > 0).sum()), int((co.pb >= 20).sum())
print(f"cohort n={len(co)} any={n_any} severe={n_sev} bg={int((co.pb==0).sum())}")
assert (len(co), n_any, n_sev) == (EXPECTED["n"], EXPECTED["any"], EXPECTED["severe"]), \
    "cohort mismatch"

sev = (co.pb >= 20).values.astype(int)
bg = (co.pb == 0).values.astype(int)
site = co.site.values
years = co.year.values


def auroc_sb(mask):
    m = mask & ((sev == 1) | (bg == 1))
    if sev[m].sum() == 0 or bg[m].sum() == 0:
        return np.nan
    return float(roc_auc_score(sev[m], co.pred.values[m]))


def cluster_ci(mask):
    # Restrict to severe|background, matching auroc_sb(). Without this the
    # resampled selection also contains any-bleaching rows (0 < pb < 20),
    # which enter the negative class and bias the bootstrap AUROC downward
    # relative to the point estimate.
    mask = mask & ((sev == 1) | (bg == 1))
    idx = np.where(mask)[0]
    s_names, first = np.unique(site[idx], return_index=True)
    smap = {s: idx[np.where(site[idx] == s)[0]] for s in s_names}
    rng = np.random.default_rng(BOOT_SEED)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(s_names), len(s_names))
        sel = np.concatenate([smap[s_names[k]] for k in pick])
        s, b = sev[sel], bg[sel]
        if s.sum() == 0 or b.sum() == 0:
            continue
        try:
            vals.append(roc_auc_score(s, co.pred.values[sel]))
        except ValueError:
            pass
    v = np.array(vals)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
            int(v.size))


rows = []
strata = {"all_2016_2019": np.ones(len(co), bool)}
for y in (2016, 2017, 2018, 2019):
    strata[f"year_{y}"] = years == y
for y in (2016, 2017, 2018, 2019):
    strata[f"leave_out_{y}"] = years != y
for name, mask in strata.items():
    n_s, n_b = int(sev[mask].sum()), int(bg[mask].sum())
    a = auroc_sb(mask)
    if n_s and n_b:
        lo, hi, nb = cluster_ci(mask)
    else:
        lo = hi = np.nan; nb = 0
    rows.append({"stratum": name, "n": int(mask.sum()), "n_severe": n_s,
                 "n_background": n_b, "auroc_sev_vs_bg": a,
                 "ci_lo": lo, "ci_hi": hi, "n_boot_valid": nb})
    print(f"  {name:16s} n={mask.sum():5d} sev={n_s:3d} bg={n_b:4d} "
          f"AUROC={a if np.isnan(a) else round(a,4)} "
          f"CI[{lo if np.isnan(lo) else round(lo,3)},{hi if np.isnan(hi) else round(hi,3)}]")

# within-stratum quintile RR for the two heavy years
rr_rows = []
for y in (2016, 2019):
    m = years == y
    p = co.pred.values[m]; s = sev[m]
    if s.sum() < 5:
        continue
    q = np.asarray(pd.qcut(p, 5, labels=False, duplicates="drop"))
    top, bot = q == q.max(), q == q.min()
    pt = s[top].mean() if top.sum() else np.nan
    pb_ = s[bot].mean() if bot.sum() else np.nan
    rr_rows.append({"stratum": f"year_{y}", "p_top": pt, "p_bot": pb_,
                    "rr": (pt / pb_) if (pb_ and pb_ > 0) else np.nan})
    print(f"  year {y}: Q5 rate {pt:.4f} vs Q1 {pb_ if pb_ is not None else float('nan'):.4f}")

summary = {"experiment": "gcbd_year_stratification",
           "generated": datetime.now().isoformat(),
           "cohort": {"n": len(co), "any": n_any, "severe": n_sev},
           "protocol": {"bootstrap": "site-cluster 1000 seed 42",
                        "strata": "per target year + leave-one-year-out"},
           "strata": rows, "quintile_rr": rr_rows}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
pd.DataFrame(rows).to_csv(OUT / "year_strata.csv", index=False)
pd.DataFrame(rr_rows).to_csv(OUT / "quintile_rr_by_year.csv", index=False)
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write(
"""# AUDIT - GCBD target-year stratification

- Frozen TSCF (md5 verified), verbatim step9 matching (obs-4wk, 0.5 deg,
  +/-4 wk, issue years 2016-2019); cohort re-pinned n=1898/530/78/1368.
- Strata are pre-specified: each target year separately + leave-one-year-out;
  AUROC = severe vs background; site-cluster bootstrap (1000, seed 42).
- Associative enrichment interpretation only; no threshold or calibration
  derived from GCBD.
""")
print(f"\nOutputs -> {OUT}")
print("Done.")
