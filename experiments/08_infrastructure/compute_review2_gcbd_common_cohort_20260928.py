# compute_review2_gcbd_common_cohort_20260928.py
# P0-1 GCBD enrichment BY MODEL on a LIKE-FOR-LIKE (common) cohort.
#
# PersA = clip(DHW/10) and TSAR = PersA + gate*raw*0.4 are NaN wherever the
# current DHW is undefined, whereas the TSCF network emits a score on every
# matched cell. Matching exactly one GCBD observation to one (issue_t, lat, lon)
# cell; a model is "defined" on an observation iff its score is finite there.
# We therefore report each model on the COMMON cohort (all three finite),
# so the three-way comparison is like-for-like; the canonical frozen R5 TSCF
# result (n=1898, 78 severe) is retained and reported separately as reference.
# Site-cluster bootstrap (n=1000, seed 42). Writes results/review2_revision/
# gcbd_common_cohort.json

import io, sys, json, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.stats import mannwhitneyu

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = RESULTS_DIR / "review2_revision"
CKPT_TSCF = LTTS_ROOT / "results" / "model_final_Btail3" / "model.pt"
CKPT_TSAR = RESULTS_DIR / "physics_anchored" / "model_seed_42.pt"
DATASET_FILE = "scs_causal_v2.nc"
GCBD_PATH = LTTS_ROOT / "data" / "china_bco_dmo_bleaching.csv"
TSAR_SCALE = 0.4
N_BOOT = 1000
BOOT_SEED = 42
SCS_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                 val_years=(2011, 2015), test_years=(2016, 2019))
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}


def restore_tscf():
    import ablate_a_dh_stopgrad as mod
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(CKPT_TSCF, map_location="cpu", weights_only=False))
    m.eval()
    return m


def restore_tsar():
    m = PhysicsAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(CKPT_TSAR, map_location="cpu", weights_only=False))
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


def infer_tsar(model, ds, device):
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    gt = np.full((n, H, W), np.nan, np.float32)
    rr = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    Y = ds.y_regression_full
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
            anch = torch.FloatTensor(Y[b + T - 1: be + T - 1].copy()).to(device)
            out = model(xd, xs, lm, anch, scale=1.0, tau=0.5, temp=0.1)
            gt[b + T - 1: be + T - 1] = out[2][:, -1].cpu().numpy()
            rr[b + T - 1: be + T - 1] = out[3][:, -1].cpu().numpy()
    return gt, rr


def match_all(df, fmap, lats, lons, times, valid_time_mask):
    """Return list of matched (issue_t, lat_idx, lon_idx, pb, site) for ANY
    model-agnostic forecast map (TSCF map used for the record set)."""
    rows = []
    for _, row in df.iterrows():
        try:
            lat = float(row['Latitude_Degrees']); lon = float(row['Longitude_Degrees'])
            year = int(row['Date_Year']); month = int(row['Date_Month'])
            day = int(row.get('Date_Day', 15))
            if not (1 <= month <= 12):
                continue
            day = min(max(day, 1), 28)
            date = pd.Timestamp(year, month, day)
        except Exception:
            continue
        lat_idx = int(np.argmin(np.abs(lats - lat))); lon_idx = int(np.argmin(np.abs(lons - lon)))
        if abs(lats[lat_idx] - lat) > 0.5 or abs(lons[lon_idx] - lon) > 0.5:
            continue
        target_issue = date - pd.Timedelta(weeks=4)
        tdiff = np.abs(times - target_issue).values.astype(np.float64)
        in_win = (np.abs(times - target_issue) <= pd.Timedelta(weeks=4)) & valid_time_mask
        if not in_win.any():
            continue
        masked = np.where(in_win, tdiff, np.inf)
        idx = int(np.argmin(masked))
        if not np.isfinite(fmap[idx, lat_idx, lon_idx]):
            continue
        pb = row['Percent_Bleaching']
        if pd.isna(pb) or not (0 <= pb <= 100):
            continue
        site = row.get('Site_ID')
        rows.append({"issue_t": idx, "lat_idx": lat_idx, "lon_idx": lon_idx,
                     "pb": float(pb),
                     "Site_ID": str(site) if pd.notna(site) else f"__s{len(rows)}"})
    return rows


def cliffs_delta(g1, g2):
    u = mannwhitneyu(g1, g2, alternative="greater").statistic
    return 2.0 * u / (len(g1) * len(g2)) - 1.0


def auroc_sb(sev, bg, p):
    m = (sev == 1) | (bg == 1)
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(sev[m], p[m]))


def groupby_max(s, p, sev):
    s = np.ascontiguousarray(s)
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    q = np.asarray(q)
    top = q == np.nanmax(q); bot = q == np.nanmin(q)
    pt = sev[top].mean() if top.sum() else np.nan
    pbot = sev[bot].mean() if bot.sum() else np.nan
    return (pt / pbot) if (pbot and pbot > 0) else np.nan


def analytics(preds, sev, bg, site):
    from sklearn.metrics import roc_auc_score
    try:
        mw = mannwhitneyu(preds[sev == 1], preds[bg == 1], alternative="greater")
        pval = float(mw.pvalue)
    except Exception:
        pval = np.nan
    point = {"n": int(len(preds)), "n_severe": int(sev.sum()),
             "auroc_sev_bg": auroc_sb(sev, bg, preds),
             "rr_q5q1_severe": groupby_max(site, preds, sev),
             "cliffs_delta_severe": cliffs_delta(preds[sev == 1], preds[bg == 1]),
             "mw_p": pval}
    site_names, site_to_idx = np.unique(site), {}
    for s in site_names:
        site_to_idx[s] = np.where(site == s)[0]
    def run(mode):
        rng = np.random.default_rng(BOOT_SEED)
        out = {"roc_sb": [], "cd": [], "rr": []}
        for _ in range(N_BOOT):
            if mode == "naive":
                idx = rng.integers(0, len(preds), len(preds))
            else:
                pick = rng.integers(0, len(site_names), len(site_names))
                idx = np.concatenate([site_to_idx[site_names[s]] for s in pick])
            p, s_, b_ = preds[idx], sev[idx], bg[idx]
            if len(np.unique(p)) < 2 or s_.sum() == 0 or b_.sum() == 0:
                continue
            try:
                out["roc_sb"].append(auroc_sb(s_, b_, p))
            except Exception:
                pass
            out["cd"].append(cliffs_delta(p[s_ == 1], p[b_ == 1]))
            out["rr"].append(groupby_max(site[idx], p, s_))
        def ci(v):
            v = np.array([x for x in v if np.isfinite(x)], dtype=float)
            if v.size == 0:
                return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
            return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                    "hi": float(np.percentile(v, 97.5)), "n": int(v.size)}
        return {k: ci(out[k]) for k in out}
    return {"point": point, "naive_ci": run("naive"), "site_cluster_ci": run("cluster")}


if True:
    print("[GCBD common cohort] loading SCS causal v2 ...", flush=True)
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=DATASET_FILE, **SCS_SPLIT)
    persa = ds.y_regression_full.astype(np.float32).copy()
    m_tscf = restore_tscf().to(DEVICE)
    m_tsar = restore_tsar().to(DEVICE)
    tscf_map = predict_sliding(m_tscf, ds, DEVICE)
    gt, rr = infer_tsar(m_tsar, ds, DEVICE)
    tsar_map = persa + gt * rr * TSAR_SCALE
    del gt, rr
    torch.cuda.empty_cache()

    dnc = xr.open_dataset(Path(str(DATA_DIR)) / DATASET_FILE)
    lats = dnc['lat'].values.astype(float); lons = dnc['lon'].values.astype(float)
    times = pd.DatetimeIndex(dnc['time'].values); dnc.close()
    valid_time_mask = (times.year >= 2016) & (times.year <= 2019)

    df = pd.read_csv(GCBD_PATH, low_memory=False)
    df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')

    rows = match_all(df, tscf_map, lats, lons, times, valid_time_mask)
    co = pd.DataFrame(rows)
    print("  matched (TSCF) n=", len(co), flush=True)

    def score_at(map_, issue_t, lat_idx, lon_idx):
        return map_[issue_t, lat_idx, lon_idx]

    for m, mp in (("TSCF", tscf_map), ("PersA", persa), ("TSAR", tsar_map)):
        co[m] = [score_at(mp, t, i, j) for t, i, j in
                 zip(co['issue_t'].values, co['lat_idx'].values, co['lon_idx'].values)]
    print("  severe n per model finite:",
          {m: int((co[m].notna() & (co['pb'] >= 20)).sum()) for m in ("TSCF", "PersA", "TSAR")},
          flush=True)

    # Common cohort: all three finite
    common = co[co['TSCF'].notna() & co['PersA'].notna() & co['TSAR'].notna()].copy()
    print("  common cohort n=", len(common),
          " severe=", int((common['pb'] >= 20).sum()), flush=True)

    out = {"cohort": {"matched_TSCF": int(len(co)),
                      "matched_TSCF_severe": int((co['pb'] >= 20).sum()),
                      "common": int(len(common)),
                      "common_severe": int((common['pb'] >= 20).sum())},
           "note": ("common cohort = GCBD observations where TSCF, PersA and TSAR "
                    "are all finite; canonical step9/R5 TSCF result is on the full "
                    "matched set and reported separately."),
           "models": {}}
    sev = (common['pb'] >= 20).astype(int).values
    bg = (common['pb'] == 0).astype(int).values
    site = common['Site_ID'].values
    for m in ("TSCF", "PersA", "TSAR"):
        preds = common[m].values.astype(float)
        out["models"][m] = analytics(preds, sev, bg, site)
        pt = out["models"][m]["point"]; sc = out["models"][m]["site_cluster_ci"]
        print(f"  [{m}] AUROC {pt['auroc_sev_bg']:.4f} [{sc['roc_sb']['lo']:.3f},{sc['roc_sb']['hi']:.3f}] "
              f"RR {pt['rr_q5q1_severe']} δ {pt['cliffs_delta_severe']:.3f} "
              f"p={pt['mw_p']:.1e}", flush=True)

    # Canonical R5 TSCF (full 1898) reference, site-cluster
    sevF = (co['pb'] >= 20).astype(int).values; bgF = (co['pb'] == 0).astype(int).values
    siteF = co['Site_ID'].values
    out["reference_R5_TSCF_full"] = analytics(
        co['TSCF'].values.astype(float), sevF, bgF, siteF)

    with open(OUT / "gcbd_common_cohort.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print("DONE ->", OUT / "gcbd_common_cohort.json")