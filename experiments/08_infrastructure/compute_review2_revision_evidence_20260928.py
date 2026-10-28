# compute_review2_revision_evidence_20260928.py
#
# Targeted reviewer Round-2 revision evidence (evidence-locked; no frozen
# output is modified). Reuses ONLY frozen protocols/data and writes new
# outputs to results/review2_revision/.
#
#   P0-1  GCBD model-wise enrichment (PersA_DHW, TSAR, frozen TSCF) on the
#         EXACT R5/step9 record set + obs-4wk matching + site-cluster bootstrap:
#         AUROC (severe vs background), top-quintile RR (Q5/Q1), Cliff's delta,
#         each with 95% CI.  Interpretation: thermal-risk-coordinate enrichment,
#         NOT biological prediction.
#   P0-4  Marginal per-model 95% CI AND paired-difference 95% CI for the
#         structural contrasts A/B/C/D (SCS/GBR/CARIB), week-block bootstrap
#         (1000, seed 42), shared resample matrix.
#   P1-7  CARIB C-A spatial-cluster (grid-cell) bootstrap sensitivity for
#         ΔROC/ΔPR/ΔCSI (robustness; does NOT replace the week-block bootstrap).
#   P1-6  CARIB environment-channel spatial permutation (Step-8/S5 protocol,
#         K=100, seed 42, frozen D1): original vs permuted metrics + CI.
#
# Run (piecf_env):
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_review2_revision_evidence_20260928.py

import io, sys, os, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = RESULTS_DIR / "review2_revision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- frozen inputs (read-only) ----
CKPT_TSCF = LTTS_ROOT / "results" / "model_final_Btail3" / "model.pt"
CKPT_TSAR = RESULTS_DIR / "physics_anchored" / "model_seed_42.pt"   # formal seed-42 TSAR
REC_STRUCT = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_CARIB  = RESULTS_DIR / "physics_anchored" / "audit" / "rec_CARIB.npz"
DATASET_FILE = "scs_causal_v2.nc"
CARIB_FILE = "caribbean_causal_v2.nc"
GCBD_PATH = LTTS_ROOT / "data" / "china_bco_dmo_bleaching.csv"

LEAD = 4
SEV = 0.8
TAU_EVAL = 0.9
TSAR_SCALE = 0.4                        # formal TSAR scale (SCS-val argmax CSI@0.9)
SCALES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
N_BOOT = 1000
BOOT_SEED = 42
N_PERM = 100
PERM_SEED = 42
SCS_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                 val_years=(2011, 2015), test_years=(2016, 2019))
CARIB_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                   val_years=(2014, 2016), test_years=(2017, 2020))
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}      # TSCF scoring cfg

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
ENV_NAMES = ["wind_speed", "ssr", "chlorophyll", "par", "seasonal_sin", "seasonal_cos"]
ENV_IDX = [DATASET_ORDER.index(c) for c in ENV_NAMES]

TSCF_MD5 = "7cdbca26cae03fa53be53210940e8ce1"
TSAR_MD5 = "ae86e017bc5ba634b0c68e5d13b4f9f0"


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cliffs_delta(g1, g2):
    n1, n2 = len(g1), len(g2)
    u = mannwhitneyu(g1, g2, alternative="greater").statistic
    return 2.0 * u / (n1 * n2) - 1.0


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Reviewer Round-2 revision evidence — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
print(f"Device: {DEVICE}")

# guard: verify frozen checkpoints havehes
print(f"TSCF ckpt md5 {md5_file(CKPT_TSCF)} (expected {TSCF_MD5})")
print(f"TSAR ckpt md5 {md5_file(CKPT_TSAR)} (expected {TSAR_MD5})")

# ───────────────────────── SECTION P0-1: GCBD per model ─────────────────────────
def restore_tscf():
    import ablate_a_dh_stopgrad as mod
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(CKPT_TSCF, map_location="cpu", weights_only=False))
    m.eval()
    return m


def restore_tsar(path):
    m = PhysicsAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def predict_sliding(model, ds, device, cfg):
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
            out = model(xd, xs, lm, **cfg)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc


def predict_from_X(model, X_full, land_mask, H, W, C, device, cfg):
    """Sliding inference over a caller-supplied input tensor (e.g. permuted)."""
    n = X_full.shape[0]
    T = 24
    fc = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(land_mask).to(device)
    bs = 16
    with torch.no_grad():
        for b in range(0, n - T + 1, bs):
            be = min(b + bs, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X_full[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.zeros((be - b, 1), dtype=torch.float32, device=device)
            out = model(xd, xs, lm, **cfg)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc


def infer_tsar(model, ds, device):
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    yf = np.full((n, H, W), np.nan, np.float32)
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
            out = model(xd, xs, lm, anch, scale=1.0, tau=0.5, temp=0.1)   # raw
            yf[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
            gt[b + T - 1: be + T - 1] = out[2][:, -1].cpu().numpy()
            rr[b + T - 1: be + T - 1] = out[3][:, -1].cpu().numpy()
    return yf, gt, rr


def match_gcbd(row, fmap, lats, lons, times, valid_time_mask):
    """Verbatim step9/gcbd_independent_validation matching."""
    try:
        lat = float(row['Latitude_Degrees']); lon = float(row['Longitude_Degrees'])
        year = int(row['Date_Year']); month = int(row['Date_Month'])
        day = int(row.get('Date_Day', 15))
        if not (1 <= month <= 12):
            return None
        day = min(max(day, 1), 28)
        date = pd.Timestamp(year, month, day)
    except Exception:
        return None
    lat_idx = int(np.argmin(np.abs(lats - lat))); lon_idx = int(np.argmin(np.abs(lons - lon)))
    if abs(lats[lat_idx] - lat) > 0.5 or abs(lons[lon_idx] - lon) > 0.5:
        return None
    target_issue = date - pd.Timedelta(weeks=LEAD)
    tdiff = np.abs(times - target_issue).values.astype(np.float64)
    in_win = (np.abs(times - target_issue) <= pd.Timedelta(weeks=4)) & valid_time_mask
    if not in_win.any():
        return None
    masked = np.where(in_win, tdiff, np.inf)
    idx = int(np.argmin(masked))
    v = fmap[idx, lat_idx, lon_idx]
    if not np.isfinite(v):
        return None
    return {'pred': float(v), 'issue_t': idx, 'lat_idx': lat_idx, 'lon_idx': lon_idx}


def gcbd_analytics(preds, pb, site, label="", k_bins=5):
    """AUROC severe-vs-background, top-quintile RR, Cliff's delta, all with
    naive-observation and site-cluster bootstrap 95% CI (1000, seed 42)."""
    sev = (pb >= 20.0).astype(int)
    bg = (pb == 0).astype(int)
    a = (pb > 0).astype(int)
    mask_sb = (sev == 1) | (bg == 1)
    def auroc(l, p):
        return float(roc_auc_score(l, p))
    def rr(ps, s):
        q = np.asarray(pd.qcut(ps, k_bins, labels=False, duplicates="drop"))
        top = q == np.nanmax(q); bot = q == np.nanmin(q)
        pt = s[top].mean() if top.sum() else np.nan
        pb_ = s[bot].mean() if bot.sum() else np.nan
        return (pt / pb_) if (pb_ and pb_ > 0) else np.nan
    point = {
        "auroc_severe_vs_background": auroc(sev[mask_sb], preds[mask_sb]),
        "auroc_any_vs_background": auroc(a, preds),
        "rr_quintile_severe": rr(preds, sev),
        "cliffs_delta_severe": cliffs_delta(preds[sev == 1], preds[bg == 1]),
        "mw_p_severe": float(mannwhitneyu(preds[sev == 1], preds[bg == 1],
                                          alternative="greater").pvalue),
        "n_severe": int(sev.sum()), "n_any": int(a.sum()), "n": int(len(preds)),
    }
    site_names = np.unique(site)
    site_to_idx = {s: np.where(site == s)[0] for s in site_names}
    def boot_idx(rng, mode):
        if mode == "naive":
            return rng.integers(0, len(preds), len(preds))
        pick = rng.integers(0, len(site_names), len(site_names))
        return np.concatenate([site_to_idx[site_names[s]] for s in pick])
    def one(rng, mode):
        idx = boot_idx(rng, mode)
        p, s, b, aa = preds[idx], sev[idx], bg[idx], a[idx]
        if len(np.unique(p)) < 2 or s.sum() == 0 or b.sum() == 0:
            return None
        try:
            return {"roc_sb": auroc(s[(s == 1) | (b == 1)], p[(s == 1) | (b == 1)]),
                    "cd": cliffs_delta(p[s == 1], p[b == 1]),
                    "rr": rr(p, s)}
        except ValueError:
            return None
    def run(mode):
        rng = np.random.default_rng(BOOT_SEED)
        rows = [one(rng, mode) for _ in range(N_BOOT)]
        rows = [r for r in rows if r is not None]
        def ci(key):
            v = np.array([r[key] for r in rows if np.isfinite(r[key])])
            if v.size < 100:
                return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": int(v.size)}
            return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                    "hi": float(np.percentile(v, 97.5)), "n": int(v.size)}
        return {k: ci(k) for k in ("roc_sb", "cd", "rr")}
    return {"point": point, "naive_obs_ci": run("naive"),
            "site_cluster_ci": run("cluster")}


if True:
    print("\n[P0-1] GCBD model-wise enrichment (record set = step9) ...")
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=LEAD, dataset_file=DATASET_FILE, **SCS_SPLIT)
    H, W = ds.n_lat, ds.n_lon
    persa = ds.y_regression_full.astype(np.float32).copy()
    m_tscf = restore_tscf().to(DEVICE)
    m_tsar = restore_tsar(CKPT_TSAR).to(DEVICE)
    tscf_map = predict_sliding(m_tscf, ds, DEVICE, MODEL_CFG)
    yf, gt, rr = infer_tsar(m_tsar, ds, DEVICE)
    tsar_map = persa + gt * rr * TSAR_SCALE          # scale=0.4 formal
    del yf
    torch.cuda.empty_cache()

    dnc = xr.open_dataset(Path(str(DATA_DIR)) / DATASET_FILE)
    lats = dnc['lat'].values.astype(float); lons = dnc['lon'].values.astype(float)
    times = pd.DatetimeIndex(dnc['time'].values); dnc.close()
    valid_time_mask = (times.year >= 2016) & (times.year <= 2019)

    df = pd.read_csv(GCBD_PATH, low_memory=False)
    df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')

    results_p01 = {}
    for name, fmap in (("TSCF", tscf_map), ("PersA", persa), ("TSAR", tsar_map)):
        recs = []
        for _, row in df.iterrows():
            m = match_gcbd(row, fmap, lats, lons, times, valid_time_mask)
            if m is None:
                continue
            pb = row['Percent_Bleaching']
            if pd.isna(pb) or not (0 <= pb <= 100):
                continue
            recs.append({"pred": m['pred'], "pb": float(pb),
                         "Site_ID": str(row.get('Site_ID')) if pd.notna(row.get('Site_ID')) else f"__s{len(recs)}"})
        co = pd.DataFrame(recs)
        preds = co['pred'].values.astype(float); pb = co['pb'].values.astype(float)
        site = co['Site_ID'].values
        analy = gcbd_analytics(preds, pb, site, label=name)
        n_sev = int((pb >= 20).sum()); n_any = int((pb > 0).sum()); n_bg = int((pb == 0).sum())
        print(f"  [{name}] n={len(co)} any={n_any} sev={n_sev} bg={n_bg} "
              f"AUROC(sb)={analy['point']['auroc_severe_vs_background']:.4f} "
              f"RR={analy['point']['rr_quintile_severe']:.2f} "
              f"δ={analy['point']['cliffs_delta_severe']:.3f}")
        results_p01[name] = {"cohort": {"n": int(len(co)), "any": n_any,
                                        "severe": n_sev, "background": n_bg},
                             **analy}
    with open(OUT_DIR / "gcbd_model_wise_enrichment.json", "w", encoding="utf-8") as f:
        json.dump(results_p01, f, indent=2, ensure_ascii=False, default=float)

# ───────────────────────── SECTION P0-4 + P1-7: structural contrasts ─────────────
print("\n[P0-4 / P1-7] structural-ablation CIs (rec_structural_ablation.npz) ...")
z = np.load(REC_STRUCT)
ALL_C = ["A_PersA", "B_ungated", "C_anchored", "D_TSCF"]
KEY_MAP = {dom: {m: f"{dom}_{m}_pred" for m in ["A_PersA", "B_ungated", "C_anchored", "D_TSCF"]}
           for dom in ["SCS", "GBR", "CARIB"]}


def ci95(v):
    v = np.array([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"lo": np.nan, "hi": np.nan}
    return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}


def week_block(dom, preds, target, t_idx):
    """Shared-resample week-block bootstrap (1000, seed42). Returns marginal CI
    per model and paired-difference CI per contrast, for roc/pr/csi."""
    rng = np.random.default_rng(BOOT_SEED)
    weeks = np.unique(t_idx)
    n_w = len(weeks)
    tgt_wk = {w: target[t_idx == w] for w in weeks}
    model_wk = {m: {w: preds[m][t_idx == w] for w in weeks} for m in ALL_C}
    keys = ["roc", "pr", "csi"]
    reps = {m: {k: [] for k in keys} for m in ALL_C}
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_w, n_w)
        for m in ALL_C:
            p = np.concatenate([model_wk[m][weeks[w]] for w in pick])
            t = np.concatenate([tgt_wk[weeks[w]] for w in pick])
            l = (t >= SEV).astype(int)
            roc, pr = (float(np.nan), float(np.nan))
            if len(np.unique(l)) >= 2:
                roc = float(roc_auc_score(l, p)); pr = float(average_precision_score(l, p))
            h = int(np.sum((p >= 0.9) & (t >= SEV)))
            mi = int(np.sum((p < 0.9) & (t >= SEV)))
            fa = int(np.sum((p >= 0.9) & (t < SEV)))
            csi = h / (h + mi + fa) if (h + mi + fa) else np.nan
            reps[m]["roc"].append(roc); reps[m]["pr"].append(pr); reps[m]["csi"].append(csi)
    marginal = {m: {k: ci95(reps[m][k]) for k in keys} for m in ALL_C}
    contrasts = [("C_anchored", "A_PersA"), ("B_ungated", "A_PersA"),
                 ("C_anchored", "B_ungated"), ("D_TSCF", "A_PersA"),
                 ("C_anchored", "D_TSCF")]
    paired = {}
    for a, b in contrasts:
        paired[f"{a}_-_{b}"] = {k: ci95([reps[a][k][r] - reps[b][k][r] for r in range(N_BOOT)])
                                for k in keys}
    point = {m: {} for m in ALL_C}
    for m in ALL_C:
        p, t = preds[m], target
        l = (t >= SEV).astype(int)
        try:
            point[m]["roc"] = float(roc_auc_score(l, p))
        except ValueError:
            point[m]["roc"] = np.nan
        try:
            point[m]["pr"] = float(average_precision_score(l, p))
        except Exception:
            point[m]["pr"] = np.nan
        h = int(np.sum((p >= 0.9) & (t >= SEV)))
        mi = int(np.sum((p < 0.9) & (t >= SEV)))
        fa = int(np.sum((p >= 0.9) & (t < SEV)))
        point[m]["csi"] = h / (h + mi + fa) if (h + mi + fa) else np.nan
    return {"point": point, "marginal_ci": marginal, "paired_ci": paired, "n": int(len(target))}


results_p04 = {"domains": {}}
results_p17 = {}
for dom in ["SCS", "GBR", "CARIB"]:
    target = z[f"{dom}_target"]
    t_idx = z[f"{dom}_A_PersA_t"]
    preds = {m: z[KEY_MAP[dom][m]] for m in ALL_C}
    wb = week_block(dom, preds, target, t_idx)
    results_p04["domains"][dom] = wb
    print(f"\n  [{dom}] n={wb['n']}")
    for m in ALL_C:
        pt = wb["point"][m]
        mg = wb["marginal_ci"][m]
        print(f"    {m:10s} ROC {pt['roc']:.4f} [{mg['roc']['lo']:.4f},{mg['roc']['hi']:.4f}]  "
              f"PR {pt['pr']:.4f} [{mg['pr']['lo']:.4f},{mg['pr']['hi']:.4f}]  "
              f"CSI {pt['csi']:.4f} [{mg['csi']['lo']:.4f},{mg['csi']['hi']:.4f}]")
    if dom == "CARIB":
        for cname in ("C_anchored_-_A_PersA", "B_ungated_-_A_PersA"):
            pd_ = wb["paired_ci"][cname]
            pt_c, pt_a = wb["point"]["C_anchored"], wb["point"]["A_PersA"]
            if cname.startswith("B"):
                pt_a = wb["point"]["B_ungated"]
            print(f"    {cname}: ΔROC {pt_c['roc']-pt_a['roc']:+.4f} "
                  f"[{pd_['roc']['lo']:+.4f},{pd_['roc']['hi']:+.4f}]  "
                  f"ΔPR [{pd_['pr']['lo']:+.4f},{pd_['pr']['hi']:+.4f}]  "
                  f"ΔCSI [{pd_['csi']['lo']:+.4f},{pd_['csi']['hi']:+.4f}]")
    # ---- P1-7 CARIB spatial-cluster bootstrap (C-A) ----
    if dom == "CARIB":
        i_idx = z[f"{dom}_i"]; j_idx = z[f"{dom}_j"]
        pc = preds["C_anchored"]; pa = preds["A_PersA"]; tg = target
        cells = np.unique(np.stack([i_idx.astype(int), j_idx.astype(int)], axis=1), axis=0)
        cell_to_idx = {tuple(c): np.where((i_idx.astype(int) == c[0]) & (j_idx.astype(int) == c[1]))[0]
                       for c in cells}
        rng = np.random.default_rng(BOOT_SEED)
        def sclust_metric(pc_, pa_, tg_):
            l = (tg_ >= SEV).astype(int)
            try:
                roc_c = roc_auc_score(l, pc_); roc_a = roc_auc_score(l, pa_)
            except ValueError:
                roc_c = roc_a = np.nan
            def csi(p_):
                h = int(np.sum((p_ >= 0.9) & (tg_ >= SEV)))
                mi = int(np.sum((p_ < 0.9) & (tg_ >= SEV)))
                fa = int(np.sum((p_ >= 0.9) & (tg_ < SEV)))
                return h / (h + mi + fa) if (h + mi + fa) else np.nan
            def pr(p_):
                try:
                    return average_precision_score(l, p_)
                except Exception:
                    return np.nan
            return {"droc": roc_c - roc_a, "dpr": pr(pc_) - pr(pa_),
                    "dcsi": csi(pc_) - csi(pa_)}
        rows = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(cells), len(cells))
            idx = np.concatenate([cell_to_idx[tuple(cells[s])] for s in pick])
            rows.append(sclust_metric(pc[idx], pa[idx], tg[idx]))
        res = {k: ci95([r[k] for r in rows]) for k in ("droc", "dpr", "dcsi")}
        pt = sclust_metric(pc, pa, tg)
        results_p17 = {"n_sites": int(len(cells)),
                       "point": {k: float(pt[k]) for k in ("droc", "dpr", "dcsi")},
                       "spatial_cluster_ci": res, "n_boot": N_BOOT, "seed": BOOT_SEED}
        print(f"  [P1-7] CARIB C-A spatial-cluster (grid-cell, {len(cells)} cells) "
              f"ΔROC {pt['droc']:+.4f} [{res['droc']['lo']:+.4f},{res['droc']['hi']:+.4f}]  "
              f"ΔPR [{res['dpr']['lo']:+.4f},{res['dpr']['hi']:+.4f}]  "
              f"ΔCSI [{res['dcsi']['lo']:+.4f},{res['dcsi']['hi']:+.4f}]")

with open(OUT_DIR / "structural_marginal_paired_ci.json", "w", encoding="utf-8") as f:
    json.dump(results_p04, f, indent=2, ensure_ascii=False, default=float)
with open(OUT_DIR / "carib_ca_spatial_cluster.json", "w", encoding="utf-8") as f:
    json.dump(results_p17, f, indent=2, ensure_ascii=False)

# ───────────────────────── SECTION P1-6: CARIB env permutation ─────────────
print("\n[P1-6] CARIB environment-channel spatial permutation (Step-8/S5 protocol) ...")
rec_c = np.load(REC_CARIB)
carib_t = rec_c['t'].astype(np.int64); carib_i = rec_c['i'].astype(np.int64)
carib_j = rec_c['j'].astype(np.int64); carib_target = rec_c['target']
base_pred = rec_c['tscf'].astype(np.float32)      # FULL_FROZEN CARIB
def full_metrics_raw(pred, target):
    l = (target >= SEV).astype(int)
    try:
        roc = float(roc_auc_score(l, pred))
    except ValueError:
        roc = np.nan
    try:
        pr = float(average_precision_score(l, pred))
    except Exception:
        pr = np.nan
    h = int(np.sum((pred >= 0.9) & (target >= SEV)))
    mi = int(np.sum((pred < 0.9) & (target >= SEV)))
    fa = int(np.sum((pred >= 0.9) & (target < SEV)))
    csi = h / (h + mi + fa) if (h + mi + fa) else np.nan
    return {"roc": roc, "pr": pr, "csi": csi}

ds_carib = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                           lead_weeks=LEAD, dataset_file=CARIB_FILE, **CARIB_SPLIT)
H, W = ds_carib.n_lat, ds_carib.n_lon
n_dates = ds_carib.n_dates
lm = ds_carib.land_mask
assert ds_carib.dyn_feature_names == DATASET_ORDER or list(ds_carib.dyn_feature_names) == DATASET_ORDER, \
    f"channel order mismatch: {ds_carib.dyn_feature_names}"
base_m = full_metrics_raw(base_pred, carib_target)
print(f"  CARIB baseline (FULL_FROZEN): n={len(carib_target)} "
      f"ROC {base_m['roc']:.4f} PR {base_m['pr']:.4f} CSI {base_m['csi']:.4f}")

Xf = np.ascontiguousarray(ds_carib.X_dynamic_full, dtype=np.float32)
model = restore_tscf().to(DEVICE)
perm_rng = np.random.default_rng(PERM_SEED)
d_roc, d_pr, d_csi = [], [], []
for k in range(N_PERM):
    perm = perm_rng.permutation(H * W)
    pi, pj = np.unravel_index(perm, (H, W))
    Xp = Xf.copy()
    env_all = Xf[..., ENV_IDX]
    Xp[..., ENV_IDX] = env_all[:, pi, pj, :].reshape(n_dates, H, W, len(ENV_IDX))
    keep = [c for c in range(13) if c not in ENV_IDX]
    assert np.array_equal(Xp[..., keep], Xf[..., keep], equal_nan=True)
    fmap = predict_from_X(model, Xp, lm, H, W, 13, DEVICE, MODEL_CFG)
    pp = fmap[carib_t, carib_i, carib_j]
    r = full_metrics_raw(pp, carib_target)
    d_roc.append(base_m["roc"] - r["roc"]); d_pr.append(base_m["pr"] - r["pr"])
    d_csi.append(base_m["csi"] - r["csi"])
    if (k + 1) % 20 == 0:
        print(f"  perm {k+1}/{N_PERM}")
def pci(v):
    v = np.array(v, dtype=float)
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5))}
results_p16 = {"domain": "CARIB", "base_metrics": base_m,
               "permuted": "env/calendar channels (spatial permutation, time-shared)",
               "n_perm": N_PERM, "seed": PERM_SEED,
               "d_roc": pci(d_roc), "d_pr": pci(d_pr), "d_csi09": pci(d_csi),
               "n_severe": int((carib_target >= SEV).sum())}
with open(OUT_DIR / "carib_env_permutation.json", "w", encoding="utf-8") as f:
    json.dump(results_p16, f, indent=2, ensure_ascii=False, default=float)
print(f"  CARIB env-perm ΔROC {results_p16['d_roc']['mean']:+.4f} "
      f"[{results_p16['d_roc']['lo']:+.4f},{results_p16['d_roc']['hi']:+.4f}]; "
      f"ΔPR {results_p16['d_pr']['mean']:+.4f} "
      f"[{results_p16['d_pr']['lo']:+.4f},{results_p16['d_pr']['hi']:+.4f}]; "
      f"ΔCSI@0.9 {results_p16['d_csi09']['mean']:+.4f} "
      f"[{results_p16['d_csi09']['lo']:+.4f},{results_p16['d_csi09']['hi']:+.4f}]")

with open(OUT_DIR / "review2_revision_protocol.json", "w", encoding="utf-8") as f:
    json.dump({
        "experiment": "review2_revision_evidence",
        "generated": datetime.now().isoformat(),
        "device": str(DEVICE),
        "tscf_ckpt_md5": md5_file(CKPT_TSCF), "tsar_ckpt_md5": md5_file(CKPT_TSAR),
        "note": "reuses frozen records (rec_structural_ablation.npz, rec_CARIB.npz); "
                "writes only under results/review2_revision/",
        "p01_gcbd": "R5/step9 matching gate; N_BOOT=1000 seed42; site-cluster bootstrap",
        "p04": "week-block bootstrap 1000 seed42 shared matrix",
        "p16": "Step-8/S5 env spatial perm, K=100 seed42",
        "p17": "CARIB C-A grid-cell spatial-cluster bootstrap 1000 seed42",
    }, f, indent=2, ensure_ascii=False)

print("\nDone. Outputs ->", OUT_DIR)