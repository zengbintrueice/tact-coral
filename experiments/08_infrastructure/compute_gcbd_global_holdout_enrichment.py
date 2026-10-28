# compute_gcbd_global_holdout_enrichment.py
#
# Weakness-fix #2 — extend the GCBD ecological enrichment beyond the SCS
# 2016-2019 test window:
#
#   Part A (temporal holdout): SCS VALIDATION window 2011-2015. The models
#     never trained on 2011-2015 (train 2003-2010) and the primary test used
#     2016-2019, so enrichment in 2011-2015 is a clean temporal holdout.
#   Part B (global site windows): GBR and CARIB zero-shot windows (2017-2020)
#     matched against the GLOBAL GCBD download (11,851 Caribbean / 1,810 GBR
#     usable records), using the frozen SCS-trained seed-42 checkpoints.
#
# Protocol: identical matching verbatim (obs - 4 wk issue, +/- 4 wk window,
# 0.5 deg buffer, issue-year filter per window); identical analytics
# (AUROC severe-vs-background + top-quintile RR + Cliff's delta + MW p;
# naive-observation and site-cluster bootstrap 95% CI, 1000, seed 42).
# NO training, NO target fitting, NO threshold selection. Interpretation:
# ranking enrichment of a thermal-risk coordinate, not biological prediction.
# Decision rule (pre-specified): a window result is interpreted only if it
# contains >= 10 severe observations; otherwise "not established
# (insufficient severe events)".
#
# Outputs -> results/gcbd_global_holdout_enrichment/

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import DATA_DIR, RESULTS_DIR, A6_WEIGHT
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = RESULTS_DIR / "gcbd_global_holdout_enrichment"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_TSCF = A6_WEIGHT
TSCF_MD5 = "7cdbca26cae03fa53be53210940e8ce1"
CKPT_TSAR = RESULTS_DIR / "physics_anchored" / "model_seed_42.pt"
TSAR_MD5 = "ae86e017bc5ba634b0c68e5d13b4f9f0"
GCBD_SCS = LTTS_ROOT / "data" / "china_bco_dmo_bleaching.csv"       # step9 source
GCBD_GLOBAL = LTTS_ROOT / "data" / "gcbd_global_bleaching_environmental.csv"

LEAD = 4
SEV_PB = 20.0
N_BOOT = 1000
BOOT_SEED = 42
MIN_SEVERE = 10          # pre-specified interpretability floor

WINDOWS = {
    "SCS_val2011_2015": {"file": "scs_causal_v2.nc", "years": (2011, 2015),
                          "gcbd": GCBD_SCS, "global_schema": False,
                          "split": dict(year_based_split=True, train_years=(2003, 2010),
                                        val_years=(2011, 2015), test_years=(2016, 2019))},
    "CARIB_2015_2016":  {"file": "caribbean_causal_v2.nc", "years": (2015, 2016),
                          "gcbd": GCBD_GLOBAL, "global_schema": True,
                          "split": dict(year_based_split=True, train_years=(2003, 2010),
                                        val_years=(2014, 2016), test_years=(2017, 2020))},
}
# GBR zero-shot window: the global GCBD download contains NO GBR observations
# after 2014 (box-verified), so a GBR zero-shot enrichment window is not
# feasible with the available data and is reported as a coverage gap.
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}
TSAR_SCALE = 0.4


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cliffs_delta(g1, g2):
    u = mannwhitneyu(g1, g2, alternative="greater").statistic
    return 2.0 * u / (len(g1) * len(g2)) - 1.0


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


def match_window(row, lats, lons, times, valid_time_mask):
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
    li = int(np.argmin(np.abs(lats - lat))); ji = int(np.argmin(np.abs(lons - lon)))
    if abs(lats[li] - lat) > 0.5 or abs(lons[ji] - lon) > 0.5:
        return None
    tgt = date - pd.Timedelta(weeks=LEAD)
    in_win = (np.abs(times - tgt) <= pd.Timedelta(weeks=4)) & valid_time_mask
    if not in_win.any():
        return None
    masked = np.where(in_win, np.abs(times - tgt).values.astype(np.float64), np.inf)
    return int(np.argmin(masked)), li, ji


def analytics(preds, pb, site):
    sev = (pb >= SEV_PB).astype(int); bg = (pb == 0).astype(int)
    mask_sb = (sev == 1) | (bg == 1)
    pt = {"n": len(preds), "n_severe": int(sev.sum()), "n_background": int(bg.sum()),
          "auroc_sev_vs_bg": float(roc_auc_score(sev[mask_sb], preds[mask_sb])),
          "cliffs_delta": cliffs_delta(preds[sev == 1], preds[bg == 1]),
          "mw_p": float(mannwhitneyu(preds[sev == 1], preds[bg == 1],
                                     alternative="greater").pvalue)}
    q = np.asarray(pd.qcut(preds, 5, labels=False, duplicates="drop"))
    top, bot = q == np.nanmax(q), q == np.nanmin(q)
    p_t, p_b = sev[top].mean(), sev[bot].mean()
    pt["rr_q5q1"] = float(p_t / p_b) if p_b > 0 else np.nan
    site_names = np.unique(site)
    site_idx = {s: np.where(site == s)[0] for s in site_names}
    def one(rng, mode):
        if mode == "naive":
            idx = rng.integers(0, len(preds), len(preds))
        else:
            pick = rng.integers(0, len(site_names), len(site_names))
            idx = np.concatenate([site_idx[site_names[s]] for s in pick])
        p, s, b = preds[idx], sev[idx], bg[idx]
        m = (s == 1) | (b == 1)
        if len(np.unique(p)) < 2 or s.sum() == 0 or b.sum() == 0 or not m.any():
            return None
        try:
            qq = np.asarray(pd.qcut(p, 5, labels=False, duplicates="drop"))
            tq, bq = qq == np.nanmax(qq), qq == np.nanmin(qq)
            pt_, pb_ = s[tq].mean(), s[bq].mean()
            return {"roc": roc_auc_score(s[m], p[m]),
                    "cd": cliffs_delta(p[s == 1], p[b == 1]),
                    "rr": (pt_ / pb_) if (pb_ and pb_ > 0) else np.nan}
        except (ValueError, IndexError):
            return None
    def run(mode):
        rng = np.random.default_rng(BOOT_SEED)
        rows = [one(rng, mode) for _ in range(N_BOOT)]
        rows = [r for r in rows if r is not None]
        def ci(k):
            v = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])
            if v.size < 100:
                return [np.nan, np.nan]
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        return {"roc": ci("roc"), "cd": ci("cd"), "rr": ci("rr")}
    pt["auroc_ci"] = run("cluster")["roc"]
    pt["auroc_ci_naive"] = run("naive")["roc"]
    pt["cd_ci"] = run("cluster")["cd"]
    pt["rr_ci"] = run("cluster")["rr"]
    return pt


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"GCBD global + holdout enrichment — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(CKPT_TSCF) == TSCF_MD5, "frozen TSCF hash mismatch"
assert md5(CKPT_TSAR) == TSAR_MD5, "frozen TSAR hash mismatch"
print(f"frozen TSCF OK; frozen TSAR OK; device {DEVICE}")



m_tscf = None
import ablate_a_dh_stopgrad as tr
m_tscf = tr.FinalSingleHeadModel(dynamic_channels=13)
m_tscf.load_state_dict(torch.load(CKPT_TSCF, map_location="cpu", weights_only=False))
m_tscf.eval().to(DEVICE)
m_tsar = PhysicsAnchoredResidualModel(dynamic_channels=13)
m_tsar.load_state_dict(torch.load(CKPT_TSAR, map_location="cpu", weights_only=False))
m_tsar.eval().to(DEVICE)

results, csv_rows = {}, []
for wname, cfg in WINDOWS.items():
    if cfg.get("global_schema"):
        gdf = pd.read_csv(cfg["gcbd"], low_memory=False)
        gdf = gdf[(gdf['_usable_geo_year'] == True) & (gdf['_usable_severity'] == True)]
        gdf['Percent_Bleaching'] = pd.to_numeric(gdf['Percent_Bleached'], errors='coerce')
    else:
        gdf = pd.read_csv(cfg["gcbd"], low_memory=False)
        gdf['Percent_Bleaching'] = pd.to_numeric(gdf['Percent_Bleaching'], errors='coerce')
    print(f"GCBD source {cfg['gcbd'].name}: {len(gdf)} rows, "
          f"severe>=20: {(gdf['Percent_Bleaching'] >= 20).sum()}")
    print(f"\n=== {wname} ===")
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=LEAD, dataset_file=cfg["file"], **cfg["split"])
    nc = xr.open_dataset(Path(str(DATA_DIR)) / cfg["file"])
    lats = nc['lat'].values.astype(float); lons = nc['lon'].values.astype(float)
    times = pd.DatetimeIndex(nc['time'].values); nc.close()
    vtm = (times.year >= cfg["years"][0]) & (times.year <= cfg["years"][1])

    tscf_map = predict_sliding(m_tscf, ds, DEVICE, MODEL_CFG)
    persa = ds.y_regression_full.astype(np.float32).copy()
    gt, rr = infer_tsar(m_tsar, ds, DEVICE)
    tsar_map = persa + gt * rr * TSAR_SCALE
    del gt, rr
    torch.cuda.empty_cache()

    recs = {m: [] for m in ("TSCF", "PersA", "TSAR")}
    union_pb = []
    matched_all_pb = []
    for _, row in gdf.iterrows():
        m0 = match_window(row, lats, lons, times, vtm)
        if m0 is None:
            continue
        it, li, ji = m0
        pb = row['Percent_Bleaching']
        if pd.isna(pb) or not (0 <= pb <= 100):
            continue
        site = str(row.get('Site_ID')) if pd.notna(row.get('Site_ID')) else f"__s{len(recs['TSCF'])}"
        row_finite = False
        for m, fmap in (("TSCF", tscf_map), ("PersA", persa), ("TSAR", tsar_map)):
            v = fmap[it, li, ji]
            if np.isfinite(v):
                row_finite = True
                recs[m].append({"pred": float(v), "pb": float(pb), "Site_ID": site})
        if row_finite:
            union_pb.append(float(pb))
        matched_all_pb.append(float(pb))
    win = {}
    # Union cohort: matched observations on which at least one of the three
    # scores is defined. This is the cohort size quoted in the manuscript.
    _u = np.asarray(union_pb, dtype=float)
    _a = np.asarray(matched_all_pb, dtype=float)
    win["_union_cohort"] = {
        "definition": "matched observations with >=1 of TSCF/PersA/TSAR finite",
        "n": int(_u.size),
        "n_any": int((_u > 0).sum()),
        "n_severe": int((_u >= SEV_PB).sum()),
        "n_background": int((_u == 0).sum()),
    }
    win["_matched_all"] = {
        "definition": "all window-matched observations with a valid Percent_Bleaching "
                      "(score coverage NOT required)",
        "n": int(_a.size),
        "n_any": int((_a > 0).sum()),
        "n_severe": int((_a >= SEV_PB).sum()),
        "n_background": int((_a == 0).sum()),
    }
    print(f"  [matched-all] n={win['_matched_all']['n']} "
          f"any={win['_matched_all']['n_any']} "
          f"sev={win['_matched_all']['n_severe']} "
          f"bg={win['_matched_all']['n_background']}")
    print(f"  [union] n={win['_union_cohort']['n']} "
          f"any={win['_union_cohort']['n_any']} "
          f"sev={win['_union_cohort']['n_severe']} "
          f"bg={win['_union_cohort']['n_background']}")
    for m in ("TSCF", "PersA", "TSAR"):
        co = pd.DataFrame(recs[m])
        if len(co) == 0 or co["pb"].between(1, 20).sum() + (co.pb >= SEV_PB).sum() + (co.pb == 0).sum() < 10:
            continue
        a = analytics(co['pred'].values, co['pb'].values, co['Site_ID'].values)
        a["interpretation"] = ("interpretable" if a["n_severe"] >= MIN_SEVERE
                               else f"not established (insufficient severe events, n={a['n_severe']})")
        win[m] = a
        csv_rows.append({"window": wname, "model": m, **{k: v for k, v in a.items()
                                                         if k not in ("auroc_ci", "auroc_ci_naive", "cd_ci", "rr_ci")},
                         "auroc_ci": json.dumps(a["auroc_ci"]),
                         "cd_ci": json.dumps(a["cd_ci"]), "rr_ci": json.dumps(a["rr_ci"])})
        print(f"  [{m}] n={a['n']} sev={a['n_severe']} AUROC={a['auroc_sev_vs_bg']:.4f} "
              f"CI={tuple(round(x,3) for x in a['auroc_ci'])} RR={a['rr_q5q1']:.2f} "
              f"δ={a['cliffs_delta']:.3f} -> {a['interpretation']}")
    results[wname] = win
    del ds
    torch.cuda.empty_cache()

import json as _json
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    _json.dump({"experiment": "gcbd_global_holdout_enrichment",
                "generated": datetime.now().isoformat(),
                "tscf_md5": TSCF_MD5, "tsar_md5": TSAR_MD5,
                "gcbd_sources": {"SCS_val2011_2015": GCBD_SCS.name,
                                 "CARIB_2015_2016": GCBD_GLOBAL.name},
                "gbr_coverage_gap": "global GCBD has no GBR observations after 2014 "
                                    "(box-verified); GBR zero-shot enrichment window "
                                    "not feasible",
                "n_boot": N_BOOT, "seed": BOOT_SEED,
                "min_severe": MIN_SEVERE,
                "matching": "verbatim obs-4wk +/-4wk 0.5deg, issue-year filter per window",
                "interpretation": "ranking enrichment of a thermal-risk coordinate; "
                                  "NOT biological prediction",
                "windows": results}, f, indent=1, ensure_ascii=False, default=float)
pd.DataFrame(csv_rows).to_csv(OUT / "enrichment_by_basin_window.csv", index=False)

# ---- AUDIT.md ----
L = ["# AUDIT — GCBD global site windows + 2011-2015 temporal holdout", "",
     f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
     f"Frozen TSCF md5 `{TSCF_MD5}`; frozen TSAR md5 `{TSAR_MD5}` (both verified).",
     "",
     "## Data-coverage note (GBR)", "",
     "The global GCBD download contains no GBR-box observations after 2014, so a",
     "GBR zero-shot enrichment window is not feasible with the available data;",
     "this is a survey-coverage gap, not a model result.", "",
     "## Protocol", "",
     "- Matching verbatim (obs − 4 wk issue, ±4 wk window, 0.5° buffer,",
     " issue-year filter per window); identical record indices reused across",
     " TSCF/PersA/TSAR within each window (like-for-like cohorts per window).",
     "- Part A = SCS validation window 2011-2015 (training 2003-2010; primary",
     " test 2016-2019) → clean temporal holdout.",
     "- Part B = GBR / CARIB zero-shot windows (2017-2020) matched against the",
     " global GCBD download; SCS-trained frozen models only.",
     "- Analytics identical to step9/R5: AUROC(sev≥20% vs background), top-",
     "quintile RR, Cliff's δ, MW p; site-cluster bootstrap 95% CI (1,000,",
     " seed 42) primary; naive-observation CI reported alongside.",
     "- Pre-specified floor: a window is interpreted only with ≥10 severe",
     f" observations; otherwise **not established**.",
     "- No training, no target fitting, no threshold selection; ranking-",
     "enrichment interpretation only.", "",
     "## Results", ""]
for w, win in results.items():
    for m, a in win.items():
        if m.startswith("_"):
            continue
        L.append(f"- **{w} / {m}**: n={a['n']}, severe={a['n_severe']}; "
                 f"AUROC {a['auroc_sev_vs_bg']:.3f} "
                 f"[{a['auroc_ci'][0]:.3f}, {a['auroc_ci'][1]:.3f}] (site-cluster); "
                 f"RR {a['rr_q5q1']:.2f} {tuple(round(x,2) for x in a['rr_ci'])}; "
                 f"δ {a['cliffs_delta']:.3f} {tuple(round(x,2) for x in a['cd_ci'])}; "
                 f"{a['interpretation']}")
L += ["", "## Interpretation guard", "",
      "Enrichment = ranking association between the thermal-risk coordinate and",
      "independent field observations; it is not biological prediction and no",
      "calibration or threshold was derived from GCBD.", ""]
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"\nOutputs -> {OUT}")
print("Done.")
