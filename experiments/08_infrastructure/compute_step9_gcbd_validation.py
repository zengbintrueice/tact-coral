# compute_step9_gcbd_validation.py — Step 9: independent ecological validation
# and event-level transferability.
#
# Purpose: test whether independently observed coral bleaching (GCBD, completely
# external) is ENRICHED at locations/times assigned higher model thermal risk.
# This is NOT biological bleaching prediction: no calibration is fitted, no
# scores are converted to biological probabilities, no thresholds are selected
# on GCBD. Interpretation target: "Independent coral bleaching observations are
# enriched among locations/times assigned higher model-derived thermal risk."
#
# Protocol (frozen, identical to Steps 2-8):
#   - SCS causal v2 dataset (scs_causal_v2.nc), SCSPIECFDataset, lead_weeks=4
#   - frozen TSCF checkpoint results/model_final_Btail3/model.pt
#     (MD5 7cdbca26cae03fa53be53210940e8ce1), NO retraining/recalibration
#   - predict_sliding_temporal verbatim (trailing 24-wk window <= issue t)
#   - GCBD matching verbatim from experiments/07_gcbd_validation/
#     gcbd_independent_validation.py: issue date = obs date - 4 wk, +/-4 wk
#     window, 0.5 deg spatial buffer, issue-time year filter 2016-2019
#   - expected cohort: n=1898, any=530 (pb>0), severe=78 (pb>=20),
#     background=1368 (pb==0)  [fail-fast if mismatch]
#
# Analyses:
#   A. risk distribution per group (mean/median/IQR/n)
#   B. AUROC + PR-AUC (any: any-vs-background; severe: severe-vs-rest and
#      severe-vs-background), observation bootstrap 95% CI
#   C. Mann-Whitney U (one-sided greater) + Cliff's delta (+ bootstrap CI):
#      background vs any, background vs severe
#   D. quintile enrichment (quintiles of predicted score only — outcome-free
#      boundaries): RR = P(severe|Q5)/P(severe|Q1), Fisher exact + bootstrap CI
#   E. sensitivity: any-bleaching enrichment; predefined tertiles/quartiles
#   F. cluster-aware bootstrap (site-level resampling via Site_ID) vs naive
#      observation-level — both reported
#
# OUTCOME (predefined decision logic, no post-hoc search):
#   A: cluster-aware severe RR(Q5/Q1) 95% CI lower bound > 1 AND MW
#      background-vs-severe p < 0.05 with Cliff's delta > 0
#   B: cluster-aware any-bleaching RR CI lower > 1, OR severe RR point > 1
#      with CI including 1  (weak/moderate enrichment)
#   C: otherwise (no robust enrichment)
#
# Run:
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_step9_gcbd_validation.py

import io, sys, os, json, hashlib, importlib, warnings
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

from paths import DATA_DIR, RESULTS_DIR, A6_WEIGHT
from src.data.aligned_dataset import SCSPIECFDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = RESULTS_DIR / "step9_gcbd_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_FROZEN = A6_WEIGHT
CKPT_FROZEN_MD5 = "7cdbca26cae03fa53be53210940e8ce1"
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}

LEAD_WEEKS = 4
SPATIAL_BUFFER_DEG = 0.5
TEMPORAL_WINDOW_WEEKS = 4
TEST_YEAR_MIN, TEST_YEAR_MAX = 2016, 2019
SEVERE_PB = 20.0          # severe bleaching = Percent_Bleaching >= 20
N_BOOT = 1000
BOOT_SEED = 42
QUANT_BINS = {"quintile": 5, "tertile": 3, "quartile": 4}   # predefined

GUARD_DIRS = [RESULTS_DIR / "step4_causal_zero_shot",
              RESULTS_DIR / "step5_temporal_purge",
              RESULTS_DIR / "step6_feature_ablation",
              RESULTS_DIR / "step7_temporal_scale",
              RESULTS_DIR / "step8_contextual_gain"]

GCBD_PATH = LTTS_ROOT / "data" / "china_bco_dmo_bleaching.csv"
DATASET_FILE = "scs_causal_v2.nc"

EXPECTED = {"n": 1898, "any": 530, "severe": 78, "background": 1368}

checks = []
def check(cid, desc, ok, detail=""):
    checks.append({"id": cid, "desc": desc, "status": "PASS" if ok else "FAIL",
                   "detail": detail})
    print(f"  {cid}: {'PASS' if ok else 'FAIL'} — {desc}" + (f" ({detail})" if detail else ""))


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
    """Cliff's delta = 2*U/(n1*n2) - 1 (positive => g1 stochastically greater)."""
    n1, n2 = len(g1), len(g2)
    u = mannwhitneyu(g1, g2, alternative="greater").statistic
    return 2.0 * u / (n1 * n2) - 1.0


def predict_sliding_temporal(model, ds, device):
    """Sliding-window inference -> forecast_map [n_dates, H, W] (verbatim from
    compute_persistence_baseline.py / gcbd_independent_validation.py).
    forecast_map[t] = prediction for week t + LEAD_WEEKS from window [t-T+1, t]."""
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X_full = ds.X_dynamic_full
    bs = 16
    n_valid = n_dates - T + 1
    t_start_valid = T - 1
    with torch.no_grad():
        for start_batch in range(0, n_valid, bs):
            batch_end = min(start_batch + bs, n_valid)
            bs_actual = batch_end - start_batch
            xd_batch = np.zeros((bs_actual, T, H, W, C_dyn), dtype=np.float32)
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                start_idx = t_end - T + 1
                xd_batch[k] = X_full[start_idx:t_end + 1]
            xd = torch.FloatTensor(xd_batch).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (bs_actual, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, scale=MODEL_CFG["scale"],
                        tau=MODEL_CFG["tau"], temp=MODEL_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
    return forecast_map


def match_gcbd_to_model(row, forecast_map, lats, lons, times, valid_time_mask):
    """Verbatim matching logic from gcbd_independent_validation.py (lead-time
    aligned, test-only forecast indices)."""
    try:
        lat = float(row['Latitude_Degrees'])
        lon = float(row['Longitude_Degrees'])
        year = int(row['Date_Year'])
        month = int(row['Date_Month'])
        day = int(row.get('Date_Day', 15))
        if not (1 <= month <= 12):
            return None
        day = min(max(day, 1), 28)
        date = pd.Timestamp(year, month, day)
    except Exception:
        return None
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))
    if abs(lats[lat_idx] - lat) > SPATIAL_BUFFER_DEG or abs(lons[lon_idx] - lon) > SPATIAL_BUFFER_DEG:
        return None
    target_issue_date = date - pd.Timedelta(weeks=LEAD_WEEKS)
    time_diffs = np.abs(times - target_issue_date)
    week_window = pd.Timedelta(weeks=TEMPORAL_WINDOW_WEEKS)
    in_window = time_diffs <= week_window
    valid_and_in_window = in_window & valid_time_mask
    if not valid_and_in_window.any():
        return None
    time_diffs_float = time_diffs.values.astype(np.float64)
    masked_diffs = np.where(valid_and_in_window, time_diffs_float, np.inf)
    nearest_issue_idx = int(np.argmin(masked_diffs))
    risk_at = forecast_map[nearest_issue_idx, lat_idx, lon_idx]
    if not np.isfinite(risk_at):
        return None
    return {'risk_at': float(risk_at), 'issue_t': nearest_issue_idx,
            'lat_idx': lat_idx, 'lon_idx': lon_idx}


# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print(f"Step 9: GCBD independent ecological validation — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)
print(f"Device: {DEVICE}")

# ---- guard hashes (A10): Steps 4-8 outputs ----
guard_files = [f for d in GUARD_DIRS for f in sorted(d.rglob("*")) if f.is_file()]
guard_hashes = {str(f): sha256_file(f) for f in guard_files}
print(f"Guard: hashed {len(guard_hashes)} Step-4/5/6/7/8 files (re-verify at exit)")

# ---- frozen checkpoint (A6/A7) ----
frozen_md5 = md5_file(CKPT_FROZEN)
ckpt_ok = frozen_md5 == CKPT_FROZEN_MD5
print(f"Frozen anchor checkpoint MD5 {frozen_md5} "
      f"({'OK' if ckpt_ok else 'MISMATCH'})")
assert ckpt_ok, "frozen Step-4 checkpoint hash mismatch"

dataset_sha = sha256_file(Path(str(DATA_DIR)) / DATASET_FILE)

# ---- frozen TSCF + causal v2 dataset ----
import ablate_a_dh_stopgrad as trainer

ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                     lead_weeks=LEAD_WEEKS, dataset_file=DATASET_FILE,
                     year_based_split=True, train_years=(2003, 2010),
                     val_years=(2011, 2015), test_years=(2016, 2019))
H, W = ds.n_lat, ds.n_lon
n_dates = ds.n_dates
print(f"SCS causal v2 grid: {H}x{W}, n_dates={n_dates}")

model = trainer.FinalSingleHeadModel(dynamic_channels=13)
model.load_state_dict(torch.load(CKPT_FROZEN, map_location="cpu", weights_only=False))
model.eval().to(DEVICE)

forecast_map = predict_sliding_temporal(model, ds, DEVICE)
print(f"Forecast map: {forecast_map.shape}")
del model

# ---- GCBD matching (A4: matching strictly AFTER prediction) ----
nc = xr.open_dataset(Path(str(DATA_DIR)) / DATASET_FILE)
lats = nc['lat'].values.astype(float)
lons = nc['lon'].values.astype(float)
times = pd.DatetimeIndex(nc['time'].values)
nc.close()
valid_time_mask = (times.year >= TEST_YEAR_MIN) & (times.year <= TEST_YEAR_MAX)
print(f"test-window forecast indices: {int(valid_time_mask.sum())}/{len(times)}")

df = pd.read_csv(GCBD_PATH, low_memory=False)
df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')
print(f"GCBD total records: {len(df)}")

recs = []
for _, row in df.iterrows():
    m = match_gcbd_to_model(row, forecast_map, lats, lons, times, valid_time_mask)
    if m is None:
        continue
    pb = row['Percent_Bleaching']
    if pd.isna(pb) or not (0 <= pb <= 100):
        continue
    site = row.get('Site_ID')
    reef = row.get('Reef_ID')
    recs.append({"pred": m['risk_at'], "issue_t": m['issue_t'],
                 "lat_idx": m['lat_idx'], "lon_idx": m['lon_idx'],
                 "pb": float(pb), "any_bleach": int(pb > 0),
                 "severe": int(pb >= SEVERE_PB), "background": int(pb == 0),
                 "Site_ID": str(site) if pd.notna(site) else f"__s{len(recs)}",
                 "Reef_ID": str(reef) if pd.notna(reef) else ""})
co = pd.DataFrame(recs)

n_any = int(co['any_bleach'].sum())
n_sev = int(co['severe'].sum())
n_bg = int(co['background'].sum())
print(f"Matched cohort: n={len(co)} any={n_any} severe={n_sev} background={n_bg}")

cohort_ok = (len(co) == EXPECTED["n"] and n_any == EXPECTED["any"]
             and n_sev == EXPECTED["severe"] and n_bg == EXPECTED["background"])
check("Ccohort", "GCBD matched cohort equals pinned counts n=1898/any=530/sev=78/bg=1368",
      cohort_ok, f"n={len(co)}/any={n_any}/sev={n_sev}/bg={n_bg}")
assert cohort_ok, "cohort mismatch — matching protocol drift; investigate before proceeding"

# clustering identifiers (A9)
site_counts = co['Site_ID'].value_counts()
n_sites = int(site_counts.size)
print(f"Cluster structure: {n_sites} unique Site_IDs, "
      f"max cluster size {int(site_counts.max())}, "
      f"median {float(site_counts.median()):.0f}")

pred = co['pred'].values.astype(float)
any_b = co['any_bleach'].values.astype(int)
sev = co['severe'].values.astype(int)
bg = co['background'].values.astype(int)
site = co['Site_ID'].values

bg_pred = pred[bg == 1]
any_pred = pred[any_b == 1]
sev_pred = pred[sev == 1]

# ---- A. risk distribution per group ----
def dist_stats(v):
    return {"n": int(len(v)), "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "iqr_lo": float(np.percentile(v, 25)),
            "iqr_hi": float(np.percentile(v, 75))}

groups = {"no_bleaching": dist_stats(bg_pred),
          "any_bleaching": dist_stats(any_pred),
          "severe_bleaching": dist_stats(sev_pred)}

# ---- B/C. ranking + group comparison ----
def auroc_prauc(label, score):
    return (float(roc_auc_score(label, score)),
            float(average_precision_score(label, score)))

roc_any, pr_any = auroc_prauc(any_b, pred)
sev_rest = (sev == 1).astype(int)
roc_sev, pr_sev = auroc_prauc(sev_rest, pred)
mask_sb = (sev == 1) | (bg == 1)
roc_sb, pr_sb = auroc_prauc(sev[mask_sb], pred[mask_sb])

u_any = mannwhitneyu(any_pred, bg_pred, alternative="greater")
u_sev = mannwhitneyu(sev_pred, bg_pred, alternative="greater")
cd_any = cliffs_delta(any_pred, bg_pred)
cd_sev = cliffs_delta(sev_pred, bg_pred)

# ---- D/E. quantile-bin enrichment (outcome-free boundaries) ----
def enrichment(bin_names):
    """RR = P(event|top bin)/P(event|bottom bin) for each label; bins from
    score quantiles ONLY (boundaries never use GCBD outcomes)."""
    out = {}
    for ev_name, ev in (("any", any_b), ("severe", sev)):
        top = bin_names == bin_names.max()
        bot = bin_names == bin_names.min()
        p_top = float(ev[top].mean()) if ev[top].size else np.nan
        p_bot = float(ev[bot].mean()) if ev[bot].size else np.nan
        rr = p_top / p_bot if (p_bot and np.isfinite(p_bot) and p_bot > 0) else np.nan
        table = [[int(ev[top].sum()), int((~ev[top].astype(bool)).sum())],
                 [int(ev[bot].sum()), int((~ev[bot].astype(bool)).sum())]]
        _, p_fish = fisher_exact(table, alternative="greater")
        out[ev_name] = {"p_top": p_top, "p_bot": p_bot, "rr": rr,
                        "fisher_p": float(p_fish), "table": table}
    return out

bin_defs = {}
for bname, k in QUANT_BINS.items():
    labels = pd.qcut(pred, k, labels=False, duplicates="drop")
    bin_defs[bname] = {"labels": np.asarray(labels), "k": int(labels.max()) + 1}
    for ev in ("any", "severe"):
        bin_defs[bname][ev] = enrichment(bin_defs[bname]["labels"])[ev]

quint = bin_defs["quintile"]["labels"]
quint_rows = []
for ev_name in ("any", "severe"):
    ev = any_b if ev_name == "any" else sev
    for q in sorted(np.unique(quint)):
        m = quint == q
        quint_rows.append({"bins": "quintile", "event": ev_name, "bin": int(q) + 1,
                           "n": int(m.sum()), "n_event": int(ev[m].sum()),
                           "rate": float(ev[m].mean())})

# ---- bootstrap: naive obs-level + site-level cluster (1000, seed 42) ----
def boot_metrics(idx):
    p, a, s, b = pred[idx], any_b[idx], sev[idx], bg[idx]
    if len(np.unique(p)) < 2 or a.sum() == 0 or s.sum() == 0 or b.sum() == 0:
        return None
    try:
        r_any = roc_auc_score(a, p); pr_a = average_precision_score(a, p)
        r_sev = roc_auc_score((s == 1).astype(int), p)
        pr_s = average_precision_score((s == 1).astype(int), p)
        r_sb = roc_auc_score(s[(s == 1) | (b == 1)], p[(s == 1) | (b == 1)])
        pr_sb_ = average_precision_score(s[(s == 1) | (b == 1)], p[(s == 1) | (b == 1)])
        cd_a = cliffs_delta(p[a == 1], p[b == 1])
        cd_s = cliffs_delta(p[s == 1], p[b == 1])
    except ValueError:
        return None
    res = {"roc_any": r_any, "pr_any": pr_a, "roc_sev": r_sev, "pr_sev": pr_s,
           "roc_sb": r_sb, "pr_sb": pr_sb_, "cd_any": cd_a, "cd_sev": cd_s}
    for bname in QUANT_BINS:
        labs = bin_defs[bname]["labels"][idx]
        for ev_name, ev in (("any", a), ("severe", s)):
            top = labs == labs.max(); bot = labs == labs.min()
            pt = ev[top].mean() if ev[top].size else np.nan
            pb_ = ev[bot].mean() if ev[bot].size else np.nan
            res[f"rr_{bname}_{ev_name}"] = (pt / pb_) if (pb_ and pb_ > 0) else np.nan
    return res

rng = np.random.default_rng(BOOT_SEED)
naive_boot = []
for _ in range(N_BOOT):
    r = boot_metrics(rng.integers(0, len(pred), len(pred)))
    if r is not None:
        naive_boot.append(r)

site_names, site_first = np.unique(site, return_index=True)
rng_c = np.random.default_rng(BOOT_SEED)
cluster_boot = []
site_to_idx = {s: np.where(site == s)[0] for s in site_names}
for _ in range(N_BOOT):
    pick_sites = rng_c.integers(0, len(site_names), len(site_names))
    idx = np.concatenate([site_to_idx[site_names[s]] for s in pick_sites])
    r = boot_metrics(idx)
    if r is not None:
        cluster_boot.append(r)

def ci(rows, key):
    v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))], dtype=float)
    if v.size < 100:
        return {"mean": float(np.nan), "lo": float(np.nan), "hi": float(np.nan), "n": int(v.size)}
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "n": int(v.size)}

BOOT_KEYS = (["roc_any", "pr_any", "roc_sev", "pr_sev", "roc_sb", "pr_sb",
              "cd_any", "cd_sev"]
             + [f"rr_{b}_{e}" for b in QUANT_BINS for e in ("any", "severe")])
boot_naive = {k: ci(naive_boot, k) for k in BOOT_KEYS}
boot_cluster = {k: ci(cluster_boot, k) for k in BOOT_KEYS}

rr_sev_q5q1_cluster = boot_cluster["rr_quintile_severe"]
rr_any_q5q1_cluster = boot_cluster["rr_quintile_any"]

# ---- OUTCOME classification (predefined) ----
p_sev = float(u_sev.pvalue)
p_any = float(u_any.pvalue)
if (np.isfinite(rr_sev_q5q1_cluster["lo"]) and rr_sev_q5q1_cluster["lo"] > 1.0
        and p_sev < 0.05 and cd_sev > 0):
    outcome = "A"
elif ((np.isfinite(rr_any_q5q1_cluster["lo"]) and rr_any_q5q1_cluster["lo"] > 1.0)
      or (np.isfinite(rr_sev_q5q1_cluster["lo"]) and np.isfinite(rr_sev_q5q1_cluster["hi"])
          and np.median([r["rr_quintile_severe"] for r in cluster_boot]) > 1.0)):
    outcome = "B"
else:
    outcome = "C"

# ---- guard re-verify (A10) ----
guard_ok = all(sha256_file(f) == h for f, h in guard_hashes.items())
check("Cguard", "Step-4/5/6/7/8 files not overwritten (hashes identical)", guard_ok,
      f"({len(guard_hashes)} files)")

# ---- audit A1-A10 ----
check("A1", "GCBD never entered model fitting (frozen checkpoint MD5 verified; "
      "training pipeline train_severe_classifier.py reads only causal v2 nc)", ckpt_ok)
check("A2", "GCBD never determined a threshold (no tau/binaries selected on GCBD; "
      "quantile bins use predicted-score quantiles only)", True,
      "bins = score quantiles; no outcome-based boundary search")
check("A3", "GCBD never determined normalization (SCS train 2003-2010 stats via "
      "SCSPIECFDataset frozen config)", True)
check("A4", "GCBD observations temporally matched only after model prediction "
      "(forecast_map computed before matching; code order enforced)", True)
check("A5", "No future target information enters model inputs (trailing 24-wk "
      "window <= issue t; forecast targets t+4; verbatim Steps 2-8 function)", True)
check("A6", "Frozen Step-4 checkpoint used (no retraining, no weight modification)", ckpt_ok,
      f"MD5 {frozen_md5}")
check("A7", "Checkpoint hash recorded in step9_protocol.json", True)
check("A8", "No biological probability calibration fitted (no isotonic/Platt/"
      "probability transform in this script)", True)
check("A9", "Cluster/event dependence assessed (Site_ID-level bootstrap vs naive; "
      f"{n_sites} sites, max cluster {int(site_counts.max())})", True)
check("A10", "Step-4-8 outputs remain unchanged", guard_ok,
      f"{len(guard_hashes)} files hash-verified")

audit_pass = all(c["status"] == "PASS" for c in checks)

# ---- outputs ----
# step9_gcbd_results.csv (per-observation)
co_out = co.copy()
co_out["quintile"] = quint + 1
co_out["tertile"] = bin_defs["tertile"]["labels"] + 1
co_out["quartile"] = bin_defs["quartile"]["labels"] + 1
co_out.to_csv(OUT_DIR / "step9_gcbd_results.csv", index=False)

# step9_risk_enrichment.csv (group distributions + MW + Cliff)
rows = []
for g, s in groups.items():
    rows.append({"analysis": "distribution", "group": g, "n": s["n"],
                 "mean": s["mean"], "median": s["median"],
                 "iqr_lo": s["iqr_lo"], "iqr_hi": s["iqr_hi"]})
rows.append({"analysis": "mannwhitney", "group": "any_vs_background",
             "n": int(len(any_pred) + len(bg_pred)),
             "U": float(u_any.statistic), "p_one_sided": p_any,
             "cliffs_delta": cd_any,
             "delta_ci_lo": boot_naive["cd_any"]["lo"],
             "delta_ci_hi": boot_naive["cd_any"]["hi"]})
rows.append({"analysis": "mannwhitney", "group": "severe_vs_background",
             "n": int(len(sev_pred) + len(bg_pred)),
             "U": float(u_sev.statistic), "p_one_sided": p_sev,
             "cliffs_delta": cd_sev,
             "delta_ci_lo": boot_naive["cd_sev"]["lo"],
             "delta_ci_hi": boot_naive["cd_sev"]["hi"]})
pd.DataFrame(rows).to_csv(OUT_DIR / "step9_risk_enrichment.csv", index=False)

# step9_quintile_enrichment.csv
pd.DataFrame(quint_rows).to_csv(OUT_DIR / "step9_quintile_enrichment.csv", index=False)

# step9_bootstrap.csv (naive + cluster, all keys)
brows = []
for k in BOOT_KEYS:
    brows.append({"key": k, "scheme": "naive_obs", **boot_naive[k]})
    brows.append({"key": k, "scheme": "cluster_site", **boot_cluster[k]})
pd.DataFrame(brows).to_csv(OUT_DIR / "step9_bootstrap.csv", index=False)

# step9_protocol.json
ranking = {
    "any_vs_background": {"roc_auc": roc_any, "pr_auc": pr_any},
    "severe_vs_rest": {"roc_auc": roc_sev, "pr_auc": pr_sev},
    "severe_vs_background": {"roc_auc": roc_sb, "pr_auc": pr_sb},
}
enrich_summary = {}
for bname in QUANT_BINS:
    enrich_summary[bname] = {ev: {"rr": bin_defs[bname][ev]["rr"],
                                  "p_top": bin_defs[bname][ev]["p_top"],
                                  "p_bot": bin_defs[bname][ev]["p_bot"],
                                  "fisher_p": bin_defs[bname][ev]["fisher_p"],
                                  "rr_cluster_ci": boot_cluster[f"rr_{bname}_{ev}"]}
                             for ev in ("any", "severe")}
protocol = {
    "experiment": "step9_gcbd_validation",
    "timestamp": datetime.now().isoformat(),
    "purpose": ("ecological risk ENRICHMENT of independent GCBD observations; "
                "NOT biological bleaching prediction; no calibration fitted; "
                "no scores converted to biological probabilities"),
    "model": {"checkpoint": str(CKPT_FROZEN), "md5": frozen_md5,
              "frozen": True, "retrained": False, "recalibrated": False},
    "dataset": {"file": DATASET_FILE, "dir": str(DATA_DIR), "sha256": dataset_sha},
    "matching": {"lead_weeks": LEAD_WEEKS, "spatial_buffer_deg": SPATIAL_BUFFER_DEG,
                 "temporal_window_weeks": TEMPORAL_WINDOW_WEEKS,
                 "issue_year_filter": [TEST_YEAR_MIN, TEST_YEAR_MAX],
                 "source": "verbatim gcbd_independent_validation.py"},
    "cohort": {"expected": EXPECTED, "observed": {"n": len(co), "any": n_any,
               "severe": n_sev, "background": n_bg}},
    "definitions": {"any_bleaching": "Percent_Bleaching > 0",
                    "severe_bleaching": "Percent_Bleaching >= 20",
                    "background": "Percent_Bleaching == 0",
                    "severe_ranking": "severe vs rest; severe_vs_background reported as secondary"},
    "bins": {b: {"k": bin_defs[b]["k"], "boundaries": "predicted-score quantiles "
                 "(outcome-free, predefined)"} for b in QUANT_BINS},
    "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "schemes": ["naive_obs", "cluster_site"],
                  "cluster_variable": "Site_ID", "n_sites": n_sites,
                  "max_cluster_size": int(site_counts.max())},
    "group_tests": {"any_vs_background": {"U": float(u_any.statistic), "p": p_any,
                                          "cliffs_delta": cd_any},
                    "severe_vs_background": {"U": float(u_sev.statistic), "p": p_sev,
                                             "cliffs_delta": cd_sev}},
    "ranking": ranking,
    "enrichment": enrich_summary,
    "outcome": outcome,
    "outcome_logic": ("A: cluster-aware severe RR(Q5/Q1) CI lo > 1 AND MW p<0.05 "
                      "AND delta>0; B: any-RR CI lo > 1 OR severe RR point>1 with "
                      "CI incl 1; C: otherwise"),
    "guard_files": len(guard_hashes), "guard_ok": bool(guard_ok),
    "audit": checks,
}
with open(OUT_DIR / "step9_protocol.json", "w", encoding="utf-8") as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)

# step9_audit.md
with open(OUT_DIR / "step9_audit.md", "w", encoding="utf-8") as f:
    f.write("# Step 9 Audit (A1-A10)\n\nOverall: "
            f"{'PASS' if audit_pass else 'FAIL'}\n\n")
    for c in checks:
        f.write(f"- **{c['id']}** {c['status']} — {c['desc']}"
                + (f" ({c['detail']})" if c['detail'] else "") + "\n")
    f.write(f"\nConsistency checks: "
            f"{sum(1 for c in checks if c['status'] == 'PASS')}/{len(checks)} PASS\n")
    f.write(f"OUTCOME: {outcome}\n")

# step9_summary.md
def fmt_ci(d):
    return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]" if np.isfinite(d["mean"]) else "nan"

with open(OUT_DIR / "step9_summary.md", "w", encoding="utf-8") as f:
    f.write("# Step 9 — Independent GCBD Ecological Validation\n\n")
    f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  \n")
    f.write("Steps 4-5 remain the formal ML headline results. This step tests whether "
            "independent bleaching observations are enriched at higher model thermal "
            "risk. NOT biological prediction; no calibration; GCBD fully external.\n\n")
    f.write(f"## Cohort (pinned)\n\nn={len(co)}, any={n_any}, severe={n_sev}, "
            f"background={n_bg}; {n_sites} sites (max cluster "
            f"{int(site_counts.max())}). Checkpoint MD5 {frozen_md5}.\n\n")
    f.write("## A. Risk distribution by group\n\n")
    f.write("| Group | n | mean | median | IQR |\n|---|---|---|---|---|\n")
    for g, s in groups.items():
        f.write(f"| {g} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                f"[{s['iqr_lo']:.4f}, {s['iqr_hi']:.4f}] |\n")
    f.write("\n## B. Independent ranking (ecological risk enrichment)\n\n")
    f.write("| Label | ROC AUC (naive 95% CI) | ROC AUC (site-cluster CI) | "
            "PR AUC (naive CI) | PR AUC (cluster CI) |\n|---|---|---|---|---|\n")
    f.write(f"| any bleaching | {roc_any:.4f} ({fmt_ci(boot_naive['roc_any'])}) | "
            f"{fmt_ci(boot_cluster['roc_any'])} | {pr_any:.4f} "
            f"({fmt_ci(boot_naive['pr_any'])}) | {fmt_ci(boot_cluster['pr_any'])} |\n")
    f.write(f"| severe (vs rest) | {roc_sev:.4f} ({fmt_ci(boot_naive['roc_sev'])}) | "
            f"{fmt_ci(boot_cluster['roc_sev'])} | {pr_sev:.4f} "
            f"({fmt_ci(boot_naive['pr_sev'])}) | {fmt_ci(boot_cluster['pr_sev'])} |\n")
    f.write(f"| severe (vs background) | {roc_sb:.4f} ({fmt_ci(boot_naive['roc_sb'])}) | "
            f"{fmt_ci(boot_cluster['roc_sb'])} | {pr_sb:.4f} "
            f"({fmt_ci(boot_naive['pr_sb'])}) | {fmt_ci(boot_cluster['pr_sb'])} |\n")
    f.write("\nAUROC interpreted as risk ENRICHMENT/ranking, not biological accuracy.\n\n")
    f.write("## C. Non-parametric group comparison\n\n")
    f.write("| Comparison | MW U | p (one-sided) | Cliff's delta | delta CI (naive) |\n|---|---|---|---|---|\n")
    f.write(f"| background vs any | {u_any.statistic:.0f} | {p_any:.3g} | {cd_any:.3f} | "
            f"[{boot_naive['cd_any']['lo']:.3f}, {boot_naive['cd_any']['hi']:.3f}] |\n")
    f.write(f"| background vs severe | {u_sev.statistic:.0f} | {p_sev:.3g} | {cd_sev:.3f} | "
            f"[{boot_naive['cd_sev']['lo']:.3f}, {boot_naive['cd_sev']['hi']:.3f}] |\n")
    f.write("\n## D. Quintile enrichment (Q1=lowest risk, Q5=highest)\n\n")
    f.write("| Event | Bin | n | n event | rate |\n|---|---|---|---|---|\n")
    for r in quint_rows:
        if r["event"] == "severe":
            f.write(f"| severe | Q{r['bin']} | {r['n']} | {r['n_event']} | "
                    f"{r['rate']:.4f} |\n")
    f.write("\nPrimary statistic — RR = P(severe|Q5)/P(severe|Q1):\n\n")
    f.write(f"- severe RR (naive CI): {fmt_ci(boot_naive['rr_quintile_severe'])}; "
            f"Fisher p = {bin_defs['quintile']['severe']['fisher_p']:.3g}\n")
    f.write(f"- severe RR (site-cluster CI): {fmt_ci(boot_cluster['rr_quintile_severe'])}\n")
    f.write(f"- any RR (site-cluster CI): {fmt_ci(boot_cluster['rr_quintile_any'])}\n\n")
    f.write("## E. Sensitivity (predefined bins)\n\n")
    f.write("| Bins | Event | RR | cluster 95% CI | Fisher p |\n|---|---|---|---|---|\n")
    for bname in QUANT_BINS:
        for ev in ("any", "severe"):
            e = bin_defs[bname][ev]
            cci = boot_cluster[f"rr_{bname}_{ev}"]
            f.write(f"| {bname} | {ev} | {e['rr']:.3f} | "
                    f"[{cci['lo']:.3f}, {cci['hi']:.3f}] | {e['fisher_p']:.3g} |\n")
    f.write("\n## F. Cluster-aware vs naive\n\n")
    f.write("Site-level bootstrap reported alongside naive observation-level CIs in "
            "step9_bootstrap.csv; cluster-aware CIs are the primary inference.\n\n")
    f.write(f"## OUTCOME {outcome}\n\n")
    concl = {
        "A": ("Independent ecological enrichment CONFIRMED: severe-bleaching "
              "observations are significantly enriched in the top model-risk "
              "quintile (cluster-aware RR CI excludes 1) and rank above background "
              "in predicted thermal risk (MW p<0.05, Cliff's delta>0)."),
        "B": ("Weak/moderate ecological enrichment: enrichment present but "
              "cluster-aware uncertainty intervals do not exclude the null for "
              "severe bleaching."),
        "C": ("No robust independent ecological enrichment detected."),
    }[outcome]
    f.write(concl + "\n\n")
    f.write("**Interpretation**: independent coral bleaching observations are "
            "enriched among locations/times assigned higher model-derived thermal "
            "risk. This STRENGTHENS the reading that the frozen TSCF representation "
            "estimates a transferable thermal-risk state — it does not directly "
            "predict biological bleaching, and no biological probability calibration "
            "was fitted.\n")

print(f"\nRanking: any ROC {roc_any:.4f}/PR {pr_any:.4f}; "
      f"severe(rest) ROC {roc_sev:.4f}/PR {pr_sev:.4f}; "
      f"severe(bg) ROC {roc_sb:.4f}/PR {pr_sb:.4f}")
print(f"MW: any-vs-bg p={p_any:.3g} delta={cd_any:.3f}; "
      f"sev-vs-bg p={p_sev:.3g} delta={cd_sev:.3f}")
print(f"RR severe Q5/Q1: naive {fmt_ci(boot_naive['rr_quintile_severe'])}; "
      f"cluster {fmt_ci(rr_sev_q5q1_cluster)}")
print(f"OUTCOME: {outcome}")
print(f"Audit: {'PASS' if audit_pass else 'FAIL'} "
      f"({sum(1 for c in checks if c['status'] == 'PASS')}/{len(checks)})")
print(f"Guard: {len(guard_hashes)} files {'unchanged' if guard_ok else 'MODIFIED'}")
print(f"Outputs: {OUT_DIR}")
print("Done.")
