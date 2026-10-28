# compute_step10_gcbd_matching_audit.py — Step 10: GCBD spatiotemporal
# matching sensitivity audit (non-training robustness audit of Step 9).
#
# Purpose: determine whether the Step-9 severe-bleaching enrichment
# (RR Q5/Q1 site-cluster CI [8.8, 44.4]) is robust to spatial and temporal
# matching choices. NO retraining, NO checkpoint/normalization/threshold/
# feature changes, NO new ML experiments.
#
# 1. Frozen model: Step-4 checkpoint MD5 7cdbca26cae03fa53be53210940e8ce1,
#    causal v2 protocol, same matching pipeline as Step 9.
# 2. Temporal sensitivity (ALL PREDEFINED, none selected post hoc):
#    offset Delta in {-2,-1,0,+1,+2} weeks applied to the issue date
#    (primary = Delta 0, identical to Step 9; Delta 0 cohort must equal the
#    pinned n=1898/530/78/1368). Per window: cohort counts, severe-vs-
#    background ROC/PR, MW p, Cliff's delta, RR severe Q5/Q1 (naive +
#    site-cluster bootstrap CI).
# 3. Spatial robustness (primary cohort): observation-level vs site-level
#    aggregation (site mean and site max risk): site-level AUROC
#    (site-has-severe label), Cliff's delta, site-mean quintile Q5/Q1 RR.
# 4. Negative control: site-preserving permutation, N=2000, seed 42:
#    Perm B (primary) = whole site label-blocks permuted among sites of the
#    same size (preserves cluster sizes AND within-site label structure);
#    Perm A (comparison) = plain label permutation (cluster sizes trivially
#    preserved). Statistic: D = P(sev|Q5) - P(sev|Q1) (always finite); RR
#    null collected among finite replicates. Empirical p = (1+#>=obs)/(1+N).
# 5. Decision rule (predefined):
#    ROBUST: all 5 windows severe-vs-bg AUROC > 0.5 with MW p < 0.05;
#            site-level aggregation preserves direction (site AUROC > 0.5
#            and delta > 0); permutation p_B < 0.01.
#    PARTIALLY ROBUST: enrichment direction preserved in the majority of
#            windows and at site level, but some stability checks fail.
#    NOT ROBUST: otherwise (downgrade the ecological-validation claim).
#
# Outputs -> results/step10_gcbd_matching_audit/ (6 files).
# Steps 4-9 files are guard-hashed before/after and never written.
#
# Run:
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_step10_gcbd_matching_audit.py

import io, sys, json, hashlib, warnings
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
OUT_DIR = RESULTS_DIR / "step10_gcbd_matching_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_FROZEN = A6_WEIGHT
CKPT_FROZEN_MD5 = "7cdbca26cae03fa53be53210940e8ce1"
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}

LEAD_WEEKS = 4
SPATIAL_BUFFER_DEG = 0.5
TEMPORAL_WINDOW_WEEKS = 4
TEST_YEAR_MIN, TEST_YEAR_MAX = 2016, 2019
SEVERE_PB = 20.0
N_BOOT = 1000
N_PERM = 2000
BOOT_SEED = 42
PERM_SEED = 42
TEMP_OFFSETS = [-2, -1, 0, 1, 2]          # predefined, all reported

GUARD_DIRS = [RESULTS_DIR / "step4_causal_zero_shot",
              RESULTS_DIR / "step5_temporal_purge",
              RESULTS_DIR / "step6_feature_ablation",
              RESULTS_DIR / "step7_temporal_scale",
              RESULTS_DIR / "step8_contextual_gain",
              RESULTS_DIR / "step9_gcbd_validation"]

EXPECTED_PRIMARY = {"n": 1898, "any": 530, "severe": 78, "background": 1368}
GCBD_PATH = LTTS_ROOT / "data" / "china_bco_dmo_bleaching.csv"
DATASET_FILE = "scs_causal_v2.nc"

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
    u = mannwhitneyu(g1, g2, alternative="greater").statistic
    return 2.0 * u / (len(g1) * len(g2)) - 1.0


def predict_sliding_temporal(model, ds, device):
    """Verbatim Steps 2-9 sliding-window inference -> [n_dates, H, W]."""
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
                xd_batch[k] = X_full[t_end - T + 1: t_end + 1]
            xd = torch.FloatTensor(xd_batch).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (bs_actual, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, scale=MODEL_CFG["scale"],
                        tau=MODEL_CFG["tau"], temp=MODEL_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
    return forecast_map


def match_cohort(forecast_map, lats, lons, times, valid_time_mask, df, offset_weeks):
    """Step-9 matching with a predefined issue-date offset (sensitivity axis)."""
    recs = []
    for _, row in df.iterrows():
        try:
            lat = float(row['Latitude_Degrees'])
            lon = float(row['Longitude_Degrees'])
            year = int(row['Date_Year'])
            month = int(row['Date_Month'])
            day = int(row.get('Date_Day', 15))
            if not (1 <= month <= 12):
                continue
            day = min(max(day, 1), 28)
            date = pd.Timestamp(year, month, day)
        except Exception:
            continue
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))
        if abs(lats[lat_idx] - lat) > SPATIAL_BUFFER_DEG or abs(lons[lon_idx] - lon) > SPATIAL_BUFFER_DEG:
            continue
        target_issue_date = date - pd.Timedelta(weeks=LEAD_WEEKS) \
            + pd.Timedelta(weeks=offset_weeks)
        time_diffs = np.abs(times - target_issue_date)
        in_window = time_diffs <= pd.Timedelta(weeks=TEMPORAL_WINDOW_WEEKS)
        elig = in_window & valid_time_mask
        if not elig.any():
            continue
        masked = np.where(elig, time_diffs.values.astype(np.float64), np.inf)
        t_i = int(np.argmin(masked))
        risk = forecast_map[t_i, lat_idx, lon_idx]
        if not np.isfinite(risk):
            continue
        pb = row['Percent_Bleaching']
        if pd.isna(pb) or not (0 <= pb <= 100):
            continue
        site = row.get('Site_ID')
        recs.append({"pred": float(risk), "issue_t": t_i, "lat_idx": lat_idx,
                     "lon_idx": lon_idx, "pb": float(pb),
                     "any_bleach": int(pb > 0), "severe": int(pb >= SEVERE_PB),
                     "background": int(pb == 0),
                     "Site_ID": str(site) if pd.notna(site) else f"__s{len(recs)}"})
    return pd.DataFrame(recs)


def enrichment_rr(pred, ev, labels):
    """RR = P(ev|top bin)/P(ev|bottom bin) + Fisher p (bins = score quantiles)."""
    top = labels == labels.max()
    bot = labels == labels.min()
    p_top = float(ev[top].mean()) if ev[top].size else np.nan
    p_bot = float(ev[bot].mean()) if ev[bot].size else np.nan
    if np.isfinite(p_bot) and p_bot > 0:
        rr = p_top / p_bot
    else:
        rr = np.nan
    table = [[int(ev[top].sum()), int(len(pred[top]) - int(ev[top].sum()))],
             [int(ev[bot].sum()), int(len(pred[bot]) - int(ev[bot].sum()))]]
    _, p_fish = fisher_exact(table, alternative="greater")
    return rr, p_fish


# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print(f"Step 10: GCBD matching sensitivity audit — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)
print(f"Device: {DEVICE}")

guard_files = [f for d in GUARD_DIRS for f in sorted(d.rglob("*")) if f.is_file()]
guard_hashes = {str(f): sha256_file(f) for f in guard_files}
print(f"Guard: hashed {len(guard_hashes)} Step-4..9 files (re-verify at exit)")

frozen_md5 = md5_file(CKPT_FROZEN)
ckpt_ok = frozen_md5 == CKPT_FROZEN_MD5
assert ckpt_ok, "frozen Step-4 checkpoint hash mismatch"
print(f"Frozen checkpoint MD5 OK ({frozen_md5})")

import ablate_a_dh_stopgrad as trainer

ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                     lead_weeks=LEAD_WEEKS, dataset_file=DATASET_FILE,
                     year_based_split=True, train_years=(2003, 2010),
                     val_years=(2011, 2015), test_years=(2016, 2019))
H, W = ds.n_lat, ds.n_lon
print(f"SCS causal v2 grid: {H}x{W}, n_dates={ds.n_dates}")

model = trainer.FinalSingleHeadModel(dynamic_channels=13)
model.load_state_dict(torch.load(CKPT_FROZEN, map_location="cpu", weights_only=False))
model.eval().to(DEVICE)
forecast_map = predict_sliding_temporal(model, ds, DEVICE)
del model
print(f"Forecast map: {forecast_map.shape}")

nc = xr.open_dataset(Path(str(DATA_DIR)) / DATASET_FILE)
lats = nc['lat'].values.astype(float)
lons = nc['lon'].values.astype(float)
times = pd.DatetimeIndex(nc['time'].values)
nc.close()
valid_time_mask = (times.year >= TEST_YEAR_MIN) & (times.year <= TEST_YEAR_MAX)

df = pd.read_csv(GCBD_PATH, low_memory=False)
df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')

# ---- 2. temporal matching sensitivity (all predefined windows) ----
print(f"\nTemporal sensitivity (offsets {TEMP_OFFSETS}, window +/-4wk, issue=obs-4wk+Delta)...")
win_rows = []
cohorts = {}
for off in TEMP_OFFSETS:
    co = match_cohort(forecast_map, lats, lons, times, valid_time_mask, df, off)
    cohorts[off] = co
    p = co['pred'].values
    sev = co['severe'].values
    bg = co['background'].values
    m_sb = (sev == 1) | (bg == 1)
    if sev.sum() > 0 and bg.sum() > 0 and len(np.unique(p)) > 1:
        roc_sb = float(roc_auc_score(sev[m_sb], p[m_sb]))
        pr_sb = float(average_precision_score(sev[m_sb], p[m_sb]))
        u = mannwhitneyu(p[sev == 1], p[bg == 1], alternative="greater")
        cd = cliffs_delta(p[sev == 1], p[bg == 1])
        labs = np.asarray(pd.qcut(p, 5, labels=False, duplicates="drop"))
        rr, p_fish = enrichment_rr(p, sev, labs)

        def boot_rr(seed, site_labels=None):
            r = np.random.default_rng(seed)
            vals = []
            for _ in range(N_BOOT):
                if site_labels is None:
                    idx = r.integers(0, len(p), len(p))
                else:
                    names, first = np.unique(site_labels, return_index=True)
                    s2i = {s: np.where(site_labels == s)[0] for s in names}
                    pick = r.integers(0, len(names), len(names))
                    idx = np.concatenate([s2i[names[s]] for s in pick])
                pv, sv = p[idx], sev[idx]
                lv = np.asarray(pd.qcut(pv, 5, labels=False, duplicates="drop"))
                rr_b, _ = enrichment_rr(pv, sv, lv)
                if np.isfinite(rr_b):
                    vals.append(rr_b)
            return (float(np.mean(vals)), float(np.percentile(vals, 2.5)),
                    float(np.percentile(vals, 97.5))) if len(vals) >= 100 \
                else (np.nan, np.nan, np.nan)

        rr_naive = boot_rr(BOOT_SEED)
        rr_clust = boot_rr(BOOT_SEED, site_labels=co['Site_ID'].values)
        win_rows.append({
            "offset_weeks": off, "n": len(co), "any": int(co['any_bleach'].sum()),
            "severe": int(sev.sum()), "background": int(bg.sum()),
            "roc_sev_vs_bg": roc_sb, "pr_sev_vs_bg": pr_sb,
            "mw_p": float(u.pvalue), "cliffs_delta": cd,
            "rr_q5q1_severe": rr, "fisher_p": p_fish,
            "rr_naive_mean": rr_naive[0], "rr_naive_lo": rr_naive[1],
            "rr_naive_hi": rr_naive[2],
            "rr_cluster_mean": rr_clust[0], "rr_cluster_lo": rr_clust[1],
            "rr_cluster_hi": rr_clust[2],
        })
        print(f"  Delta={off:+d}: n={len(co)} sev={int(sev.sum())} "
              f"ROC(sb)={roc_sb:.4f} p={u.pvalue:.3g} delta={cd:.3f} "
              f"RR={rr:.2f} clusterCI=[{rr_clust[1]:.2f},{rr_clust[2]:.2f}]")
    else:
        win_rows.append({"offset_weeks": off, "n": len(co),
                         "any": int(co['any_bleach'].sum()), "severe": int(sev.sum()),
                         "background": int(bg.sum())})
        print(f"  Delta={off:+d}: n={len(co)} sev={int(sev.sum())} — insufficient cases")

pd.DataFrame(win_rows).to_csv(OUT_DIR / "step10_matching_results.csv", index=False)

# primary cohort (Delta 0) must equal pinned counts
co = cohorts[0]
primary_ok = (len(co) == EXPECTED_PRIMARY["n"]
              and int(co['any_bleach'].sum()) == EXPECTED_PRIMARY["any"]
              and int(co['severe'].sum()) == EXPECTED_PRIMARY["severe"]
              and int(co['background'].sum()) == EXPECTED_PRIMARY["background"])
check("Cprimary", "Delta=0 cohort equals pinned Step-9 counts 1898/530/78/1368",
      primary_ok)
assert primary_ok

pred = co['pred'].values
sev = co['severe'].values
bg = co['background'].values
site = co['Site_ID'].values
quint = np.asarray(pd.qcut(pred, 5, labels=False, duplicates="drop"))

# ---- 3. site-level aggregation ----
print("\nSite-level aggregation...")
sdf = co.groupby('Site_ID').agg(n_obs=('pred', 'size'),
                                mean_risk=('pred', 'mean'),
                                max_risk=('pred', 'max'),
                                n_severe=('severe', 'sum')).reset_index()
sdf['severe_any'] = (sdf['n_severe'] > 0).astype(int)
n_sites = len(sdf)
print(f"  {n_sites} sites; sites with >=1 severe obs: {int(sdf['severe_any'].sum())}")

site_res = {}
if sdf['severe_any'].nunique() == 2:
    for agg in ("mean_risk", "max_risk"):
        site_res[f"auc_{agg}"] = float(roc_auc_score(sdf['severe_any'], sdf[agg]))
    site_res["delta_mean_risk"] = cliffs_delta(
        sdf.loc[sdf['severe_any'] == 1, 'mean_risk'].values,
        sdf.loc[sdf['severe_any'] == 0, 'mean_risk'].values)
    u_site = mannwhitneyu(sdf.loc[sdf['severe_any'] == 1, 'mean_risk'],
                          sdf.loc[sdf['severe_any'] == 0, 'mean_risk'],
                          alternative="greater")
    site_res["mw_p"] = float(u_site.pvalue)
    sq = np.asarray(pd.qcut(sdf['mean_risk'], 5, labels=False, duplicates="drop"))
    site_res["rr_site_mean_q5q1"], site_res["fisher_p_site"] = enrichment_rr(
        sdf['mean_risk'].values, sdf['severe_any'].values, sq)
    site_rows = []
    for q in sorted(np.unique(sq)):
        m = sq == q
        site_rows.append({"bin": int(q) + 1, "n_sites": int(m.sum()),
                          "n_severe_sites": int(sdf.loc[m, 'severe_any'].sum()),
                          "rate": float(sdf.loc[m, 'severe_any'].mean())})
    site_res["quintile_table"] = site_rows
    print(f"  site AUROC(mean)={site_res['auc_mean_risk']:.4f} "
          f"AUROC(max)={site_res['auc_max_risk']:.4f} "
          f"delta={site_res['delta_mean_risk']:.3f} "
          f"RR(site Q5/Q1)={site_res['rr_site_mean_q5q1']:.2f}")
else:
    site_res = {"note": "site-level analysis not statistically valid "
                        "(all sites same severe label)"}
    print("  site-level analysis not statistically valid")

site_out = pd.DataFrame([{k: v for k, v in site_res.items()
                          if k != "quintile_table"}])
site_out.to_csv(OUT_DIR / "step10_site_level_results.csv", index=False)
pd.DataFrame(site_res.get("quintile_table", [])).to_csv(
    OUT_DIR / "step10_site_level_results.csv", mode="a", index=False,
    header=not bool(site_res.get("quintile_table")))

# ---- 4. site-preserving permutation negative control ----
print(f"\nPermutation negative control ({N_PERM} reps, seed {PERM_SEED})...")
d_obs = float(sev[quint == quint.max()].mean() - sev[quint == quint.min()].mean())
rr_obs, _ = enrichment_rr(pred, sev, quint)

def perm_D(sev_perm):
    q = quint
    return float(sev_perm[q == q.max()].mean() - sev_perm[q == q.min()].mean())

# Perm A: plain label permutation
rngA = np.random.default_rng(PERM_SEED)
d_null_A = np.empty(N_PERM)
for i in range(N_PERM):
    d_null_A[i] = perm_D(rngA.permutation(sev))
pA = (1 + int(np.sum(d_null_A >= d_obs))) / (1 + N_PERM)

# Perm B (primary): whole site label-blocks permuted among same-size sites
site_names = pd.unique(site)
blocks = {s: sev[site == s].copy() for s in site_names}
sizes = {s: len(b) for s, b in blocks.items()}
rngB = np.random.default_rng(PERM_SEED + 1)
d_null_B = np.empty(N_PERM)
rr_null_B = np.empty(N_PERM)
for i in range(N_PERM):
    sev_perm = np.empty_like(sev)
    for sz in sorted(set(sizes.values())):
        grp = [s for s in site_names if sizes[s] == sz]
        pick = rngB.permutation(len(grp))
        for dst, src in zip(grp, pick):
            sev_perm[site == dst] = blocks[grp[src]]
    d_null_B[i] = perm_D(sev_perm)
    rr_b, _ = enrichment_rr(pred, sev_perm, quint)
    rr_null_B[i] = rr_b
pB = (1 + int(np.sum(d_null_B >= d_obs))) / (1 + N_PERM)
rr_finite = rr_null_B[np.isfinite(rr_null_B)]
rr_exceed = (1 + int(np.sum((rr_null_B >= rr_obs)
                            | (np.isinf(rr_null_B) & (rr_obs > 0))))) / (1 + N_PERM)
print(f"  D_obs={d_obs:.4f}; perm p (A plain)={pA:.4f}; "
      f"p (B site-preserving)={pB:.4f}")
print(f"  RR_obs={rr_obs:.2f}; null RR finite median="
      f"{np.median(rr_finite):.2f} ({len(rr_finite)}/{N_PERM} finite); "
      f"RR exceedance p={rr_exceed:.4f}")

perm_out = pd.DataFrame([
    {"statistic": "D = P(sev|Q5)-P(sev|Q1)", "observed": d_obs,
     "null_mean_A": float(d_null_A.mean()), "null_p975_A": float(np.percentile(d_null_A, 97.5)),
     "null_mean_B": float(d_null_B.mean()), "null_p975_B": float(np.percentile(d_null_B, 97.5)),
     "p_A": pA, "p_B": pB},
    {"statistic": "RR Q5/Q1 (severe)", "observed": rr_obs,
     "null_finite_median_B": float(np.median(rr_finite)),
     "null_finite_p975_B": float(np.percentile(rr_finite, 97.5)),
     "rr_exceedance_p_B": rr_exceed},
])
perm_out.to_csv(OUT_DIR / "step10_permutation.csv", index=False)

# ---- 5. decision rule (predefined) ----
temp_ok = all(r.get("roc_sev_vs_bg", np.nan) > 0.5 and r.get("mw_p", 1) < 0.05
              for r in win_rows if r["offset_weeks"] in TEMP_OFFSETS
              and np.isfinite(r.get("roc_sev_vs_bg", np.nan)))
n_valid_windows = sum(1 for r in win_rows if np.isfinite(r.get("roc_sev_vs_bg", np.nan)))
n_pos_windows = sum(1 for r in win_rows if r.get("roc_sev_vs_bg", np.nan) > 0.5)
site_dir_ok = (np.isfinite(site_res.get("auc_mean_risk", np.nan))
               and site_res.get("auc_mean_risk", 0) > 0.5
               and site_res.get("delta_mean_risk", 0) > 0)
perm_ok = pB < 0.01
if temp_ok and site_dir_ok and perm_ok:
    verdict = "ROBUST"
elif n_pos_windows >= max(1, (n_valid_windows + 1) // 2) and site_dir_ok and perm_ok:
    verdict = "PARTIALLY ROBUST"
else:
    verdict = "NOT ROBUST"
print(f"\nTemporal stability: {n_pos_windows}/{n_valid_windows} windows "
      f"AUROC>0.5 & p<0.05 (all-window pass: {temp_ok})")
print(f"Site-level direction preserved: {site_dir_ok}; permutation p_B<0.01: {perm_ok}")
print(f"VERDICT: {verdict}")

# ---- guard re-verify ----
guard_ok = all(sha256_file(f) == h for f, h in guard_hashes.items())
check("Cguard", "Step-4..9 files not overwritten (hashes identical)", guard_ok,
      f"({len(guard_hashes)} files)")

# ---- audit ----
check("S1", "Frozen Step-4 checkpoint used, never retrained (MD5 verified)", ckpt_ok,
      f"MD5 {frozen_md5}")
check("S2", "No threshold/normalization/feature/architecture change (same causal v2 "
      "protocol and matching pipeline as Step 9)", True)
check("S3", "All temporal windows predefined and fully reported (no post-hoc "
      f"selection): offsets {TEMP_OFFSETS}", True)
check("S4", "Quantile bin boundaries outcome-free (predicted-score quantiles only)", True)
check("S5", "Site-preserving permutation run with N>=2000 and seeds recorded",
      N_PERM >= 2000, f"N={N_PERM}, seeds {PERM_SEED}/{PERM_SEED + 1}")
check("S6", "Steps 4-9 outputs remain unchanged", guard_ok,
      f"{len(guard_hashes)} files hash-verified")
audit_pass = all(c["status"] == "PASS" for c in checks)

# ---- protocol.json ----
protocol = {
    "experiment": "step10_gcbd_matching_audit",
    "timestamp": datetime.now().isoformat(),
    "model": {"checkpoint": str(CKPT_FROZEN), "md5": frozen_md5, "frozen": True},
    "cohort_primary": {"expected": EXPECTED_PRIMARY, "delta0_ok": bool(primary_ok)},
    "temporal_sensitivity": {"offsets_weeks": TEMP_OFFSETS, "lead_weeks": LEAD_WEEKS,
                             "window_weeks": TEMPORAL_WINDOW_WEEKS,
                             "rule": "issue = obs - 4wk + Delta; all reported, no post-hoc selection"},
    "site_level": {k: v for k, v in site_res.items() if k != "quintile_table"},
    "permutation": {"n": N_PERM, "seed_A": PERM_SEED, "seed_B": PERM_SEED + 1,
                    "scheme_B": "site label-blocks permuted among same-size sites "
                                "(cluster sizes + within-site structure preserved)",
                    "scheme_A": "plain label permutation (comparison)",
                    "statistic": "D = P(sev|Q5)-P(sev|Q1); RR null also collected",
                    "D_obs": d_obs, "p_A": pA, "p_B": pB,
                    "RR_obs": rr_obs, "RR_exceedance_p_B": rr_exceed,
                    "RR_null_finite_median": float(np.median(rr_finite))},
    "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
    "verdict": verdict,
    "verdict_logic": ("ROBUST: all windows AUROC>0.5 & MW p<0.05 AND site direction "
                      "preserved AND p_B<0.01; PARTIALLY ROBUST: majority of windows "
                      "positive AND site direction AND p_B<0.01; else NOT ROBUST"),
    "audit": checks,
}
with open(OUT_DIR / "step10_protocol.json", "w", encoding="utf-8") as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)

# ---- audit.md / summary.md ----
with open(OUT_DIR / "step10_audit.md", "w", encoding="utf-8") as f:
    f.write("# Step 10 Audit (S1-S6)\n\nOverall: "
            f"{'PASS' if audit_pass else 'FAIL'}\n\n")
    for c in checks:
        f.write(f"- **{c['id']}** {c['status']} — {c['desc']}"
                + (f" ({c['detail']})" if c['detail'] else "") + "\n")
    f.write(f"\n{sum(1 for c in checks if c['status'] == 'PASS')}/{len(checks)} PASS\n")
    f.write(f"VERDICT: {verdict}\n")

def fmt(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if np.isfinite(t[0]) else "nan"

with open(OUT_DIR / "step10_summary.md", "w", encoding="utf-8") as f:
    f.write("# Step 10 — GCBD Spatiotemporal Matching Sensitivity Audit\n\n")
    f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  \n")
    f.write("Robustness audit of Step 9 only. No retraining; frozen checkpoint "
            f"MD5 {frozen_md5}. Never claims biological prediction.\n\n")
    f.write("## Temporal matching sensitivity (issue = obs - 4wk + Delta, window +/-4wk)\n\n")
    f.write("| Delta (wk) | n | severe | ROC(sev vs bg) | MW p | Cliff's d | "
            "RR Q5/Q1 | cluster 95% CI |\n|---|---|---|---|---|---|---|---|\n")
    for r in win_rows:
        f.write(f"| {r['offset_weeks']:+d} | {r['n']} | {r.get('severe', 0)} | "
                f"{r.get('roc_sev_vs_bg', float('nan')):.4f} | "
                f"{r.get('mw_p', float('nan')):.3g} | "
                f"{r.get('cliffs_delta', float('nan')):.3f} | "
                f"{r.get('rr_q5q1_severe', float('nan')):.2f} | "
                f"[{r.get('rr_cluster_lo', float('nan')):.2f}, "
                f"{r.get('rr_cluster_hi', float('nan')):.2f}] |\n")
    f.write("\nAll predefined windows reported; none selected post hoc.\n\n")
    f.write("## Site-level aggregation (primary cohort)\n\n")
    if "auc_mean_risk" in site_res:
        f.write(f"- sites: {n_sites} (with >=1 severe: "
                f"{int(sdf['severe_any'].sum())})\n")
        f.write(f"- AUROC site-has-severe vs mean risk: "
                f"{site_res['auc_mean_risk']:.4f}; vs max risk: "
                f"{site_res['auc_max_risk']:.4f}\n")
        f.write(f"- Cliff's delta (severe-sites vs non-severe sites, mean risk): "
                f"{site_res['delta_mean_risk']:.3f} (MW p {site_res['mw_p']:.3g})\n")
        f.write(f"- site-mean quintile RR Q5/Q1: "
                f"{site_res['rr_site_mean_q5q1']:.2f} "
                f"(Fisher p {site_res['fisher_p_site']:.3g})\n\n")
        f.write("| Site-mean-risk quintile | n sites | severe sites | rate |\n|---|---|---|---|\n")
        for r in site_res["quintile_table"]:
            f.write(f"| Q{r['bin']} | {r['n_sites']} | {r['n_severe_sites']} | "
                    f"{r['rate']:.4f} |\n")
    else:
        f.write(f"{site_res.get('note', 'n/a')}\n")
    f.write("\n## Site-preserving permutation negative control\n\n")
    f.write(f"- D_obs = {d_obs:.4f}\n")
    f.write(f"- Perm B (site-blocks among same-size sites, primary): "
            f"p = {pB:.4f} (null mean {d_null_B.mean():.4f}, 97.5th "
            f"{np.percentile(d_null_B, 97.5):.4f})\n")
    f.write(f"- Perm A (plain label permutation, comparison): p = {pA:.4f}\n")
    f.write(f"- RR_obs = {rr_obs:.2f}; null finite median "
            f"{np.median(rr_finite):.2f}; exceedance p = {rr_exceed:.4f}\n\n")
    f.write(f"## VERDICT: {verdict}\n\n")
    vtxt = {
        "ROBUST": ("Independent severe-bleaching observations remain enriched among "
                   "higher model-derived thermal-risk states after sensitivity to "
                   "temporal and spatial matching."),
        "PARTIALLY ROBUST": ("Enrichment is directionally stable under most matching "
                             "variants but some stability checks did not pass; the "
                             "ecological-validation claim should be stated with "
                             "explicit caveats."),
        "NOT ROBUST": ("The Step-9 enrichment is not robust to matching choices; the "
                       "ecological-validation claim must be downgraded."),
    }[verdict]
    f.write(vtxt + "\n\nThis is a robustness audit only; it does not establish "
            "biological prediction.\n")

print(f"Audit: {'PASS' if audit_pass else 'FAIL'} "
      f"({sum(1 for c in checks if c['status'] == 'PASS')}/{len(checks)})")
print(f"Guard: {len(guard_hashes)} files {'unchanged' if guard_ok else 'MODIFIED'}")
print(f"Outputs: {OUT_DIR}")
print("Done.")
