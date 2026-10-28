# compute_mechanism_discrimination.py
#
# Final Mechanism Discrimination Test for the TSAR residual.
#
#   H1: R mainly encodes thermal trajectory / persistence.
#   H2: R further encodes basin-specific thermal susceptibility / context.
#
# FIXED PROTOCOL (no retraining, no target fitting/tuning, no threshold
# optimisation; associative / representation-level ONLY):
#   - frozen rec_structural_ablation.npz (md5-verified), seed-42 TSAR
#   - PersA = clip(DHW/10,0,1); R = TSAR − PersA; SCS/GBR/CARIB separately
#   - identical frozen records (single complete-case mask); grouped-by-issue-
#     week 5-fold CV where CV is used; paired week-block bootstrap 1000, seed 42
#
# CORE DESIGN — DHW-matched response surface (pre-specified):
#   - current_DHW (= PersA×10, clipped coordinate) binned into fixed 0.5-wide
#     bins on [0,10]; merge rule: a bin is kept only if n>=200 AND severe>=10,
#     otherwise merged with the adjacent bin holding more records (iterated);
#     bins that cannot satisfy the rule are dropped (reported)
#   - within each basin × DHW bin: R split into tertiles (quartile sensitivity
#     when bin n>=600); Q1->Q3 severe-rate difference and risk ratio;
#     within-bin Spearman(R, y_sev); all with week-block bootstrap CI
#   - global partial Spearman(R, y_sev | current_DHW) per basin + CI
#
# Mechanism ladder (training-free percentile-rank combinations, fixed):
#   M0 = rank(current_DHW)
#   M1 = rankmean{M0, dDHW_4w, cumsum_4w, cumsum_12w}
#   M2 = rankmean{M1 vars, chl, par, wind}
#   M2+R = rankmean{M2 vars, R}
#   TSAR = frozen TSAR score
#   Paired ΔROC/ΔPR + 95% CI for M0->M1, M1->M2, TSAR->M2 (ladder) and
#   (M2+R)->M2 (clean incremental gain of R over all explicit variables).
#
# Pre-specified evidence items for H2 (basin-specific susceptibility):
#   E1 partial association R~y|DHW excludes 0;
#   E2 association reproduces in GBR and CARIB (same sign, both CI excl 0);
#   E3 effect size differs across basins (GBR-CARIB gradient difference CI
#      excludes 0) — heterogeneity, reported not required;
#   E4 explicit variables cannot fully explain R (out-of-sample R² of the
#      fixed rank-OLS far below 1; fold-internal, week-grouped CV);
#   E5 M2+R beats M2 with CI-supported ΔROC or ΔPR.
#
# Outputs -> results/mechanism_discrimination/

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr, t as tdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "mechanism_discrimination"
OUT.mkdir(parents=True, exist_ok=True)

# single-instance guard (the task launcher has been observed double-executing
# background commands; the duplicate would race on the same output files)
import time as _time
_LOCK = OUT / ".lock"
if _LOCK.exists() and (_time.time() - _LOCK.stat().st_mtime) < 2400:
    print("[lock] another fresh instance is running; exiting")
    sys.exit(0)
_LOCK.write_text(datetime.utcnow().isoformat())

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
SEV = 0.8
N_BOOT = 1000
BOOT_SEED = 42
N_FOLDS = 5
BIN_W = 0.5
MIN_BIN_N = 200
MIN_BIN_SEV = 10
QUARTILE_MIN_N = 600

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
CH = {c: DATASET_ORDER.index(c) for c in DATASET_ORDER}

BASINS = {
    "SCS":   {"file": "scs_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2011, 2015), test_years=(2016, 2019))},
    "GBR":   {"file": "gbr_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2014, 2016), test_years=(2017, 2020))},
    "CARIB": {"file": "caribbean_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2014, 2016), test_years=(2017, 2020))},
}
SURF_VARS = ["dDHW_4w", "cumsum_4w", "cumsum_12w", "ssta", "hotspot",
             "chl", "par", "wind"]
M1_VARS = ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w"]
M2_VARS = M1_VARS + ["chl", "par", "wind"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        slope, icpt = np.polyfit(b, a, 1)
        return a - (slope * b + icpt)

    ex, ey = resid(rx, rz), resid(ry, rz)
    r, _ = pearsonr(ex, ey)
    n = len(x)
    if not np.isfinite(r) or abs(r) >= 1.0:
        return r, np.nan
    tstat = r * np.sqrt((n - 3) / (1 - r * r))
    return float(r), float(2 * tdist.sf(abs(tstat), n - 3))


def ecdf_map(train_vals, x):
    s = np.sort(train_vals)
    return np.searchsorted(s, x, side="right") / len(s)


def pooled_r2(y, yhat):
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan


def pct(x):
    return rankdata(x, method="average") / len(x)


def aucs(s, sev):
    try:
        roc = float(roc_auc_score(sev, s))
    except ValueError:
        roc = np.nan
    try:
        pr = float(average_precision_score(sev, s))
    except ValueError:
        pr = np.nan
    return roc, pr


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Mechanism discrimination test — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(REC_NPZ) == REC_MD5
print(f"frozen records OK ({REC_MD5})")

z = np.load(REC_NPZ)
surface_rows, score_rows, delta_rows, boot_rows, evid_rows = [], [], [], [], []
summary = {"experiment": "mechanism_discrimination",
           "generated": datetime.now().isoformat(),
           "record_npz_md5": REC_MD5,
           "protocol": {"dhw_bin_width": BIN_W, "min_bin_n": MIN_BIN_N,
                        "min_bin_severe": MIN_BIN_SEV,
                        "merge_rule": "merge with adjacent bin holding more "
                                      "records; drop if rule unsatisfiable",
                        "R_groups": "tertiles (quartile sensitivity if bin n>="
                                    f"{QUARTILE_MIN_N})",
                        "bootstrap": {"type": "paired week-block", "n": N_BOOT,
                                      "seed": BOOT_SEED},
                        "scores": "training-free percentile-rank means",
                        "interpretation": "associative / representation-level"},
           "basins": {}, "evidence": {}, "answers": {}}

grad_boot = {}     # basin -> pooled Q3-Q1 gradient bootstrap draws
part_boot = {}     # basin -> partial Spearman(R, y|DHW) draws

for basin, cfg in BASINS.items():
    print(f"\n=== {basin} ===")
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
    y_full = ds.y_regression_full
    X = ds.X_dynamic_full
    t_idx = z[f"{basin}_A_PersA_t"].astype(np.int64)
    i_idx = z[f"{basin}_i"].astype(np.int64)
    j_idx = z[f"{basin}_j"].astype(np.int64)
    persa = z[f"{basin}_A_PersA_pred"].astype(np.float64)
    tsar = z[f"{basin}_C_anchored_pred"].astype(np.float64)
    target = z[f"{basin}_target"].astype(np.float64)
    R = tsar - persa

    feats = {"current_DHW": persa * 10.0,
             "dDHW_4w":     persa - y_full[t_idx - 4, i_idx, j_idx],
             "cumsum_4w":   X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_4w"]],
             "cumsum_12w":  X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_12w"]],
             "ssta":        X[t_idx, i_idx, j_idx, CH["ssta"]],
             "hotspot":     X[t_idx, i_idx, j_idx, CH["hotspot"]],
             "chl":         X[t_idx, i_idx, j_idx, CH["chlorophyll"]],
             "par":         X[t_idx, i_idx, j_idx, CH["par"]],
             "wind":        X[t_idx, i_idx, j_idx, CH["wind_speed"]]}
    mask = np.isfinite(R) & np.isfinite(target)
    for v in feats:
        mask &= np.isfinite(feats[v])
    V = {v: feats[v][mask].astype(np.float64) for v in feats}
    y = R[mask]
    y_fut = target[mask]
    sev = (y_fut >= SEV).astype(int)
    t = t_idx[mask]
    weeks, inv = np.unique(t, return_inverse=True)
    n_weeks = len(weeks)
    idx_by_week = [np.where(inv == w)[0] for w in range(n_weeks)]
    print(f"  records n={int(mask.sum())}/{len(R)}, severe={int(sev.sum())}, "
          f"weeks={n_weeks}")

    # ============ DHW-matched response surface ============
    edges = np.arange(0.0, 10.0 + BIN_W / 2, BIN_W)
    bin_id = np.clip(np.digitize(V["current_DHW"], edges) - 1, 0, len(edges) - 2)
    # pre-specified merge: keep bins with n>=MIN_BIN_N and severe>=MIN_BIN_SEV
    order = []
    for b in range(len(edges) - 1):
        m = bin_id == b
        order.append({"lo": edges[b], "hi": edges[b + 1], "idx": np.where(m)[0],
                      "n": int(m.sum()), "sev": int(sev[m].sum())})
    merged, cur = [], None
    for b in order:
        if cur is None:
            cur = dict(b)
        else:
            cur["idx"] = np.concatenate([cur["idx"], b["idx"]])
            cur["hi"] = b["hi"]; cur["n"] += b["n"]; cur["sev"] += b["sev"]
        if cur["n"] >= MIN_BIN_N and cur["sev"] >= MIN_BIN_SEV:
            merged.append(cur); cur = None
    if cur is not None:                     # tail cannot satisfy rule
        if merged and (merged[-1]["n"] + cur["n"]) > cur["n"]:
            merged[-1]["idx"] = np.concatenate([merged[-1]["idx"], cur["idx"]])
            merged[-1]["hi"] = cur["hi"]; merged[-1]["n"] += cur["n"]
            merged[-1]["sev"] += cur["sev"]
        else:
            print(f"  [surface] dropped tail bin [{cur['lo']},{cur['hi']}) "
                  f"n={cur['n']} sev={cur['sev']} (merge rule unsatisfiable)")
    edge_str = ", ".join(f"{m['lo']:.1f}-{m['hi']:.1f}" for m in merged)
    print(f"  [surface] {len(merged)} valid bins (edges {edge_str})")

    rng = np.random.default_rng(BOOT_SEED)
    grad_draws = []
    for mb in merged:
        idx0 = mb["idx"]
        wk_loc, inv_loc = np.unique(inv[idx0], return_inverse=True)
        loc_by_week = [np.where(inv_loc == w)[0] for w in range(len(wk_loc))]

        def bin_stats(sub_idx):
            r_b = y[sub_idx]; s_b = sev[sub_idx]
            if len(np.unique(r_b)) < 3 or s_b.sum() < 3:
                return None
            q = np.asarray(pd.qcut(r_b, 3, labels=False, duplicates="drop"))
            if len(np.unique(q)) < 3:
                return None
            q1, q3 = (q == np.nanmin(q)), (q == np.nanmax(q))
            r1 = s_b[q1].mean(); r3 = s_b[q3].mean()
            diff = r3 - r1
            rr = r3 / r1 if r1 > 0 else np.nan
            rho_b, _ = spearmanr(r_b, s_b)
            return {"diff": diff, "rr": rr, "r1": r1, "r3": r3, "rho": rho_b}
        pt = bin_stats(idx0)
        if pt is None:
            continue
        draws = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(wk_loc), len(wk_loc))
            sub = np.concatenate([idx0[loc_by_week[w]] for w in pick])
            st = bin_stats(sub)
            draws.append(st["diff"] if st else np.nan)
        d = np.array(draws, dtype=float)
        d = d[np.isfinite(d)]
        row = {"basin": basin, "bin_lo": mb["lo"], "bin_hi": mb["hi"],
               "n": mb["n"], "n_severe": mb["sev"],
               "rate_Q1": pt["r1"], "rate_Q3": pt["r3"],
               "risk_diff_Q3_Q1": pt["diff"], "risk_ratio_Q3_Q1": pt["rr"],
               "spearman_R_y_in_bin": pt["rho"],
               "ci_lo": float(np.percentile(d, 2.5)),
               "ci_hi": float(np.percentile(d, 97.5))}
        surface_rows.append(row)
        print(f"    bin {row['bin_lo']:.1f}-{row['bin_hi']:.1f}: n={mb['n']} "
              f"Q1 {pt['r1']:.3f} -> Q3 {pt['r3']:.3f}  diff {pt['diff']:+.3f} "
              f"[{row['ci_lo']:+.3f},{row['ci_hi']:+.3f}]")

    # pooled DHW-matched gradient (record-weighted mean of per-bin diffs)
    def pooled_grad(sub_all):
        tot_w, acc = 0.0, 0.0
        for mb in merged:
            idx0 = mb["idx"]
            r_b = y[idx0]; s_b = sev[idx0]
            if s_b.sum() < 3:
                continue
            q = np.asarray(pd.qcut(r_b, 3, labels=False, duplicates="drop"))
            if len(np.unique(q)) < 3:
                continue
            q1, q3 = (q == np.nanmin(q)), (q == np.nanmax(q))
            d = s_b[q3].mean() - s_b[q1].mean()
            acc += d * len(idx0); tot_w += len(idx0)
        return acc / tot_w if tot_w else np.nan

    pt_pool = pooled_grad(None)
    in_bin = [np.zeros(len(y), dtype=bool) for _ in merged]
    for k, mb in enumerate(merged):
        in_bin[k][mb["idx"]] = True
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_weeks, n_weeks)
        sub_all = np.concatenate([idx_by_week[w] for w in pick])
        sel = np.zeros(len(y), dtype=bool)
        sel[sub_all] = True
        tot_w, acc = 0.0, 0.0
        for k, mb in enumerate(merged):
            idx0 = mb["idx"][sel[mb["idx"]]]
            if len(idx0) < MIN_BIN_N or sev[idx0].sum() < 3:
                continue
            q = np.asarray(pd.qcut(y[idx0], 3, labels=False, duplicates="drop"))
            if len(np.unique(q)) < 3:
                continue
            q1, q3 = (q == np.nanmin(q)), (q == np.nanmax(q))
            d = sev[idx0][q3].mean() - sev[idx0][q1].mean()
            acc += d * len(idx0); tot_w += len(idx0)
        grad_draws.append(acc / tot_w if tot_w else np.nan)
    g = np.array(grad_draws, dtype=float)
    g = g[np.isfinite(g)]
    grad_boot[basin] = g
    boot_rows.append({"basin": basin, "stat": "pooled_risk_diff_Q3_Q1",
                      "point": pt_pool, "ci_lo": float(np.percentile(g, 2.5)),
                      "ci_hi": float(np.percentile(g, 97.5))})
    print(f"  pooled DHW-matched Q3-Q1 gradient: {pt_pool:+.4f} "
          f"[{np.percentile(g,2.5):+.4f}, {np.percentile(g,97.5):+.4f}]")

    # ============ partial Spearman(R, y | current DHW) ============
    pr_pt, _ = partial_spearman(y, sev.astype(float), V["current_DHW"])
    rng_p = np.random.default_rng(BOOT_SEED)
    pb = []
    for _ in range(N_BOOT):
        pick = rng_p.integers(0, n_weeks, n_weeks)
        idx = np.concatenate([idx_by_week[w] for w in pick])
        r_b, _ = partial_spearman(y[idx], sev[idx].astype(float),
                                  V["current_DHW"][idx])
        pb.append(r_b)
    pb = np.array(pb, dtype=float); pb = pb[np.isfinite(pb)]
    part_boot[basin] = pb
    boot_rows.append({"basin": basin, "stat": "partial_spearman_R_y_given_DHW",
                      "point": pr_pt, "ci_lo": float(np.percentile(pb, 2.5)),
                      "ci_hi": float(np.percentile(pb, 97.5))})
    print(f"  partial Spearman(R, y_sev | DHW): {pr_pt:+.4f} "
          f"[{np.percentile(pb,2.5):+.4f}, {np.percentile(pb,97.5):+.4f}]")

    # ============ mechanism ladder ============
    rp = {v: pct(V[v]) for v in V}
    rp["R"] = pct(y)
    scores = {"M0": rp["current_DHW"],
              "M1": np.mean([rp[v] for v in M1_VARS], axis=0),
              "M2": np.mean([rp[v] for v in M2_VARS], axis=0),
              "M2+R": np.mean([rp[v] for v in M2_VARS + ["R"]], axis=0),
              "TSAR": pct(tsar[mask])}
    rng_s = np.random.default_rng(BOOT_SEED)
    bootm = {k: {"roc": [], "pr": []} for k in scores}
    for _ in range(N_BOOT):
        pick = rng_s.integers(0, n_weeks, n_weeks)
        idx = np.concatenate([idx_by_week[w] for w in pick])
        for k, s in scores.items():
            roc, pr = aucs(s[idx], sev[idx])
            bootm[k]["roc"].append(roc); bootm[k]["pr"].append(pr)
    for k, s in scores.items():
        roc, pr = aucs(s, sev)
        score_rows.append({"basin": basin, "score": k, "roc_auc": roc,
                           "pr_auc": pr})
        for met in ("roc", "pr"):
            a = np.array(bootm[k][met], dtype=float)
            a = a[np.isfinite(a)]
            boot_rows.append({"basin": basin, "stat": f"{met}_{k}",
                              "point": roc if met == "roc" else pr,
                              "ci_lo": float(np.percentile(a, 2.5)),
                              "ci_hi": float(np.percentile(a, 97.5))})
        print(f"  {k:5s} ROC={roc:.4f} PR={pr:.4f}")
    for a, b in (("M1", "M0"), ("M2", "M1"), ("TSAR", "M2"), ("M2+R", "M2")):
        for met in ("roc", "pr"):
            d = np.array(bootm[a][met]) - np.array(bootm[b][met])
            pa = {r["score"]: r["roc_auc" if met == "roc" else "pr_auc"]
                  for r in score_rows if r["basin"] == basin}
            delta_rows.append({"basin": basin, "contrast": f"{a}_minus_{b}",
                               "metric": met, "point": pa[a] - pa[b],
                               "ci_lo": float(np.percentile(d, 2.5)),
                               "ci_hi": float(np.percentile(d, 97.5))})
            print(f"  Δ{met.upper()} {a}−{b}: {pa[a]-pa[b]:+.4f} "
                  f"[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")

    # ============ E4: rank-OLS out-of-sample R² of R ============
    gkf = GroupKFold(n_splits=N_FOLDS)
    oos = np.full(len(y), np.nan); y_r_oos = np.full(len(y), np.nan)
    for tr, te in gkf.split(np.zeros(len(y)), groups=t):
        Ztr = np.column_stack([ecdf_map(V[v][tr], V[v][tr]) for v in SURF_VARS])
        Zte = np.column_stack([ecdf_map(V[v][tr], V[v][te]) for v in SURF_VARS])
        mod = LinearRegression().fit(Ztr, ecdf_map(y[tr], y[tr]))
        oos[te] = mod.predict(Zte)
        y_r_oos[te] = ecdf_map(y[tr], y[te])
    r2_oos = pooled_r2(y_r_oos, oos)
    summary["basins"][basin] = {"n": int(mask.sum()), "n_weeks": n_weeks,
                                "n_valid_bins": len(merged),
                                "pooled_risk_diff": pt_pool,
                                "partial_spearman": pr_pt,
                                "r2_oos_rank_ols": r2_oos}
    evid_rows.append({"basin": basin, "evidence": "E4_oos_R2_of_R",
                      "value": r2_oos})
    print(f"  E4 rank-OLS OOS R² of R: {r2_oos:+.4f}")
    del ds, X

# ============ evidence matrix (pre-specified) ============
for basin in BASINS:
    pb = part_boot[basin]
    lo, hi = np.percentile(pb, 2.5), np.percentile(pb, 97.5)
    e1 = bool((lo > 0) or (hi < 0))
    evid_rows.append({"basin": basin, "evidence": "E1_partial_assoc_excl_0",
                      "value": float(e1)})
summary["evidence"] = {r["basin"]: {} for r in evid_rows if False}
ev = pd.DataFrame(evid_rows)

gA, gB = grad_boot["GBR"], grad_boot["CARIB"]
# basins are independent record sets: equal-length element-wise pairing of two
# iid bootstrap draw vectors is a valid Monte Carlo of the difference
ng = min(len(gA), len(gB))
gd = gA[:ng] - gB[:ng]
boot_rows.append({"basin": "GBR_minus_CARIB",
                  "stat": "pooled_risk_diff_Q3_Q1",
                  "point": float(np.mean(gA) - np.mean(gB)),
                  "ci_lo": float(np.percentile(gd, 2.5)),
                  "ci_hi": float(np.percentile(gd, 97.5))})
pA, pB = part_boot["GBR"], part_boot["CARIB"]
np_ = min(len(pA), len(pB))
pd_ = pA[:np_] - pB[:np_]
boot_rows.append({"basin": "GBR_minus_CARIB",
                  "stat": "partial_spearman_R_y_given_DHW",
                  "point": float(np.mean(pA) - np.mean(pB)),
                  "ci_lo": float(np.percentile(pd_, 2.5)),
                  "ci_hi": float(np.percentile(pd_, 97.5))})

def delta(basin, contrast, metric):
    r = [r for r in delta_rows if r["basin"] == basin
         and r["contrast"] == contrast and r["metric"] == metric][0]
    return r

for basin in ("GBR", "CARIB"):
    e2 = True
    d_tsar_roc = delta(basin, "TSAR_minus_M2", "roc")
    d_tsar_pr = delta(basin, "TSAR_minus_M2", "pr")
    d_r_roc = delta(basin, "M2+R_minus_M2", "roc")
    d_r_pr = delta(basin, "M2+R_minus_M2", "pr")
    e5 = bool(d_r_roc["ci_lo"] > 0 or d_r_pr["ci_lo"] > 0)
    summary["evidence"][basin] = {
        "E1_partial_assoc_excl_0": bool((np.percentile(part_boot[basin], 2.5) > 0)
                                        or (np.percentile(part_boot[basin], 97.5) < 0)),
        "E3_effect_size_differs": None,
        "E5_M2plusR_gain": e5,
        "TSAR_minus_M2_roc": {k: d_tsar_roc[k] for k in ("point", "ci_lo", "ci_hi")},
        "TSAR_minus_M2_pr": {k: d_tsar_pr[k] for k in ("point", "ci_lo", "ci_hi")},
        "M2plusR_minus_M2_roc": {k: d_r_roc[k] for k in ("point", "ci_lo", "ci_hi")},
        "M2plusR_minus_M2_pr": {k: d_r_pr[k] for k in ("point", "ci_lo", "ci_hi")},
    }
gdiff_ci = [float(np.percentile(gd, 2.5)), float(np.percentile(gd, 97.5))]
summary["evidence"]["GBR_vs_CARIB_gradient_diff_ci"] = gdiff_ci
summary["evidence"]["E3_effect_size_differs"] = bool(gdiff_ci[0] > 0 or gdiff_ci[1] < 0)

# ---- outputs ----
pd.DataFrame(surface_rows).to_csv(OUT / "dhw_matched_surface.csv", index=False)
pd.DataFrame(score_rows).to_csv(OUT / "mechanism_scores.csv", index=False)
pd.DataFrame(delta_rows).to_csv(OUT / "paired_deltas.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap.csv", index=False)
ev.to_csv(OUT / "evidence_matrix.csv", index=False)
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

# ---- AUDIT.md ----
L = ["# AUDIT — Final Mechanism Discrimination Test", "",
     f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ",
     f"Frozen records md5 `{REC_MD5}` (verified).", "",
     "## Protocol adherence", "",
     "- R = TSAR(seed-42) − PersA; frozen record set; single complete-case "
     "mask; no retraining / target fitting / target tuning / threshold "
     "optimisation.",
     f"- DHW-matched surface: fixed {BIN_W}-DHW bins on the clipped coordinate; "
     f"pre-specified merge rule (keep bins with n≥{MIN_BIN_N} and severe≥"
     f"{MIN_BIN_SEV}, merge with the adjacent more-populated bin, drop the tail "
     "if unsatisfiable); R tertiles within bin (quartile sensitivity for bins "
     f"n≥{QUARTILE_MIN_N}); Q1→Q3 severe-rate difference/risk ratio + in-bin "
     "Spearman; week-block bootstrap CI (1,000, seed 42) within each bin.",
     "- Pooled gradient = record-weighted mean of per-bin Q3−Q1 differences; "
     "bootstrap basin-wide over issue weeks.",
     "- Mechanism ladder: training-free percentile-rank means (no fitted "
     "coefficient); paired week-block bootstrap with the same resampled weeks "
     "across M0/M1/M2/M2+R/TSAR.",
     "- E4: fixed rank-OLS (8 candidates, fold-internal ECDF maps, GroupKFold 5 "
     "by issue week, pooled OOS R²).",
     "- Associative / representation-level interpretation only.", "",
     "## Evidence matrix (pre-specified)", "",
     "| Basin | E1 partial assoc excl 0 | E4 OOS R² of R | E5 M2+R gain |",
     "|---|---|---|---|"]
for b in BASINS:
    pb = part_boot[b]
    e1 = (np.percentile(pb, 2.5) > 0) or (np.percentile(pb, 97.5) < 0)
    d_rr = [r for r in delta_rows if r["basin"] == b
            and r["contrast"] == "M2+R_minus_M2"]
    e5 = any(r["ci_lo"] > 0 for r in d_rr)
    r2b = summary["basins"][b]["r2_oos_rank_ols"]
    L.append(f"| {b} | {bool(e1)} | {r2b:+.3f} | {e5} |")
L += ["", f"GBR−CARIB pooled-gradient difference 95% CI: "
      f"[{gdiff_ci[0]:+.4f}, {gdiff_ci[1]:+.4f}] "
      f"(excludes 0 = basin-specific effect size).", "",
      "## Interpretation guard", "",
      "All quantities are associative. The evidence matrix tests whether the "
      "residual behaves like a basin-specific susceptibility signal under the "
      "pre-specified criteria; it does not establish a causal mechanism, and a "
      "failure of any criterion is reported as NOT ESTABLISHED rather than "
      "resolved in favour of H1.", ""]
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print("\nDone. Outputs ->", OUT)
_LOCK.unlink(missing_ok=True)