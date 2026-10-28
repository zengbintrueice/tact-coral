# compute_residual_mechanism_decomposition.py
#
# Residual Mechanism Decomposition + Nested Incremental Explanation Test.
#
# Question: what does the TSAR residual R = TSAR − PersA_DHW encode?  Is it
# mainly thermal trajectory/persistence, or nonthermal environmental context,
# rather than a simple re-encoding of current DHW?
#
# FIXED PROTOCOL (no retraining, no target fitting, no target tuning, no
# threshold optimisation; associative / representation-level interpretation
# ONLY):
#   - frozen records  : results/physics_anchored/structural_ablation/
#                       rec_structural_ablation.npz (SCS/GBR/CARIB, seed-42
#                       TSAR at scale 0.4, PersA_DHW, target y(t+4))
#   - R = TSAR − PersA per record
#   - candidate mechanism variables read VERBATIM from the causal v2 datasets
#     at the record's issue time (t, i, j); no new windows/labels:
#       current_DHW   = PersA*10 (the clipped DHW/10 coordinate; DHW>10 clipped)
#       dDHW_4w       = y(t) − y(t−4)   (4-week DHW change, clipped coordinate)
#       d_hotspot_dt  = existing dataset channel (alternate derivative, reported)
#       cumsum_4w/12w = existing trailing HotSpot exposure channels (persistence)
#       ssta, hotspot = existing thermal state channels
#       chl, par, wind= existing nonthermal context channels
#   - complete-case mask over ALL candidate variables (identical records for
#     every part; n dropped reported in AUDIT)
#   - all uncertainty: paired week-block bootstrap (1000, seed 42); FDR = BH
#
# Parts:
#   A  Spearman rho + partial Spearman (controlling current_DHW), raw and
#      FDR-adjusted p, per basin
#   B  strictly nested explanatory models for R:
#        M0: current_DHW
#        M1: M0 + dDHW_4w + cumsum_4w + cumsum_12w
#        M2: M1 + chl + par + wind
#      OLS (primary) + pre-fixed rank-OLS (robust fallback); 5-fold CV
#      grouped by week (fixed form, no tuning); CV R², CV MAE
#   C  mechanism contribution ΔR²(M1−M0), ΔR²(M2−M1) with bootstrap CI;
#      standardized coefficients; GBR-vs-CARIB direction consistency
#   D  severe y(t+4) ranking: PersA → +thermal trajectory (pre-specified
#      training-free equal-weight rank mean) → +nonthermal context → TSAR;
#      ROC-AUC/PR-AUC increments + paired bootstrap CI; closure fraction
#
# Outputs -> results/residual_mechanism_decomposition/
#
# Run (piecf_env):
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_residual_mechanism_decomposition.py

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

OUT = RESULTS_DIR / "residual_mechanism_decomposition"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"

SEV = 0.8
N_BOOT = 1000
BOOT_SEED = 42
N_FOLDS = 5

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
CH = {c: DATASET_ORDER.index(c) for c in DATASET_ORDER}

BASINS = {
    "SCS":   {"file": "scs_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2011, 2015), test_years=(2016, 2019)),
              "years": (2016, 2019)},
    "GBR":   {"file": "gbr_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2014, 2016), test_years=(2017, 2020)),
              "years": (2017, 2020)},
    "CARIB": {"file": "caribbean_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2014, 2016), test_years=(2017, 2020)),
              "years": (2017, 2020)},
}

# variable groups (pre-specified)
RAW_VARS = ["current_DHW", "dDHW_4w", "d_hotspot_dt", "cumsum_4w", "cumsum_12w",
            "ssta", "hotspot", "chl", "par", "wind"]
THERMAL_TRAJ = ["dDHW_4w", "cumsum_4w", "cumsum_12w"]      # + current_DHW in M1
PARTIAL_VARS = ["dDHW_4w", "d_hotspot_dt", "cumsum_4w", "cumsum_12w",
                "ssta", "hotspot", "chl", "par", "wind"]    # controlling current_DHW
NONThermal = ["chl", "par", "wind"]

M1_VARS = ["current_DHW"] + THERMAL_TRAJ
M2_VARS = M1_VARS + NONThermal


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def partial_spearman(x, y, z):
    """Partial Spearman of x,y controlling z (rank-OLS residual method)."""
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
    p = 2 * tdist.sf(abs(tstat), n - 3)
    return float(r), float(p)


def week_index_arrays(t_idx):
    weeks, inv = np.unique(t_idx, return_inverse=True)
    return weeks, inv


def week_boot_idx(idx_by_week, rng, n_weeks):
    pick = rng.integers(0, n_weeks, n_weeks)
    return np.concatenate([idx_by_week[w] for w in pick])


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Residual mechanism decomposition — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
rec_md5 = md5(REC_NPZ)
print(f"rec_structural_ablation.npz md5 {rec_md5} "
      f"({'OK' if rec_md5 == REC_MD5 else 'MISMATCH'})")
assert rec_md5 == REC_MD5, "frozen record set hash mismatch"

z = np.load(REC_NPZ)
results = {"experiment": "residual_mechanism_decomposition",
           "generated": datetime.now().isoformat(),
           "record_npz_md5": rec_md5,
           "protocol": {"R": "TSAR(C_anchored, scale 0.4) − PersA_DHW",
                        "bootstrap": {"type": "paired week-block", "n": N_BOOT,
                                      "seed": BOOT_SEED},
                        "cv": {"folds": N_FOLDS, "grouped_by": "issue week",
                               "model_form": "fixed OLS + pre-fixed rank-OLS; no tuning"},
                        "fdr": "Benjamini-Hochberg within family per basin",
                        "interpretation": "associative / representation-level only"},
           "basins": {}}

assoc_rows, nested_rows, boot_rows, mech_rows = [], [], [], []
panel_scores = {}

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

    # ---- candidate mechanism variables at issue time (verbatim channels) ----
    feats = {
        "current_DHW":  persa * 10.0,                       # clipped DHW/10 ×10
        "dDHW_4w":      np.full(len(t_idx), np.nan),
        "d_hotspot_dt": X[t_idx, i_idx, j_idx, CH["d_hotspot_dt"]],
        "cumsum_4w":    X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_4w"]],
        "cumsum_12w":   X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_12w"]],
        "ssta":         X[t_idx, i_idx, j_idx, CH["ssta"]],
        "hotspot":      X[t_idx, i_idx, j_idx, CH["hotspot"]],
        "chl":          X[t_idx, i_idx, j_idx, CH["chlorophyll"]],
        "par":          X[t_idx, i_idx, j_idx, CH["par"]],
        "wind":         X[t_idx, i_idx, j_idx, CH["wind_speed"]],
    }
    y_prev4 = y_full[t_idx - 4, i_idx, j_idx]               # t−4 ≥ 19 for all records
    feats["dDHW_4w"] = persa - y_prev4                      # 4-week change, clipped coord

    # ---- complete-case mask over ALL candidates (identical records everywhere) ----
    mask = np.isfinite(R) & (target >= 0)
    for v in RAW_VARS:
        mask &= np.isfinite(feats[v])
    n_total, n_kept = len(R), int(mask.sum())
    n_sev = int((target[mask] >= SEV).sum())
    nan_counts = {v: int((~np.isfinite(feats[v])).sum()) for v in RAW_VARS}
    print(f"  records n={n_total} -> complete-case n={n_kept} (dropped {n_total-n_kept}), "
          f"severe={n_sev}; NaN per var: {nan_counts}")

    t = t_idx[mask]
    weeks, inv = np.unique(t, return_inverse=True)
    n_weeks = len(weeks)
    idx_by_week = [np.where(inv == w)[0] for w in range(n_weeks)]
    V = {v: feats[v][mask].astype(np.float64) for v in RAW_VARS}
    r_res = R[mask]
    tgt = target[mask]

    rng = np.random.default_rng(BOOT_SEED)

    # ---------------- Part A: associations ----------------
    print("  [A] Spearman + partial Spearman (FDR) ...")
    raw_p, part_p = [], []
    for v in RAW_VARS:
        rho, p = spearmanr(V[v], r_res)
        assoc_rows.append({"basin": basin, "variable": v, "type": "raw",
                           "rho": rho, "p_raw": p})
        raw_p.append(p)
    for v in PARTIAL_VARS:
        pr, pp = partial_spearman(V[v], r_res, V["current_DHW"])
        assoc_rows.append({"basin": basin, "variable": v, "type": "partial_ctrl_DHW",
                           "rho": pr, "p_raw": pp})
        part_p.append(pp)
    # FDR per family per basin
    raw_rows = [r for r in assoc_rows if r["basin"] == basin and r["type"] == "raw"]
    part_rows = [r for r in assoc_rows if r["basin"] == basin and r["type"] == "partial_ctrl_DHW"]
    for r_, q_ in zip(raw_rows, bh_fdr(raw_p)):
        r_["p_fdr"] = q_
    for r_, q_ in zip(part_rows, bh_fdr(part_p)):
        r_["p_fdr"] = q_
    # bootstrap CIs for rho (raw + partial), same week resamples
    for rep in range(N_BOOT):
        idx = week_boot_idx(idx_by_week, rng, n_weeks)
        for r_ in raw_rows:
            rho_b, _ = spearmanr(V[r_["variable"]][idx], r_res[idx])
            r_.setdefault("boot", []).append(rho_b)
        for r_ in part_rows:
            pr_b, _ = partial_spearman(V[r_["variable"]][idx], r_res[idx],
                                       V["current_DHW"][idx])
            r_.setdefault("boot", []).append(pr_b)
    for r_ in raw_rows + part_rows:
        b = np.array(r_.pop("boot"), dtype=float)
        b = b[np.isfinite(b)]
        r_["ci_lo"] = float(np.percentile(b, 2.5))
        r_["ci_hi"] = float(np.percentile(b, 97.5))

    # ---------------- Part B: nested models ----------------
    print("  [B] nested M0/M1/M2 (OLS + rank-OLS, week-grouped CV) ...")
    def fit_models(Xd, yv, folds):
        oof = np.zeros_like(yv); ins = []
        gkf = GroupKFold(n_splits=folds)
        for tr, te in gkf.split(Xd, yv, groups=inv):
            m = LinearRegression().fit(Xd[tr], yv[tr])
            oof[te] = m.predict(Xd[te])
            ins.append(m)
        full = LinearRegression().fit(Xd, yv)
        r2_cv = 1 - np.sum((yv - oof) ** 2) / np.sum((yv - yv.mean()) ** 2)
        mae_cv = float(np.mean(np.abs(yv - oof)))
        r2_in = full.score(Xd, yv)
        return full, r2_in, r2_cv, mae_cv, oof

    def design(names):
        Xm = np.column_stack([V[n] for n in names])
        mu, sd = Xm.mean(0), Xm.std(0)
        sd[sd == 0] = 1.0
        return (Xm - mu) / sd

    models = {"M0": ["current_DHW"], "M1": M1_VARS, "M2": M2_VARS}
    oof_store, coefs = {}, {}
    for mname, names in models.items():
        Xd = design(names)
        full, r2_in, r2_cv, mae_cv, oof = fit_models(Xd, r_res, N_FOLDS)
        oof_store[mname] = oof
        coefs[mname] = dict(zip(names, full.coef_.round(6).tolist()))
        nested_rows.append({"basin": basin, "model": mname, "variant": "OLS",
                            "predictors": "+".join(names), "n": n_kept,
                            "r2_insample": r2_in, "r2_cv": r2_cv, "mae_cv": mae_cv})
        print(f"    {mname} OLS  R2cv={r2_cv:.4f} MAEcv={mae_cv:.5f} R2in={r2_in:.4f}")
        # pre-fixed rank-OLS robust fallback
        Xr = np.column_stack([rankdata(V[n]) for n in names])
        yr = rankdata(r_res)
        full_r, r2_in_r, r2_cv_r, _, _ = fit_models(Xr, yr, N_FOLDS)
        nested_rows.append({"basin": basin, "model": mname, "variant": "rank_OLS",
                            "predictors": "+".join(names), "n": n_kept,
                            "r2_insample": r2_in_r, "r2_cv": r2_cv_r, "mae_cv": np.nan})
        print(f"    {mname} rank R2cv={r2_cv_r:.4f} R2in={r2_in_r:.4f}")

    # ---------------- Part C: mechanism contribution ----------------
    print("  [C] ΔR² contribution + direction consistency ...")
    X0, X1, X2 = design(models["M0"]), design(models["M1"]), design(models["M2"])
    m0, r0i, _, _, _ = fit_models(X0, r_res, N_FOLDS)
    m1, r1i, _, _, _ = fit_models(X1, r_res, N_FOLDS)
    m2, r2i, _, _, _ = fit_models(X2, r_res, N_FOLDS)
    d1_pt, d2_pt = r1i - r0i, r2i - r1i
    d1_b, d2_b = [], []
    rng_c = np.random.default_rng(BOOT_SEED)
    for _ in range(N_BOOT):
        idx = week_boot_idx(idx_by_week, rng_c, n_weeks)
        r2s = []
        for Xd in (X0, X1, X2):
            mm = LinearRegression().fit(Xd[idx], r_res[idx])
            r2s.append(mm.score(Xd[idx], r_res[idx]))
        d1_b.append(r2s[1] - r2s[0]); d2_b.append(r2s[2] - r2s[1])
    for step, pt, bb in (("M1_minus_M0", d1_pt, d1_b), ("M2_minus_M1", d2_pt, d2_b)):
        boot_rows.append({"basin": basin, "stat": f"deltaR2_{step}",
                          "point": pt, "ci_lo": float(np.percentile(bb, 2.5)),
                          "ci_hi": float(np.percentile(bb, 97.5))})
        print(f"    ΔR² {step}: {pt:+.4f} [{np.percentile(bb,2.5):+.4f}, {np.percentile(bb,97.5):+.4f}]")
    # standardized coefficients from M2 + partial associations (direction check)
    for v in M2_VARS:
        cand = [r["rho"] for r in part_rows if r["variable"] == v]
        mech_rows.append({"basin": basin, "variable": v, "std_coef_M2": coefs["M2"][v],
                          "partial_rho": cand[0] if cand else np.nan})

    # ---------------- Part D: training-free incremental ranking ----------------
    print("  [D] PersA → +thermal → +context → TSAR (severe y(t+4)) ...")
    sev = (tgt >= SEV).astype(int)
    def pct(x):
        r = rankdata(x, method="average") / len(x)
        return r
    s_persa = pct(V["current_DHW"])
    s_thermal = np.mean([s_persa] + [pct(V[v]) for v in THERMAL_TRAJ], axis=0)
    s_full = np.mean([s_thermal] + [pct(V[v]) for v in NONThermal], axis=0)
    s_tsar = pct(tsar[mask])                                # TSAR score (raw; AUC invariant to monotone map)
    scores = {"PersA": s_persa, "PersA+thermal": s_thermal,
              "PersA+thermal+context": s_full, "TSAR": s_tsar}
    panel_scores[basin] = {"target": tgt, **{k: v for k, v in scores.items()}}
    rng_d = np.random.default_rng(BOOT_SEED)
    boot_auc = {k: {"roc": [], "pr": []} for k in scores}
    for _ in range(N_BOOT):
        idx = week_boot_idx(idx_by_week, rng_d, n_weeks)
        for k, s in scores.items():
            try:
                boot_auc[k]["roc"].append(roc_auc_score(sev[idx], s[idx]))
                boot_auc[k]["pr"].append(average_precision_score(sev[idx], s[idx]))
            except ValueError:
                pass
    for k in scores:
        for met in ("roc", "pr"):
            pt = (roc_auc_score(sev, scores[k]) if met == "roc"
                  else average_precision_score(sev, scores[k]))
            b = np.array(boot_auc[k][met])
            boot_rows.append({"basin": basin, "stat": f"{met}_{k}",
                              "point": float(pt), "ci_lo": float(np.percentile(b, 2.5)),
                              "ci_hi": float(np.percentile(b, 97.5))})
    # paired increments
    pairs = [("PersA+thermal", "PersA"), ("PersA+thermal+context", "PersA+thermal"),
             ("TSAR", "PersA"), ("TSAR", "PersA+thermal")]
    for a, b_ in pairs:
        for met in ("roc", "pr"):
            key = f"{met}_{a}_minus_{b_}"
            # paired diffs need per-replicate arrays; recompute diffs from stored boots
            d = np.array(boot_auc[a][met]) - np.array(boot_auc[b_][met])
            pt = ((roc_auc_score(sev, scores[a]) if met == "roc"
                   else average_precision_score(sev, scores[a]))
                  - (roc_auc_score(sev, scores[b_]) if met == "roc"
                     else average_precision_score(sev, scores[b_])))
            boot_rows.append({"basin": basin, "stat": key, "point": float(pt),
                              "ci_lo": float(np.percentile(d, 2.5)),
                              "ci_hi": float(np.percentile(d, 97.5))})
    roc_persa = roc_auc_score(sev, s_persa); roc_th = roc_auc_score(sev, s_thermal)
    roc_tsar = roc_auc_score(sev, s_tsar)
    denom = roc_tsar - roc_persa
    # closure is only meaningful when TSAR has a positive ROC increment over
    # PersA to explain; otherwise undefined (pre-specified floor 1e-3)
    closure = ((roc_th - roc_persa) / denom) if denom > 1e-3 else np.nan
    print(f"    ROC closure (thermal / TSAR increment): "
          f"{'undefined (TSAR increment <= 0.001)' if not np.isfinite(closure) else format(closure, '.3f')}")
    boot_rows.append({"basin": basin, "stat": "roc_closure_thermal_over_TSAR",
                      "point": float(closure), "ci_lo": np.nan, "ci_hi": np.nan})

    results["basins"][basin] = {
        "n_total": n_total, "n_complete_case": n_kept, "n_dropped": n_total - n_kept,
        "n_severe": n_sev, "nan_counts": nan_counts,
        "n_weeks": n_weeks, "coefs_M2": coefs["M2"],
        "r2_insample": {"M0": r0i, "M1": r1i, "M2": r2i},
    }
    del ds, X
    import torch
    torch.cuda.empty_cache()

# ---------------- write outputs ----------------
pd.DataFrame(assoc_rows).to_csv(OUT / "associations.csv", index=False)
pd.DataFrame(nested_rows).to_csv(OUT / "nested_models.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap.csv", index=False)
pd.DataFrame(mech_rows).to_csv(OUT / "mechanism_contribution.csv", index=False)
np.savez_compressed(OUT / "panel_d_scores.npz", **{
    f"{b}_{k}": v for b, d in panel_scores.items() for k, v in d.items()})

with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=float)

print(f"\nOutputs -> {OUT}")
print("Done.")