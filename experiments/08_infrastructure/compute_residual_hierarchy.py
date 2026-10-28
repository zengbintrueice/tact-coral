# compute_residual_hierarchy.py
#
# Residual Hierarchy / Incremental Information Test.
#
# Question: does the TSAR residual R = TSAR − PersA contain predictive
# information BEYOND current DHW, thermal trajectory/persistence and
# nonthermal context — i.e. is R a re-combination of known variables or does
# it carry additional representation-level information?
#
# FIXED PROTOCOL (no retraining, no target fitting, no target tuning, no
# threshold optimisation; predictive/representation-level interpretation ONLY):
#   - frozen records: rec_structural_ablation.npz (md5-verified), seed-42 TSAR
#   - PersA = clip(DHW/10,0,1); R = TSAR − PersA; per-basin analyses
#   - identical frozen records + identical issue-week grouped 5-fold CV
#   - bootstrap: paired week-block, 1000, seed 42 (same resampled weeks across
#     all hierarchy levels — paired deltas)
#
# PRE-SPECIFIED hierarchy scores (training-free; no coefficient is fitted to
# the target — equal-weight mean of within-basin percentile ranks):
#   H0 = PersA
#   H1 = rankmean{PersA, dDHW_4w, cumsum_4w, cumsum_12w}
#   H2 = rankmean{H1 vars, chl, par, wind}
#   H3 = rankmean{H2 vars, R}
#   (TSAR frozen score reported alongside for the consistency check H3 vs TSAR)
#
# Metrics per level: ROC-AUC, PR-AUC, Brier, clipped log-loss (diagnostic:
# H1–H3 are rank means, not calibrated probabilities).
# Paired deltas: H1−H0, H2−H1, H3−H2, H3−TSAR for ΔROC/ΔPR/ΔBrier + 95% CI.
#
# Residual-explanation test (fixed form, week-grouped 5-fold CV):
#   R ~ H1 predictors            {PersA, dDHW_4w, cumsum_4w, cumsum_12w}
#   R ~ H1 + H2 predictors       (+ chl, par, wind)
#   out-of-sample R², ΔR²(out-of-sample) + paired bootstrap CI
#
# Outputs -> results/residual_hierarchy/
#
# Run (piecf_env):
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_residual_hierarchy.py

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "residual_hierarchy"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"

SEV = 0.8
N_BOOT = 1000
BOOT_SEED = 42
N_FOLDS = 5
LOGLOSS_CLIP = 1e-6                      # pre-specified clipping for log-loss

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

H1_VARS = ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w"]
H2_VARS = H1_VARS + ["chl", "par", "wind"]
LEVELS = {"H0": None, "H1": H1_VARS, "H2": H2_VARS, "H3": H2_VARS + ["R"]}
ADJ_PAIRS = [("H1", "H0"), ("H2", "H1"), ("H3", "H2"), ("H3", "TSAR")]
EXPL_H1 = ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w"]
EXPL_H2 = EXPL_H1 + ["chl", "par", "wind"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x):
    return rankdata(x, method="average") / len(x)


def metrics_all(score, sev_label, target):
    y = sev_label
    out = {}
    try:
        out["roc"] = float(roc_auc_score(y, score))
    except ValueError:
        out["roc"] = np.nan
    try:
        out["pr"] = float(average_precision_score(y, score))
    except ValueError:
        out["pr"] = np.nan
    out["brier"] = float(np.mean((score - y) ** 2))
    try:
        out["logloss"] = float(log_loss(y, np.clip(score, LOGLOSS_CLIP, 1 - LOGLOSS_CLIP),
                                        labels=[0, 1]))
    except Exception:
        out["logloss"] = np.nan
    return out


def cv_oof_r2(Xd, yv, groups, n_folds=N_FOLDS):
    """Out-of-sample R² via deterministic week-grouped 5-fold CV."""
    oof = np.zeros_like(yv)
    for tr, te in GroupKFold(n_splits=n_folds).split(Xd, yv, groups=groups):
        m = LinearRegression().fit(Xd[tr], yv[tr])
        oof[te] = m.predict(Xd[te])
    return float(1 - np.sum((yv - oof) ** 2) / np.sum((yv - yv.mean()) ** 2))


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Residual hierarchy / incremental information — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
rec_md5 = md5(REC_NPZ)
print(f"rec_structural_ablation.npz md5 {rec_md5} "
      f"({'OK' if rec_md5 == REC_MD5 else 'MISMATCH'})")
assert rec_md5 == REC_MD5

z = np.load(REC_NPZ)
results = {"experiment": "residual_hierarchy",
           "generated": datetime.now().isoformat(),
           "record_npz_md5": rec_md5,
           "protocol": {"hierarchy_scores": "training-free equal-weight rank means "
                        "(pre-specified; no coefficient fitted to target)",
                        "R": "TSAR(C_anchored, scale 0.4) − PersA",
                        "bootstrap": {"type": "paired week-block", "n": N_BOOT,
                                      "seed": BOOT_SEED},
                        "cv": {"folds": N_FOLDS, "grouped_by": "issue week"},
                        "logloss_clip": LOGLOSS_CLIP,
                        "interpretation": "predictive / representation-level only"},
           "basins": {}}

hier_rows, delta_rows, expl_rows, boot_rows = [], [], [], []
panel = {}

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

    feats = {
        "current_DHW":  persa * 10.0,
        "dDHW_4w":      persa - y_full[t_idx - 4, i_idx, j_idx],
        "cumsum_4w":    X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_4w"]],
        "cumsum_12w":   X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_12w"]],
        "chl":          X[t_idx, i_idx, j_idx, CH["chlorophyll"]],
        "par":          X[t_idx, i_idx, j_idx, CH["par"]],
        "wind":         X[t_idx, i_idx, j_idx, CH["wind_speed"]],
    }
    mask = np.isfinite(R) & np.isfinite(target)
    for v in feats:
        mask &= np.isfinite(feats[v])
    n_total, n_kept = len(R), int(mask.sum())
    n_sev = int((target[mask] >= SEV).sum())
    print(f"  records n={n_total} -> complete-case n={n_kept} (dropped {n_total-n_kept}), severe={n_sev}")

    V = {v: feats[v][mask].astype(np.float64) for v in feats}
    r_res = R[mask]
    tgt = target[mask]
    sev = (tgt >= SEV).astype(int)
    tsar_k = tsar[mask]
    weeks, inv = np.unique(t_idx[mask], return_inverse=True)
    n_weeks = len(weeks)
    idx_by_week = [np.where(inv == w)[0] for w in range(n_weeks)]

    def boot_idx(rng):
        pick = rng.integers(0, n_weeks, n_weeks)
        return np.concatenate([idx_by_week[w] for w in pick])

    # ---- hierarchy scores (pre-specified rank means) ----
    rp = {v: pct(V[v]) for v in V}
    rp["R"] = pct(r_res)
    scores = {"H0": pct(V["current_DHW"]),
              "H1": np.mean([rp[v] for v in H1_VARS], axis=0),
              "H2": np.mean([rp[v] for v in H2_VARS], axis=0),
              "H3": np.mean([rp[v] for v in H2_VARS + ["R"]], axis=0),
              "TSAR": pct(tsar_k)}
    panel[basin] = {**{f"score_{k}": v for k, v in scores.items()},
                    "target": tgt, "R": r_res,
                    **{f"feat_{v}": V[v] for v in V}, "inv_week": inv}

    # ---- point metrics per level ----
    for lname, s in scores.items():
        m = metrics_all(s, sev, tgt)
        hier_rows.append({"basin": basin, "level": lname, "n": n_kept,
                          "n_severe": n_sev, **m})
        print(f"  {lname:5s} ROC={m['roc']:.4f} PR={m['pr']:.4f} "
              f"Brier={m['brier']:.5f} LogLoss={m['logloss']:.4f}")

    # ---- paired week-block bootstrap (shared picks across levels) ----
    rng = np.random.default_rng(BOOT_SEED)
    bootm = {lname: {"roc": [], "pr": [], "brier": [], "logloss": []}
             for lname in scores}
    for _ in range(N_BOOT):
        idx = boot_idx(rng)
        y_b = sev[idx]
        for lname, s in scores.items():
            m = metrics_all(s[idx], y_b, tgt[idx])
            for k in bootm[lname]:
                bootm[lname][k].append(m[k])
    for lname in scores:
        for k, arr in bootm[lname].items():
            a = np.array(arr, dtype=float)
            a = a[np.isfinite(a)]
            boot_rows.append({"basin": basin, "stat": f"{k}_{lname}",
                              "point": dict((r["level"], r[k]) for r in hier_rows
                                            if r["basin"] == basin)[lname],
                              "ci_lo": float(np.percentile(a, 2.5)),
                              "ci_hi": float(np.percentile(a, 97.5))})

    # ---- paired deltas ----
    for a, b in ADJ_PAIRS:
        for k in ("roc", "pr", "brier"):
            d = np.array(bootm[a][k]) - np.array(bootm[b][k])
            pa = dict((r["level"], r[k]) for r in hier_rows if r["basin"] == basin)
            delta_rows.append({"basin": basin, "contrast": f"{a}_minus_{b}",
                               "metric": k, "point": pa[a] - pa[b],
                               "ci_lo": float(np.percentile(d, 2.5)),
                               "ci_hi": float(np.percentile(d, 97.5))})
            if k in ("roc", "pr"):
                print(f"  Δ{k.upper()} {a}−{b}: {pa[a]-pa[b]:+.4f} "
                      f"[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")

    # ---- residual-explanation test (fixed form, week-grouped CV) ----
    def design(names):
        Xm = np.column_stack([V[n] for n in names])
        mu, sd = Xm.mean(0), Xm.std(0)
        sd[sd == 0] = 1.0
        return (Xm - mu) / sd

    r2_h1 = cv_oof_r2(design(EXPL_H1), r_res, inv)
    r2_h2 = cv_oof_r2(design(EXPL_H2), r_res, inv)
    # paired bootstrap of the out-of-sample ΔR² (same folds per replicate)
    rng_e = np.random.default_rng(BOOT_SEED)
    d_boot, h1_boot, h2_boot = [], [], []
    X1, X2 = design(EXPL_H1), design(EXPL_H2)
    for _ in range(N_BOOT):
        idx = boot_idx(rng_e)
        g = inv[idx]
        # deterministic grouped folds on the resampled week set
        uw = np.unique(g)
        fold_of = {w: i % N_FOLDS for i, w in enumerate(uw)}
        folds = np.array([fold_of[w] for w in g])
        r2s = []
        for Xd in (X1, X2):
            oof = np.zeros(len(idx))
            for f in range(N_FOLDS):
                tr, te = folds != f, folds == f
                if te.sum() == 0:
                    continue
                m = LinearRegression().fit(Xd[idx][tr], r_res[idx][tr])
                oof[te] = m.predict(Xd[idx][te])
            r2s.append(1 - np.sum((r_res[idx] - oof) ** 2)
                       / np.sum((r_res[idx] - r_res[idx].mean()) ** 2))
        h1_boot.append(r2s[0]); h2_boot.append(r2s[1]); d_boot.append(r2s[1] - r2s[0])
    expl_rows.append({"basin": basin, "model": "R~H1", "r2_oos": r2_h1,
                      "ci_lo": float(np.percentile(h1_boot, 2.5)),
                      "ci_hi": float(np.percentile(h1_boot, 97.5))})
    expl_rows.append({"basin": basin, "model": "R~H1+H2", "r2_oos": r2_h2,
                      "ci_lo": float(np.percentile(h2_boot, 2.5)),
                      "ci_hi": float(np.percentile(h2_boot, 97.5))})
    expl_rows.append({"basin": basin, "model": "deltaR2_H2_over_H1",
                      "r2_oos": r2_h2 - r2_h1,
                      "ci_lo": float(np.percentile(d_boot, 2.5)),
                      "ci_hi": float(np.percentile(d_boot, 97.5))})
    print(f"  R~H1 R2oos={r2_h1:.4f}; R~H1+H2 R2oos={r2_h2:.4f} "
          f"(Δ={r2_h2-r2_h1:+.4f} [{np.percentile(d_boot,2.5):+.4f},"
          f"{np.percentile(d_boot,97.5):+.4f}])")

    results["basins"][basin] = {
        "n_total": n_total, "n_complete_case": n_kept, "n_dropped": n_total - n_kept,
        "n_severe": n_sev, "n_weeks": n_weeks,
        "r2_oos_R_given_H1": r2_h1, "r2_oos_R_given_H1H2": r2_h2,
        "delta_r2_oos": r2_h2 - r2_h1,
    }
    del ds, X
    import torch
    torch.cuda.empty_cache()

# ---- verdicts (pre-specified) ----
def get_delta(basin, contrast, metric):
    row = [r for r in delta_rows if r["basin"] == basin
           and r["contrast"] == contrast and r["metric"] == metric][0]
    return row

verdicts = {}
for basin in BASINS:
    d32_roc = get_delta(basin, "H3_minus_H2", "roc")
    d32_pr = get_delta(basin, "H3_minus_H2", "pr")
    established = bool((d32_roc["ci_lo"] > 0) or (d32_pr["ci_lo"] > 0))
    verdicts[basin] = {
        "H3_over_H2_roc": {k: d32_roc[k] for k in ("point", "ci_lo", "ci_hi")},
        "H3_over_H2_pr": {k: d32_pr[k] for k in ("point", "ci_lo", "ci_hi")},
        "residual_extra_info_established": established,
    }
results["verdicts"] = verdicts

# ---- outputs ----
pd.DataFrame(hier_rows).to_csv(OUT / "hierarchy_metrics.csv", index=False)
pd.DataFrame(delta_rows).to_csv(OUT / "paired_deltas.csv", index=False)
pd.DataFrame(expl_rows).to_csv(OUT / "residual_explanation.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap.csv", index=False)
np.savez_compressed(OUT / "panel_scores.npz", **{
    f"{b}_{k}": v for b, d in panel.items() for k, v in d.items()})
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=float)
print(f"\nOutputs -> {OUT}")
print("Done.")