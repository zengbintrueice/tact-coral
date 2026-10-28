# compute_oos_deltas_mechanism.py
#
# Strict OUT-OF-SAMPLE upgrade of the in-sample ΔR² from the Residual
# Mechanism Decomposition (results/residual_mechanism_decomposition/).
#
# FIXED PROTOCOL:
#   - frozen rec_structural_ablation.npz (md5-verified); R = TSAR(seed-42) − PersA
#   - per basin SCS/GBR/CARIB; M0 = DHW; M1 = M0 + dDHW_4w + cumsum_4w +
#     cumsum_12w; M2 = M1 + chl + par + wind
#   - fixed OLS (primary) + pre-fixed rank-OLS (robustness); NO tuning
#   - 5-fold GroupKFold, groups = issue week; per fold: models fitted
#     independently on training weeks, predicting ONLY unseen issue weeks
#   - per-record OOS predictions saved; M0/M1/M2 use IDENTICAL test records
#   - pooled OOS R² = 1 − SSE/SST (SST from the full OOS mean); the fold-mean
#     R² is NOT the primary statistic (reported as a stability check)
#   - ΔR²(M1−M0), ΔR²(M2−M1) on the common pooled OOS records
#   - paired week-block bootstrap (1000, seed 42): the SAME resampled weeks are
#     applied to M0/M1/M2 predictions (paired); CIs for pooled OOS R² + both ΔR²
#   - OOS RMSE/MAE; OOS R² may be negative (never truncated/replaced)
#   - standardization fitted WITHIN each training fold, applied to that fold's
#     test weeks; no global preprocessing
#   - no future target / test-week information in any model choice
#
# Compute environment: piecf_env. No GPU is used BY DESIGN — this is a pure
# numpy/sklearn statistics task with no neural-network inference.
#
# Outputs -> results/residual_mechanism_decomposition/oos/

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "residual_mechanism_decomposition" / "oos"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
N_FOLDS = 5
N_BOOT = 1000
BOOT_SEED = 42

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

# complete-case mask over ALL mechanism-decomposition variables (identical
# record set as the in-sample analysis; models use only their own columns)
RAW_VARS = ["current_DHW", "dDHW_4w", "d_hotspot_dt", "cumsum_4w", "cumsum_12w",
            "ssta", "hotspot", "chl", "par", "wind"]
MODELS = {"M0": ["current_DHW"],
          "M1": ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w"],
          "M2": ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w",
                 "chl", "par", "wind"]}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ecdf_map(train_vals, x):
    """Map x through the training-fold empirical CDF (rank in [0,1])."""
    s = np.sort(train_vals)
    return np.searchsorted(s, x, side="right") / len(s)


def pooled_r2(y, yhat):
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Strict OOS ΔR² upgrade — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(REC_NPZ) == REC_MD5, "frozen record set hash mismatch"
print(f"frozen records OK ({REC_MD5})")

z = np.load(REC_NPZ)
pred_frames, metric_rows, delta_rows, boot_rows = [], [], [], []
summary = {"experiment": "oos_delta_upgrade",
           "generated": datetime.now().isoformat(),
           "record_npz_md5": REC_MD5,
           "protocol": {"cv": f"GroupKFold {N_FOLDS} grouped by issue week",
                        "standardization": "fit within training fold only",
                        "primary_stat": "pooled OOS R² (1−SSE/SST), fold R² as stability check",
                        "bootstrap": {"type": "paired week-block on pooled OOS predictions",
                                      "n": N_BOOT, "seed": BOOT_SEED},
                        "negative_r2": "allowed, never truncated",
                        "env": "piecf_env (CUDA available; GPU unused by design — pure statistics)"},
           "basins": {}}

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
    R = tsar - persa

    feats = {"current_DHW": persa * 10.0,
             "dDHW_4w":     persa - y_full[t_idx - 4, i_idx, j_idx],
             "d_hotspot_dt": X[t_idx, i_idx, j_idx, CH["d_hotspot_dt"]],
             "cumsum_4w":   X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_4w"]],
             "cumsum_12w":  X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_12w"]],
             "ssta":        X[t_idx, i_idx, j_idx, CH["ssta"]],
             "hotspot":     X[t_idx, i_idx, j_idx, CH["hotspot"]],
             "chl":         X[t_idx, i_idx, j_idx, CH["chlorophyll"]],
             "par":         X[t_idx, i_idx, j_idx, CH["par"]],
             "wind":        X[t_idx, i_idx, j_idx, CH["wind_speed"]]}
    mask = np.isfinite(R)
    for v in RAW_VARS:
        mask &= np.isfinite(feats[v])
    V = {v: feats[v][mask].astype(np.float64) for v in RAW_VARS}
    y = R[mask]
    t = t_idx[mask]; ii = i_idx[mask]; jj = j_idx[mask]
    print(f"  records n={len(y)} (complete-case; dropped {len(R)-len(y)}), "
          f"weeks={len(np.unique(t))}")

    # ---- fold-internal CV: identical test records across M0/M1/M2 ----
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(np.zeros(len(y)), groups=t))
    oos = {m: np.full(len(y), np.nan) for m in MODELS}
    oos_rank = {m: np.full(len(y), np.nan) for m in MODELS}
    y_rank_oos = np.full(len(y), np.nan)   # fold-internal train-ECDF response
    fold_r2 = {m: [] for m in MODELS}
    fold_r2_rank = {m: [] for m in MODELS}
    for f, (tr, te) in enumerate(folds):
        assert len(set(t[tr]) & set(t[te])) == 0, "week leakage across folds"
        for m, names in MODELS.items():
            Xtr = np.column_stack([V[v][tr] for v in names])
            Xte = np.column_stack([V[v][te] for v in names])
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            mod = LinearRegression().fit(Ztr, y[tr])          # OLS, fold-internal
            oos[m][te] = mod.predict(Zte)
            # rank-OLS: ECDF maps fitted on the training fold only
            Ztr_r = np.column_stack([ecdf_map(V[v][tr], V[v][tr]) for v in names])
            Zte_r = np.column_stack([ecdf_map(V[v][tr], V[v][te]) for v in names])
            ytr_r = ecdf_map(y[tr], y[tr])
            yte_r = ecdf_map(y[tr], y[te])
            mod_r = LinearRegression().fit(Ztr_r, ytr_r)
            oos_rank[m][te] = mod_r.predict(Zte_r)
            y_rank_oos[te] = yte_r                            # fold-internal scale
            fold_r2[m].append(pooled_r2(y[te], oos[m][te]))
            fold_r2_rank[m].append(pooled_r2(yte_r, oos_rank[m][te]))
        print(f"  fold {f+1}/{N_FOLDS}: test weeks={len(np.unique(t[te]))}, "
              f"records={len(te)}")

    # ---- pooled OOS metrics (primary) ----
    yhat_df = pd.DataFrame({"basin": basin, "t": t, "i": ii, "j": jj, "y_R": y})
    for m in MODELS:
        yhat_df[f"pred_{m}"] = oos[m]
        yhat_df[f"pred_{m}_rank"] = oos_rank[m]
    pred_frames.append(yhat_df)

    base = {"basin": basin, "n": len(y), "n_folds": N_FOLDS}
    for m in MODELS:
        r2p = pooled_r2(y, oos[m])
        r2p_rank = pooled_r2(y_rank_oos, oos_rank[m])
        rmse = float(np.sqrt(np.mean((y - oos[m]) ** 2)))
        mae = float(np.mean(np.abs(y - oos[m])))
        metric_rows.append({"basin": basin, "model": m, "variant": "OLS",
                            "oos_r2_pooled": r2p, "oos_rmse": rmse, "oos_mae": mae,
                            **{f"fold{k+1}_r2": fold_r2[m][k] for k in range(N_FOLDS)}})
        metric_rows.append({"basin": basin, "model": m, "variant": "rank_OLS",
                            "oos_r2_pooled": r2p_rank,
                            "oos_rmse": np.nan, "oos_mae": np.nan,
                            **{f"fold{k+1}_r2": fold_r2_rank[m][k] for k in range(N_FOLDS)}})
        print(f"  {m} OLS  pooled OOS R²={r2p:+.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
        print(f"  {m} rank pooled OOS R²={r2p_rank:+.4f} (rank scale)")
        print(f"       per-fold R²: " + " ".join(f"{v:+.4f}" for v in fold_r2[m]))

    # ---- paired deltas on common OOS records ----
    r2 = {m: pooled_r2(y, oos[m]) for m in MODELS}
    r2r = {m: pooled_r2(y_rank_oos, oos_rank[m]) for m in MODELS}
    for step, a, b in (("M1_minus_M0", "M1", "M0"), ("M2_minus_M1", "M2", "M1")):
        delta_rows.append({"basin": basin, "contrast": f"deltaR2_{step}",
                           "variant": "OLS", "point": r2[a] - r2[b]})
        delta_rows.append({"basin": basin, "contrast": f"deltaR2_{step}",
                           "variant": "rank_OLS", "point": r2r[a] - r2r[b]})
        print(f"  ΔR² {step} OLS={r2[a]-r2[b]:+.4f}  rank={r2r[a]-r2r[b]:+.4f}")

    # ---- paired week-block bootstrap on pooled OOS predictions ----
    weeks, inv = np.unique(t, return_inverse=True)
    idx_by_week = [np.where(inv == w)[0] for w in range(len(weeks))]
    rng = np.random.default_rng(BOOT_SEED)
    boots = {m: {"r2": []} for m in MODELS}
    boots_rank = {m: {"r2": []} for m in MODELS}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(weeks), len(weeks))
        idx = np.concatenate([idx_by_week[w] for w in pick])
        for m in MODELS:
            boots[m]["r2"].append(pooled_r2(y[idx], oos[m][idx]))
            boots_rank[m]["r2"].append(pooled_r2(y_rank_oos[idx], oos_rank[m][idx]))
    for m in MODELS:
        a = np.array(boots[m]["r2"]); ar = np.array(boots_rank[m]["r2"])
        boot_rows.append({"basin": basin, "stat": f"oos_r2_{m}", "variant": "OLS",
                          "point": r2[m], "ci_lo": float(np.percentile(a, 2.5)),
                          "ci_hi": float(np.percentile(a, 97.5))})
        boot_rows.append({"basin": basin, "stat": f"oos_r2_{m}", "variant": "rank_OLS",
                          "point": r2r[m], "ci_lo": float(np.percentile(ar, 2.5)),
                          "ci_hi": float(np.percentile(ar, 97.5))})
    for step, a, b in (("M1_minus_M0", "M1", "M0"), ("M2_minus_M1", "M2", "M1")):
        d = np.array(boots[a]["r2"]) - np.array(boots[b]["r2"])
        dr = np.array(boots_rank[a]["r2"]) - np.array(boots_rank[b]["r2"])
        for variant, dd, pt in (("OLS", d, r2[a] - r2[b]),
                                ("rank_OLS", dr, r2r[a] - r2r[b])):
            boot_rows.append({"basin": basin, "stat": f"deltaR2_{step}",
                              "variant": variant, "point": pt,
                              "ci_lo": float(np.percentile(dd, 2.5)),
                              "ci_hi": float(np.percentile(dd, 97.5))})
        print(f"  ΔR² {step} OOS OLS 95% CI [{np.percentile(d,2.5):+.4f}, "
              f"{np.percentile(d,97.5):+.4f}]; rank [{np.percentile(dr,2.5):+.4f}, "
              f"{np.percentile(dr,97.5):+.4f}]")

    summary["basins"][basin] = {
        "n": len(y), "n_weeks": len(weeks),
        "oos_r2_pooled": {m: r2[m] for m in MODELS},
        "oos_r2_pooled_rank": {m: r2r[m] for m in MODELS},
        "deltaR2_M1_minus_M0": r2["M1"] - r2["M0"],
        "deltaR2_M2_minus_M1": r2["M2"] - r2["M1"],
        "fold_r2": {m: fold_r2[m] for m in MODELS},
    }
    del ds, X

# ---- outputs ----
pd.concat(pred_frames, ignore_index=True).to_csv(OUT / "oos_predictions.csv", index=False)
pd.DataFrame(metric_rows).to_csv(OUT / "oos_metrics.csv", index=False)
pd.DataFrame(delta_rows).to_csv(OUT / "oos_deltas.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "oos_bootstrap.csv", index=False)
with open(OUT / "summary_oos.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

# ---- AUDIT_OOS.md ----
B = summary["basins"]
def g(basin, stat, variant="OLS"):
    r = [r for r in boot_rows if r["basin"] == basin and r["stat"] == stat
         and r["variant"] == variant][0]
    return r

L = ["# AUDIT_OOS — Strict out-of-sample ΔR² upgrade", "",
     f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ",
     f"Frozen records: `rec_structural_ablation.npz` md5 `{REC_MD5}` (verified).",
     f"Environment: piecf_env; **GPU unused by design** — pure numpy/sklearn "
     f"statistics, no neural-network inference.", "",
     "## Protocol adherence", "",
     "- R = TSAR(seed-42 C_anchored, scale 0.4) − PersA_DHW; per-basin analyses "
     "on the same complete-case record set as the in-sample decomposition "
     "(SCS dropped 87 PAR-NaN records; GBR/CARIB none).",
     f"- 5-fold GroupKFold grouped by issue week; fold-internal assertion that "
     "train/test weeks are disjoint (no week leakage).",
     "- Standardization (means/SDs) fitted **within each training fold** and "
     "applied to that fold's test weeks; no global preprocessing.",
     "- Rank-OLS variant: training-fold empirical-CDF maps for predictors and "
     "response, also fold-internal; pooled on the rank scale.",
     "- Primary statistic: **pooled OOS R² = 1 − SSE/SST** over all OOS "
     "predictions; per-fold R² reported only as a stability check; the fold "
     "average is not used.",
     "- ΔR²(M1−M0), ΔR²(M2−M1) computed on the pooled OOS records, which are "
     "identical across M0/M1/M2 by construction (same folds, same records).",
     "- Bootstrap: paired week-block (1,000, seed 42) on the pooled OOS "
     "predictions — the SAME resampled weeks are applied to M0/M1/M2, so the "
     "ΔR² CIs are paired.",
     "- Negative OOS R² is allowed and reported without truncation.",
     "- No future target or test-week information enters any model choice "
     "(model form fixed a priori; no tuning anywhere).", "",
     "## Results (pooled OOS)", "",
     "| Basin | Model | OLS OOS R² [95% CI] | rank-OLS OOS R² [95% CI] |",
     "|---|---|---|---|"]
for b in B:
    for m in ("M0", "M1", "M2"):
        o, r_ = g(b, f"oos_r2_{m}"), g(b, f"oos_r2_{m}", "rank_OLS")
        L.append(f"| {b} | {m} | {o['point']:+.4f} [{o['ci_lo']:+.4f}, {o['ci_hi']:+.4f}] "
                 f"| {r_['point']:+.4f} [{r_['ci_lo']:+.4f}, {r_['ci_hi']:+.4f}] |")
L += ["", "| Basin | ΔR² contrast | OLS point [95% CI] | rank-OLS point [95% CI] |",
      "|---|---|---|---|"]
for b in B:
    for st in ("deltaR2_M1_minus_M0", "deltaR2_M2_minus_M1"):
        o, r_ = g(b, st), g(b, st, "rank_OLS")
        L.append(f"| {b} | {st.replace('deltaR2_','')} | {o['point']:+.4f} "
                 f"[{o['ci_lo']:+.4f}, {o['ci_hi']:+.4f}] | {r_['point']:+.4f} "
                 f"[{r_['ci_lo']:+.4f}, {r_['ci_hi']:+.4f}] |")
L += ["", "## Interpretation guard", "",
     "OOS ΔR² quantifies out-of-sample incremental explanatory/predictive "
     "information about the residual R. It is NOT a causal mechanism claim: no "
     "intervention, no structural model, and no claim that R 'is produced by' "
     "these variables.", ""]
with open(OUT / "AUDIT_OOS.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print(f"\nOutputs -> {OUT}")
print("Done.")