# compute_combination_rule_sensitivity.py
#
# Combination-rule sensitivity of the residual-hierarchy conclusions.
#
# The published hierarchy (results/residual_hierarchy/) builds H1/H2/H3 with a
# pre-specified TRAINING-FREE equal-weight mean of within-basin percentile
# ranks. A reviewer may ask whether the conclusions depend on that single
# choice. This script recomputes the hierarchy under five fixed, training-free
# combination rules and reports the three decision-relevant contrasts per rule:
#   C1 = H1 − H0 (thermal trajectory adds ranking information)
#   C2 = H2 − H1 (nonthermal context adds ranking information)
#   C3 = H3 − H2 (the residual adds information beyond explicit context)
# plus the H3−TSAR consistency check.
#
# Rules (all fixed a priori, no tuning, no target information):
#   R1 mean_rank      : equal-weight mean of percentile ranks  (primary)
#   R2 median_rank    : componentwise median of ranks
#   R3 zscore_mean    : equal-weight mean of z-scored variables (monotone
#                       per variable; AUC unaffected by scaling)
#   R4 borda          : rank-sum (equivalent to mean-rank up to scaling, so
#                       it is identical to R1 for AUC — kept for completeness)
#   R5 weighted_2to1  : PersA weight 1/2, remaining components share 1/2
#                       (anchor-forward rule)
# Metrics: ROC-AUC, PR-AUC; paired week-block bootstrap (1,000, seed 42) for
# ΔROC / ΔPR; verdict per rule = same pre-specified CI rule as R8.
#
# Outputs -> results/residual_hierarchy/rule_sensitivity/
#
# Run (piecf_env):
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_combination_rule_sensitivity.py

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(LTTS_ROOT))
from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "residual_hierarchy" / "rule_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
SEV = 0.8
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
H1_VARS = ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w"]
H2_VARS = H1_VARS + ["chl", "par", "wind"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x):
    return rankdata(x, method="average") / len(x)


def combine(rule, comps, dhw):
    """comps: dict name -> raw variable array (current_DHW included)."""
    if rule == "R1_mean_rank":
        return np.mean([pct(v) for v in comps.values()], axis=0)
    if rule == "R2_median_rank":
        return np.median([pct(v) for v in comps.values()], axis=0)
    if rule == "R3_zscore_mean":
        zs = []
        for v in comps.values():
            mu, sd = v.mean(), v.std()
            zs.append((v - mu) / (sd if sd > 0 else 1.0))
        return np.mean(zs, axis=0)
    if rule == "R4_borda":
        return np.sum([pct(v) for v in comps.values()], axis=0)
    if rule == "R5_weighted_2to1":
        n = len(comps)
        w = {"current_DHW": 0.5}
        wo = (1 - 0.5) / max(n - 1, 1)
        return sum((w.get(k, wo)) * pct(v) for k, v in comps.items())
    raise ValueError(rule)


if __name__ == "__main__":
    print("=" * 66)
    print(f"Combination-rule sensitivity — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 66)
    assert md5(REC_NPZ) == REC_MD5
    print(f"frozen records OK ({REC_MD5})")

    z = np.load(REC_NPZ)
    rows, verdict_rows = [], []
    RULES = ["R1_mean_rank", "R2_median_rank", "R3_zscore_mean",
             "R4_borda", "R5_weighted_2to1"]

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
        tgt = z[f"{basin}_target"].astype(np.float64)
        R = tsar - persa
        feats = {"current_DHW": persa * 10.0,
                 "dDHW_4w":     persa - y_full[t_idx - 4, i_idx, j_idx],
                 "cumsum_4w":   X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_4w"]],
                 "cumsum_12w":  X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_12w"]],
                 "chl":         X[t_idx, i_idx, j_idx, CH["chlorophyll"]],
                 "par":         X[t_idx, i_idx, j_idx, CH["par"]],
                 "wind":        X[t_idx, i_idx, j_idx, CH["wind_speed"]]}
        mask = np.isfinite(R) & np.isfinite(tgt)
        for v in feats.values():
            mask &= np.isfinite(v)
        V = {k: v[mask].astype(np.float64) for k, v in feats.items()}
        sev = (tgt[mask] >= SEV).astype(int)
        tsar_k = tsar[mask]
        weeks, inv = np.unique(t_idx[mask], return_inverse=True)
        n_weeks = len(weeks)
        idx_by_week = [np.where(inv == w)[0] for w in range(n_weeks)]

        def boot_idx(rng):
            pick = rng.integers(0, n_weeks, n_weeks)
            return np.concatenate([idx_by_week[w] for w in pick])

        for rule in RULES:
            s0 = combine(rule, {"current_DHW": V["current_DHW"]}, V["current_DHW"])
            s1 = combine(rule, {k: V[k] for k in H1_VARS}, V["current_DHW"])
            s2 = combine(rule, {k: V[k] for k in H2_VARS}, V["current_DHW"])
            s3 = combine(rule, {**{k: V[k] for k in H2_VARS}, "R": R[mask]},
                         V["current_DHW"])
            scores = {"H0": s0, "H1": s1, "H2": s2, "H3": s3,
                      "TSAR": pct(tsar_k)}
            rng = np.random.default_rng(BOOT_SEED)
            bm = {k: {"roc": [], "pr": []} for k in scores}
            for _ in range(N_BOOT):
                idx = boot_idx(rng)
                yb = sev[idx]
                for k, s in scores.items():
                    try:
                        bm[k]["roc"].append(roc_auc_score(yb, s[idx]))
                        bm[k]["pr"].append(average_precision_score(yb, s[idx]))
                    except ValueError:
                        pass
            pt = {k: {"roc": float(roc_auc_score(sev, s)),
                      "pr": float(average_precision_score(sev, s))}
                  for k, s in scores.items()}
            for k, s in scores.items():
                for met in ("roc", "pr"):
                    a = np.array(bm[k][met])
                    rows.append({"basin": basin, "rule": rule, "level": k,
                                 "metric": met, "point": pt[k][met],
                                 "ci_lo": float(np.percentile(a, 2.5)),
                                 "ci_hi": float(np.percentile(a, 97.5))})
            for a_, b_ in (("H1", "H0"), ("H2", "H1"), ("H3", "H2"), ("H3", "TSAR")):
                for met in ("roc", "pr"):
                    d = np.array(bm[a_][met]) - np.array(bm[b_][met])
                    rows.append({"basin": basin, "rule": rule,
                                 "level": f"{a_}_minus_{b_}", "metric": met,
                                 "point": pt[a_][met] - pt[b_][met],
                                 "ci_lo": float(np.percentile(d, 2.5)),
                                 "ci_hi": float(np.percentile(d, 97.5))})
                    if (a_, b_) == ("H3", "H2"):
                        verdict_rows.append({
                            "basin": basin, "rule": rule, "metric": met,
                            "point": rows[-1]["point"],
                            "ci_excl_0_pos": bool(rows[-1]["ci_lo"] > 0)})
            print(f"  [{rule}] H3−H2 ΔROC {pt['H3']['roc']-pt['H2']['roc']:+.4f} "
                  f"ΔPR {pt['H3']['pr']-pt['H2']['pr']:+.4f}")
        del ds
        import torch
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rule_sensitivity.csv", index=False)
    vd = pd.DataFrame(verdict_rows)
    # verdict table: per rule per basin — is C3 established (ROC or PR)?
    pivot = vd.pivot_table(index=["rule", "basin"], columns="metric",
                           values="ci_excl_0_pos").reset_index()
    roc_col = pivot["roc"] if "roc" in pivot.columns else pd.Series(False, index=pivot.index)
    pr_col = pivot["pr"] if "pr" in pivot.columns else pd.Series(False, index=pivot.index)
    pivot["C3_established"] = roc_col.fillna(False).astype(bool) | pr_col.fillna(False).astype(bool)
    pivot.to_csv(OUT / "verdicts.csv", index=False)
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"experiment": "combination_rule_sensitivity",
                   "generated": datetime.now().isoformat(),
                   "record_npz_md5": REC_MD5,
                   "rules": RULES,
                   "note": "training-free rules only; R4≡R1 for AUC (both "
                           "monotone in the rank sum)",
                   "interpretation": "predictive / representation-level only"},
                  f, indent=2, ensure_ascii=False)
    print("\nC3 (H3−H2) verdicts by rule:")
    print(pivot.to_string(index=False))
    print(f"\nOutputs -> {OUT}")
    print("Done.")
