# compute_cross_basin_residual_mechanism.py
#
# Cross-basin Residual Mechanism Consistency Test.
#
# Question: does the TSAR residual R = TSAR − PersA correspond to the SAME
# basin-specific susceptibility/context signature in GBR and CARIB, or does it
# encode entirely different signals in different basins?
#
# FIXED PROTOCOL (no retraining, no target fitting/tuning, no threshold
# optimisation; associative / representation-level interpretation ONLY):
#   - frozen rec_structural_ablation.npz (md5-verified), seed-42 TSAR
#   - R = TSAR − PersA; SCS/GBR/CARIB analysed separately (SCS = source
#     reference only, never required evidence for the transfer-target claim)
#   - candidate variables (verbatim project definitions, at issue time):
#     dDHW_4w, cumsum_4w, cumsum_12w, ssta, hotspot (thermal trajectory /
#     persistence / state) and chl, par, wind (nonthermal context)
#   - unified complete-case mask (identical records across all parts)
#   - Spearman + partial Spearman (controlling current DHW) with paired
#     week-block bootstrap CI (1000, seed 42) + BH-FDR per family per basin
#   - GBR−CARIB effect difference per variable with bootstrap 95% CI
#     (independent week-block bootstrap streams per basin; element-wise
#     difference of replicates — documented in AUDIT)
#   - fixed rank-OLS: R ~ thermal trajectory/persistence + nonthermal context
#     (8 candidate variables, fold-internal ECDF maps, GroupKFold 5 by issue
#     week, pooled OOS R²); standardized (rank-scale) coefficients compared
#     across basins for sign consistency
#   - verdict rule (pre-specified): a variable is "consistent" iff GBR and
#     CARIB partial associations share sign AND both are FDR-significant
#     (q<0.05); the GBR−CARIB difference CI is reported as homogeneity
#     information (excluding 0 = heterogeneous effect size, still same sign)
#
# Outputs -> results/cross_basin_residual_mechanism/

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

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "cross_basin_residual_mechanism"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
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
                            val_years=(2011, 2015), test_years=(2016, 2019))},
    "GBR":   {"file": "gbr_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2014, 2016), test_years=(2017, 2020))},
    "CARIB": {"file": "caribbean_causal_v2.nc",
              "split": dict(year_based_split=True, train_years=(2003, 2010),
                            val_years=(2014, 2016), test_years=(2017, 2020))},
}

CANDIDATES = ["dDHW_4w", "cumsum_4w", "cumsum_12w", "ssta", "hotspot",
              "chl", "par", "wind"]
THERMAL = ["dDHW_4w", "cumsum_4w", "cumsum_12w", "ssta", "hotspot"]
NONTHERMAL = ["chl", "par", "wind"]
MODEL_VARS = THERMAL + NONTHERMAL
TEST_PAIR = ("GBR", "CARIB")


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


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Cross-basin residual mechanism consistency — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(REC_NPZ) == REC_MD5, "frozen record set hash mismatch"
print(f"frozen records OK ({REC_MD5})")

z = np.load(REC_NPZ)
assoc_rows, coef_rows, boot_rows = [], [], []
per_basin_boot = {}          # basin -> var -> bootstrap partial-rho draws
summary = {"experiment": "cross_basin_residual_mechanism",
           "generated": datetime.now().isoformat(),
           "record_npz_md5": REC_MD5,
           "protocol": {"candidates": CANDIDATES,
                        "partial_control": "current_DHW (= PersA*10)",
                        "bootstrap": {"type": "paired week-block per basin",
                                      "n": N_BOOT, "seed": BOOT_SEED},
                        "model": "fixed rank-OLS (fold-internal ECDF), "
                                 "GroupKFold 5 by issue week, pooled OOS R²",
                        "fdr": "Benjamini-Hochberg per family per basin",
                        "verdict_rule": "consistent iff GBR & CARIB share sign "
                                        "AND both q<0.05; difference CI reported "
                                        "as homogeneity information",
                        "interpretation": "associative / representation-level only"},
           "basins": {}, "verdicts": {}}

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
             "cumsum_4w":   X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_4w"]],
             "cumsum_12w":  X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_12w"]],
             "ssta":        X[t_idx, i_idx, j_idx, CH["ssta"]],
             "hotspot":     X[t_idx, i_idx, j_idx, CH["hotspot"]],
             "chl":         X[t_idx, i_idx, j_idx, CH["chlorophyll"]],
             "par":         X[t_idx, i_idx, j_idx, CH["par"]],
             "wind":        X[t_idx, i_idx, j_idx, CH["wind_speed"]]}
    mask = np.isfinite(R) & np.isfinite(feats["current_DHW"])
    for v in CANDIDATES:
        mask &= np.isfinite(feats[v])
    V = {v: feats[v][mask].astype(np.float64) for v in feats}
    y = R[mask]
    t = t_idx[mask]
    print(f"  records n={int(mask.sum())}/{len(R)} (complete-case), "
          f"weeks={len(np.unique(t))}")

    weeks, inv = np.unique(t, return_inverse=True)
    n_weeks = len(weeks)
    idx_by_week = [np.where(inv == w)[0] for w in range(n_weeks)]

    # ---- Part 1: raw + partial Spearman with bootstrap CI + FDR ----
    raw_p, part_p = [], []
    for v in CANDIDATES:
        rho, p = spearmanr(V[v], y)
        assoc_rows.append({"basin": basin, "variable": v, "type": "raw",
                           "rho": rho, "p_raw": p})
        raw_p.append(p)
    for v in CANDIDATES:
        pr, pp = partial_spearman(V[v], y, V["current_DHW"])
        assoc_rows.append({"basin": basin, "variable": v,
                           "type": "partial_ctrl_DHW", "rho": pr, "p_raw": pp})
        part_p.append(pp)
    raw_rows = [r for r in assoc_rows if r["basin"] == basin and r["type"] == "raw"]
    part_rows = [r for r in assoc_rows if r["basin"] == basin
                 and r["type"] == "partial_ctrl_DHW"]
    for r_, q_ in zip(raw_rows, bh_fdr(raw_p)):
        r_["p_fdr"] = q_
    for r_, q_ in zip(part_rows, bh_fdr(part_p)):
        r_["p_fdr"] = q_

    rng = np.random.default_rng(BOOT_SEED)
    boot_draws = {v: {"raw": [], "partial": []} for v in CANDIDATES}
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_weeks, n_weeks)
        idx = np.concatenate([idx_by_week[w] for w in pick])
        for v in CANDIDATES:
            rb, _ = spearmanr(V[v][idx], y[idx])
            boot_draws[v]["raw"].append(rb)
            pb, _ = partial_spearman(V[v][idx], y[idx], V["current_DHW"][idx])
            boot_draws[v]["partial"].append(pb)
    per_basin_boot[basin] = boot_draws
    for r_ in raw_rows + part_rows:
        key = "raw" if r_["type"] == "raw" else "partial"
        b = np.array(per_basin_boot[basin][r_["variable"]][key], dtype=float)
        b = b[np.isfinite(b)]
        r_["ci_lo"] = float(np.percentile(b, 2.5))
        r_["ci_hi"] = float(np.percentile(b, 97.5))

    # ---- Part 4: fixed rank-OLS, fold-internal, pooled OOS ----
    gkf = GroupKFold(n_splits=N_FOLDS)
    oos_pred = np.full(len(y), np.nan)
    y_rank_oos = np.full(len(y), np.nan)
    coefs_full = None
    for tr, te in gkf.split(np.zeros(len(y)), groups=t):
        Ztr = np.column_stack([ecdf_map(V[v][tr], V[v][tr]) for v in MODEL_VARS])
        Zte = np.column_stack([ecdf_map(V[v][tr], V[v][te]) for v in MODEL_VARS])
        ytr_r = ecdf_map(y[tr], y[tr])
        yte_r = ecdf_map(y[tr], y[te])
        mod = LinearRegression().fit(Ztr, ytr_r)
        oos_pred[te] = mod.predict(Zte)
        y_rank_oos[te] = yte_r
        if coefs_full is None:
            full = LinearRegression().fit(
                np.column_stack([rankdata(V[v], method="average") / len(y)
                                 for v in MODEL_VARS]),
                rankdata(y, method="average") / len(y))
            coefs_full = dict(zip(MODEL_VARS, full.coef_.round(6).tolist()))
    r2_oos = pooled_r2(y_rank_oos, oos_pred)
    print(f"  rank-OLS pooled OOS R²={r2_oos:+.4f}")
    for v in MODEL_VARS:
        coef_rows.append({"basin": basin, "variable": v,
                          "rank_coef": coefs_full[v], "r2_oos_rank_ols": r2_oos})

    # bootstrap of rank coefficients (same resampled weeks)
    rng_c = np.random.default_rng(BOOT_SEED)
    coef_boot = {v: [] for v in MODEL_VARS}
    for _ in range(N_BOOT):
        pick = rng_c.integers(0, n_weeks, n_weeks)
        idx = np.concatenate([idx_by_week[w] for w in pick])
        Zb = np.column_stack([rankdata(V[v][idx], method="average") / len(idx)
                              for v in MODEL_VARS])
        yb = rankdata(y[idx], method="average") / len(idx)
        mb = LinearRegression().fit(Zb, yb)
        for j, v in enumerate(MODEL_VARS):
            coef_boot[v].append(mb.coef_[j])
    for v in MODEL_VARS:
        b = np.array(coef_boot[v])
        boot_rows.append({"basin": basin, "stat": f"rank_coef_{v}",
                          "point": coefs_full[v],
                          "ci_lo": float(np.percentile(b, 2.5)),
                          "ci_hi": float(np.percentile(b, 97.5))})

    summary["basins"][basin] = {"n": int(mask.sum()), "n_weeks": n_weeks,
                                "r2_oos_rank_ols": r2_oos,
                                "rank_coefs": coefs_full}
    del ds, X

# ---- Part 2/3 + 5: GBR−CARIB differences and verdicts ----
def part_row(basin, var):
    return [r for r in assoc_rows if r["basin"] == basin
            and r["variable"] == var and r["type"] == "partial_ctrl_DHW"][0]

bA, bB = TEST_PAIR
for v in CANDIDATES:
    rA, rB = part_row(bA, v), part_row(bB, v)
    dA = np.array(per_basin_boot[bA][v]["partial"], dtype=float)
    dB = np.array(per_basin_boot[bB][v]["partial"], dtype=float)
    d = dA - dB                                    # element-wise, independent streams
    diff_lo, diff_hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    assoc_rows.append({"basin": f"{bA}_minus_{bB}", "variable": v,
                       "type": "partial_ctrl_DHW", "rho": rA["rho"] - rB["rho"],
                       "p_raw": np.nan, "p_fdr": np.nan,
                       "ci_lo": diff_lo, "ci_hi": diff_hi})
    # cA: rank coefficient difference
    cA = summary["basins"][bA]["rank_coefs"][v]
    cB = summary["basins"][bB]["rank_coefs"][v]
    boot_rows.append({"basin": f"{bA}_minus_{bB}", "stat": f"rank_coef_{v}",
                      "point": cA - cB, "ci_lo": np.nan, "ci_hi": np.nan})
    # verdict (pre-specified rule)
    same_sign = np.sign(rA["rho"]) == np.sign(rB["rho"]) and rA["rho"] != 0
    both_sig = (rA["p_fdr"] < 0.05) and (rB["p_fdr"] < 0.05)
    homogeneous = (diff_lo <= 0) and (diff_hi >= 0)
    if same_sign and both_sig and homogeneous:
        verdict = "consistent"
    elif same_sign and both_sig:
        verdict = "same-sign but heterogeneous effect size"
    else:
        verdict = "not consistent"
    summary["verdicts"][v] = {
        "GBR_partial_rho": rA["rho"], "CARIB_partial_rho": rB["rho"],
        "GBR_q": rA["p_fdr"], "CARIB_q": rB["p_fdr"],
        "diff": rA["rho"] - rB["rho"], "diff_ci": [diff_lo, diff_hi],
        "verdict": verdict}

# ---- outputs ----
pd.DataFrame(assoc_rows).to_csv(OUT / "mechanism_associations.csv", index=False)
pd.DataFrame(coef_rows).to_csv(OUT / "coefficient_comparison.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap.csv", index=False)
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

# ---- AUDIT.md ----
L = ["# AUDIT — Cross-basin Residual Mechanism Consistency Test", "",
     f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ",
     f"Frozen records: `rec_structural_ablation.npz` md5 `{REC_MD5}` (verified).", "",
     "## Protocol adherence", "",
     "- R = TSAR(seed-42) − PersA; frozen record set; no retraining, no target "
     "fitting/tuning, no threshold optimisation.",
     "- Candidate variables read verbatim from the causal v2 datasets at the "
     "record issue time; unified complete-case mask (identical records across "
     "all parts).",
     "- Partial Spearman controls current_DHW (= PersA×10, the clipped "
     "coordinate); raw and partial families FDR-corrected (BH) per basin.",
     "- Paired week-block bootstrap (1,000, seed 42) per basin; the GBR−CARIB "
     "difference CI is formed from independent per-basin bootstrap streams "
     "(element-wise difference of replicates; the two basins are independent "
     "record sets, so no cross-basin pairing is assumed).",
     "- Fixed rank-OLS (8 candidates, fold-internal ECDF maps, GroupKFold 5 by "
     "issue week, pooled OOS R²); rank-scale coefficients reported with "
     "week-block bootstrap CIs.",
     "- SCS is reported as the source/reference only; the consistency claim is "
     "evaluated exclusively on GBR vs CARIB.",
     "- Associative / representation-level interpretation only; no causal claim.",
     "",
     "## Pre-specified verdict rule", "",
     "A variable is **consistent** iff GBR and CARIB partial associations share "
     "sign AND both are FDR-significant (q<0.05) AND the GBR−CARIB difference "
     "95% CI includes 0 (homogeneous effect size). Same-sign + both-significant "
     "with the difference CI excluding 0 is reported as *same-sign but "
     "heterogeneous effect size*; anything else is **not consistent**.", "",
     "## Verdicts (GBR vs CARIB, partial | current DHW)", "",
     "| Variable | GBR ρ (q) | CARIB ρ (q) | diff [95% CI] | verdict |",
     "|---|---|---|---|---|"]
for v in CANDIDATES:
    s = summary["verdicts"][v]
    L.append(f"| {v} | {s['GBR_partial_rho']:+.3f} ({s['GBR_q']:.1e}) | "
             f"{s['CARIB_partial_rho']:+.3f} ({s['CARIB_q']:.1e}) | "
             f"{s['diff']:+.3f} [{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}] "
             f"| {s['verdict']} |")
L += ["", "## Interpretation guard", "",
      "Statistical association is not mechanism: no claim is made that the "
      "residual is produced by, or causally depends on, these variables. "
      "Consistency of association across two basins is representation-level "
      "evidence only and does not establish a shared causal pathway.", ""]
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print("\nVerdicts (GBR vs CARIB):")
for v in CANDIDATES:
    s = summary["verdicts"][v]
    print(f"  {v:12s} GBR {s['GBR_partial_rho']:+.3f}  CARIB {s['CARIB_partial_rho']:+.3f}"
          f"  -> {s['verdict']}")
print(f"\nOutputs -> {OUT}")
print("Done.")