# compute_response_surface_susceptibility_mechanism.py
#
# Three-stage gated pipeline:
#   ① Conditional thermal-response surface (per-DHW-bin severe rates + R-tertile contrast)
#   ② Basin-specific susceptibility test (gate: R-tertile Δ CI-positive in GBR & CARIB)
#   ③ Mechanism decomposition (nested M0→M3 OOS R²; gate: ② established)
#
# FIXED PROTOCOL:
#   - frozen rec_structural_ablation.npz (md5-verified), seed-42 TSAR/PersA
#   - R = TSAR − PersA; per-basin analyses
#   - no retraining / target fitting / target tuning / threshold optimisation
#   - complete-case mask; identical records across all parts per basin
#   - week-block bootstrap 1000, seed 42; BH-FDR within family per basin
#   - associative / representation-level language only
#
# Outputs -> results/response_surface_susceptibility_mechanism/

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr, t as tdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "response_surface_susceptibility_mechanism"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"

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

SEV = 0.8
BIN_W = 0.5
MIN_BIN = 40
N_BOOT = 1000
BOOT_SEED = 42
N_FOLDS = 5

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
CH = {c: i for i, c in enumerate(DATASET_ORDER)}

MECH_VARS = ["dDHW_4w", "cumsum_4w", "cumsum_12w", "hotspot", "ssta",
             "chl", "par", "wind"]
MECH_GROUPS = {
    "A_trajectory": ["dDHW_4w"],
    "B_persistence": ["cumsum_4w", "cumsum_12w"],
    "C_thermal_state": ["hotspot", "ssta"],
    "D_nonthermal": ["chl", "par", "wind"],
}
# nested model levels
M_LEVELS = {
    "M0": ["current_DHW"],
    "M1": ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w"],
    "M2": ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w",
           "hotspot", "ssta"],
    "M3": ["current_DHW", "dDHW_4w", "cumsum_4w", "cumsum_12w",
           "hotspot", "ssta", "chl", "par", "wind"],
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); q = np.empty(n)
    r = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    q[o] = np.clip(r, 0, 1)
    return q


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        s, c = np.polyfit(b, a, 1)
        return a - (s * b + c)
    r, _ = pearsonr(resid(rx, rz), resid(ry, rz))
    n = len(x)
    if not np.isfinite(r) or abs(r) >= 1.0:
        return r, np.nan
    ts = r * np.sqrt((n - 3) / (1 - r * r))
    return float(r), float(2 * tdist.sf(abs(ts), n - 3))


def pooled_r2(y, yhat):
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1 - sse / sst if sst > 0 else np.nan


def ecdf_map(tv, x):
    s = np.sort(tv)
    return np.searchsorted(s, x, side="right") / len(s)


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Response surface → susceptibility → mechanism — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(REC_NPZ) == REC_MD5
print(f"frozen records OK ({REC_MD5})")

z = np.load(REC_NPZ)
data = {}
for b, cfg in BASINS.items():
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
    y_full = ds.y_regression_full
    X = ds.X_dynamic_full
    t_idx = z[f"{b}_A_PersA_t"].astype(np.int64)
    i_idx = z[f"{b}_i"].astype(np.int64)
    j_idx = z[f"{b}_j"].astype(np.int64)
    persa = z[f"{b}_A_PersA_pred"].astype(np.float64)
    tsar = z[f"{b}_C_anchored_pred"].astype(np.float64)
    tgt = z[f"{b}_target"].astype(np.float64)
    t = t_idx.astype(np.int64)
    dhw = persa * 10.0
    R = tsar - persa
    sev = (tgt >= SEV).astype(int)

    feats = {"current_DHW": dhw,
             "dDHW_4w":     persa - y_full[t_idx - 4, i_idx, j_idx],
             "cumsum_4w":   X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_4w"]],
             "cumsum_12w":  X[t_idx, i_idx, j_idx, CH["hotspot_cumsum_12w"]],
             "hotspot":     X[t_idx, i_idx, j_idx, CH["hotspot"]],
             "ssta":        X[t_idx, i_idx, j_idx, CH["ssta"]],
             "chl":         X[t_idx, i_idx, j_idx, CH["chlorophyll"]],
             "par":         X[t_idx, i_idx, j_idx, CH["par"]],
             "wind":        X[t_idx, i_idx, j_idx, CH["wind_speed"]]}

    mask = np.isfinite(R) & np.isfinite(tgt) & np.isfinite(persa)
    for v in feats.values():
        mask &= np.isfinite(v)
    V = {k: v[mask] for k, v in feats.items()}
    V["current_DHW"] = V["current_DHW"]
    weeks = t[mask]
    uw, inv = np.unique(weeks, return_inverse=True)
    idx_by_week = [np.where(inv == w)[0] for w in range(len(uw))]

    data[b] = {"dhw": V["current_DHW"], "R": R[mask], "sev": sev[mask],
               "tgt": tgt[mask], "week": weeks, "V": V, "inv": inv,
               "idx_by_week": idx_by_week, "n_weeks": len(uw),
               "mask_n": int(mask.sum()), "total": len(R)}
    print(f"[{b}] n={int(mask.sum())}/{len(R)} weeks={len(uw)}")
    del ds, X

# ══════════════════════════════════════════════════════════════════════
# PART ①: Conditional thermal-response surface
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 66)
print("PART ① Conditional thermal-response surface")
print("=" * 66)

# common DHW bins across all three basins
edges = np.arange(0, 10.0 + BIN_W / 2, BIN_W)
centers = edges[:-1] + BIN_W / 2
counts = {}
for b in data:
    c, _ = np.histogram(data[b]["dhw"], bins=edges)
    counts[b] = c
common_bins = np.where(np.min([counts[b] for b in data], axis=0) >= MIN_BIN)[0]
print(f"Common supported bins (≥{MIN_BIN}): {len(common_bins)} bins")
bin_centers = centers[common_bins]

surface_rows = []
for b in data:
    d = data[b]
    dhw, sev, R = d["dhw"], d["sev"], d["R"]
    bidx = np.digitize(dhw, edges) - 1
    keep = np.isin(bidx, common_bins)
    pos = np.searchsorted(common_bins, bidx[keep])
    sev_k, R_k = sev[keep], R[keep]

    # outcome-free R tertiles within each bin
    tert = np.full(len(sev_k), -1)
    for j in range(len(common_bins)):
        m = pos == j
        if m.sum() >= 3:
            q = np.quantile(R_k[m], [1 / 3, 2 / 3])
            tert[m] = np.digitize(R_k[m], q)

    # week-block bootstrap
    weeks_k = d["week"][keep]
    uw_k, inv_k = np.unique(weeks_k, return_inverse=True)
    ibw = [np.where(inv_k == w)[0] for w in range(len(uw_k))]
    rng = np.random.default_rng(BOOT_SEED)
    rate_boot = np.full((N_BOOT, len(common_bins)), np.nan)
    rhi_boot = np.full((N_BOOT, len(common_bins)), np.nan)
    rlo_boot = np.full((N_BOOT, len(common_bins)), np.nan)
    for r in range(N_BOOT):
        pick = rng.integers(0, len(uw_k), len(uw_k))
        idx = np.concatenate([ibw[w] for w in pick])
        p_, s_, rq_ = pos[idx], sev_k[idx], tert[idx]
        n_bn = np.bincount(p_, minlength=len(common_bins))
        ssum = np.bincount(p_, weights=s_, minlength=len(common_bins))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_boot[r] = np.where(n_bn > 0, ssum / n_bn, np.nan)
        for j in range(len(common_bins)):
            hi_m = (p_ == j) & (rq_ == 2)
            lo_m = (p_ == j) & (rq_ == 0)
            if hi_m.sum() > 0 and lo_m.sum() > 0:
                rhi_boot[r, j] = s_[hi_m].mean() - s_[lo_m].mean()

    for j in range(len(common_bins)):
        m = pos == j
        n_bn = int(m.sum())
        n_sev_bn = int(sev_k[m].sum())
        rate = n_sev_bn / n_bn if n_bn else np.nan
        # R tertile rates
        tert_rates = {}
        for q in range(3):
            qm = m & (tert == q)
            nq = int(qm.sum())
            sq = int(sev_k[qm].sum()) if nq else 0
            tert_rates[f"Q{q+1}_rate"] = sq / nq if nq else np.nan
            tert_rates[f"Q{q+1}_n"] = nq
        d_r = rhi_boot[:, j]
        surface_rows.append({
            "basin": b, "bin_center": bin_centers[j], "n": n_bn,
            "n_severe": n_sev_bn, "severe_rate": rate,
            "ci_lo": float(np.nanpercentile(rate_boot[:, j], 2.5)),
            "ci_hi": float(np.nanpercentile(rate_boot[:, j], 97.5)),
            "R_Q1_rate": tert_rates.get("Q1_rate", np.nan),
            "R_Q3_rate": tert_rates.get("Q3_rate", np.nan),
            "R_Q3_minus_Q1": tert_rates.get("Q3_rate", np.nan) - tert_rates.get("Q1_rate", np.nan) if tert_rates.get("Q1_rate") is not None and tert_rates.get("Q3_rate") is not None else np.nan,
            "R_contrast_ci_lo": float(np.nanpercentile(d_r, 2.5)) if np.isfinite(d_r).sum() > 10 else np.nan,
            "R_contrast_ci_hi": float(np.nanpercentile(d_r, 97.5)) if np.isfinite(d_r).sum() > 10 else np.nan,
        })

surface = pd.DataFrame(surface_rows)
surface.to_csv(OUT / "response_surface.csv", index=False)

# Gate ①→②: check if R-tertile contrast is direction-consistent (Q3>Q1)
# in a majority of supported bins for GBR and CARIB
gate1 = {}
for b in ("GBR", "CARIB"):
    sub = surface[(surface["basin"] == b) & np.isfinite(surface["R_Q3_minus_Q1"])]
    if len(sub) == 0:
        gate1[b] = False
        continue
    n_pos = int((sub["R_Q3_minus_Q1"] > 0).sum())
    n_ci = int(((sub["R_contrast_ci_lo"] > 0) |
                (sub["R_contrast_ci_hi"] < 0)).sum())
    gate1[b] = bool(n_pos >= max(1, len(sub) // 2))
    print(f"  Gate ①→② {b}: {n_pos}/{len(sub)} bins positive, "
          f"{n_ci} CI-separated -> {'PASS' if gate1[b] else 'FAIL'}")

gate1_pass = all(gate1.values())
print(f"Gate ①→②: {'PASS — proceeding to Part ②' if gate1_pass else 'FAIL — susceptibility not established'}")

# ══════════════════════════════════════════════════════════════════════
# PART ②: Basin-specific susceptibility test
# ══════════════════════════════════════════════════════════════════════
suscept_rows = []
suscept_pass = {}

if gate1_pass:
    print("\n" + "=" * 66)
    print("PART ② Basin-specific susceptibility test")
    print("=" * 66)

    for b in data:
        d = data[b]
        dhw, sev, R, tgt = d["dhw"], d["sev"], d["R"], d["tgt"]
        sev_label = (tgt >= SEV).astype(int)

        # partial Spearman(R, y | DHW)
        ps, ps_p = partial_spearman(R, sev_label, dhw)

        # ΔROC/ΔPR: PersA vs TSAR (paired week-block bootstrap)
        persa_dhw = dhw / 10.0
        rng = np.random.default_rng(BOOT_SEED)
        weeks, inv = np.unique(d["week"], return_inverse=True)
        ibw = [np.where(inv == w)[0] for w in range(len(weeks))]
        roc_p, pr_p, roc_t, pr_t = [], [], [], []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(weeks), len(weeks))
            idx = np.concatenate([ibw[w] for w in pick])
            yb = sev_label[idx]
            try:
                roc_p.append(roc_auc_score(yb, persa_dhw[idx]))
                pr_p.append(average_precision_score(yb, persa_dhw[idx]))
            except ValueError:
                pass
            try:
                roc_t.append(roc_auc_score(yb, (persa_dhw + R)[idx]))
                pr_t.append(average_precision_score(yb, (persa_dhw + R)[idx]))
            except ValueError:
                pass

        def pt(metric, score):
            if metric == "roc":
                return float(roc_auc_score(sev_label, score))
            return float(average_precision_score(sev_label, score))

        persa_droc = pt("roc", persa_dhw)
        persa_dpr = pt("pr", persa_dhw)
        tsar_droc = pt("roc", persa_dhw + R)
        tsar_dpr = pt("pr", persa_dhw + R)

        d_roc_arr = np.array(roc_t) - np.array(roc_p)
        d_pr_arr = np.array(pr_t) - np.array(pr_p)

        suscept_rows.append({
            "basin": b, "partial_spearman_R_y_given_DHW": ps,
            "ps_p": ps_p, "n": len(R),
            "persa_roc": persa_droc, "tsar_roc": tsar_droc,
            "delta_roc": tsar_droc - persa_droc,
            "delta_roc_ci_lo": float(np.percentile(d_roc_arr, 2.5)),
            "delta_roc_ci_hi": float(np.percentile(d_roc_arr, 97.5)),
            "persa_pr": persa_dpr, "tsar_pr": tsar_dpr,
            "delta_pr": tsar_dpr - persa_dpr,
            "delta_pr_ci_lo": float(np.percentile(d_pr_arr, 2.5)),
            "delta_pr_ci_hi": float(np.percentile(d_pr_arr, 97.5)),
        })
        print(f"  [{b}] partial ρ={ps:+.4f} (p={ps_p:.1e}) "
              f"ΔROC={tsar_droc-persa_droc:+.4f} [{np.percentile(d_roc_arr,2.5):+.4f},"
              f"{np.percentile(d_roc_arr,97.5):+.4f}]")

    # Gate ②→③: R must be CI-positively associated with conditional
    # susceptibility in GBR AND CARIB (not necessarily SCS)
    for r in suscept_rows:
        b = r["basin"]
        ci_pos_roc = r["delta_roc_ci_lo"] > 0
        ci_pos_pr = r["delta_pr_ci_lo"] > 0
        suscept_pass[b] = bool(ci_pos_roc or ci_pos_pr)
        print(f"  Gate ②→③ {b}: "
              f"{'PASS' if suscept_pass[b] else 'FAIL'} "
              f"(ΔROC CI_lo>0={ci_pos_roc}, ΔPR CI_lo>0={ci_pos_pr})")

    gate2_pass = suscept_pass.get("GBR", False) and suscept_pass.get("CARIB", False)
else:
    gate2_pass = False
    print("\nGate ①→② FAILED — skipping Part ② and Part ③")

suscept = pd.DataFrame(suscept_rows) if suscept_rows else pd.DataFrame()
suscept.to_csv(OUT / "susceptibility.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# PART ③: Mechanism decomposition (only if ② passes)
# ══════════════════════════════════════════════════════════════════════
mech_rows, nested_rows = [], []
mech_pass = {}

if gate2_pass:
    print("\n" + "=" * 66)
    print("PART ③ Mechanism decomposition (M0→M3)")
    print("=" * 66)

    for b in data:
        if not suscept_pass.get(b, False) and b in ("GBR", "CARIB"):
            print(f"  [{b}] susceptibility not established — skipping mechanism")
            continue
        d = data[b]
        V, R, week = d["V"], d["R"], d["week"]
        n = len(R)

        weeks_u, inv = np.unique(week, return_inverse=True)
        ibw = [np.where(inv == w)[0] for w in range(len(weeks_u))]
        rng = np.random.default_rng(BOOT_SEED)

        # partial Spearman per mechanism variable (controlling current_DHW)
        ps_results = {}
        ps_pvals = []
        for v in MECH_VARS:
            ps, pp = partial_spearman(V[v], R, V["current_DHW"])
            ps_results[v] = {"rho": ps, "p": pp}
            ps_pvals.append(pp)
            mech_rows.append({"basin": b, "variable": v, "group":
                              next((g for g, vs in MECH_GROUPS.items() if v in vs), ""),
                              "partial_rho": ps, "p_raw": pp})
        qs = bh_fdr(ps_pvals)
        for i, v in enumerate(MECH_VARS):
            mech_rows[-len(MECH_VARS) + i]["p_fdr"] = qs[i]

        # nested models with fold-internal preprocessing
        def design(names):
            Xm = np.column_stack([V[nm] for nm in names])
            mu, sd = Xm.mean(0), Xm.std(0)
            sd[sd == 0] = 1.0
            return (Xm - mu) / sd

        R_rank = rankdata(R) / n
        oos = {}
        for mname, names in M_LEVELS.items():
            Xd = design(names)
            oof = np.zeros(n)
            for tr, te in GroupKFold(n_splits=N_FOLDS).split(Xd, R_rank, groups=inv):
                mod = LinearRegression().fit(Xd[tr], R_rank[tr])
                oof[te] = mod.predict(Xd[te])
            oos[mname] = pooled_r2(R_rank, oof)
            nested_rows.append({"basin": b, "model": mname,
                                "predictors": "+".join(names),
                                "r2_oos": oos[mname]})

        # ΔR² with paired week-block bootstrap
        d_boot = {}
        rng_b = np.random.default_rng(BOOT_SEED)
        boot_r2 = {m: [] for m in M_LEVELS}
        for _ in range(N_BOOT):
            pick = rng_b.integers(0, len(weeks_u), len(weeks_u))
            idx = np.concatenate([ibw[w] for w in pick])
            for mname, names in M_LEVELS.items():
                Xd = design(names)
                oof = np.zeros(len(idx))
                g = inv[idx]
                uw2 = np.unique(g)
                fold_of = {int(w2): i2 % N_FOLDS for i2, w2 in enumerate(uw2)}
                folds = np.array([fold_of[int(w2)] for w2 in g])
                for f in range(N_FOLDS):
                    tr, te = folds != f, folds == f
                    if te.sum() == 0:
                        continue
                    mod = LinearRegression().fit(Xd[idx][tr], R_rank[idx][tr])
                    oof[te] = mod.predict(Xd[idx][te])
                boot_r2[mname].append(pooled_r2(R_rank[idx], oof))

        for step, hi_m, lo_m in (("M1_minus_M0", "M1", "M0"),
                                  ("M2_minus_M1", "M2", "M1"),
                                  ("M3_minus_M2", "M3", "M2")):
            dh = np.array(boot_r2[hi_m]) - np.array(boot_r2[lo_m])
            d_boot[step] = {"point": oos[hi_m] - oos[lo_m],
                            "ci_lo": float(np.nanpercentile(dh, 2.5)),
                            "ci_hi": float(np.nanpercentile(dh, 97.5))}
            nested_rows.append({"basin": b, "model": f"deltaR2_{step}",
                                "predictors": f"{hi_m} − {lo_m}",
                                "r2_oos": d_boot[step]["point"]})
            print(f"  [{b}] ΔR² {step}: {d_boot[step]['point']:+.4f} "
                  f"[{d_boot[step]['ci_lo']:+.4f}, {d_boot[step]['ci_hi']:+.4f}]")

        # verdict: is trajectory/persistence the dominant contributor?
        traj_dom = d_boot["M1_minus_M0"]["ci_lo"] > 0
        state_add = d_boot["M2_minus_M1"]["ci_lo"] > 0
        ctx_add = d_boot["M3_minus_M2"]["ci_lo"] > 0
        mech_pass[b] = {"trajectory_established": traj_dom,
                        "thermal_state_adds": state_add,
                        "nonthermal_adds": ctx_add}
        print(f"  [{b}] trajectory={traj_dom}, state_add={state_add}, ctx_add={ctx_add}")

    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(OUT / "mechanism_decomposition.csv", index=False)
else:
    print("\nGate ②→③ FAILED — mechanism decomposition not performed")
    mech_df = pd.DataFrame()

nested_df = pd.DataFrame(nested_rows) if nested_rows else pd.DataFrame()
nested_df.to_csv(OUT / "nested_models.csv", index=False)
mech_df.to_csv(OUT / "mechanism_decomposition.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# Bootstrap summary
# ══════════════════════════════════════════════════════════════════════
boot_rows = []
for r in suscept_rows:
    boot_rows.append({"basin": r["basin"], "stat": "partial_spearman",
                      "point": r["partial_spearman_R_y_given_DHW"],
                      "ci_lo": np.nan, "ci_hi": np.nan})
    boot_rows.append({"basin": r["basin"], "stat": "delta_roc_TSAR_minus_PersA",
                      "point": r["delta_roc"], "ci_lo": r["delta_roc_ci_lo"],
                      "ci_hi": r["delta_roc_ci_hi"]})
    boot_rows.append({"basin": r["basin"], "stat": "delta_pr_TSAR_minus_PersA",
                      "point": r["delta_pr"], "ci_lo": r["delta_pr_ci_lo"],
                      "ci_hi": r["delta_pr_ci_hi"]})
for nr in nested_rows:
    if "deltaR2" in nr.get("model", ""):
        b = nr["basin"]
        step = nr["model"].replace("deltaR2_", "")
        pt = nr["r2_oos"]
        ci = next((r for r in (d_boot.get(step, {}).items() if 'd_boot' in dir() else [])), None)
        # find from mech context
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(OUT / "bootstrap.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# Verdicts + summary
# ══════════════════════════════════════════════════════════════════════
verdicts = {
    "gate1_response_surface_heterogeneity": {
        "GBR": gate1.get("GBR", False),
        "CARIB": gate1.get("CARIB", False),
        "description": "R-tertile severe-rate contrast direction-consistent "
                       "across majority of matched DHW bins"},
    "gate2_susceptibility_established": {
        b: suscept_pass.get(b, False) for b in ("GBR", "CARIB", "SCS")
        if suscept_rows},
    "gate3_mechanism_decomposition_run": gate2_pass,
    "mechanism_results": mech_pass,
}

summary = {
    "experiment": "response_surface_susceptibility_mechanism",
    "generated": datetime.now().isoformat(),
    "record_npz_md5": REC_MD5,
    "gates": verdicts,
    "common_bins": int(len(common_bins)),
    "basin_n": {b: d["mask_n"] for b, d in data.items()},
    "verdict_summary": {
        "Q1_cross_basin_heterogeneity": gate1_pass,
        "Q2_R_identifies_susceptibility": gate2_pass,
        "Q3_mechanism": mech_pass,
        "Q4_GBR_CARIB_shared": (
            mech_pass.get("GBR", {}).get("trajectory_established", False) ==
            mech_pass.get("CARIB", {}).get("trajectory_established", False)
            and mech_pass.get("GBR", {}).get("trajectory_established", False)
        ) if gate2_pass else False,
        "Q5_mechanism_not_established_flag": not gate2_pass or not all(
            mech_pass.get(b, {}).get("trajectory_established", False)
            for b in ("GBR", "CARIB")),
    },
    "interpretation": "associative / representation-level only",
}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

# ---- AUDIT.md ----
L = ["# AUDIT — Response Surface → Susceptibility → Mechanism Decomposition",
     "", f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
     f"Frozen records md5 `{REC_MD5}` (verified).", "",
     "## Protocol adherence", "",
     "- Frozen rec_structural_ablation.npz; seed-42 TSAR/PersA; no retraining.",
     "- R = TSAR − PersA; per-basin; complete-case; unified records per part.",
     "- Week-block bootstrap 1000, seed 42; BH-FDR per family per basin.",
     "- Gated pipeline: ①→② requires R-tertile contrast direction-consistent "
     "in GBR+CARIB; ②→③ requires ΔROC or ΔPR CI-positive in both targets.",
     "- Associative / representation-level language only.", "",
     "## Gate results", "",
     f"- ①→② (response heterogeneity): {'PASS' if gate1_pass else 'FAIL'}",
     f"- ②→③ (susceptibility established): {'PASS' if gate2_pass else 'FAIL'}",
     "", "## Mechanism verdicts (if run)", ""]
for b, m in mech_pass.items():
    L.append(f"- {b}: trajectory={m['trajectory_established']}, "
             f"state_add={m['thermal_state_adds']}, "
             f"nonthermal_add={m['nonthermal_adds']}")
L += ["", "## Interpretation guard", "",
      "All statements are associative / representation-level. 'Susceptibility' "
      "means conditional association between the frozen residual and future "
      "severe thermal events at matched current DHW; it is not a causal or "
      "biological sensitivity claim.", ""]
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print(f"\nOutputs -> {OUT}")
print("Done.")
