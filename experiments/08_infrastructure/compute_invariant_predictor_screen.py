# compute_invariant_predictor_screen.py
#
# EXPLORATORY / SUPPORTING (clearly labelled): treat the three basins as
# "environments" (ICP/IRM-inspired) and screen which input channels are
# invariant predictors of the future thermal target y(t+4).
#
# Method: per basin, Spearman rho_b(channel, y(t+4)) on the frozen record
# set (complete-case over all candidates); invariance = pairwise cross-basin
# |rho_b - rho_b'| with week-block bootstrap 95% CI (independent per-basin
# streams, seed 42). Pre-specified classification: a channel is an
# "invariant predictor" iff (a) sign-stable across basins, (b) homogeneous
# (all pairwise difference CIs include 0), (c) FDR-significant (BH, within
# basin family) in >= 2 of the 3 basins.
#
# CAVEATS (mandatory): (i) the basins share global ocean dynamics, so
# invariance across three dependent environments is weaker evidence than
# across truly independent environments; (ii) invariance is a property of
# the predictor-target ASSOCIATION, not of a causal mechanism; (iii) the
# anchor PersA is included as a reference (it is deterministic given DHW).
#
# Outputs -> results/invariant_predictor_screen/

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

OUT = RESULTS_DIR / "invariant_predictor_screen"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
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

CANDS = ["PersA_anchor", "hotspot", "ssta", "sst", "d_hotspot_dt",
         "hotspot_cumsum_4w", "hotspot_cumsum_12w",
         "chlorophyll", "par", "wind_speed", "ssr",
         "seasonal_sin", "seasonal_cos", "sst_anomaly"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); q = np.empty(n)
    r = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    q[o] = np.clip(r, 0, 1)
    return q


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Invariant predictor screen (exploratory) — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(REC_NPZ) == REC_MD5
z = np.load(REC_NPZ)

rho_boot = {}       # basin -> var -> boot draws
assoc_rows = []
for basin, cfg in BASINS.items():
    print(f"=== {basin} ===")
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
    X = ds.X_dynamic_full
    t_idx = z[f"{basin}_A_PersA_t"].astype(np.int64)
    i_idx = z[f"{basin}_i"].astype(np.int64)
    j_idx = z[f"{basin}_j"].astype(np.int64)
    persa = z[f"{basin}_A_PersA_pred"].astype(np.float64)
    tgt = z[f"{basin}_target"].astype(np.float64)
    feats = {"PersA_anchor": persa}
    for c in CANDS[1:]:
        feats[c] = X[t_idx, i_idx, j_idx, CH[c]]
    mask = np.isfinite(tgt) & np.isfinite(persa)
    for v in feats.values():
        mask &= np.isfinite(v)
    feats = {v: arr[mask] for v, arr in feats.items()}   # complete-case, aligned
    weeks, inv = np.unique(t_idx[mask], return_inverse=True)
    idx_by_week = [np.where(inv == w)[0] for w in range(len(weeks))]
    y = tgt[mask]
    rho_boot[basin] = {}
    # deterministic per-basin seed (str hash() is salted per process)
    rng = np.random.default_rng(BOOT_SEED + 1000 * list(BASINS).index(basin))
    boot = {v: [] for v in feats}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(weeks), len(weeks))
        idx = np.concatenate([idx_by_week[w] for w in pick])
        for v in feats:
            boot[v].append(spearmanr(feats[v][idx], y[idx]).statistic)
    for v in feats:
        r, pval = spearmanr(feats[v], y)
        rho_boot[basin][v] = np.array(boot[v])
        assoc_rows.append({"basin": basin, "variable": v, "rho": r, "p": pval,
                           "n": int(mask.sum())})
    del ds, X
    torch = __import__("torch")
    torch.cuda.empty_cache()

assoc = pd.DataFrame(assoc_rows)
# FDR within basin family
for b in BASINS:
    m = assoc["basin"] == b
    assoc.loc[m, "q"] = bh_fdr(assoc.loc[m, "p"].values)

inv_rows = []
for v in CANDS:
    rs = {b: assoc[(assoc.basin == b) & (assoc.variable == v)]["rho"].iloc[0]
          for b in BASINS}
    qs = {b: assoc[(assoc.basin == b) & (assoc.variable == v)]["q"].iloc[0]
          for b in BASINS}
    sign_stable = len({np.sign(rs[b]) for b in BASINS}) == 1
    pairs = [("SCS", "GBR"), ("SCS", "CARIB"), ("GBR", "CARIB")]
    hom = True
    for bA, bB in pairs:
        d = rho_boot[bA][v] - rho_boot[bB][v]
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        if not (lo <= 0 <= hi):
            hom = False
    nsig = sum(1 for b in BASINS if qs[b] < 0.05)
    is_inv = bool(sign_stable and hom and nsig >= 2)
    inv_rows.append({"variable": v, **{f"rho_{b}": rs[b] for b in BASINS},
                     **{f"q_{b}": qs[b] for b in BASINS},
                     "sign_stable": sign_stable, "homogeneous_ci": hom,
                     "n_sig_basins": nsig, "invariant_predictor": is_inv})
    print(f"  {v:20s} " + " ".join(f"{b[:1]}{rs[b]:+.3f}" for b in BASINS)
          + f"  sign={sign_stable} hom={hom} nsig={nsig} -> "
          f"{'INVARIANT' if is_inv else 'environment-dependent'}")

pd.DataFrame(inv_rows).to_csv(OUT / "invariance.csv", index=False)
assoc.to_csv(OUT / "associations.csv", index=False)
json.dump({"experiment": "invariant_predictor_screen",
           "generated": datetime.now().isoformat(),
           "record_npz_md5": REC_MD5,
           "status": "EXPLORATORY / SUPPORTING",
           "classification_rule": "sign-stable + homogeneous (all pairwise "
                                  "difference CIs include 0) + FDR-significant "
                                  "in >=2 basins",
           "caveats": ["basins share global ocean dynamics (dependent "
                       "environments)", "association-level, not causal",
                       "invariance across 3 dependent environments is weak "
                       "evidence compared to truly independent environments"],
           "invariant_set": [r["variable"] for r in inv_rows
                             if r["invariant_predictor"]]},
          open(OUT / "summary.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(f"\nOutputs -> {OUT}")
print("Done.")
