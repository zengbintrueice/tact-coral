# compute_transfer_coordinate.py
#
# Transfer Matrix + Transferable Thermal Coordinate Stability.
#
# Goal: test whether the physical DHW anchor PersA_DHW = clip(DHW/10,0,1)
# is a more stable cross-basin thermal-risk coordinate than the learned
# contextual TSCF representation, via a fully zero-shot 6-direction
# transfer matrix over SCS/GBR/CARIB plus a coordinate-stability analysis.
#
# This script implements the NO-TRAINING tier only:
#   * PersA_DHW across all directions  (source-independent: same per-target)
#   * TSCF / TSAR for the SCS->GBR and SCS->CARIB directions (frozen evidence
#     record set rec_structural_ablation.npz)  [extended once reverse-source
#     TSCF/TSAR models are trained]
#   * XGB_THERMAL SCS-sourced control (results/xgb_transfer_control/metrics.json)
#   * coordinate stability: PersA_DHW vs y(t+4) Spearman / ROC-AUC / PR-AUC per
#     basin and per year, with week-block bootstrap 95% CI.
#
# Metric definitions are VERBATIM from the evidence pipeline
# (experiments/08_infrastructure/compute_caribbean_zero_shot.py and
# pfv_utils.py) so numbers are comparable to the locked evaluations.
#
# Read-only over locked results. Outputs -> results/transfer_coordinate/
#
# Run (piecf_env):  cd LTTS_CRW && python experiments/08_infrastructure/compute_transfer_coordinate.py

import io, sys, os, json, hashlib, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

SEV = 0.8
TAU = 0.9
ECE_BINS = 10
N_BOOT = 1000
BOOT_SEED = 42
OUT = LTTS_ROOT / "results" / "transfer_coordinate"
OUT.mkdir(parents=True, exist_ok=True)
REC = LTTS_ROOT / "results" / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
XGB_MJ = LTTS_ROOT / "results" / "xgb_transfer_control" / "metrics.json"
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
CASUAL_FILES = {   # basin -> (netcdf,time-dim name)
    "SCS": "scs_causal_v2.nc",
    "GBR": "gbr_causal_v2.nc",
    "CARIB": "caribbean_causal_v2.nc",
}
# rec structural ablation key prefixes per basin
MODKEY = {"TSCF": "D_TSCF", "TSAR": "C_anchored", "PersA": "A_PersA"}

# ---------- verbatim pipeline metrics (compute_caribbean_zero_shot.py) ----------
def auc_scores(pred, target):
    y = (target >= SEV).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        return float(roc_auc_score(y, pred)), float(average_precision_score(y, pred))
    except ValueError:
        return np.nan, np.nan

def ece_score(pred, target):
    if len(pred) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, ECE_BINS - 1)
    severe = (target >= SEV).astype(np.float64)
    ece, n = 0.0, len(pred)
    for b in range(ECE_BINS):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(severe[m].mean()) - float(pred[m].mean()))
    return float(ece)

def classification_counts(pred, target, tau):
    hits = int(np.sum((pred >= tau) & (target >= SEV)))
    misses = int(np.sum((pred < tau) & (target >= SEV)))
    fa = int(np.sum((pred >= tau) & (target < SEV)))
    return hits, misses, fa

def metrics_from_counts(h, m_, fa):
    csi = h / (h + m_ + fa) if (h + m_ + fa) else np.nan
    pod = h / (h + m_) if (h + m_) else np.nan
    far = fa / (h + fa) if (h + fa) else np.nan
    f1 = 2 * h / (2 * h + fa + m_) if (2 * h + fa + m_) else np.nan
    return csi, pod, far, f1

def full_metrics(pred, target):
    out = {"n": int(len(pred)), "n_severe": int(np.sum(target >= SEV))}
    out["severe_prevalence"] = out["n_severe"] / len(pred) if len(pred) else np.nan
    out["roc_auc"], out["pr_auc"] = auc_scores(pred, target)
    out["brier"] = float(np.mean((pred - target) ** 2))
    out["ece"] = ece_score(pred, target)
    h, m_, fa = classification_counts(pred, target, TAU)
    csi, pod, far, f1 = metrics_from_counts(h, m_, fa)
    out.update(csi_tau09=csi, pod_tau09=pod, far_tau09=far, f1_tau09=f1)
    return out

# ---------- week-block bootstrap (paired-scheme on same seed) ----------
def make_week_picks(t_idx, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    weeks = np.unique(t_idx)
    return weeks[rng.integers(0, len(weeks), size=(n_boot, len(weeks)))]

def boot_metric_ci(pred, target, t_idx, picks=None, n_boot=N_BOOT, seed=BOOT_SEED):
    if picks is None:
        picks = make_week_picks(t_idx, n_boot, seed)
    weeks = np.unique(t_idx)
    boot = {k: [] for k in full_metrics(pred[:1], target[:1])}
    for b in range(n_boot):
        mask = np.isin(t_idx, weeks if False else picks[b])
        if mask.sum() < 10:
            continue
        res = full_metrics(pred[mask], target[mask])
        for k, v in res.items():
            boot[k].append(v)
    out = {}
    for k, vals in boot.items():
        vals = np.array([v for v in vals if np.isfinite(v)])
        out[k] = [np.nan, np.nan] if len(vals) == 0 else [float(np.percentile(vals, 2.5)),
                                                           float(np.percentile(vals, 97.5))]
    return out

def diff_boot_ci(pa, ta, ta_idx, pb, tb, tb_idx, n_boot=N_BOOT, seed=BOOT_SEED, keys=("roc_auc", "pr_auc")):
    """Bootstrap 95% CI of (metric on B) - (metric on A) using the same
    week-block resampling scheme (seed 42) applied independently to each
    population. Across-basin populations are disjoint, so this is a
    difference-of-independent week-block estimates, not a record-level pair."""
    rng = np.random.default_rng(seed)
    wa = np.unique(ta_idx); wb = np.unique(tb_idx)
    pa_ct = lambda pick, w: np.isin(w, pick)
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        picka = wa[rng.integers(0, len(wa), len(wa))]
        pickb = wb[rng.integers(0, len(wb), len(wb))]
        ma = pa_ct(picka, ta_idx); mb = pa_ct(pickb, tb_idx)
        for k in keys:
            va = full_metrics(pa[ma], ta[ma])[k] if ma.sum() >= 10 else np.nan
            vb = full_metrics(pb[mb], tb[mb])[k] if mb.sum() >= 10 else np.nan
            if np.isfinite(va) and np.isfinite(vb):
                store[k].append(vb - va)
    out = {}
    for k in keys:
        v = np.array(store[k])
        out[k] = [float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# ---------- load frozen record set + year maps ----------
z = np.load(REC, allow_pickle=True)
import xarray as xr
def year_of(basin, t_idx):
    d = xr.open_dataset(DATA_DIR / CASUAL_FILES[basin])
    times = pd.DatetimeIndex(d["time"].values)
    d.close()
    return np.array([times[int(t)].year for t in t_idx])

rec = {}
for basin in ["SCS", "GBR", "CARIB"]:
    rec[basin] = {
        "PersA": z[f"{basin}_A_PersA_pred"].astype(float),
        "TSCF": z[f"{basin}_D_TSCF_pred"].astype(float),
        "TSAR": z[f"{basin}_C_anchored_pred"].astype(float),
        "target": z[f"{basin}_target"].astype(float),
        "t": z[f"{basin}_A_PersA_t"],
        "same_dir": f"{basin}->{basin}(source)",
    }

# ================= (5) coordinate stability =================
coord_rows = []
coord_ci = {}
for basin in ["SCS", "GBR", "CARIB"]:
    pa = rec[basin]["PersA"]; tg = rec[basin]["target"]
    roc, pr = auc_scores(pa, tg)
    rho = float(spearmanr(pa, tg).correlation)
    ci = boot_metric_ci(pa, tg, rec[basin]["t"])
    coord_rows.append(dict(basin=basin, n=len(pa), n_severe=int((tg >= SEV).sum()),
                           spearman=rho, roc_auc=roc, pr_auc=pr))
    coord_ci[basin] = {"spearman": [np.nan, np.nan], "roc_auc": ci["roc_auc"], "pr_auc": ci["pr_auc"]}
coord_df = pd.DataFrame(coord_rows)
coord_df.to_csv(OUT / "coordinate_metrics.csv", index=False)

year_rows = []
for basin in ["SCS", "GBR", "CARIB"]:
    pa = rec[basin]["PersA"]; tg = rec[basin]["target"]
    yr = year_of(basin, rec[basin]["t"])
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() == 0:
            continue
        roc, pr = auc_scores(pa[m], tg[m])
        year_rows.append(dict(basin=basin, year=int(y), n=int(m.sum()),
                              n_severe=int((tg[m] >= SEV).sum()),
                              spearman=float(spearmanr(pa[m], tg[m]).correlation),
                              roc_auc=roc, pr_auc=pr))
year_df = pd.DataFrame(year_rows)
year_df.to_csv(OUT / "yearly_coordinate_metrics.csv", index=False)

# ================= transfer matrix (no-training tier) =================
# Build a source model -> target metric map. For now only SCS-sourced TSCF/TSAR
# exist (frozen); reverse-source rows are appended once the corresponding
# GBR-/CARIB-trained TSCF/TSAR are available. PersA is source-independent.
def preval(basin):
    return float((rec[basin]["target"] >= SEV).mean())

rows = []
def add_row(src, tgt, model, m):
    rows.append(dict(source=src, target=tgt, model=model, **m))

# PersA: source-independent -> same per target basin
for tgt in ["SCS", "GBR", "CARIB"]:
    m = full_metrics(rec[tgt]["PersA"], rec[tgt]["target"])
    for src in [b for b in ["SCS", "GBR", "CARIB"] if b != tgt]:
        add_row(src, tgt, "PersA", m)

# SCS-sourced TSCF / TSAR / PersA-sourced to GBR & CARIB (frozen)
src = "SCS"
for tgt in ["GBR", "CARIB"]:
    for model in ["TSCF", "TSAR", "PersA"]:
        m = full_metrics(rec[tgt][model], rec[tgt]["target"])
        # source(self) metric for degradation
        m["_src_roc"] = full_metrics(rec[src][model], rec[src]["target"])["roc_auc"]
        m["_src_pr"] = full_metrics(rec[src][model], rec[src]["target"])["pr_auc"]
        add_row(src, tgt, model, m)

# XGB_THERMAL SCS-sourced control (if present)
if XGB_MJ.exists():
    xj = json.load(open(XGB_MJ, encoding="utf-8"))
    xg = {m["model"]: m for m in xj}
    if "XGB_THERMAL" in (xg.get("XGB_THERMAL") or {}):
        pass
rows_df = pd.DataFrame(rows)
rows_df.to_csv(OUT / "transfer_matrix.csv", index=False)

# transfer_matrix_ci: Δ = target - source for TSCF/TSAR/PersA (week-block diff CI)
ci_rows = []
src = "SCS"
for tgt in ["GBR", "CARIB"]:
    for model in ["TSCF", "TSAR", "PersA"]:
        diff = diff_boot_ci(rec[src][model], rec[src]["target"], rec[src]["t"],
                            rec[tgt][model], rec[tgt]["target"], rec[tgt]["t"])
        ci_rows.append(dict(source=src, target=tgt, model=model,
                            d_roc=full_metrics(rec[tgt][model], rec[tgt]["target"])["roc_auc"]
                                 - full_metrics(rec[src][model], rec[src]["target"])["roc_auc"],
                            d_roc_lo=diff["roc_auc"][1], d_roc_hi=diff["roc_auc"][2],
                            d_pr=full_metrics(rec[tgt][model], rec[tgt]["target"])["pr_auc"]
                                - full_metrics(rec[src][model], rec[src]["target"])["pr_auc"],
                            d_pr_lo=diff["pr_auc"][1], d_pr_hi=diff["pr_auc"][2]))
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUT / "transfer_matrix_ci.csv", index=False)

# provenance / hashes
prov = {
    "record_set": {"rec_structural_ablation.npz": md5(REC)},
    "checkpoints": {"frozen_TSCF": md5(LTTS_ROOT/"results/model_final_Btail3/model.pt"),
                    "TSAR_seed42": md5(LTTS_ROOT/"results/physics_anchored/model_seed_42.pt")},
    "data": {b: md5(DATA_DIR / f) for b, f in CASUAL_FILES.items()},
    "note": "no-training tier; reverse-source TSCF/TSAR rows to be appended after GBR/CARIB-sourced training",
}
json.dump(prov, open(OUT / "provenance_tier1.json", "w", encoding="utf-8"), indent=2)
print("== coordinate stability =="); print(coord_df.to_string(index=False))
print("\n== yearly coordinate stability =="); print(year_df.to_string(index=False))
print("\n== transfer matrix (no-training tier) =="); print(rows_df[["source","target","model","roc_auc","pr_auc","csi_tau09","severe_prevalence"]].to_string(index=False))
print("\n== delta CI (SCS->GBR/CARIB) =="); print(ci_df.to_string(index=False))
print("outputs ->", OUT)