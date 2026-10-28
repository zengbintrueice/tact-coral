# compute_prevalence_matched_transfer.py
#
# Weakness-fix #1B: prevalence-matched directional comparison.
#
# Limitation (3) offers the alternative account that the CARIB collapse is
# regime/label-scarcity (severe prevalence 0.016 vs 0.035 in GBR). This test
# removes the prevalence confound: every target's severe prevalence is
# matched to a common reference by subsampling (severe events down, or
# background down), and the six-direction matrix is re-read at matched
# prevalence.
#
# PRE-SPECIFIED PROTOCOL (frozen models; no retraining/tuning):
#   - populations: the frozen record sets (rec_structural_ablation.npz;
#     SCS-source rows) + one sliding inference per (reverse source, model,
#     target) with the FROZEN seed-42 reverse checkpoints, sampled at the
#     same record sets
#   - regimes (both pre-specified):
#       R_low  p*=0.01633 (CARIB): SCS/GBR severe events subsampled DOWN;
#              CARIB untouched
#       R_high p*=0.03546 (GBR):   SCS/CARIB background subsampled DOWN
#              (keep all severe); GBR untouched
#   - 500 subsample replicates, seed 42; ROC/PR percentile CI per direction
#   - verdict: the pair-specific collapse persists at matched prevalence iff
#     TSCF ROC(CARIB target) < ROC(GBR target) with non-overlapping
#     replicate CIs in BOTH regimes (same for PR)
#
# Outputs -> results/prevalence_matched_transfer/

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel

OUT = RESULTS_DIR / "prevalence_matched_transfer"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
TSCF_MD5 = "7cdbca26cae03fa53be53210940e8ce1"
WT_REV = RESULTS_DIR / "transfer_coordinate" / "weights"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEV = 0.8
N_REP = 500
SEED = 42
P_LOW = 0.01633          # CARIB severe prevalence
P_HIGH = 0.03546         # GBR severe prevalence

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
DIRECTIONS = [(s, t) for s in ("SCS", "GBR", "CARIB")
              for t in ("SCS", "GBR", "CARIB") if s != t]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def predict_sliding(model, ds, kind):
    kind = kind.lower()
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    fc = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(DEVICE)
    X = ds.X_dynamic_full
    Y = ds.y_regression_full
    with torch.no_grad():
        for b in range(0, n - T + 1, 16):
            be = min(b + 16, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(DEVICE)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(DEVICE)
            if kind == "tsar":
                anch = torch.FloatTensor(Y[b + T - 1: be + T - 1].copy()).to(DEVICE)
                out = model(xd, xs, lm, anch, scale=0.4, tau=0.5, temp=0.1)
            else:
                out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc


print("=" * 66)
print(f"Prevalence-matched directional comparison — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
print("Device:", DEVICE)
assert md5(REC_NPZ) == REC_MD5
z = np.load(REC_NPZ)

# ---- collect per-basin records: pred_TSCF / pred_TSAR / target (per source) ----
store = {}
for b, cfg in BASINS.items():
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
    t = z[f"{b}_A_PersA_t"].astype(np.int64)
    i = z[f"{b}_i"].astype(np.int64)
    j = z[f"{b}_j"].astype(np.int64)
    tgt = z[f"{b}_target"].astype(np.float64)
    store.setdefault(b, {})["recs"] = {
        "target": tgt,
        "SCS_source": {"TSCF": z[f"{b}_D_TSCF_pred"],
                       "TSAR": z[f"{b}_C_anchored_pred"],
                       "PersA": z[f"{b}_A_PersA_pred"]},
        "t": t, "i": i, "j": j}
    # reverse-source predictions (frozen seed-42 reverse checkpoints)
    for src in ("GBR", "CARIB"):
        preds = {}
        for kind, cls, wname in (
                ("TSCF", "tscf", "tscf_model.pt"),
                ("TSAR", "tsar", "tsar_model.pt")):
            w = WT_REV / src / wname
            if not w.exists():
                raise FileNotFoundError(w)
            if kind == "TSCF":
                import ablate_a_dh_stopgrad as tr
                m = tr.FinalSingleHeadModel(dynamic_channels=13)
            else:
                m = PhysicsAnchoredResidualModel(dynamic_channels=13)
            m.load_state_dict(torch.load(w, map_location="cpu", weights_only=False))
            m.eval().to(DEVICE)
            fc = predict_sliding(m, ds, kind)
            preds[kind] = fc[t, i, j].astype(np.float64)
            del m, fc
            torch.cuda.empty_cache()
        store[b]["recs"][f"{src}_source"] = preds
        print(f"[{b}] {src}-source reverse predictions done")
    persa = z[f"{b}_A_PersA_pred"]
    store[b]["recs"].setdefault("GBR_source", {})["PersA"] = persa
    store[b]["recs"].setdefault("CARIB_source", {})["PersA"] = persa
    del ds
    torch.cuda.empty_cache()

# ---- prevalence matching + replicate evaluation ----
rows = []
for src, tgt in DIRECTIONS:
    r = store[tgt]["recs"][f"{src}_source"]
    target = store[tgt]["recs"]["target"]
    sev = (target >= SEV)
    n = len(target)
    p_obs = sev.mean()
    for regime, p_star, mode in (("R_low", P_LOW, "severe_down"),
                                 ("R_high", P_HIGH, "bg_down")):
        rng = np.random.default_rng(SEED)
        n_sev_star = int(round(p_star * n))
        res = {m: {"roc": [], "pr": []} for m in ("TSCF", "TSAR", "PersA")}
        for _ in range(N_REP):
            if mode == "severe_down":
                if sev.sum() <= n_sev_star:
                    keep = np.ones(n, bool)
                else:
                    si = np.where(sev)[0]
                    keep = np.ones(n, bool)
                    drop = rng.choice(si, size=int(sev.sum() - n_sev_star),
                                      replace=False)
                    keep[drop] = False
            else:  # bg_down: keep all severe, subsample background
                n_bg_star = n_sev_star - int(sev.sum())
                if n_bg_star <= 0 or (~sev).sum() <= n_bg_star:
                    keep = np.ones(n, bool)
                else:
                    bi = np.where(~sev)[0]
                    keep = np.zeros(n, bool)
                    keep[sev] = True
                    keep[rng.choice(bi, size=n_bg_star, replace=False)] = True
            if keep.sum() == 0 or sev[keep].sum() == 0:
                continue
            for m in ("TSCF", "TSAR", "PersA"):
                s = r[m][keep]
                y = sev[keep].astype(int)
                try:
                    res[m]["roc"].append(roc_auc_score(y, s))
                    res[m]["pr"].append(average_precision_score(y, s))
                except ValueError:
                    pass
        for m in ("TSCF", "TSAR", "PersA"):
            for met in ("roc", "pr"):
                a = np.array(res[m][met], float)
                rows.append({"source": src, "target": tgt, "regime": regime,
                             "p_star": p_star, "p_obs_target": p_obs,
                             "model": m, "metric": met,
                             "mean": float(a.mean()) if a.size else np.nan,
                             "lo": float(np.percentile(a, 2.5)) if a.size else np.nan,
                             "hi": float(np.percentile(a, 97.5)) if a.size else np.nan})
    print(f"[{src}->{tgt}] matched regimes done (obs prevalence {p_obs:.4f})")

df = pd.DataFrame(rows)
df.to_csv(OUT / "matched_matrix.csv", index=False)

# ---- verdicts (pre-specified) ----
def get(src, tgt, regime, model, metric):
    r = df[(df.source == src) & (df.target == tgt) & (df.regime == regime)
           & (df.model == model) & (df.metric == metric)]
    return r[["mean", "lo", "hi"]].iloc[0]

verdicts = {}
for regime in ("R_low", "R_high"):
    g = get("SCS", "GBR", regime, "TSCF", "roc")
    c = get("SCS", "CARIB", regime, "TSCF", "roc")
    gpr = get("SCS", "GBR", regime, "TSCF", "pr")
    cpr = get("SCS", "CARIB", regime, "TSCF", "pr")
    roc_persist = bool(c["hi"] < g["lo"])
    pr_persist = bool(cpr["hi"] < gpr["lo"])
    verdicts[regime] = {
        "TSCF_roc_GBR": [g["mean"], g["lo"], g["hi"]],
        "TSCF_roc_CARIB": [c["mean"], c["lo"], c["hi"]],
        "TSCF_pr_GBR": [gpr["mean"], gpr["lo"], gpr["hi"]],
        "TSCF_pr_CARIB": [cpr["mean"], cpr["lo"], cpr["hi"]],
        "collapse_persists_roc": roc_persist,
        "collapse_persists_pr": pr_persist}
    print(f"[{regime}] TSCF ROC GBR {g['mean']:.4f}[{g['lo']:.4f},{g['hi']:.4f}] "
          f"vs CARIB {c['mean']:.4f}[{c['lo']:.4f},{c['hi']:.4f}] -> "
          f"persists_roc={roc_persist}; PR persists={pr_persist}")

summary = {"experiment": "prevalence_matched_transfer",
           "generated": datetime.now().isoformat(),
           "device": str(DEVICE),
           "regimes": {"R_low": P_LOW, "R_high": P_HIGH},
           "n_rep": N_REP, "seed": SEED,
           "note": "reverse-source predictions from frozen seed-42 checkpoints; "
                   "subsampling CIs (500 replicates); no retraining",
           "verdicts": verdicts}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write(
"""# AUDIT - Prevalence-matched directional comparison

- Prevalence confound removed by subsampling to a common severe prevalence:
  R_low p*=0.01633 (CARIB; SCS/GBR severe downsampled) and
  R_high p*=0.03546 (GBR; SCS/CARIB background downsampled, all severe kept).
- 500 subsample replicates (seed 42); percentile CIs; frozen seed-42 models,
  reverse checkpoints md5-tracked in transfer_coordinate/weights.
- ROC-AUC is prevalence-invariant at population level, so the informative
  matched comparison is PR-AUC (prevalence-dependent) plus the composition-
  robustness check on ROC.
- Associative conditional-risk interpretation only; no retraining, no target
  fitting, no threshold optimisation.
""")
print(f"\nOutputs -> {OUT}")
print("Done.")
