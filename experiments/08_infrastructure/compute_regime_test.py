# compute_regime_test.py
#
# Regime-confound test for the six-direction transfer matrix (shortcoming:
# "CARIB-as-source transfers well possibly because its 2017-2020 test window
# is low-severity / no mass-bleaching year").
#
# Two pre-specified tests, frozen seed-42 checkpoints, no retraining:
#
#   T1. Target-year severity stratification. For every direction
#       (source model x target basin, TSCF + TSAR): ROC AUC and severe
#       prevalence PER TARGET YEAR. Killer cells: CARIB->SCS and GBR->SCS
#       on SCS 2016 and 2019 (the mass-bleaching years, prevalence >> 0).
#
#   T2. Prevalence-matched direction comparison. Within each target basin,
#       downsample background records to fix severe prevalence at a grid
#       p* (0.005..0.05), recompute every direction's ROC at the SAME p*
#       (100 subsample replicates, seed 42). If the direction ordering
#       (CARIB-source best) is preserved at every matched p*, the regime
#       account does not explain the source-identity effect.
#
#   Verdicts (pre-specified):
#     V1 regime dies if CARIB->SCS ROC >= 0.90 on BOTH SCS 2016 and 2019.
#     V2 ordering robust if CARIB-source mean ROC >= others at all p*.
#     V3 report Spearman(per-year ROC, per-year prevalence) pooled over
#        directions: if ROC declines with prevalence for ALL sources
#        roughly equally, prevalence is task difficulty, not source identity.
#
# SCS-source per-record predictions come from the frozen
# rec_structural_ablation.npz; GBR/CARIB-source predictions are generated
# here (full-map sliding inference, seed-42 reverse checkpoints, md5-verified).
#
# Outputs -> results/transfer_coordinate/regime_test/
#
# Run (piecf_env, GPU):
#   cd LTTS_CRW && python experiments/08_infrastructure/compute_regime_test.py

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "04_structural_ablation"))
from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
import ablate_a_dh_stopgrad as tscf_trainer
from physics_anchored_model import PhysicsAnchoredResidualModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEV = 0.8
N_REPS = 100
SEED = 42

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
WTK = RESULTS_DIR / "transfer_coordinate" / "weights"
REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
OUT = RESULTS_DIR / "transfer_coordinate" / "regime_test"
OUT.mkdir(parents=True, exist_ok=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def restore(kind, source):
    if kind == "TSCF":
        m = tscf_trainer.FinalSingleHeadModel(dynamic_channels=13)
    else:
        m = PhysicsAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(WTK / source / f"{'tscf' if kind == 'TSCF' else 'tsar'}_model.pt",
                                 map_location="cpu", weights_only=False))
    return m.eval().to(DEVICE)


def infer_full_map(model, kind, basin):
    """Full sliding-window forecast map for a frozen model on a target basin
    (issue t predicts t+4), matching compute_physics_anchored_audit.infer_tscf."""
    cfg = BASINS[basin]
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    fc = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(DEVICE)
    X = ds.X_dynamic_full
    bs = 16
    with torch.no_grad():
        for b in range(0, n - T + 1, bs):
            be = min(b + bs, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(DEVICE)
            xs = torch.zeros((be - b, 1), device=DEVICE)
            if kind == "TSCF":
                out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            else:
                anch = torch.FloatTensor(ds.y_regression_full[b + T - 1: be + T - 1].copy()).to(DEVICE)
                out = model(xd, xs, lm, anch, scale=0.4, tau=0.5, temp=0.1)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc, ds


def collect(fc, basin, kind, source):
    """PFV record set for one direction (issue-year filter, ocean, finite target)."""
    cfg = BASINS[basin]
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
    y_full = ds.y_regression_full
    years_lo, years_hi = cfg["split"]["test_years"]
    T, n = ds.history_weeks, ds.n_dates
    rec = {"pred": [], "target": [], "year": []}
    for t in range(T - 1, n):
        yr = int(ds.date_strings[t][:4])
        if not (years_lo <= yr <= years_hi):
            continue
        tt = t + 4
        if tt >= n:
            continue
        v = ~np.isnan(y_full[tt]) & ~ds.land_mask & ~np.isnan(fc[t])
        if not v.any():
            continue
        rec["pred"].append(fc[t][v])
        rec["target"].append(y_full[tt][v])
        rec["year"].append(np.full(int(v.sum()), yr))
    del ds
    return (np.concatenate(rec["pred"]), np.concatenate(rec["target"]),
            np.concatenate(rec["year"]))


if __name__ == "__main__":
    print("=" * 66)
    print(f"Regime-confound test — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Device:", DEVICE)
    print("=" * 66)

    # ---------- gather per-record predictions for all 12 cells ----------
    P = {}   # (source, kind, target) -> (pred, target, year)
    # SCS-source rows from the frozen record set
    assert md5(REC_NPZ) == REC_MD5, "frozen record set hash mismatch"
    z = np.load(REC_NPZ)
    years_map = {}
    for tgt in ["SCS", "GBR", "CARIB"]:
        cfg = BASINS[tgt]
        ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                             lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
        years_map[tgt] = np.array([int(s[:4]) for s in ds.date_strings])
        del ds
    for tgt in ["SCS", "GBR", "CARIB"]:
        t_idx = z[f"{tgt}_A_PersA_t"].astype(int)
        yr = years_map[tgt][t_idx]
        for kind, key in (("TSCF", "D_TSCF"), ("TSAR", "C_anchored")):
            P[("SCS", kind, tgt)] = (z[f"{tgt}_{key}_pred"].astype(float),
                                     z[f"{tgt}_target"].astype(float), yr)
        print(f"[loaded] SCS-source on {tgt} (frozen npz)", flush=True)

    # GBR/CARIB-source rows: fresh inference (seed-42 checkpoints)
    for src in ["GBR", "CARIB"]:
        for kind in ("TSCF", "TSAR"):
            model = restore(kind, src)
            for tgt in ["SCS", "GBR", "CARIB"]:
                fc, _ = infer_full_map(model, kind, tgt)
                P[(src, kind, tgt)] = collect(fc, tgt, kind, src)
                n, sev = len(P[(src, kind, tgt)][1]), int((P[(src, kind, tgt)][1] >= SEV).sum())
                print(f"[inferred] {src} {kind} -> {tgt}: n={n} sev={sev}", flush=True)
            del model
            torch.cuda.empty_cache()

    # ---------- T1: per-year ROC + prevalence ----------
    rows = []
    for (src, kind, tgt), (pred, target, yr) in P.items():
        for y in np.unique(yr):
            m = yr == y
            pv = float((target[m] >= SEV).mean())
            roc = (float(roc_auc_score((target[m] >= SEV).astype(int), pred[m]))
                   if len(np.unique((target[m] >= SEV).astype(int))) >= 2 else np.nan)
            rows.append({"source": src, "model": kind, "target": tgt, "year": int(y),
                         "n": int(m.sum()), "prevalence": pv, "roc": roc,
                         "role": "in_source" if src == tgt else "zero_shot",
                         "d_roc_vs_own_source": np.nan})
    # attach dROC vs own source per year
    df = pd.DataFrame(rows)
    for (src, kind) in {(r["source"], r["model"]) for r in rows}:
        own = df[(df["source"] == src) & (df["model"] == kind) & (df["target"] == src)]
        own = own.set_index("year")["roc"]
        m = (df["source"] == src) & (df["model"] == kind)
        df.loc[m, "d_roc_vs_own_source"] = df.loc[m, "roc"].values - df.loc[m, "year"].map(own).values
    df.to_csv(OUT / "per_year_roc.csv", index=False)

    print("\n===== T1: mass-bleaching-year cells (SCS 2016 / 2019) =====")
    v1 = {}
    for src in ["SCS", "GBR", "CARIB"]:
        for kind in ("TSCF", "TSAR"):
            cells = {}
            for y in (2016, 2019):
                r = df[(df["source"] == src) & (df["model"] == kind)
                       & (df["target"] == "SCS") & (df["year"] == y)]
                cells[y] = float(r["roc"].iloc[0]) if len(r) else np.nan
            holds = all(np.isfinite(v) and v >= 0.90 for v in cells.values())
            v1[f"{src}_{kind}"] = {"roc_2016": cells[2016], "roc_2019": cells[2019],
                                   "holds_ge_090": bool(holds)}
            print(f"  {src:5s} {kind:5s} SCS2016 {cells[2016]:.4f}  SCS2019 "
                  f"{cells[2019]:.4f}  -> {'holds' if holds else 'fails'} (V1: ROC>=0.90)")

    # ---------- T2: prevalence-matched ROC ----------
    pm_rows = []
    p_grid = [0.005, 0.010, 0.020, 0.035, 0.050]
    rng = np.random.default_rng(SEED)
    for tgt in ["SCS", "GBR", "CARIB"]:
        for kind in ("TSCF", "TSAR"):
            dirs = [(s, k) for (s, k, t) in P if t == tgt and k == kind]
            base = {d: P[d] for d in [(s, k, tgt) for (s, k) in dirs]}
            for p_star in p_grid:
                rocs = {f"{s}_{k}": [] for (s, k) in dirs}
                for rep in range(N_REPS):
                    for (s, k) in dirs:
                        pred, target, _ = P[(s, k, tgt)]
                        sev_m = target >= SEV
                        n_sev = int(sev_m.sum())
                        n_bg = int(round(n_sev * (1 - p_star) / p_star))
                        bg_idx = np.where(~sev_m)[0]
                        if n_bg > len(bg_idx):
                            n_bg = len(bg_idx)
                        pick_bg = rng.choice(bg_idx, n_bg, replace=False)
                        idx = np.concatenate([np.where(sev_m)[0], pick_bg])
                        yy = (target[idx] >= SEV).astype(int)
                        rocs[f"{s}_{k}"].append(
                            roc_auc_score(yy, pred[idx]) if len(np.unique(yy)) >= 2 else np.nan)
                for key, vals in rocs.items():
                    v = np.array([x for x in vals if np.isfinite(x)])
                    pm_rows.append({"target": tgt, "model": kind, "p_star": p_star,
                                    "source": key.split("_")[0],
                                    "roc_median": float(np.median(v)),
                                    "roc_lo": float(np.percentile(v, 2.5)),
                                    "roc_hi": float(np.percentile(v, 97.5))})
    pm = pd.DataFrame(pm_rows)
    pm.to_csv(OUT / "prevalence_matched_roc.csv", index=False)

    print("\n===== T2: direction ordering at matched prevalence (TSCF) =====")
    v2 = {}
    for tgt in ["SCS", "GBR", "CARIB"]:
        sub = pm[(pm["target"] == tgt) & (pm["model"] == "TSCF")]
        piv = sub.pivot_table(index="p_star", columns="source", values="roc_median")
        ok_all = True
        for p_star, r in piv.iterrows():
            best = r.idxmax()
            if best != "CARIB":
                ok_all = False
            print(f"  {tgt:5s} p*={p_star:.3f}: " +
                  "  ".join(f"{c} {r[c]:.4f}" for c in piv.columns) +
                  f"   best={best}")
        v2[tgt] = {"carib_best_at_every_p_star": bool(ok_all)}

    # ---------- V3: ROC-vs-prevalence gradient ----------
    g = []
    for (src, kind, tgt), (pred, target, yr) in P.items():
        ytab = df[(df["source"] == src) & (df["model"] == kind) & (df["target"] == tgt)]
        rho = spearmanr(ytab["prevalence"], ytab["roc"]).statistic
        g.append({"source": src, "model": kind, "target": tgt,
                  "spearman_roc_vs_prev": float(rho)})
    gdf = pd.DataFrame(g)
    gdf.to_csv(OUT / "roc_prevalence_gradient.csv", index=False)
    print("\n===== V3: per-direction ROC vs prevalence gradient =====")
    print(gdf.round(3).to_string(index=False))

    summary = {"experiment": "regime_confound_test",
               "generated": datetime.now().isoformat(),
               "device": str(DEVICE),
               "record_npz_md5": REC_MD5,
               "protocol": {"stratification": "per target year",
                            "prevalence_matching": f"background downsampling to p*, {N_REPS} reps, seed 42",
                            "no_retraining": True},
               "verdicts": {"V1_mass_bleaching_years": v1,
                            "V2_prevalence_matched_ordering": v2,
                            "V3_gradient": gdf.to_dict(orient="records")},
               "interpretation": ("V1/V2 support -> the CARIB-source advantage is not a "
                                  "low-severity regime artifact; week-block remains the "
                                  "primary inference basis; associative language only")}
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nOutputs -> {OUT}")
    print("DONE")
