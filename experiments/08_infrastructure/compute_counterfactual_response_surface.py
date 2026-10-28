# compute_counterfactual_response_surface.py
#
# Counterfactual Thermal-Response Surface for the frozen TSAR.
#
# Question: at FIXED current DHW, does changing the thermal trajectory
# (ΔDHW over the last 4 weeks) change the model response TSAR and its
# residual R = TSAR − PersA — and is that response basin-specific?
#
# STRICT PROTOCOL:
#   - frozen rec_structural_ablation.npz (md5-verified) + frozen seed-42 TSAR
#     checkpoint (md5-verified); NO retraining, NO target fitting/tuning, NO
#     threshold optimisation; frozen-model counterfactual inference ONLY
#   - real data are never modified: counterfactual thermal states are built on
#     copies of the model input windows
#   - results are model-response / associative evidence, NOT causal effects
#
# PRE-SPECIFIED counterfactual intervention (identical in all basins):
#   target (d, δ) in DHW units: d = current DHW (°C-weeks),
#   δ = ΔDHW over the last 4 weeks (°C-weeks).
#   For a supporting real record (window of 24 weeks, observed anchor a):
#     1. anchor' = clip(d/10, 0, 1)                       (PersA' = anchor')
#     2. HotSpot channel scaled over the whole window by r = a'/max(a, 0.02)
#        (level shift preserving the record's trajectory shape)
#     3. trajectory override: add Δh = δ/28 per week to HotSpot over the LAST
#        4 window weeks (CRW-consistent approx: 4 weeks x 7 days x HS = DHW
#        gain), clipped at 0
#     4. cumulative channels shifted consistently:
#        cumsum_W' = max(cumsum_W * r + 4*Δh, 0)  for W in {4, 12}
#     5. d_hotspot_dt recomputed as h'(last) - h'(second-last); sst/ssta and
#        all nonthermal channels UNCHANGED (documented in AUDIT)
#     6. frozen TSAR inference -> TSAR' = anchor' + gate'*res'*0.4; R' = TSAR' - anchor'
#
# Support restriction (anti-extrapolation): a (d, δ) cell is evaluated only on
# real records with |10a − d| <= 0.5 and |10*δ_obs − δ| <= 1.0 (δ_obs from the
# frozen record set); cells with < 100 supporting records are MASKED;
# at most 256 records per cell (seed-42 subsample). Grids share the common
# supported range across basins.
#
# Uncertainty: per-cell week-block bootstrap (1000, seed 42) over the
# supporting records' counterfactual responses (re-averaging saved
# per-record results; no re-inference inside the bootstrap). Cross-basin
# differences use independent per-basin bootstrap streams (element-wise
# difference), as in the cross-basin consistency test.
#
# Outputs -> results/counterfactual_response_surface/
#
# Run (piecf_env, GPU):
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_counterfactual_response_surface.py

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel

OUT = RESULTS_DIR / "counterfactual_response_surface"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"
CKPT_TSAR = RESULTS_DIR / "physics_anchored" / "model_seed_42.pt"
TSAR_MD5 = "ae86e017bc5ba634b0c68e5d13b4f9f0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
CH = {c: i for i, c in enumerate(DATASET_ORDER)}
HS, C4, C12, DHD = CH["hotspot"], CH["hotspot_cumsum_4w"], CH["hotspot_cumsum_12w"], CH["d_hotspot_dt"]

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

DHW_TOL = 0.5          # |10a − d| tolerance (°C-weeks)
DELTA_TOL = 1.0        # |10*δ_obs − δ| tolerance (°C-weeks)
MIN_SUPPORT = 100
MAX_PER_CELL = 256
N_BOOT = 1000
BOOT_SEED = 42
D_LEVELS = 25          # grid resolution along current DHW
DELTA_LEVELS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=float)  # °C-wk / 4wk
D_MAX_QUANTILE = 0.995  # per-basin cap; grid uses the min across basins
ANCHOR_FLOOR = 0.02     # coordinate floor to avoid level-ratio blowup


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"Counterfactual thermal-response surface — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
print(f"Device: {DEVICE}")
assert md5(REC_NPZ) == REC_MD5, "frozen record set hash mismatch"
assert md5(CKPT_TSAR) == TSAR_MD5, "frozen TSAR checkpoint hash mismatch"
print(f"frozen records OK; frozen TSAR OK ({TSAR_MD5})")

z = np.load(REC_NPZ)
model = PhysicsAnchoredResidualModel(dynamic_channels=13)
model.load_state_dict(torch.load(CKPT_TSAR, map_location="cpu", weights_only=False))
model.eval().to(DEVICE)

# ---- pass 1: common supported DHW range across basins ----
rec_meta = {}
for basin, cfg in BASINS.items():
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=4, dataset_file=cfg["file"], **cfg["split"])
    y_full = ds.y_regression_full
    X = ds.X_dynamic_full
    t_idx = z[f"{basin}_A_PersA_t"].astype(np.int64)
    i_idx = z[f"{basin}_i"].astype(np.int64)
    j_idx = z[f"{basin}_j"].astype(np.int64)
    a = z[f"{basin}_A_PersA_pred"].astype(np.float64)          # y(t), coordinate
    d_obs = a * 10.0
    delta_obs = (a - y_full[t_idx - 4, i_idx, j_idx]) * 10.0    # °C-wk / 4wk
    ok = np.isfinite(a) & np.isfinite(delta_obs) & (delta_obs >= -6) & (delta_obs <= 6)
    rec_meta[basin] = {"ds": ds, "t": t_idx[ok], "i": i_idx[ok], "j": j_idx[ok],
                       "a": a[ok], "d": d_obs[ok], "delta": delta_obs[ok]}
    print(f"[{basin}] records {ok.sum()}/{len(a)} in δ range [−6, +6] °C-wk; "
          f"DHW p99.5 = {np.quantile(d_obs[ok], D_MAX_QUANTILE):.2f}")

d_max = min(float(np.quantile(m["d"], D_MAX_QUANTILE)) for m in rec_meta.values())
d_grid = np.linspace(0.0, d_max, D_LEVELS)
print(f"common counterfactual grid: d ∈ [0, {d_max:.2f}] °C-wk "
      f"({D_LEVELS} levels) × δ ∈ {list(DELTA_LEVELS)} °C-wk")

# ---- pass 2: counterfactual inference per cell ----
cell_rows, rec_rows = [], []
for basin, m in rec_meta.items():
    print(f"\n=== {basin} ===")
    ds = m["ds"]
    X = ds.X_dynamic_full
    T = ds.history_weeks
    t_r, i_r, j_r, a_r, d_r, dl_r = m["t"], m["i"], m["j"], m["a"], m["d"], m["delta"]
    n_cells = 0
    for d in d_grid:
        for delta in DELTA_LEVELS:
            sel = np.where((np.abs(d_r - d) <= DHW_TOL)
                           & (np.abs(dl_r - delta) <= DELTA_TOL))[0]
            if len(sel) < MIN_SUPPORT:
                continue
            rng = np.random.default_rng(BOOT_SEED)
            if len(sel) > MAX_PER_CELL:
                sel = np.sort(rng.choice(sel, MAX_PER_CELL, replace=False))
            # ---- build counterfactual batch (FULL spatial window, only the
            # target pixel's thermal channels modified — preserves the conv
            # context exactly as in the frozen evaluation) ----
            B = len(sel)
            anchor = np.clip(d / 10.0, 0.0, 1.0)
            r_scale = anchor / np.maximum(a_r[sel], ANCHOR_FLOOR)
            dh = delta / 28.0                                     # per-week HS add
            H, W = X.shape[1], X.shape[2]
            batch = np.empty((B, T, H, W, 13), np.float32)
            i_sel = i_r[sel]; j_sel = j_r[sel]
            for k in range(B):
                te = t_r[sel[k]]
                win = X[te - T + 1: te + 1].copy()                # [T,H,W,13]
                px = win[:, i_sel[k], j_sel[k], :]                # [T,13] view
                px[..., HS] = np.maximum(px[..., HS] * r_scale[k], 0.0)
                px[-4:, HS] = np.maximum(px[-4:, HS] + dh, 0.0)
                px[..., C4] = np.maximum(px[..., C4] * r_scale[k] + 4 * dh, 0.0)
                px[..., C12] = np.maximum(px[..., C12] * r_scale[k] + 4 * dh, 0.0)
                px[..., DHD] = px[-1, HS] - px[-2, HS]
                win[:, i_sel[k], j_sel[k], :] = px
                batch[k] = win
            lm = torch.BoolTensor(ds.land_mask).to(DEVICE)
            xd = torch.FloatTensor(batch).to(DEVICE)
            xs = torch.zeros((B, 1), device=DEVICE)
            anch = torch.full((B,), anchor, dtype=torch.float32,
                              device=DEVICE).view(B, 1, 1).expand(B, H, W)
            with torch.no_grad():
                out = model(xd, xs, lm, anch, scale=0.4, tau=0.5, temp=0.1)
            yfinal_map = out[0][:, -1]                            # [B,H,W]
            yfinal = yfinal_map[torch.arange(B),
                                torch.as_tensor(i_sel, device=DEVICE),
                                torch.as_tensor(j_sel, device=DEVICE)].cpu().numpy()
            persa_cf = np.full(B, anchor)
            r_cf = yfinal - anchor
            week = t_r[sel]
            rec_rows.append(pd.DataFrame({
                "basin": basin, "d": d, "delta": delta, "week": week,
                "tsar_cf": yfinal, "persa_cf": persa_cf, "r_cf": r_cf}))
            cell_rows.append({"basin": basin, "d": d, "delta": delta,
                              "support": B,
                              "tsar_mean": float(yfinal.mean()),
                              "persa_mean": anchor,
                              "r_mean": float(r_cf.mean())})
            n_cells += 1
            del xd, out
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    print(f"  evaluated cells: {n_cells}")

    # free dataset
    del ds, X
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

cells = pd.DataFrame(cell_rows)
recs = pd.concat(rec_rows, ignore_index=True)

# ---- per-cell week-block bootstrap CI (on saved per-record responses) ----
print("\nbootstrap per-cell CIs ...")
rng_boot = np.random.default_rng(BOOT_SEED)
boot_rows = []
for (basin, d, delta), grp in recs.groupby(["basin", "d", "delta"]):
    weeks, inv = np.unique(grp["week"].values, return_inverse=True)
    idx_by_week = [np.where(inv == w)[0] for w in range(len(weeks))]
    vals = grp["r_cf"].values
    ts_vals = grp["tsar_cf"].values
    means, ts_means = [], []
    for _ in range(N_BOOT):
        pick = rng_boot.integers(0, len(weeks), len(weeks))
        idx = np.concatenate([idx_by_week[w] for w in pick])
        means.append(vals[idx].mean())
        ts_means.append(ts_vals[idx].mean())
    boot_rows.append({"basin": basin, "d": d, "delta": delta,
                      "r_lo": float(np.percentile(means, 2.5)),
                      "r_hi": float(np.percentile(means, 97.5)),
                      "tsar_lo": float(np.percentile(ts_means, 2.5)),
                      "tsar_hi": float(np.percentile(ts_means, 97.5))})
boots = pd.DataFrame(boot_rows)
cells = cells.merge(boots, on=["basin", "d", "delta"], how="left")

# ---- same-DHW trajectory contrast (pre-specified summary) ----
traj_rows = []
for basin in BASINS:
    sub = cells[cells["basin"] == basin]
    for d, g in sub.groupby("d"):
        if len(g) >= 4:
            rise = g.loc[g["delta"] == 4, "r_mean"]
            fall = g.loc[g["delta"] == -4, "r_mean"]
            if len(rise) and len(fall):
                traj_rows.append({"basin": basin, "d": d,
                                  "contrast": "R(rise +4) - R(fall -4)",
                                  "point": float(rise.iloc[0] - fall.iloc[0])})
traj = pd.DataFrame(traj_rows)

# ---- cross-basin difference on common support (CARIB − GBR) ----
piv = cells.pivot_table(index=["d", "delta"], columns="basin", values="r_mean")
common = piv.dropna(subset=["GBR", "CARIB"])
cross_rows = []
if len(common):
    recs_g = recs[recs["basin"] == "GBR"]
    recs_c = recs[recs["basin"] == "CARIB"]
    for (d, delta) in common.index:
        m_g = (recs_g["d"] == d) & (recs_g["delta"] == delta)
        m_c = (recs_c["d"] == d) & (recs_c["delta"] == delta)
        gg = recs_g.loc[m_g, "r_cf"].values
        cc = recs_c.loc[m_c, "r_cf"].values
        wk_g = recs_g.loc[m_g, "week"].values
        wk_c = recs_c.loc[m_c, "week"].values
        rng_g = np.random.default_rng(BOOT_SEED)
        rng_c = np.random.default_rng(BOOT_SEED + 1)
        def boot_mean(v, wk, rng):
            weeks, inv = np.unique(wk, return_inverse=True)
            ibw = [np.where(inv == w)[0] for w in range(len(weeks))]
            out = []
            for _ in range(N_BOOT):
                pick = rng.integers(0, len(weeks), len(weeks))
                out.append(v[np.concatenate([ibw[w] for w in pick])].mean())
            return np.array(out)
        bg = boot_mean(gg, wk_g, rng_g)
        bc = boot_mean(cc, wk_c, rng_c)
        diff = bc - bg
        cross_rows.append({"d": d, "delta": delta,
                           "r_carib_minus_gbr": float(piv.loc[(d, delta), "CARIB"]
                                                      - piv.loc[(d, delta), "GBR"]),
                           "lo": float(np.percentile(diff, 2.5)),
                           "hi": float(np.percentile(diff, 97.5))})
cross = pd.DataFrame(cross_rows)

# ---- outputs ----
cells.to_csv(OUT / "surface.csv", index=False)
recs.to_csv(OUT / "counterfactual_records.csv", index=False)
traj.to_csv(OUT / "trajectory_contrast.csv", index=False)
cross.to_csv(OUT / "cross_basin_difference.csv", index=False)
pd.DataFrame([{"basin": b, "stat": f"traj_contrast_R(+4 minus -4)_median_over_d",
               "point": float(traj[traj['basin'] == b]['point'].median())}
              for b in BASINS if len(traj[traj['basin'] == b])]).to_csv(
    OUT / "bootstrap.csv", index=False)

summary = {
    "experiment": "counterfactual_thermal_response_surface",
    "generated": datetime.now().isoformat(),
    "device": str(DEVICE),
    "frozen": {"records_md5": REC_MD5, "tsar_md5": TSAR_MD5},
    "grid": {"d_max_common": d_max, "n_d": D_LEVELS,
             "delta_levels": DELTA_LEVELS.tolist()},
    "support": {"dhw_tol": DHW_TOL, "delta_tol": DELTA_TOL,
                "min_support": MIN_SUPPORT, "max_per_cell": MAX_PER_CELL},
    "intervention": ("anchor'=clip(d/10); HotSpot window scaled by anchor'/max(a,0.02); "
                     "+δ/28 per week on last 4 weeks; cumsum_4/12 shifted by +4Δh; "
                     "d_hotspot_dt recomputed; sst/ssta/nonthermal unchanged"),
    "cells_evaluated": {b: int((cells["basin"] == b).sum()) for b in BASINS},
    "interpretation": "counterfactual MODEL RESPONSE / associative evidence; "
                      "NOT a causal effect",
}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

# ---- AUDIT.md ----
L = ["# AUDIT — Counterfactual Thermal-Response Surface", "",
     f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ",
     f"Frozen records md5 `{REC_MD5}`; frozen TSAR (seed-42) md5 `{TSAR_MD5}` "
     "(both verified). Device: "
     f"{DEVICE}.", "",
     "## Protocol adherence", "",
     "- Frozen-model counterfactual inference ONLY: no retraining, no target "
     "fitting/tuning, no threshold optimisation. Real data are never modified; "
     "counterfactual thermal states are built on copies of model input windows.",
     "- **These are model-response surfaces, i.e. associative evidence about "
     "the frozen model's input-output map — NOT causal effects.**",
     "- Intervention is identical in all basins and pre-specified (see "
     "summary.json / script header): anchor + level-scaled HotSpot window + "
     "δ/28 per week on the last 4 weeks + consistent cumsum shift + "
     "recomputed d_hotspot_dt. sst/ssta and nonthermal channels are held at "
     "the observed values (documented limitation: the counterfactual is "
     "defined on the thermal-trajectory channels only).",
     "- Anti-extrapolation: cells require ≥100 real records within "
     "|10a−d| ≤ 0.5 and |10δ_obs−δ| ≤ 1.0 °C-weeks; ≤256 records per cell "
     "(seed-42 subsample); grid restricted to the common supported DHW range "
     f"[0, {d_max:.2f}] °C-weeks.",
     "- Uncertainty: per-cell week-block bootstrap (1,000, seed 42) over saved "
     "per-record counterfactual responses (no re-inference). Cross-basin "
     "differences use independent per-basin bootstrap streams (no cross-basin "
     "pairing is assumed).",
     "- δ/28 weekly-HotSpot approximation to the CRW 84-day DHW accumulation "
     "is an approximation; it is applied identically everywhere and does not "
     "use target information.", "",
     "## Trajectory contrast at fixed DHW (R at δ=+4 minus R at δ=−4)", ""]
for b in BASINS:
    tb = traj[traj["basin"] == b]
    if len(tb):
        L.append(f"- {b}: median over supported d = "
                 f"{tb['point'].median():+.4f} (coordinate units); "
                 f"positive at {int((tb['point'] > 0).sum())}/{len(tb)} levels.")
L += ["", "## Interpretation guard", "",
      "The surface describes what the FROZEN MODEL would output under "
      "hypothetical thermal trajectories; it does not describe how real reefs "
      "respond to heat stress. Any 'susceptibility' language is shorthand for "
      "basin differences in the model's response map.", ""]
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print("\nTrajectory contrast (R at δ=+4 minus δ=−4, median over d):")
print(traj.groupby("basin")["point"].median().round(4).to_string())
print(f"\nOutputs -> {OUT}")
print("Done.")