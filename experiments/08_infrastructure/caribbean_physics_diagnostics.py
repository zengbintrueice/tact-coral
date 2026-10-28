#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caribbean_physics_diagnostics.py — decisive physics-constrained experiments
============================================================================

Two cheap, no-retraining diagnostics to test WHY the frozen SCS-trained TSCF
fails to transfer to the Caribbean (ROC 0.60), and whether physics-grounded
fixes recover skill:

  EXP A — TEST-TIME RE-NORMALIZATION (distribution-shift hypothesis):
      The model's ChannelNormalizer z-scores inputs with SCS train (2003-2010)
      mean/std. If the Caribbean failure is driven by input distribution shift,
      recomputing those stats on the CARIBBEAN train-period weeks (2003-2010,
      pre-test, no future leakage) and re-running the FROZEN weights should
      recover most skill.

  EXP B — PHYSICS-ANCHORED FUSION (structural prior):
      y_final(t+4) = a * PersA_DHW(t) + (1-a) * TSCF(t),  a in [0,1].
      PersA_DHW = clip(DHW(t)/10) = current severe thermal stress, which
      transfers perfectly (ROC 0.98, basin-invariant CRW definition). This
      makes the strong physics-based persistence baseline a STRUCTURAL PRIOR
      rather than just a benchmark: TSCF can only add skill, and its transfer
      floor is persistence ranking. a is selected on Caribbean VAL (2014-2016),
      reported on TEST (2017-2020).

No training, no tuning of model weights, no threshold selection on test
(tau=0.90 pinned). Outputs -> results/caribbean_zero_shot/phys_diagnostics.

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' caribbean_physics_diagnostics.py
"""
import io, sys, os, json, warnings, importlib
from pathlib import Path

# do NOT pre-wrap stdout — ablate_a_dh_stopgrad re-wraps it on import
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))
from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from sklearn.metrics import roc_auc_score, average_precision_score

LEAD_WEEKS = 4
SEVERE = 0.8
TAU = 0.90
OUT_DIR = RESULTS_DIR / "caribbean_zero_shot" / "phys_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CARIB_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                   val_years=(2014, 2016), test_years=(2017, 2020))

RAW_DYN = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par', 'seasonal_sin',
           'wind_speed', 'ssr', 'sst_anomaly']
ALL13 = RAW_DYN + ['d_hotspot_dt', 'hotspot_cumsum_4w', 'hotspot_cumsum_12w', 'seasonal_cos']


def restore():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(A6_WEIGHT, map_location='cpu', weights_only=False))
    m.eval()
    return m


def infer_all(m, ds, device):
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n = ds.n_dates
    C = ds.C_dyn
    fm = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    with torch.no_grad():
        for b in range(0, n - T + 1, 16):
            be = min(b + 16, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1:te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            out = m(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            for k in range(be - b):
                fm[b + k + T - 1] = out[0][k, -1].cpu().numpy()
    return fm


def caribbean_stats_13(ds, years=(2003, 2010)):
    """13-channel mean/std on Caribbean train-period weeks (causal v2), same
    semantics as compute_normalization_v2 (derived recomputed trailing)."""
    import xarray as xr
    dsn = xr.open_dataset(os.path.join(ds.data_dir, ds.dataset_file))
    times = pd.DatetimeIndex(dsn.time.values)
    msk = (times.year >= years[0]) & (times.year <= years[1])
    raw = [dsn[n].values.astype(np.float32) for n in RAW_DYN]
    hs = raw[2].copy()
    nanm = np.isnan(hs)
    hsc = np.where(nanm, 0.0, hs)
    cum = np.cumsum(hsc, axis=0)
    d_hs = np.zeros_like(hs); d_hs[1:] = hsc[1:] - hsc[:-1]; d_hs[nanm] = np.nan
    cs4 = np.zeros_like(hs); cs12 = np.zeros_like(hs)
    for i in range(hs.shape[0]):
        cs4[i] = cum[i] if i < 4 else cum[i] - cum[i - 4]
        cs12[i] = cum[i] if i < 12 else cum[i] - cum[i - 12]
    cs4[nanm] = np.nan; cs12[nanm] = np.nan
    woy = np.array([t.isocalendar()[1] for t in times], dtype=np.float32)
    sc = np.broadcast_to(np.cos(2 * np.pi * woy / 52.0)[:, None, None].astype(np.float32),
                         hs.shape).copy()
    arrs = raw + [d_hs, cs4, cs12, sc]
    means, stds = [], []
    for a in arrs:
        sl = a[msk]
        means.append(float(np.nanmean(sl)))
        stds.append(float(np.nanstd(sl)))
    return np.array(means, np.float32), np.array(stds, np.float32)


def collect(ds, fm, y, y0, y1):
    T = ds.history_weeks
    n = ds.n_dates
    ocean = ~ds.land_mask                       # ocean-masked (aligned label_mask)
    p, t = [], []
    for ti in range(T - 1, n):
        yy = int(ds.date_strings[ti][:4])
        if not (y0 <= yy <= y1):
            continue
        tt = ti + LEAD_WEEKS
        if tt >= n:
            continue
        v = ~np.isnan(y[tt]) & ocean
        if v.sum() == 0:
            continue
        p.append(fm[ti][v]); t.append(y[tt][v])
    return np.concatenate(p), np.concatenate(t)


def metrics(p, t):
    sev = t >= SEVERE
    h = int(((p >= TAU) & sev).sum()); ms = int(((p < TAU) & sev).sum())
    fa = int(((p >= TAU) & ~sev).sum())
    csi = h / (h + ms + fa) if (h + ms + fa) else np.nan
    if len(np.unique(sev)) < 2:
        roc, pr = np.nan, np.nan
    else:
        roc = roc_auc_score(sev, p); pr = average_precision_score(sev, p)
    return dict(n=len(p), severe=int(sev.sum()), roc=float(roc), pr=float(pr), csi=float(csi))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=LEAD_WEEKS, dataset_file="caribbean_causal_v2.nc",
                         **CARIB_SPLIT)
    y = ds.y_regression_full
    persA = y.astype(np.float32)  # current-DHW persistence

    # --- EXP A: re-normalization ---
    print("\n[EXP A] test-time re-normalization (Caribbean train-period stats)...")
    m_a = restore().to(device)
    fm_orig = infer_all(m_a, ds, device)
    pv_orig, tv = collect(ds, fm_orig, y, 2014, 2016)
    pt_orig, tt = collect(ds, fm_orig, y, 2017, 2020)
    print("  [raw SCS-norm] val n=%d sev=%d ; test %s" % (
        len(pv_orig), int((tv >= SEVERE).sum()), metrics(pt_orig, tt)))

    means_c, stds_c = caribbean_stats_13(ds)
    with torch.no_grad():
        m_a.channel_norm.mean.copy_(torch.tensor(means_c).view(1, 1, 1, 1, -1))
        m_a.channel_norm.std.copy_(torch.tensor(stds_c).view(1, 1, 1, 1, -1))
    fm_renorm = infer_all(m_a, ds, device)
    pt_ren, _ = collect(ds, fm_renorm, y, 2017, 2020)
    print("  [Caribbean-renorm] test %s" % (metrics(pt_ren, tt),))
    del m_a

    # --- EXP B: physics-anchored fusion (alpha on val CSI@0.9) ---
    print("\n[EXP B] physics-anchored fusion  y = a*PersA + (1-a)*TSCF ...")
    pA_val, _ = collect(ds, persA, y, 2014, 2016)
    pA_test, _ = collect(ds, persA, y, 2017, 2020)
    best = (None, -1)
    sweep = []
    for a in np.arange(0.0, 1.0001, 0.05):
        pv_f = a * pA_val + (1 - a) * pv_orig
        csi_val = metrics(pv_f, tv)["csi"]
        sweep.append((float(a), float(csi_val)))
        if csi_val > best[1]:
            best = (float(a), float(csi_val))
    a_opt = best[0]
    pt_f = a_opt * pA_test + (1 - a_opt) * pt_orig
    print("  alpha* (val CSI-max) = %.2f (val CSI=%.4f)" % (a_opt, best[1]))
    print("  test fusion      %s" % (metrics(pt_f, tt),))
    print("  test PersA only  %s" % (metrics(pA_test, tt),))
    print("  test TSCF only   %s" % (metrics(pt_orig, tt),))

    # also pure alpha=1 (persistence) and alpha=0 (TSCF)
    out = {
        "tau_pinned": TAU, "severe": SEVERE, "alpha_selected_on": "Caribbean val 2014-2016",
        "expA": {"scs_norm": metrics(pt_orig, tt),
                 "caribbean_train_renorm": metrics(pt_ren, tt)},
        "expB": {"alpha_opt": a_opt, "val_csi_at_alpha": best[1],
                 "alpha_sweep": sweep,
                 "test_persA_only": metrics(pA_test, tt),
                 "test_tscf_only": metrics(pt_orig, tt),
                 "test_fusion": metrics(pt_f, tt)},
    }
    with open(OUT_DIR / "phys_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    pd.DataFrame(out["expB"]["alpha_sweep"], columns=["alpha", "val_csi"]).to_csv(
        OUT_DIR / "fusion_alpha_sweep.csv", index=False)
    print(f"\nOutputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
