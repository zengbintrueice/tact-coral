#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review-response analyses (B + C):

  B. Clip-ungated control (B_clip = clip(PersA + res, 0, 1)) — evaluate the
     existing un-gated residual predictions range-clipped to [0,1] on the same
     PFV record sets, to test whether the un-gated residual's calibration
     degradation is purely a score-out-of-range artifact.

  C. XGBoost week-block bootstrap CIs — reconstruct the exact (t, i, j) week
     identity of every XGB_FULL / XGB_THERMAL / XGB_NO_THERMAL prediction from
     results/xgb_transfer_control/predictions.npz (same loader + label_mask
     ordering as the record-set audit), then compute paired week-block bootstrap
     95% CIs (N=1000, seed 42) for ROC/PR/Brier/ECE per variant per domain.
     No retraining.

Outputs:
  B -> results/physics_anchored/structural_ablation/clip_ungated.csv
      (B_clip metrics per domain + paired CIs vs PersA and vs C)
  C -> results/xgb_transfer_control/bootstrap_cis.csv
"""
import io, os, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
if not DATA_DIR.is_dir():
    DATA_DIR = LTTS_ROOT / "data"

SEV = 0.8
TAU = 0.90
N_BOOT = 1000
BOOT_SEED = 42

from src.data.aligned_dataset import SCSPIECFDataset  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

DOMAIN_NC = {"SCS": "scs_coral_bleaching_dataset_v9_1deg_raw.nc",
             "GBR": "gbr_coral_bleaching_dataset_v9_1deg_raw.nc",
             "CARIB": "caribbean_coral_bleaching_dataset_v9_1deg_raw.nc"}
SPLIT = {"SCS": dict(mode="test", years=(2003, 2010, 2011, 2015, 2016, 2019)),
         "GBR": dict(mode="test", years=(2003, 2010, 2014, 2016, 2017, 2020)),
         "CARIB": dict(mode="test", years=(2003, 2010, 2014, 2016, 2017, 2020))}
XGB_VARS = ["XGB_FULL", "XGB_THERMAL", "XGB_NO_THERMAL"]
XGB_META_KEY = {"XGB_FULL": "XGB_FULL", "XGB_THERMAL": "XGB_THERMAL",
                "XGB_NO_THERMAL": "XGB_NO_THERMAL"}


def metrics(pred, target):
    sev = target >= SEV
    if len(np.unique(sev)) < 2:
        roc = pr = np.nan
    else:
        roc = roc_auc_score(sev, pred)
        pr = average_precision_score(sev, pred)
    pb = pred >= TAU
    h = int(np.sum(pb & sev)); ms = int(np.sum(~pb & sev)); fa = int(np.sum(pb & ~sev))
    csi = h / (h + ms + fa) if (h + ms + fa) > 0 else np.nan
    brier = float(np.mean((pred - target) ** 2))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        msk = (pred >= lo) & (pred < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / len(pred)) * abs(sev[msk].mean() - pred[msk].mean())
    return dict(roc_auc=float(roc), pr_auc=float(pr), csi_tau09=float(csi),
                brier=float(brier), ece=float(ece))


def boot_week(pred, target, week, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    weeks = np.unique(week)
    picks = weeks[rng.integers(0, len(weeks), size=(n_boot, len(weeks)))]
    ks = list(metrics(pred[:1], target[:1]))
    boot = {k: [] for k in ks}
    for b in range(n_boot):
        m = np.isin(week, picks[b])
        if m.sum() < 10:
            continue
        r = metrics(pred[m], target[m])
        for k in ks:
            boot[k].append(r[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in boot.items()}


def load_dataset(dom):
    ty = (2016, 2019) if dom == "SCS" else (2017, 2020)
    return SCSPIECFDataset(data_dir=str(DATA_DIR), dataset_file=DOMAIN_NC[dom],
                           mode="test", random_split=False, lead_weeks=4,
                           year_based_split=True, train_years=(2003, 2010),
                           val_years=(2011, 2015) if dom == "SCS" else (2014, 2016),
                           test_years=ty)


def xgb_week_ids(dom):
    """Return (week, target, n_per_domain) arrays aligning with xgb predictions
    flat ordering (window-major, then H*W pixel-major of the selected mask)."""
    ds = load_dataset(dom)
    T = ds.history_weeks
    y_full = ds.y_regression_full
    ocean = ~ds.land_mask
    weeks, targets = [], []
    for w, s in enumerate(ds.window_indices):
        t = s + T - 1
        li = t + 4
        if li >= ds.n_dates:
            continue
        yt = y_full[li]
        valid = np.isfinite(yt) & ocean
        ii, jj = np.where(valid)
        weeks.append(np.full(len(ii), t, np.int32))
        targets.append(yt[valid])
    return (np.concatenate(weeks), np.concatenate(targets))


def main():
    # ---------------- B. clip-ungated control ----------------
    zr = np.load(LTTS_ROOT / "results" / "physics_anchored" / "structural_ablation" /
                 "rec_structural_ablation.npz")
    b_rows = []
    for dom in ["SCS", "GBR", "CARIB"]:
        b_pred = zr[f"{dom}_B_ungated_pred"]
        b_clip = np.clip(b_pred, 0.0, 1.0).astype(np.float32)
        tgt = zr[f"{dom}_target"]
        t_idx = zr[f"{dom}_A_PersA_t"]   # same week ids across models (shared record set)
        week = t_idx.astype(int)
        a_pred = zr[f"{dom}_A_PersA_pred"]
        c_pred = zr[f"{dom}_C_anchored_pred"]
        mb = metrics(b_clip, tgt)
        ma = metrics(a_pred, tgt)
        mc = metrics(c_pred, tgt)
        # paired-difference CIs: B_clip - A  and  B_clip - C
        rng = np.random.default_rng(BOOT_SEED)
        weeks = np.unique(week)
        picks = weeks[rng.integers(0, len(weeks), size=(N_BOOT, len(weeks)))]
        ks = list(mb.keys())
        def pairci(x, y):
            out = {}
            for k in ks:
                dv = []
                for b in range(N_BOOT):
                    m = np.isin(week, picks[b])
                    if m.sum() < 10:
                        continue
                    dv.append(metrics(x[m], tgt[m])[k] - metrics(y[m], tgt[m])[k])
                dv = np.array([v for v in dv if np.isfinite(v)])
                out[k+"_ci"] = ([] if len(dv) == 0 else
                                [float(np.percentile(dv, 2.5)),
                                 float(np.percentile(dv, 97.5))])
            return out
        b_rows.append(dict(domain=dom, variant="B_clip", n=len(tgt)) | mb |
                      {f"Bclip_A_{k[:3]}": (ma[k] - mb[k]) for k in ks} |
                      {f"Bclip_C_{k[:3]}": (mc[k] - mb[k]) for k in ks})
        b_rows.append(dict(domain=dom, variant="B_ungated", **metrics(b_pred, tgt)))
        b_rows.append(dict(domain=dom, variant="A_PersA", **ma))
        b_rows.append(dict(domain=dom, variant="C_anchored", **mc))
        print(f"[B] {dom}: B_clip ROC={mb['roc_auc']:.3f} PR={mb['pr_auc']:.3f} "
              f"CSI={mb['csi_tau09']:.3f} Brier={mb['brier']:.4f} ECE={mb['ece']:.3f}", flush=True)
    pd.DataFrame(b_rows).to_csv(
        LTTS_ROOT / "results" / "physics_anchored" / "structural_ablation" /
        "clip_ungated.csv", index=False)
    print("[B] clip_ungated.csv written", flush=True)

    # ---------------- C. XGBoost week-block bootstrap CIs ----------------
    up = LTTS_ROOT / "results" / "xgb_transfer_control" / "predictions.npz"
    zx = np.load(up)
    c_rows = []
    for dom in ["SCS", "GBR", "CARIB"]:
        week, targets = xgb_week_ids(dom)
        print(f"[C] {dom}: {len(week)} records, weeks={len(np.unique(week))}", flush=True)
        for var in XGB_VARS:
            pred = zx[f"{var}_{dom}_pred"]
            tgt = zx[f"{var}_{dom}_target"]
            assert len(pred) == len(week), f"{var} {dom}: count mismatch"
            m = metrics(pred, tgt)
            ci = boot_week(pred, tgt, week)
            c_rows.append(dict(domain=dom, variant=var, n=len(pred)) |
                          {k: v for k, v in m.items()} |
                          {f"{k}_lo": ci[k][0] for k in m} |
                          {f"{k}_hi": ci[k][1] for k in m})
            print(f"[C] {dom} {var}: ROC={m['roc_auc']:.3f} "
                  f"[{ci['roc_auc'][0]:.3f},{ci['roc_auc'][1]:.3f}] "
                  f"PR={m['pr_auc']:.3f} [{ci['pr_auc'][0]:.3f},{ci['pr_auc'][1]:.3f}]", flush=True)
    pd.DataFrame(c_rows).to_csv(
        LTTS_ROOT / "results" / "xgb_transfer_control" / "bootstrap_cis.csv", index=False)
    print("[C] xgb bootstrap_cis.csv written", flush=True)
    print("PART B + C DONE", flush=True)


if __name__ == "__main__":
    main()