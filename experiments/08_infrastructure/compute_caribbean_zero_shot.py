#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_caribbean_zero_shot.py — TSCF zero-shot evaluation on the Caribbean
============================================================================

Independent-domain (Caribbean / Wider Tropical Atlantic) zero-shot test of the
frozen SCS-trained TSCF model, mirroring the GBR protocol (Steps 2/4).

  - Frozen state: SCS-trained checkpoint (results/model_final_Btail3/model.pt),
    SCS-train-only normalization (baked in checkpoint), causal v2 Caribbean
    inputs, severe y>=0.8, tau=0.90 pinned, lead 4wk, history 24wk.
  - Pure Forecasting Verification: sliding-window inference, verify y_hat(t+4)
    vs y(t+4), prediction-year filter on the ISSUE step (2017-2020 test).
  - Baselines (like-for-like): PersistenceA_DHW (y_regression at issue time),
    PersistenceB_cumsum12 (trailing 12-wk HotSpot sum /10).
  - Paired week-block bootstrap (1000, seed 42) for TSCF vs baseline deltas.

Terminology: thermal-stress-aware, NOT "physics-informed"; persistence baselines
are "strong physics-based persistence baselines", never "naive".

Outputs -> results/caribbean_zero_shot/:
  caribbean_zero_shot_results.json / .csv / .md
  caribbean_zero_shot_predictions.npz
  caribbean_zero_shot_protocol.json

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_caribbean_zero_shot.py
"""
import io, sys, os, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

# NOTE: do NOT wrap sys.stdout here — importing ablate_a_dh_stopgrad re-wraps it
# (its own TextIOWrapper at module level); a double wrap raises
# "ValueError: I/O operation on closed file" at exit. All output below is ASCII.
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
TAUS = [0.8, 0.9]
TAU_PRIMARY = 0.90
HISTORY_WEEKS = 24
N_BOOT = 1000
BOOT_SEED = 42
ECE_BINS = 10
OUT_DIR = RESULTS_DIR / "caribbean_zero_shot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}
DATASET_FILE = "caribbean_causal_v2.nc"
CARIB_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                   val_years=(2014, 2016), test_years=(2017, 2020))
TEST_YEARS = (2017, 2020)
METHODS = ["TSCF", "PersistenceA_DHW", "PersistenceB_cumsum12"]


def restore_tscf():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A6_WEIGHT, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_sliding_temporal(model, ds, device):
    """PFV sliding-window inference -> forecast_map [n_dates, H, W]."""
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X_full = ds.X_dynamic_full
    bs = 16
    n_valid = n_dates - T + 1
    t_start_valid = T - 1
    with torch.no_grad():
        for start_batch in range(0, n_valid, bs):
            batch_end = min(start_batch + bs, n_valid)
            bs_actual = batch_end - start_batch
            xd_batch = np.zeros((bs_actual, T, H, W, C_dyn), dtype=np.float32)
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                start_idx = t_end - T + 1
                xd_batch[k] = X_full[start_idx:t_end + 1]
            xd = torch.FloatTensor(xd_batch).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (bs_actual, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, scale=MODEL_CFG["scale"],
                        tau=MODEL_CFG["tau"], temp=MODEL_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
    return forecast_map


def compute_cumsum12_map(hotspot):
    hs = np.ascontiguousarray(hotspot, dtype=np.float64)
    D = hs.shape[0]
    hs_clean = np.nan_to_num(hs, nan=0.0)
    nan_mask = np.isnan(hs)
    cumsum = np.cumsum(hs_clean, axis=0)
    cs = np.zeros_like(hs)
    cs[:12] = cumsum[:12]
    if D > 12:
        cs[12:] = cumsum[12:] - cumsum[:-12]
    cs[nan_mask] = np.nan
    return cs.astype(np.float32)


def classification_counts(pred, target, severe_thresh, pred_thresh):
    pred_bin = pred >= pred_thresh
    actual_bin = target >= severe_thresh
    return (int(np.sum(pred_bin & actual_bin)),
            int(np.sum(~pred_bin & actual_bin)),
            int(np.sum(pred_bin & ~actual_bin)))


def metrics_from_counts(hits, misses, false_alarms):
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    f1 = 2 * hits / (2 * hits + false_alarms + misses) if (2 * hits + false_alarms + misses) > 0 else np.nan
    return csi, pod, far, f1


def auc_scores(pred, target):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = (target >= SEVERE_THRESHOLD).astype(int)
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
    severe = (target >= SEVERE_THRESHOLD).astype(np.float64)
    ece, n = 0.0, len(pred)
    for b in range(ECE_BINS):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(severe[m].mean()) - float(pred[m].mean()))
    return float(ece)


def full_metrics(pred, target):
    out = {"n": int(len(pred)),
           "n_severe": int(np.sum(target >= SEVERE_THRESHOLD))}
    out["severe_prevalence"] = out["n_severe"] / len(pred) if len(pred) else np.nan
    out["roc_auc"], out["pr_auc"] = auc_scores(pred, target)
    out["brier"] = float(np.mean((pred - target) ** 2))
    out["ece"] = ece_score(pred, target)
    for tau in TAUS:
        h, m_, fa = classification_counts(pred, target, SEVERE_THRESHOLD, tau)
        csi, pod, far, f1 = metrics_from_counts(h, m_, fa)
        tkey = f"tau{int(round(tau * 10)):02d}"
        out[f"csi_{tkey}"] = csi
        out[f"pod_{tkey}"] = pod
        out[f"far_{tkey}"] = far
        out[f"f1_{tkey}"] = f1
    return out


def make_week_picks(t_idx, n_boot=N_BOOT, seed=BOOT_SEED):
    """Shared resample matrix of week indices (block bootstrap at week level)."""
    rng = np.random.default_rng(seed)
    weeks = np.unique(t_idx)
    n_weeks = len(weeks)
    picks = rng.integers(0, n_weeks, size=(n_boot, n_weeks))
    return weeks[picks]


def boot_metric_ci(pred, target, t_idx, picks=None, n_boot=N_BOOT, seed=BOOT_SEED):
    """Week-block bootstrap 95% CI for every scalar metric in full_metrics."""
    if picks is None:
        rng = np.random.default_rng(seed)
        weeks = np.unique(t_idx)
        picks = weeks[rng.integers(0, len(weeks), size=(n_boot, len(weeks)))]
    boot = {k: [] for k in full_metrics(pred[:1], target[:1])}
    for b in range(n_boot):
        mask = np.isin(t_idx, picks[b])
        if mask.sum() < 10:
            continue
        res = full_metrics(pred[mask], target[mask])
        for k, v in res.items():
            boot[k].append(v)
    out = {}
    for k, vals in boot.items():
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) == 0:
            out[k] = [np.nan, np.nan]
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[k] = [float(lo), float(hi)]
    return out


def collect_records(maps, y_full, ds):
    """PFV record sets with prediction-year filter (issue step year in TEST_YEARS)
    and OCEAN MASK (v = finite y(t+4) & ~land_mask, aligned with label_mask)."""
    T = HISTORY_WEEKS
    n_dates = ds.n_dates
    date_strings = ds.date_strings
    H, W = y_full.shape[1], y_full.shape[2]
    ocean = ~ds.land_mask                       # [H,W]
    rec = {m: {"pred": [], "target": [], "t": [], "i": [], "j": []} for m in METHODS}
    for t in range(T - 1, n_dates):
        year = int(date_strings[t][:4])
        if not (TEST_YEARS[0] <= year <= TEST_YEARS[1]):
            continue
        t_tgt = t + LEAD_WEEKS
        if t_tgt >= n_dates:
            continue
        target_t = y_full[t_tgt]
        finite_target = ~np.isnan(target_t) & ocean
        if finite_target.sum() == 0:
            continue
        vc = finite_target.copy()
        for m in METHODS:
            vc &= ~np.isnan(maps[m][t])
        if vc.sum() == 0:
            continue
        ii, jj = np.where(vc)
        for m in METHODS:
            rec[m]["pred"].append(maps[m][t][vc])
            rec[m]["target"].append(target_t[vc])
            rec[m]["t"].append(np.full(len(ii), t, dtype=np.int32))
            rec[m]["i"].append(ii.astype(np.int16))
            rec[m]["j"].append(jj.astype(np.int16))
    return {m: {k: np.concatenate(v) if v else np.array([], dtype=np.float32)
                for k, v in r.items()} for m, r in rec.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {DATASET_FILE} (test years {TEST_YEARS[0]}-{TEST_YEARS[1]})...")
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=LEAD_WEEKS, dataset_file=DATASET_FILE,
                         **CARIB_SPLIT)
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    y_full = ds.y_regression_full
    print(f"  T={T} H={H} W={W} n_dates={n_dates}")

    print("  Restoring frozen TSCF model...")
    model = restore_tscf().to(device)

    map_A = y_full.astype(np.float32).copy()
    hs_idx = ds.dyn_feature_names.index('hotspot')
    hotspot_full = ds.X_dynamic_full[..., hs_idx].astype(np.float32)
    map_B = np.clip(compute_cumsum12_map(hotspot_full) / 10.0, 0.0, 1.0).astype(np.float32)

    print("  Running TSCF sliding-window inference...")
    t0 = time.time()
    map_T = predict_sliding_temporal(model, ds, device)
    print(f"  Inference done in {time.time() - t0:.1f}s")

    maps = {"TSCF": map_T, "PersistenceA_DHW": map_A, "PersistenceB_cumsum12": map_B}
    rec = collect_records(maps, y_full, ds)

    rows, npz = [], {}
    results = {"protocol": "Caribbean zero-shot test 2017-2020 (mirror GBR)",
               "grid": f"{H}x{W}", "n_dates": int(n_dates),
               "checkpoint_md5": None, "methods": {}}
    import hashlib
    results["checkpoint_md5"] = hashlib.md5(open(A6_WEIGHT, 'rb').read()).hexdigest()

    picks = make_week_picks(rec[METHODS[0]]["t"])
    for m in METHODS:
        r = rec[m]
        assert len(np.unique(r["t"])) == picks.shape[1], f"week count mismatch for {m}"
        res = full_metrics(r["pred"], r["target"])
        res["bootstrap"] = boot_metric_ci(r["pred"], r["target"], r["t"], picks=picks)
        results["methods"][m] = res
        print(f"\n  [{m}] n={res['n']} severe={res['n_severe']} prev={res['severe_prevalence']:.4f}")
        print(f"      ROC={res['roc_auc']:.4f} PR={res['pr_auc']:.4f} "
              f"CSI@0.9={res['csi_tau09']:.4f} Brier={res['brier']:.5f} ECE={res['ece']:.4f}")
        rows.append({"method": m, **{k: v for k, v in res.items() if k != "bootstrap"}})
        npz[f"{m}_pred"] = r["pred"]
        npz[f"{m}_target"] = r["target"]
        npz[f"{m}_t"] = r["t"]
        npz[f"{m}_i"] = r["i"]
        npz[f"{m}_j"] = r["j"]

    # TSCF minus baselines on the common record set
    results["comparison"] = {}
    for b in ["PersistenceA_DHW", "PersistenceB_cumsum12"]:
        d = {}
        for k in ["roc_auc", "pr_auc", "csi_tau09", "pod_tau09", "far_tau09", "f1_tau09", "brier", "ece"]:
            d[k] = results["methods"]["TSCF"][k] - results["methods"][b][k]
        db = {}
        for k in ["roc_auc", "pr_auc", "csi_tau09", "brier", "ece"]:
            db[k] = [results["methods"]["TSCF"]["bootstrap"][k],
                     results["methods"][b]["bootstrap"][k]]
        results["comparison"][f"TSCF_minus_{b}"] = {"point": d, "bootstrap_both": db}
        print(f"\n  TSCF − {b}: ΔROC={d['roc_auc']:+.4f} ΔPR={d['pr_auc']:+.4f} "
              f"ΔCSI@0.9={d['csi_tau09']:+.4f}")

    results["generated_utc"] = datetime.now().isoformat()
    results["tau"] = {"primary": TAU_PRIMARY, "pinned": "a priori; never selected on test"}
    results["bootstrap"] = {"type": "paired week-block", "n": N_BOOT, "seed": BOOT_SEED}

    pd.DataFrame(rows).to_csv(OUT_DIR / "caribbean_zero_shot_results.csv", index=False)
    np.savez_compressed(OUT_DIR / "caribbean_zero_shot_predictions.npz", **npz)
    with open(OUT_DIR / "caribbean_zero_shot_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    protocol = {
        "title": "Caribbean zero-shot protocol (frozen TSCF)",
        "model": "results/model_final_Btail3/model.pt (SCS-trained, frozen)",
        "checkpoint_md5": results["checkpoint_md5"],
        "normalization": "SCS-train only, baked in checkpoint (FROZEN)",
        "dataset": DATASET_FILE,
        "pfv": "sliding window X_full[t-T+1..t], verify y_hat(t+4) vs y(t+4)",
        "test_years": list(TEST_YEARS),
        "tau_primary": TAU_PRIMARY,
        "severe": f"y>= {SEVERE_THRESHOLD}",
        "baselines": {"PersistenceA_DHW": "y_regression at issue time",
                      "PersistenceB_cumsum12": "trailing 12-wk HotSpot sum /10"},
        "bootstrap": {"type": "paired week-block", "n": N_BOOT, "seed": BOOT_SEED},
        "generated_utc": results["generated_utc"],
    }
    with open(OUT_DIR / "caribbean_zero_shot_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    md = ["# Caribbean Zero-Shot Evaluation — Frozen TSCF", "",
          f"Generated: {results['generated_utc']}", "",
          f"Grid {H}x{W}, n_dates={n_dates}, test years {TEST_YEARS[0]}-{TEST_YEARS[1]}, "
          f"PFV t+4 verification.", "",
          "| Method | n | severe | prev | ROC | PR | CSI@0.9 | POD@0.9 | FAR@0.9 | Brier | ECE |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r = results["methods"][m]
        md.append(f"| {m} | {r['n']} | {r['n_severe']} | {r['severe_prevalence']:.4f} | "
                  f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['csi_tau09']:.4f} | "
                  f"{r['pod_tau09']:.4f} | {r['far_tau09']:.4f} | {r['brier']:.5f} | {r['ece']:.4f} |")
    md += ["", "TSCF minus baselines (common record set):", ""]
    for k, v in results["comparison"].items():
        md.append(f"- **{k}**: ΔROC={v['point']['roc_auc']:+.4f}, "
                  f"ΔPR={v['point']['pr_auc']:+.4f}, ΔCSI@0.9={v['point']['csi_tau09']:+.4f}")
    md += ["", "Terminology: thermal-stress-aware model; strong physics-based persistence baselines "
           "(never 'naive'). Cumulative HotSpot channels are legitimate predictors (NOAA CRW DHW).",
           "Normalization is SCS-train-only and frozen; nothing recomputed on Caribbean.", ""]
    (OUT_DIR / "caribbean_zero_shot.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nOutputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
