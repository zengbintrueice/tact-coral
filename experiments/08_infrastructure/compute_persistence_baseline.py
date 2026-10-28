# compute_persistence_baseline.py — Strong physics-based persistence baseline (v2)
#
#   Purpose: Determine how much four-week-ahead thermal-risk predictability is
#   obtained simply by carrying the current thermal state forward, evaluated on
#   the EXACT same protocol as the TSCF evaluation (same causal v2 datasets,
#   dates, pixels, valid masks, target definition, severe threshold, SCS test
#   period 2016-2019, GBR formal zero-shot test period 2017-2020).
#
#   Baseline A (DHW persistence):
#       yhat_persistence = clip(DHW_t / 10, 0, 1)
#     DHW_t is the current (issue-time) Jensen-aware DHW, which by the target
#     definition equals y_regression[t]. NaN where DHW is not valid.
#
#   Baseline B (12-week cumulative HotSpot persistence):
#       yhat_cumsum12 = clip(hotspot_cumsum_12w_t / 10, 0, 1)
#     hotspot_cumsum_12w[t] is the model's own derived channel: trailing
#     12-week sum of weekly hotspot (NaN weeks contribute 0; result NaN where
#     hotspot[t] is NaN) — replicated verbatim from
#     src/data/aligned_dataset.py::_compute_derived_channels.
#
#   TSCF: the deployed single-head model (ablate_a_dh_stopgrad.
#   FinalSingleHeadModel, weights results/model_final_Btail3/model.pt),
#   sliding-window inference, PFV verification at t+4.
#
#   Metrics (all methods, all on the same footing):
#     ROC AUC, PR AUC, CSI@tau=0.8, CSI@tau=0.9, POD, FAR, F1, Brier,
#     ECE (10-bin, diagnostic only).
#   Thresholds tau are PINNED (0.8 / 0.9) — never selected on the test set.
#
#   Evaluation sets per domain:
#     own    : each method evaluated where its own prediction is finite
#     common : all three methods evaluated on the shared valid mask
#              (finite TSCF & Baseline A & Baseline B & finite target)
#
#   Uncertainty: week-level (temporal block) bootstrap, 1000 replicates,
#   fixed seed — whole weeks are resampled, preserving within-week spatial
#   correlation and event clustering. Raw evaluation arrays (with t/i/j
#   indices) are saved for later spatial/event block bootstrap variants.
#
#   Terminology: the persistence baselines are STRONG PHYSICS-BASED
#   PERSISTENCE BASELINES (never "naive").
#
#   Outputs -> results/persistence_v2/:
#     persistence_metrics.csv
#     persistence_raw_predictions.npz
#     persistence_summary.json
#     persistence_report.md
#
#   Run:
#     cd LTTS_CRW/experiments/08_infrastructure
#     & 'D:\...\piecf_env\Scripts\python.exe' compute_persistence_baseline.py

import os, sys, io, json, time, warnings, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from pfv_utils import get_year_range_for_mode

LEAD_WEEKS = 4
SEVERE_THRESHOLD = 0.8
TAUS = [0.8, 0.9]
HISTORY_WEEKS = 24
N_BOOT = 1000
BOOT_SEED = 42
ECE_BINS = 10
OUT_DIR = RESULTS_DIR / "step2_persistence_v2"

MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}

SCS_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                 val_years=(2011, 2015), test_years=(2016, 2019))
GBR_SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
                 val_years=(2014, 2016), test_years=(2017, 2020))

DOMAINS = [
    {"name": "SCS", "dataset_file": "scs_causal_v2.nc", "split": SCS_SPLIT,
     "test_years": (2016, 2019),
     "protocol": "SCS year-based test (2016-2019)"},
    {"name": "GBR", "dataset_file": "gbr_causal_v2.nc", "split": GBR_SPLIT,
     "test_years": (2017, 2020),
     "protocol": "GBR formal zero-shot test (2017-2020)"},
]

METHODS = ["TSCF", "PersistenceA_DHW", "PersistenceB_cumsum12"]

OUT_DIR.mkdir(parents=True, exist_ok=True)


def restore_tscf():
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A6_WEIGHT, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_sliding_temporal(model, ds, device):
    """Sliding-window inference -> forecast_map [n_dates, H, W] (verbatim
    from gbr_zeroshot_evaluation.py)."""
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
    """Trailing 12-week rolling sum of weekly hotspot, verbatim semantics of
    aligned_dataset._compute_derived_channels: NaN weeks contribute 0 to the
    sum; result is NaN where hotspot[t] is NaN. hotspot: [D, H, W]."""
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
    hits = int(np.sum(pred_bin & actual_bin))
    misses = int(np.sum(~pred_bin & actual_bin))
    false_alarms = int(np.sum(pred_bin & ~actual_bin))
    return hits, misses, false_alarms


def metrics_from_counts(hits, misses, false_alarms):
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    f1 = 2 * hits / (2 * hits + false_alarms + misses) if (2 * hits + false_alarms + misses) > 0 else np.nan
    return csi, pod, far, f1


def ece_score(pred, target, severe_thresh=SEVERE_THRESHOLD, n_bins=ECE_BINS):
    """10-bin ECE of the prediction treated as P(severe). Diagnostic only."""
    if len(pred) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    severe = (target >= severe_thresh).astype(np.float64)
    ece = 0.0
    n = len(pred)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = float(pred[m].mean())
        acc = float(severe[m].mean())
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def auc_scores(pred, target, severe_thresh=SEVERE_THRESHOLD):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = (target >= severe_thresh).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        roc = float(roc_auc_score(y, pred))
        pr = float(average_precision_score(y, pred))
    except ValueError:
        return np.nan, np.nan
    return roc, pr


def brier_score(pred, target):
    if len(pred) == 0:
        return np.nan
    return float(np.mean((pred - target) ** 2))


def full_metrics(pred, target, bootstrap=True):
    """All requested metrics for one (pred, target) record set."""
    out = {
        "n_pixels": int(len(pred)),
        "n_severe": int(np.sum(target >= SEVERE_THRESHOLD)),
    }
    out["severe_prevalence"] = out["n_severe"] / len(pred) if len(pred) else np.nan
    out["roc_auc"], out["pr_auc"] = auc_scores(pred, target)
    for tau in TAUS:
        h, mi, fa = classification_counts(pred, target, SEVERE_THRESHOLD, tau)
        csi, pod, far, f1 = metrics_from_counts(h, mi, fa)
        key = f"tau{int(round(tau * 10)):02d}"
        out[f"csi_{key}"] = csi
        out[f"pod_{key}"] = pod
        out[f"far_{key}"] = far
        out[f"f1_{key}"] = f1
        out[f"hits_{key}"] = h
        out[f"misses_{key}"] = mi
        out[f"false_alarms_{key}"] = fa
    out["brier"] = brier_score(pred, target)
    out["ece"] = ece_score(pred, target)
    if bootstrap and len(pred) > 0:
        out["bootstrap"] = week_block_bootstrap(pred, target)
    return out


def make_week_picks(t_idx, n_boot=N_BOOT, seed=BOOT_SEED):
    """Shared resample matrix so all methods see the SAME resampled weeks per
    replicate (explicit pairing, not an emergent property of identical seeds)."""
    rng = np.random.default_rng(seed)
    n_w = len(np.unique(t_idx))
    return rng.integers(0, n_w, (n_boot, n_w))


def boot_metric_ci(pred, target, t_idx, picks=None, n_boot=N_BOOT, seed=BOOT_SEED):
    """Week-level block bootstrap CIs for ROC AUC, PR AUC, CSI@0.9, POD,
    FAR, F1@0.9, Brier."""
    rng = np.random.default_rng(seed)
    weeks = np.unique(t_idx)
    by_week = {w: (pred[t_idx == w], target[t_idx == w]) for w in weeks}
    n_w = len(weeks)
    rows = {"roc_auc": [], "pr_auc": [], "csi_tau09": [], "pod_tau09": [],
            "far_tau09": [], "f1_tau09": [], "brier": []}
    for rep in range(n_boot):
        pick = picks[rep] if picks is not None else rng.integers(0, n_w, n_w)
        ps, ts = [], []
        for w in pick:
            p, t = by_week[weeks[w]]
            ps.append(p)
            ts.append(t)
        p = np.concatenate(ps)
        t = np.concatenate(ts)
        roc, pr = auc_scores(p, t)
        h, mi, fa = classification_counts(p, t, SEVERE_THRESHOLD, 0.9)
        csi, pod, far, f1 = metrics_from_counts(h, mi, fa)
        rows["roc_auc"].append(roc)
        rows["pr_auc"].append(pr)
        rows["csi_tau09"].append(csi)
        rows["pod_tau09"].append(pod)
        rows["far_tau09"].append(far)
        rows["f1_tau09"].append(f1)
        rows["brier"].append(brier_score(p, t))
    ci = {}
    for k, v in rows.items():
        v = np.array([x for x in v if not (isinstance(x, float) and np.isnan(x))], dtype=float)
        if len(v) == 0:
            ci[k] = {"lo": np.nan, "hi": np.nan}
        else:
            ci[k] = {"lo": float(np.percentile(v, 2.5)),
                     "hi": float(np.percentile(v, 97.5))}
    return ci


def collect_records(domain, maps, y_full, ds):
    """Build PFV record sets for all methods.

    Returns:
      own:   {method: {"pred","target","t","i","j"}}
      common: single record set on the shared valid mask
      common_members: which methods are finite on the common mask (all)
    """
    T = HISTORY_WEEKS
    n_dates = ds.n_dates
    year_range = domain["test_years"]
    date_strings = ds.date_strings
    H, W = y_full.shape[1], y_full.shape[2]
    ocean = ~ds.land_mask                       # ocean cells only (align label_mask)

    own = {m: {"pred": [], "target": [], "t": [], "i": [], "j": []} for m in METHODS}
    com = {"pred": {m: [] for m in METHODS}, "target": [], "t": [], "i": [], "j": []}

    for t in range(T - 1, n_dates):
        if year_range is not None:
            year = int(date_strings[t][:4])
            if not (year_range[0] <= year <= year_range[1]):
                continue
        t_tgt = t + LEAD_WEEKS
        if t_tgt >= n_dates:
            continue
        target_t = y_full[t_tgt]
        finite_target = ~np.isnan(target_t) & ocean
        if finite_target.sum() == 0:
            continue
        finite = {m: ~np.isnan(maps[m][t]) for m in METHODS}
        # own sets
        for m in METHODS:
            v = finite_target & finite[m]
            if v.sum() == 0:
                continue
            ii, jj = np.where(v)
            own[m]["pred"].append(maps[m][t][v])
            own[m]["target"].append(target_t[v])
            own[m]["t"].append(np.full(len(ii), t, dtype=np.int32))
            own[m]["i"].append(ii.astype(np.int16))
            own[m]["j"].append(jj.astype(np.int16))
        # common set
        vc = finite_target.copy()
        for m in METHODS:
            vc &= finite[m]
        if vc.sum() > 0:
            ii, jj = np.where(vc)
            for m in METHODS:
                com["pred"][m].append(maps[m][t][vc])
            com["target"].append(target_t[vc])
            com["t"].append(np.full(len(ii), t, dtype=np.int32))
            com["i"].append(ii.astype(np.int16))
            com["j"].append(jj.astype(np.int16))

    def finalize(rec):
        return {k: np.concatenate(v) if v else np.array([], dtype=np.float32)
                for k, v in rec.items()}

    own = {m: finalize(r) for m, r in own.items()}
    com_final = {
        "target": np.concatenate(com["target"]) if com["target"] else np.array([], dtype=np.float32),
        "t": np.concatenate(com["t"]) if com["t"] else np.array([], dtype=np.int32),
        "i": np.concatenate(com["i"]) if com["i"] else np.array([], dtype=np.int16),
        "j": np.concatenate(com["j"]) if com["j"] else np.array([], dtype=np.int16),
        "pred": {m: np.concatenate(v) if v else np.array([], dtype=np.float32)
                 for m, v in com["pred"].items()},
    }
    return own, com_final


# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print(f"Persistence baseline v2 (strong physics-based persistence) — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Dataset dir: {DATA_DIR}")
print(f"Severe threshold: {SEVERE_THRESHOLD}; pinned taus: {TAUS}")

model = None

summary = {
    "experiment": "persistence_baseline_v2",
    "description": ("Strong physics-based persistence baselines vs TSCF on the "
                    "causal v2 datasets, identical PFV protocol "
                    "(t+4 verification, prediction-year filtering)."),
    "terminology_note": ("Baselines A and B are strong physics-based persistence "
                         "baselines, NOT naive baselines."),
    "configuration": {
        "dataset_files": {d["name"]: d["dataset_file"] for d in DOMAINS},
        "lead_weeks": LEAD_WEEKS,
        "history_weeks": HISTORY_WEEKS,
        "severe_threshold": SEVERE_THRESHOLD,
        "pinned_taus": TAUS,
        "threshold_selection": "none (pinned a priori; never selected on test)",
        "weights": str(A6_WEIGHT),
        "bootstrap": {"type": "week-level temporal block", "n_boot": N_BOOT,
                      "seed": BOOT_SEED},
        "ece_note": "ECE is a diagnostic only, not a calibrated probability.",
    },
    "domains": {},
    "timestamp": datetime.now().isoformat(),
}

csv_rows = []
npz_payload = {}

for domain in DOMAINS:
    name = domain["name"]
    print("\n" + "=" * 64)
    print(f"Domain {name}: {domain['protocol']}")
    print("=" * 64)

    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=LEAD_WEEKS, dataset_file=domain["dataset_file"],
                         **domain["split"])
    T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    y_full = ds.y_regression_full
    print(f"  T={T} H={H} W={W} n_dates={n_dates} ocean={int((~np.isnan(y_full[0])).sum())}")

    # ---- Baseline A: persistence of current DHW (= y_regression at issue time)
    map_A = y_full.astype(np.float32).copy()

    # ---- Baseline B: trailing 12-week cumulative HotSpot
    hs_idx = ds.dyn_feature_names.index('hotspot')
    hotspot_full = ds.X_dynamic_full[..., hs_idx].astype(np.float32)
    map_B = np.clip(compute_cumsum12_map(hotspot_full) / 10.0, 0.0, 1.0).astype(np.float32)

    # ---- TSCF: sliding-window inference
    if model is None:
        print("  Restoring TSCF model...")
        model = restore_tscf().to(device)
    print("  Running TSCF sliding-window inference...")
    t0 = time.time()
    map_T = predict_sliding_temporal(model, ds, device)
    print(f"  Inference done in {time.time()-t0:.1f}s")

    maps = {"TSCF": map_T, "PersistenceA_DHW": map_A, "PersistenceB_cumsum12": map_B}

    own, com = collect_records(domain, maps, y_full, ds)

    dom_results = {
        "protocol": domain["protocol"],
        "grid": f"{H}x{W}",
        "n_dates": int(n_dates),
        "ocean_pixels": int((~np.isnan(y_full[0])).sum()),
        "methods": {},
    }

    # own-set metrics
    # paired bootstrap: one shared pick matrix for all methods (same weeks
    # resampled per replicate); requires identical week counts across methods
    own_picks = make_week_picks(own[METHODS[0]]["t"])
    for m in METHODS:
        r = own[m]
        assert len(np.unique(r["t"])) == own_picks.shape[1], (
            f"own-set week count mismatch for {m}; paired bootstrap undefined")
        print(f"\n  [{m}] own valid set:")
        res = full_metrics(r["pred"], r["target"], bootstrap=False)
        res["bootstrap"] = boot_metric_ci(r["pred"], r["target"], r["t"],
                                          picks=own_picks)
        dom_results["methods"][m] = {"eval_set": "own", **res}
        print(f"    n={res['n_pixels']} severe={res['n_severe']} "
              f"prev={res['severe_prevalence']:.4f} "
              f"ROC={res['roc_auc']:.4f} PR={res['pr_auc']:.4f} "
              f"CSI@0.9={res['csi_tau09']:.4f} Brier={res['brier']:.5f}")
        csv_rows.append({"domain": name, "method": m, "eval_set": "own", **{
            k: v for k, v in res.items() if k != "bootstrap"}})
        npz_payload[f"{name}_{m}_own_pred"] = r["pred"]
        npz_payload[f"{name}_{m}_own_target"] = r["target"]
        npz_payload[f"{name}_{m}_own_t"] = r["t"]
        npz_payload[f"{name}_{m}_own_i"] = r["i"]
        npz_payload[f"{name}_{m}_own_j"] = r["j"]

    # common-set metrics
    com_picks = make_week_picks(com["t"])
    for m in METHODS:
        print(f"\n  [{m}] common valid set:")
        res = full_metrics(com["pred"][m], com["target"], bootstrap=False)
        res["bootstrap"] = boot_metric_ci(com["pred"][m], com["target"], com["t"],
                                          picks=com_picks)
        dom_results["methods"][f"{m}__common"] = {"eval_set": "common", **res}
        print(f"    n={res['n_pixels']} severe={res['n_severe']} "
              f"prev={res['severe_prevalence']:.4f} "
              f"ROC={res['roc_auc']:.4f} PR={res['pr_auc']:.4f} "
              f"CSI@0.9={res['csi_tau09']:.4f} Brier={res['brier']:.5f}")
        csv_rows.append({"domain": name, "method": m, "eval_set": "common", **{
            k: v for k, v in res.items() if k != "bootstrap"}})
        npz_payload[f"{name}_{m}_common_pred"] = com["pred"][m]
        if m == "TSCF":
            npz_payload[f"{name}_common_target"] = com["target"]
            npz_payload[f"{name}_common_t"] = com["t"]
            npz_payload[f"{name}_common_i"] = com["i"]
            npz_payload[f"{name}_common_j"] = com["j"]

    # deltas TSCF − baseline on the common set (like-for-like)
    dom_results["comparison_common"] = {}
    for m in ["PersistenceA_DHW", "PersistenceB_cumsum12"]:
        d = {}
        for k in ["roc_auc", "pr_auc", "csi_tau09", "pod_tau09", "far_tau09",
                  "f1_tau09", "brier", "ece"]:
            d[k] = (dom_results["methods"]["TSCF__common"][k]
                    - dom_results["methods"][f"{m}__common"][k])
        dom_results["comparison_common"][f"TSCF_minus_{m}"] = d

    summary["domains"][name] = dom_results

print("\nSaving outputs...")

# CSV
csv_path = OUT_DIR / "persistence_metrics.csv"
df = pd.DataFrame(csv_rows)
cols_first = ["domain", "method", "eval_set", "n_pixels", "n_severe",
              "severe_prevalence", "roc_auc", "pr_auc",
              "csi_tau08", "csi_tau09", "pod_tau09", "far_tau09", "f1_tau09",
              "brier", "ece",
              "pod_tau08", "far_tau08", "f1_tau08",
              "hits_tau09", "misses_tau09", "false_alarms_tau09"]
cols = [c for c in cols_first if c in df.columns] + \
       [c for c in df.columns if c not in cols_first]
df = df[cols]
df.to_csv(csv_path, index=False)
print(f"  {csv_path}")

# NPZ
npz_path = OUT_DIR / "persistence_raw_predictions.npz"
np.savez_compressed(npz_path, **npz_payload)
print(f"  {npz_path}")

# JSON
json_path = OUT_DIR / "persistence_summary.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
print(f"  {json_path}")


# ── Markdown report with publication-ready comparison table ──────────────
def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def fmt_ci(res, key, nd=4):
    b = res.get("bootstrap", {}) or {}
    ci = b.get(key)
    if not ci or (isinstance(ci["lo"], float) and np.isnan(ci["lo"])):
        return fmt(res.get(key))
    return f"{res[key]:.{nd}f} [{ci['lo']:.{nd}f}, {ci['hi']:.{nd}f}]"


def method_row(mname, res):
    return (f"| {mname} | {res['n_pixels']} | {res['n_severe']} "
            f"| {fmt(res['severe_prevalence'])} "
            f"| {fmt_ci(res, 'roc_auc')} | {fmt_ci(res, 'pr_auc')} "
            f"| {fmt_ci(res, 'csi_tau09')} "
            f"| {fmt(res['pod_tau09'])} | {fmt(res['far_tau09'])} "
            f"| {fmt(res['f1_tau09'])} | {fmt(res['brier'], 5)} "
            f"| {fmt(res['ece'])} |")


lines = []
lines.append("# Persistence Baseline v2 — Strong Physics-Based Persistence vs TSCF")
lines.append("")
lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ")
lines.append(f"Data: causal v2 datasets (`scs_causal_v2.nc`, `gbr_causal_v2.nc`)  ")
lines.append(f"Protocol: Pure Forecasting Verification — prediction issued at week t "
             f"using information available at t only, verified against the target at "
             f"t+{LEAD_WEEKS} weeks; prediction-year filtering; severe threshold "
             f"y ≥ {SEVERE_THRESHOLD}; detection thresholds **pinned** at τ ∈ "
             f"{{0.8, 0.9}} (never selected on the test set).")
lines.append("")
lines.append("## Baseline definitions")
lines.append("")
lines.append("**These are strong physics-based persistence baselines — not naive baselines.** "
             "Carrying current thermal stress forward is the operational industry standard "
             "(NOAA CRW DHW outlooks are persistence-anchored), so beating them is the "
             "relevant bar for added value.")
lines.append("")
lines.append("- **Baseline A — DHW persistence**: `yhat_persistence = clip(DHW_t / 10, 0, 1)` "
             "(current Jensen-aware DHW carried forward; identical target definition).")
lines.append("- **Baseline B — 12-week cumulative HotSpot**: "
             "`yhat_cumsum12 = clip(hotspot_cumsum_12w_t / 10, 0, 1)` "
             "(the model's own trailing 12-week thermal-accumulation channel, carried forward).")
lines.append("- **TSCF**: deployed single-head model, sliding-window inference "
             "(`model_final_Btail3/model.pt`).")
lines.append("")
lines.append("## Evaluation sets")
lines.append("")
lines.append("- **own**: each method evaluated where its own prediction is finite "
             "(persistence is undefined where the current DHW/HotSpot is missing).")
lines.append("- **common**: all methods evaluated on the shared valid mask "
             "(finite TSCF & Baseline A & Baseline B & finite target) — the like-for-like comparison.")
lines.append("")
lines.append("## Publication-ready comparison table (common set, τ = 0.9)")
lines.append("")
lines.append("ROC AUC / PR AUC / CSI shown as value [95% week-block bootstrap CI]. "
             "Brier = mean (ŷ − y)²; ECE = 10-bin expected calibration error (diagnostic only).")
lines.append("")
for domain in DOMAINS:
    nm = domain["name"]
    dr = summary["domains"][nm]
    lines.append(f"### {nm} — {domain['protocol']} (grid {dr['grid']})")
    lines.append("")
    lines.append("| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD | FAR | F1 | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        lines.append(method_row(m, dr["methods"][f"{m}__common"]))
    lines.append("")

    cmp_ = dr["comparison_common"]
    lines.append(f"Δ (TSCF − baseline), {nm} common set: ")
    for key, d in cmp_.items():
        lines.append(f"- {key}: ROC {fmt(d['roc_auc'])}, PR {fmt(d['pr_auc'])}, "
                     f"CSI@0.9 {fmt(d['csi_tau09'])}, Brier {fmt(d['brier'], 5)}")
    lines.append("")

    lines.append(f"### {nm} — own valid sets (per-method availability)")
    lines.append("")
    lines.append("| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD | FAR | F1 | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        lines.append(method_row(m, dr["methods"][m]))
    lines.append("")

lines.append("## Notes")
lines.append("")
lines.append("- The persistence baselines use only issue-time thermal state; they are "
             "physically grounded: 4-week DHW persistence is strongly autocorrelated "
             "during events, which is why these baselines are strong.")
lines.append("- TSCF numbers here are computed on the **causal v2** datasets; they may "
             "differ slightly from legacy JSONs computed on the pre-causal files "
             "(the sst_anomaly channel differs).")
lines.append("- Week-level (temporal block) bootstrap, 1000 replicates, seed 42: whole "
             "weeks are resampled, preserving within-week spatial correlation and event "
             "clustering. Raw per-pixel arrays with (t, i, j) indices are saved in "
             "`persistence_raw_predictions.npz` for later spatial/event block variants.")
lines.append("- ECE is reported as a diagnostic only; persistence scores are not "
             "calibrated probabilities.")
lines.append("- Thresholds τ = 0.8 / 0.9 are pinned a priori; no threshold was selected "
             "on the test set.")

md_path = OUT_DIR / "persistence_report.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"  {md_path}")

print("\nDone.")
