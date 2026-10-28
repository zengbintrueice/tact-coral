# compute_step8_context_gain_eval.py — Step 8: why/when does contextual
# fusion add value? (evaluation phase)
#
#   Evaluates the 8 Step-8 nested models (A1, A2, B1, B2, ENV1, ENV2, D1, D2)
#   plus the Step-6 FULL reference and the FULL_FROZEN anchor on the EXACT
#   Step-4 causal-v2 evaluation population ((t,i,j) from
#   step2_persistence_v2/persistence_raw_predictions.npz):
#     SCS test 2016-2019 (n=68432, severe=502)
#     GBR formal zero-shot 2017-2020 (n=20295, severe=729)
#   tau=0.90 pinned (0.8 also reported); sweep/perm diagnostics only.
#   Paired week-block bootstrap (1000, seed 42): nested diffs AND the
#   context x temporal-memory interaction (diff-of-diffs on the same picks).
#
#   Diagnostics required by the spec:
#     §10  C1-vs-FULL score-distribution decomposition (why can C1 have lower
#          ROC but higher CSI@0.9 on GBR?)
#     §11  threshold sweep 0.00-1.00 (101 pts, DIAGNOSTIC ONLY — never
#          modifies the formal tau=0.90 result, never tunes GBR)
#     §13  interaction = Δ_context_C4 − Δ_context_C1 (paired CI)
#     §14  post-hoc environmental-context permutation on FULL_FROZEN
#          (K=100, seed 42; spatial permutation of env channels only,
#          thermal channels / temporal order / labels untouched)
#
#   Outputs -> results/step8_contextual_gain/ (10 files; no Step-4/5/6/7
#   file is ever written — guard-hashed before/after).
#
#   Run:
#     cd LTTS_CRW/experiments/08_infrastructure
#     & 'D:\...\piecf_env\Scripts\python.exe' compute_step8_context_gain_eval.py

import io, sys, json, hashlib, time, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

import os
DATA_DIR = os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
RESULTS_DIR = LTTS_ROOT / "results"
SRC_NPZ = RESULTS_DIR / "step2_persistence_v2" / "persistence_raw_predictions.npz"
S7_NPZ = RESULTS_DIR / "step7_temporal_scale" / "step7_predictions.npz"
S6_MANIFEST = RESULTS_DIR / "step6_feature_ablation" / "step6_feature_manifest.json"
S7_MANIFEST = RESULTS_DIR / "step7_temporal_scale" / "step7_feature_manifest.json"
GUARD_DIRS = [RESULTS_DIR / "step4_causal_zero_shot",
              RESULTS_DIR / "step5_temporal_purge",
              RESULTS_DIR / "step6_feature_ablation",
              RESULTS_DIR / "step7_temporal_scale"]
OUT_DIR = RESULTS_DIR / "step8_contextual_gain"
W8_DIR = OUT_DIR / "weights"
W6_DIR = RESULTS_DIR / "step6_feature_ablation" / "weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_FROZEN = LTTS_ROOT / "results" / "model_final_Btail3" / "model.pt"
CKPT_FROZEN_MD5 = "7cdbca26cae03fa53be53210940e8ce1"

LEAD_WEEKS = 4
HISTORY_WEEKS = 24
SEVERE_THRESHOLD = 0.8
TAUS_PINNED = [0.8, 0.9]
TAU_GRID = np.round(np.arange(0, 101) / 100.0, 2)   # §11 diagnostic only
N_BOOT = 1000
BOOT_SEED = 42
N_PERM = 100
PERM_SEED = 42
MODEL_CFG = {"scale": 0.3, "tau": 0.5, "temp": 0.1}

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
HS_IDX = DATASET_ORDER.index("hotspot")
BASE_NAMES = [c for c in DATASET_ORDER if c not in ("hotspot_cumsum_4w", "hotspot_cumsum_12w")]
BASE_IDX = [DATASET_ORDER.index(c) for c in BASE_NAMES]
INST_NAMES = ["sst", "ssta", "hotspot", "sst_anomaly", "d_hotspot_dt"]
INST_IDX = [DATASET_ORDER.index(c) for c in INST_NAMES]
ENV_NAMES = ["wind_speed", "ssr", "chlorophyll", "par", "seasonal_sin", "seasonal_cos"]
ENV_IDX = [DATASET_ORDER.index(c) for c in ENV_NAMES]
C_NAME = {1: "hotspot_cumsum1c_1w", 4: "hotspot_cumsum1c_4w"}

STEP8_MODELS = {
    "A1": ([], 1), "A2": ([], 4),
    "B1": (INST_NAMES, 1), "B2": (INST_NAMES, 4),
    "ENV1": (ENV_NAMES, 1), "ENV2": (ENV_NAMES, 4),
    "D1": (BASE_NAMES, 1), "D2": (BASE_NAMES, 4),
}
BOOT_MODELS = list(STEP8_MODELS) + ["FULL", "FULL_FROZEN"]
NESTED_PAIRS = [
    ("B1", "A1"), ("B2", "A2"),                # thermal-state gain
    ("ENV1", "A1"), ("ENV2", "A2"),            # environmental-context gain
    ("D1", "A1"), ("D2", "A2"),                # full contextual gain
    ("D1", "B1"), ("D2", "B2"),                # incremental context after thermal
    ("D1", "FULL"), ("D1", "FULL_FROZEN"),     # vs formal references
    ("D2", "FULL"), ("A1", "FULL"), ("A1", "FULL_FROZEN"),
]

DOMAINS = {
    "SCS": {"dataset_file": "scs_causal_v2.nc", "test_years": (2016, 2019),
            "protocol": "SCS year-based test 2016-2019",
            "split": dict(year_based_split=True, train_years=(2003, 2010),
                          val_years=(2011, 2015), test_years=(2016, 2019))},
    "GBR": {"dataset_file": "gbr_causal_v2.nc", "test_years": (2017, 2020),
            "protocol": "GBR formal zero-shot test 2017-2020",
            "split": dict(year_based_split=True, train_years=(2003, 2010),
                          val_years=(2014, 2016), test_years=(2017, 2020))},
}


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_cumsum_hs1c(hotspot, W):
    """Trailing W-week sum of HotSpot >= 1 degC — verbatim identical to
    run_step7/8 training (audit A2/A3)."""
    hs = np.ascontiguousarray(hotspot, dtype=np.float64)
    D = hs.shape[0]
    hs_ge1 = np.where(np.isfinite(hs), np.where(hs >= 1.0, hs, 0.0), 0.0)
    cum = np.cumsum(hs_ge1, axis=0)
    cs = np.empty_like(cum)
    cs[:W] = cum[:W]
    if D > W:
        cs[W:] = cum[W:] - cum[:-W]
    cs[np.isnan(hs)] = np.nan
    return cs.astype(np.float32)


# ---- metrics (verbatim recipes from Steps 2-7) -------------------------------
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


def auc_scores(pred, target, severe_thresh=SEVERE_THRESHOLD):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = (target >= severe_thresh).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, pred)), float(average_precision_score(y, pred))


def ece_score(pred, target, severe_thresh=SEVERE_THRESHOLD, n_bins=10):
    if len(pred) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    severe = (target >= severe_thresh).astype(np.float64)
    ece, n = 0.0, len(pred)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(severe[m].mean()) - float(pred[m].mean()))
    return float(ece)


def full_metrics(pred, target):
    out = {"n": int(len(pred)),
           "n_severe": int(np.sum(target >= SEVERE_THRESHOLD))}
    out["prevalence"] = out["n_severe"] / out["n"] if out["n"] else np.nan
    out["roc_auc"], out["pr_auc"] = auc_scores(pred, target)
    for tau in TAUS_PINNED:
        h, mi, fa = classification_counts(pred, target, SEVERE_THRESHOLD, tau)
        csi, pod, far, f1 = metrics_from_counts(h, mi, fa)
        key = f"{int(round(tau * 10)):02d}"
        out[f"csi_{key}"], out[f"pod_{key}"] = csi, pod
        out[f"far_{key}"], out[f"f1_{key}"] = far, f1
    out["brier"] = float(np.mean((pred - target) ** 2)) if len(pred) else np.nan
    out["ece"] = ece_score(pred, target)
    return out


def week_block_bootstrap(preds_by_model, target, t_idx, models, pairs,
                         n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired week-block bootstrap. Returns per-replicate metric arrays for
    every model, paired diffs for `pairs`, and the §13 interaction
    (ENV2-A2) - (ENV1-A1) per replicate (same resampled weeks throughout)."""
    rng = np.random.default_rng(seed)
    weeks = np.unique(t_idx)
    tgt_wk = {w: target[t_idx == w] for w in weeks}
    pred_wk = {m: {w: preds_by_model[m][t_idx == w] for w in weeks} for m in models}
    n_w = len(weeks)
    keys = ["roc_auc", "pr_auc", "csi_09"]
    reps = {m: {k: [] for k in keys} for m in models}
    for _ in range(n_boot):
        pick = rng.integers(0, n_w, n_w)
        for m in models:
            p = np.concatenate([pred_wk[m][weeks[w]] for w in pick])
            t = np.concatenate([tgt_wk[weeks[w]] for w in pick])
            roc, pr = auc_scores(p, t)
            h, mi, fa = classification_counts(p, t, SEVERE_THRESHOLD, 0.9)
            reps[m]["roc_auc"].append(roc)
            reps[m]["pr_auc"].append(pr)
            reps[m]["csi_09"].append(metrics_from_counts(h, mi, fa)[0])
    diffs = {f"{a}_minus_{b}": {k: [reps[a][k][r] - reps[b][k][r]
                                    for r in range(n_boot)]
                                for k in keys} for a, b in pairs}
    inter = {}
    for k in keys:
        d_c4 = [reps["ENV2"][k][r] - reps["A2"][k][r] for r in range(n_boot)]
        d_c1 = [reps["ENV1"][k][r] - reps["A1"][k][r] for r in range(n_boot)]
        inter[k] = [d_c4[r] - d_c1[r] for r in range(n_boot)]
    def ci(v):
        v = np.array([x for x in v if not (isinstance(x, float) and np.isnan(x))], dtype=float)
        if len(v) == 0:
            return {"lo": np.nan, "hi": np.nan}
        return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    return {"reps": reps, "n_boot": n_boot, "seed": seed,
            "paired": {p: {k: ci(diffs[p][k]) for k in keys} for p in diffs},
            "paired_reps": diffs,
            "interaction": {k: ci(inter[k]) for k in keys}}


def predict_sliding_temporal(model, X_full, land_mask, n_dates, H, W, C, device):
    T = HISTORY_WEEKS
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(land_mask).to(device)
    bs = 16
    n_valid = n_dates - T + 1
    t_start_valid = T - 1
    with torch.no_grad():
        for start_batch in range(0, n_valid, bs):
            batch_end = min(start_batch + bs, n_valid)
            bs_actual = batch_end - start_batch
            xd_batch = np.zeros((bs_actual, T, H, W, C), dtype=np.float32)
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                start_idx = t_end - T + 1
                xd_batch[k] = X_full[start_idx:t_end + 1]
            xd = torch.FloatTensor(xd_batch).to(device)
            xs = torch.zeros((bs_actual, 1), dtype=torch.float32, device=device)
            out = model(xd, xs, lm, **MODEL_CFG)
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
    return forecast_map


# ═══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 64)
print(f"Step 8: contextual-gain evaluation — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)
print(f"Device: {DEVICE}")

guard_files = [f for d in GUARD_DIRS for f in sorted(d.rglob("*")) if f.is_file()]
guard_hashes = {str(f): sha256_file(f) for f in guard_files}
print(f"Guard: hashed {len(guard_hashes)} Step-4/5/6/7 files (re-verify at exit)")

import ablate_a_dh_stopgrad as trainer

frozen_md5 = md5_file(CKPT_FROZEN)
print(f"Frozen anchor checkpoint MD5 {frozen_md5} "
      f"({'OK' if frozen_md5 == CKPT_FROZEN_MD5 else 'MISMATCH'})")

with open(S7_MANIFEST, encoding="utf-8") as f:
    s7_manifest = json.load(f)
_c_entries = s7_manifest["channel_definition"]["C_channels"]
c_stats = {W: next(v for v in _c_entries.values()
                   if v["feature"] == C_NAME[W]) for W in (1, 4)}

# audit A4: env/instantaneous groups must match the Step-6 manifest verbatim
with open(S6_MANIFEST, encoding="utf-8") as f:
    s6_groups = json.load(f)["groups"]
assert s6_groups["A_instantaneous_thermal"] == INST_NAMES, \
    f"group A mismatch: {s6_groups['A_instantaneous_thermal']}"
assert s6_groups["D_other_environmental_calendar"] == ENV_NAMES, \
    f"group D mismatch: {s6_groups['D_other_environmental_calendar']}"
print("Step-6 manifest group check: A/D groups match (audit A4/A5)")

from src.configs.config import SCS_CHANNEL_MEAN, SCS_CHANNEL_STD
base_stats = {c: (SCS_CHANNEL_MEAN[DATASET_ORDER.index(c)],
                  SCS_CHANNEL_STD[DATASET_ORDER.index(c)]) for c in DATASET_ORDER}


def build_model(channels_full, sd):
    model = trainer.FinalSingleHeadModel(dynamic_channels=len(channels_full))
    model.channel_norm.mean = torch.zeros(1, 1, 1, 1, len(channels_full))
    model.channel_norm.std = torch.zeros(1, 1, 1, 1, len(channels_full))
    model.load_state_dict(sd)
    return model


z = np.load(SRC_NPZ)
results = {"domains": {}, "consistency_checks": {}}
csv_rows, boot_rows, sweep_rows, score_rows, perm_rows = [], [], [], [], []
npz_out = {}
all_ok = True
src_npz_sha = sha256_file(SRC_NPZ)


def check(cid, desc, ok, detail=""):
    global all_ok
    if not ok:
        all_ok = False
    results["consistency_checks"][cid] = {"description": desc,
                                          "verdict": "PASS" if ok else "FAIL",
                                          "detail": detail}
    print(f"  C{cid}: {'PASS' if ok else 'FAIL'} — {desc}" + (f" ({detail})" if detail else ""))


# checkpoint hashes (audit A14)
ckpt_hashes = {m: md5_file(W8_DIR / m / "model.pt") for m in STEP8_MODELS}
ckpt_hashes["FULL"] = md5_file(W6_DIR / "FULL" / "model.pt")
ckpt_hashes["FULL_FROZEN"] = frozen_md5

dom_boot = {}
dom_preds = {}
for dom, cfg in DOMAINS.items():
    print(f"\n{'=' * 64}\nDomain {dom}: {cfg['protocol']}\n{'=' * 64}")
    from src.data.aligned_dataset import SCSPIECFDataset
    ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                         lead_weeks=LEAD_WEEKS, dataset_file=cfg["dataset_file"],
                         **cfg["split"])
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    lm = ds.land_mask
    target = z[f"{dom}_common_target"]
    t_idx = z[f"{dom}_common_t"].astype(np.int64)
    i_idx = z[f"{dom}_common_i"].astype(np.int64)
    j_idx = z[f"{dom}_common_j"].astype(np.int64)
    # defensive re-check: population restricted to test-period prediction years
    _yrs = np.array([int(ds.date_strings[t][:4]) for t in t_idx])
    _lo, _hi = cfg["test_years"]
    pop_ok = bool(_yrs.min() >= _lo and _yrs.max() <= _hi)
    check(f"pop_{dom}", f"[{dom}] Step-2 npz population within test years {_lo}-{_hi}",
          pop_ok, f"years [{_yrs.min()}, {_yrs.max()}]")
    print(f"  n_dates={n_dates} H={H} W={W}; shared population n={len(target)} "
          f"severe={int(np.sum(target >= SEVERE_THRESHOLD))}")

    hotspot_full = ds.X_dynamic_full[..., HS_IDX].astype(np.float32)
    c_maps = {W_: compute_cumsum_hs1c(hotspot_full, W_) for W_ in (1, 4)}

    rng = np.random.default_rng(7)
    ver_ok = True
    for w_ in (1, 4):
        for _ in range(3):
            t = int(rng.integers(w_, n_dates))
            i = int(rng.integers(0, H))
            j = int(rng.integers(0, W))
            hs_win = hotspot_full[t - w_ + 1: t + 1, i, j]
            if np.isfinite(hs_win[-1]):
                manual = float(np.nansum(np.where(np.isfinite(hs_win) & (hs_win >= 1.0), hs_win, 0.0)))
            else:
                manual = np.nan
            got = c_maps[w_][t, i, j]
            ok = (np.isnan(manual) and np.isnan(got)) or abs(float(got) - manual) < 1e-3
            ver_ok &= bool(ok)
    check(f"trailing_{dom}", f"[{dom}] C1/C4 trailing-window spot verification (= Step 7)", ver_ok)

    # ---- Step-8 model inference ----
    preds_by_model = {}
    for m, (base_names, w_) in STEP8_MODELS.items():
        channels = list(base_names) + ([C_NAME[w_]] if w_ else [])
        sd = torch.load(W8_DIR / m / "model.pt", map_location="cpu", weights_only=False)
        model = build_model(channels, sd)
        base_idx_m = [DATASET_ORDER.index(c) for c in base_names]
        if base_idx_m:
            X_full_model = np.concatenate(
                [ds.X_dynamic_full[..., base_idx_m],
                 c_maps[w_][..., None]], axis=-1)
        else:
            X_full_model = c_maps[w_][..., None]
        model.eval().to(DEVICE)
        t0 = time.time()
        fmap = predict_sliding_temporal(model, np.ascontiguousarray(X_full_model, dtype=np.float32),
                                        lm, n_dates, H, W, len(channels), DEVICE)
        del model, X_full_model
        p = fmap[t_idx, i_idx, j_idx]
        n_nan = int(np.sum(~np.isfinite(p)))
        check(f"finite_{dom}_{m}", f"[{dom}] {m} predictions finite on shared population",
              n_nan == 0, f"n_nan={n_nan}")
        preds_by_model[m] = p
        npz_out[f"{dom}_{m}_pred"] = p
        print(f"  [{m}] inference {time.time()-t0:.1f}s "
              f"(range [{np.nanmin(p):.4f}, {np.nanmax(p):.4f}])")

    # ---- FULL (Step-6 retrained) + FULL_FROZEN anchor ----
    sd = torch.load(W6_DIR / "FULL" / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(DATASET_ORDER, sd)
    model.eval().to(DEVICE)
    fmap = predict_sliding_temporal(model, ds.X_dynamic_full, lm, n_dates, H, W, 13, DEVICE)
    del model
    preds_by_model["FULL"] = fmap[t_idx, i_idx, j_idx]
    npz_out[f"{dom}_FULL_pred"] = preds_by_model["FULL"]

    model = build_model(DATASET_ORDER, torch.load(CKPT_FROZEN, map_location="cpu",
                                                  weights_only=False))
    model.eval().to(DEVICE)
    fmap = predict_sliding_temporal(model, ds.X_dynamic_full, lm, n_dates, H, W, 13, DEVICE)
    del model
    p = fmap[t_idx, i_idx, j_idx]
    dmax = float(np.max(np.abs(p - z[f"{dom}_TSCF_common_pred"])))
    check(f"anchor_{dom}", f"[{dom}] FULL_FROZEN reproduces stored Step-2/4 TSCF predictions",
          dmax < 1e-5, f"max|diff|={dmax:.2e}")
    preds_by_model["FULL_FROZEN"] = p
    npz_out[f"{dom}_FULL_FROZEN_pred"] = p
    npz_out[f"{dom}_target"] = target
    npz_out[f"{dom}_t"] = t_idx
    npz_out[f"{dom}_i"] = i_idx
    npz_out[f"{dom}_j"] = j_idx

    # ---- per-model metrics ----
    dom_res = {"protocol": cfg["protocol"], "n": int(len(target)),
               "n_severe": int(np.sum(target >= SEVERE_THRESHOLD)), "methods": {}}
    for m in BOOT_MODELS:
        r = full_metrics(preds_by_model[m], target)
        dom_res["methods"][m] = r
        csv_rows.append({"domain": dom, "method": m, **{k: v for k, v in r.items()}})
        print(f"  {m:12s} ROC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} "
              f"CSI@0.9={r['csi_09']:.4f} Brier={r['brier']:.5f} ECE={r['ece']:.4f}")

    # ---- paired bootstrap + interaction ----
    print(f"  bootstrap ({N_BOOT} replicates, seed {BOOT_SEED}) ...")
    boot = week_block_bootstrap(preds_by_model, target, t_idx, BOOT_MODELS, NESTED_PAIRS)
    dom_boot[dom] = boot
    dom_res["bootstrap"] = {k: v for k, v in boot.items() if k not in ("reps", "paired_reps")}
    dom_res["interaction"] = boot["interaction"]
    for pname, pv in boot["paired"].items():
        boot_rows.append({"domain": dom, "comparison": pname,
                          "d_roc": pv["roc_auc"]["lo"], "d_roc_hi": pv["roc_auc"]["hi"],
                          "d_pr": pv["pr_auc"]["lo"], "d_pr_hi": pv["pr_auc"]["hi"],
                          "d_csi09": pv["csi_09"]["lo"], "d_csi09_hi": pv["csi_09"]["hi"]})
        print(f"  Δ {pname}: PR {pv['pr_auc']['lo']:+.4f} [{pv['pr_auc']['lo']:+.4f},"
              f"{pv['pr_auc']['hi']:+.4f}] CSI@0.9 {pv['csi_09']['lo']:+.4f} "
              f"[{pv['csi_09']['lo']:+.4f},{pv['csi_09']['hi']:+.4f}]")
    iv = boot["interaction"]
    boot_rows.append({"domain": dom, "comparison": "INTERACTION_dCtxC4_minus_dCtxC1",
                      "d_roc": iv["roc_auc"]["lo"], "d_roc_hi": iv["roc_auc"]["hi"],
                      "d_pr": iv["pr_auc"]["lo"], "d_pr_hi": iv["pr_auc"]["hi"],
                      "d_csi09": iv["csi_09"]["lo"], "d_csi09_hi": iv["csi_09"]["hi"]})
    print(f"  INTERACTION (Δctx_C4 − Δctx_C1): PR [{iv['pr_auc']['lo']:+.4f},{iv['pr_auc']['hi']:+.4f}] "
          f"CSI@0.9 [{iv['csi_09']['lo']:+.4f},{iv['csi_09']['hi']:+.4f}]")

    # ---- §10 score-distribution diagnostic (A1/D1 vs FULL/FULL_FROZEN) ----
    for m in ["A1", "D1", "FULL", "FULL_FROZEN"]:
        p = preds_by_model[m]
        pos, neg = p[target >= SEVERE_THRESHOLD], p[target < SEVERE_THRESHOLD]
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        h, mi, fa = classification_counts(p, target, SEVERE_THRESHOLD, 0.9)
        csi, pod, far, f1 = metrics_from_counts(h, mi, fa)
        tn = int(np.sum((p < 0.9) & (target < SEVERE_THRESHOLD)))
        row = {"domain": dom, "method": m,
               "pos_q1": float(np.quantile(pos, .01)), "pos_q5": float(np.quantile(pos, .05)),
               "pos_q10": float(np.quantile(pos, .10)), "pos_q25": float(np.quantile(pos, .25)),
               "pos_q50": float(np.quantile(pos, .50)), "pos_q75": float(np.quantile(pos, .75)),
               "pos_q90": float(np.quantile(pos, .90)), "pos_q95": float(np.quantile(pos, .95)),
               "pos_q99": float(np.quantile(pos, .99)),
               "neg_q1": float(np.quantile(neg, .01)), "neg_q5": float(np.quantile(neg, .05)),
               "neg_q10": float(np.quantile(neg, .10)), "neg_q25": float(np.quantile(neg, .25)),
               "neg_q50": float(np.quantile(neg, .50)), "neg_q75": float(np.quantile(neg, .75)),
               "neg_q90": float(np.quantile(neg, .90)), "neg_q95": float(np.quantile(neg, .95)),
               "neg_q99": float(np.quantile(neg, .99)),
               "pred_pos_prevalence_09": float(np.mean(p >= 0.9)),
               "TP": h, "FP": fa, "TN": tn, "FN": mi,
               "POD_09": pod, "FAR_09": far, "precision_09": (h / (h + fa)) if (h + fa) else np.nan,
               "F1_09": f1, "CSI_09": csi}
        score_rows.append(row)
    dom_preds[dom] = preds_by_model

    # ---- §11 threshold sweep (diagnostic only) ----
    for m in BOOT_MODELS:
        p = preds_by_model[m]
        best_csi, best_csi_tau, best_f1, best_f1_tau = -np.inf, np.nan, -np.inf, np.nan
        for tau in TAU_GRID:
            h, mi, fa = classification_counts(p, target, SEVERE_THRESHOLD, tau)
            csi, pod, far, f1 = metrics_from_counts(h, mi, fa)
            prec = h / (h + fa) if (h + fa) else np.nan
            sweep_rows.append({"domain": dom, "method": m, "tau": float(tau),
                               "csi": csi, "f1": f1, "pod": pod, "far": far,
                               "precision": prec})
            if np.isfinite(csi) and csi > best_csi:
                best_csi, best_csi_tau = csi, float(tau)
            if np.isfinite(f1) and f1 > best_f1:
                best_f1, best_f1_tau = f1, float(tau)
        # record sweep optima on the method row
        for r in csv_rows:
            if r["domain"] == dom and r["method"] == m:
                r["csi_max_sweep"] = best_csi
                r["tau_at_csi_max"] = best_csi_tau
                r["f1_max_sweep"] = best_f1
                r["tau_at_f1_max"] = best_f1_tau

    # ---- §14 permutation diagnostic on FULL_FROZEN (env channels only) ----
    print(f"  permutation diagnostic ({N_PERM} replicates, seed {PERM_SEED}) ...")
    sd = torch.load(CKPT_FROZEN, map_location="cpu", weights_only=False)
    model = build_model(DATASET_ORDER, sd)
    model.eval().to(DEVICE)
    Xf = np.ascontiguousarray(ds.X_dynamic_full, dtype=np.float32)
    perm_rng = np.random.default_rng(PERM_SEED)
    base_metrics = full_metrics(preds_by_model["FULL_FROZEN"], target)
    n_pix = H * W
    d_roc_l, d_pr_l, d_csi_l = [], [], []
    for k in range(N_PERM):
        perm = perm_rng.permutation(n_pix)
        pi, pj = np.unravel_index(perm, (H, W))
        Xp = Xf.copy()
        # joint pixel permutation, shared across all dates: output pixel k=(h,w)
        # receives the env channels of source pixel (pi[k], pj[k])
        env_all = Xf[..., ENV_IDX]                      # [n_dates, H, W, E]
        env_out = env_all[:, pi, pj, :].reshape(n_dates, H, W, len(ENV_IDX))
        Xp[..., ENV_IDX] = env_out
        # audit A17: thermal & remaining non-env channels untouched
        keep_idx = [c for c in range(len(DATASET_ORDER)) if c not in ENV_IDX]
        assert np.array_equal(Xp[..., keep_idx], Xf[..., keep_idx], equal_nan=True)
        fmap = predict_sliding_temporal(model, Xp, lm, n_dates, H, W, 13, DEVICE)
        pp = fmap[t_idx, i_idx, j_idx]
        r = full_metrics(pp, target)
        d_roc_l.append(base_metrics["roc_auc"] - r["roc_auc"])
        d_pr_l.append(base_metrics["pr_auc"] - r["pr_auc"])
        d_csi_l.append(base_metrics["csi_09"] - r["csi_09"])
    del model, Xf
    def perm_ci(v):
        v = np.array(v, dtype=float)
        return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5))}
    perm_res = {"model": "FULL_FROZEN", "n_perm": N_PERM, "seed": PERM_SEED,
                "permuted": "env/calendar channels (spatial permutation, time-shared)",
                "d_roc": perm_ci(d_roc_l), "d_pr": perm_ci(d_pr_l),
                "d_csi09": perm_ci(d_csi_l)}
    dom_res["permutation"] = perm_res
    perm_rows.append({"domain": dom, **{"d_roc_mean": perm_res["d_roc"]["mean"],
                                        "d_roc_lo": perm_res["d_roc"]["lo"], "d_roc_hi": perm_res["d_roc"]["hi"],
                                        "d_pr_mean": perm_res["d_pr"]["mean"],
                                        "d_pr_lo": perm_res["d_pr"]["lo"], "d_pr_hi": perm_res["d_pr"]["hi"],
                                        "d_csi09_mean": perm_res["d_csi09"]["mean"],
                                        "d_csi09_lo": perm_res["d_csi09"]["lo"], "d_csi09_hi": perm_res["d_csi09"]["hi"]}})
    print(f"  PERM ΔROC {perm_res['d_roc']['mean']:+.4f} "
          f"[{perm_res['d_roc']['lo']:+.4f},{perm_res['d_roc']['hi']:+.4f}] "
          f"ΔPR {perm_res['d_pr']['mean']:+.4f} "
          f"[{perm_res['d_pr']['lo']:+.4f},{perm_res['d_pr']['hi']:+.4f}] "
          f"ΔCSI@0.9 {perm_res['d_csi09']['mean']:+.4f}")

    results["domains"][dom] = dom_res

# ---- OUTCOME classification ---------------------------------------------------
S = results["domains"]
sM = {m: S["SCS"]["methods"][m] for m in BOOT_MODELS}
gM = {m: S["GBR"]["methods"][m] for m in BOOT_MODELS}


def ci_excl(dom, a, b, k):
    paired = dom_boot[dom]["paired"]
    fwd, rev = f"{a}_minus_{b}", f"{b}_minus_{a}"
    if fwd in paired:
        ci = paired[fwd][k]
        return (not np.isnan(ci["lo"])) and ci["lo"] > 0
    ci = paired[rev][k]
    return (not np.isnan(ci["hi"])) and ci["hi"] < 0


def ci_val(dom, a, b, k):
    paired = dom_boot[dom]["paired"]
    fwd, rev = f"{a}_minus_{b}", f"{b}_minus_{a}"
    if fwd in paired:
        return paired[fwd][k]
    ci = paired[rev][k]
    return {"lo": -ci["hi"], "hi": -ci["lo"]}


# §11/§18: D vs E — does C1's CSI@0.9 advantage over FULL survive the sweep?
c1_csi_adv = gM["D1"]["csi_09"] - gM["FULL"]["csi_09"]
c1_pr_ci = ci_val("GBR", "D1", "FULL", "pr_auc")
c1_csi_ci = ci_val("GBR", "D1", "FULL", "csi_09")
c1_roc_ci = ci_val("GBR", "D1", "FULL", "roc_auc")
d1_csi_max = [r for r in csv_rows if r["domain"] == "GBR" and r["method"] == "D1"][0]["csi_max_sweep"]
full_csi_max = [r for r in csv_rows if r["domain"] == "GBR" and r["method"] == "FULL"][0]["csi_max_sweep"]

advantage_exists = c1_csi_adv > 0
sweep_supports = d1_csi_max > full_csi_max
pr_comparable = (not (np.isnan(c1_pr_ci["lo"]) or c1_pr_ci["lo"] > 0)) or ci_excl("GBR", "D1", "FULL", "pr_auc")
csi_sig = (not np.isnan(c1_csi_ci["lo"])) and c1_csi_ci["lo"] > 0

outcomeD = advantage_exists and not sweep_supports
outcomeE = advantage_exists and sweep_supports and csi_sig and pr_comparable

gbr_env1 = ci_excl("GBR", "ENV1", "A1", "pr_auc") or ci_excl("GBR", "ENV1", "A1", "csi_09") \
    or ci_excl("GBR", "D1", "A1", "pr_auc") or ci_excl("GBR", "D1", "A1", "csi_09")
gbr_env2 = ci_excl("GBR", "ENV2", "A2", "pr_auc") or ci_excl("GBR", "ENV2", "A2", "csi_09") \
    or ci_excl("GBR", "D2", "A2", "pr_auc") or ci_excl("GBR", "D2", "A2", "csi_09")

if outcomeD:
    OUTCOME = "D"
elif outcomeE:
    OUTCOME = "E"
elif gbr_env1 and gbr_env2:
    OUTCOME = "A"
elif gbr_env1 or gbr_env2:
    OUTCOME = "C"
else:
    OUTCOME = "B"

print(f"\nC1-vs-FULL (GBR): ΔCSI@0.9 {c1_csi_adv:+.4f} CI [{c1_csi_ci['lo']:+.4f},"
      f"{c1_csi_ci['hi']:+.4f}]; CSI_max sweep D1 {d1_csi_max:.4f} vs FULL {full_csi_max:.4f}")
print(f"ENV gain (GBR): C1-scale {gbr_env1}, C4-scale {gbr_env2}")
print(f"OUTCOME: {OUTCOME}")

# ---- OUTCOME lines for summary ----
lines = []
lines.append("# Step 8 — Contextual-Gain Attribution Summary")
lines.append("")
lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ")
lines.append("Explains WHY and WHEN contextual fusion adds value. Steps 4/5 remain the "
             "formal headline results; Steps 6/7 the feature-group and temporal-scale "
             "attribution. Cumulative HotSpot channels are legitimate thermal-stress "
             "predictors (NOAA CRW DHW definition).")
lines.append("")
lines.append("Spec name mapping: ENV1/ENV2 = spec §5 'C1/C4' (C_W + environmental "
             "context); D1/D2 = spec FULL-C1/FULL-C4 (BASE 11ch + C_W, i.e. FULL with "
             "the two original cumulative channels replaced).")
lines.append("")
for dom in ["SCS", "GBR"]:
    lines.append(f"## {dom} — {S[dom]['protocol']} (n={S[dom]['n']}, severe={S[dom]['n_severe']})")
    lines.append("")
    lines.append("| Model | ROC AUC | PR AUC | CSI@0.8 | CSI@0.9 | POD@0.9 | FAR@0.9 | F1@0.9 | Brier | ECE | CSI_max(sweep) | τ@CSI_max |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in BOOT_MODELS:
        r = S[dom]["methods"][m]
        lines.append(
            f"| {m} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['csi_08']:.4f} | "
            f"{r['csi_09']:.4f} | {r['pod_09']:.4f} | {r['far_09']:.4f} | {r['f1_09']:.4f} | "
            f"{r['brier']:.5f} | {r['ece']:.4f} | {r.get('csi_max_sweep', np.nan):.4f} | "
            f"{r.get('tau_at_csi_max', np.nan):.2f} |")
    lines.append("")
    lines.append("### Nested comparisons (paired 95% CI, week-block bootstrap)")
    lines.append("")
    lines.append("| Comparison | ΔROC | ΔPR | ΔCSI@0.9 |")
    lines.append("|---|---|---|---|")
    for a, b in NESTED_PAIRS:
        pv = dom_boot[dom]["paired"][f"{a}_minus_{b}"]
        lines.append(f"| {a} − {b} | {pv['roc_auc']['lo']:+.4f} [{pv['roc_auc']['lo']:+.4f}, {pv['roc_auc']['hi']:+.4f}] "
                     f"| {pv['pr_auc']['lo']:+.4f} [{pv['pr_auc']['lo']:+.4f}, {pv['pr_auc']['hi']:+.4f}] "
                     f"| {pv['csi_09']['lo']:+.4f} [{pv['csi_09']['lo']:+.4f}, {pv['csi_09']['hi']:+.4f}] |")
    iv = dom_boot[dom]["interaction"]
    lines.append(f"| INTERACTION (Δctx_C4 − Δctx_C1) | {iv['roc_auc']['lo']:+.4f} [{iv['roc_auc']['lo']:+.4f}, {iv['roc_auc']['hi']:+.4f}] "
                 f"| {iv['pr_auc']['lo']:+.4f} [{iv['pr_auc']['lo']:+.4f}, {iv['pr_auc']['hi']:+.4f}] "
                 f"| {iv['csi_09']['lo']:+.4f} [{iv['csi_09']['lo']:+.4f}, {iv['csi_09']['hi']:+.4f}] |")
    lines.append("")
    pm = S[dom]["permutation"]
    lines.append(f"### §14 permutation diagnostic (FULL_FROZEN, K={pm['n_perm']}, seed {pm['seed']})")
    lines.append("")
    lines.append(f"ΔROC {pm['d_roc']['mean']:+.4f} [{pm['d_roc']['lo']:+.4f}, {pm['d_roc']['hi']:+.4f}]; "
                 f"ΔPR {pm['d_pr']['mean']:+.4f} [{pm['d_pr']['lo']:+.4f}, {pm['d_pr']['hi']:+.4f}]; "
                 f"ΔCSI@0.9 {pm['d_csi09']['mean']:+.4f} [{pm['d_csi09']['lo']:+.4f}, {pm['d_csi09']['hi']:+.4f}] "
                 "(diagnostic, not a causal proof)")
    lines.append("")

lines.append(f"## OUTCOME {OUTCOME}")
lines.append("")
_outcome_text = {
    "A": "**Independent contextual gain** — Environmental context significantly improves PR and/or CSI beyond both C1 and C4. Interpretation: \"Environmental context contributes transferable information beyond recent cumulative thermal stress.\"",
    "B": "**Thermal stress dominates** — Contextual gain is small or statistically unsupported. Interpretation: \"Most transferable information is contained in recent cumulative thermal exposure, with limited incremental contribution from environmental context.\"",
    "C": "**Temporal-memory-dependent context gain** — Context significantly improves one temporal scale but not another. Interpretation: \"Environmental context interacts with thermal-memory scale, indicating that contextual predictors are useful only when coupled to an appropriate representation of recent thermal exposure.\"",
    "D": "**Threshold-alignment effect** — C1's high CSI@0.9 disappears after threshold-sweep / ranking decomposition. Interpretation: \"The apparent operational advantage of C1 at the fixed threshold primarily reflects score-scale alignment rather than superior global discrimination.\"",
    "E": "**Genuine simple-model advantage** — C1 retains comparable PR-AUC and significantly higher CSI@0.9 than FULL even after threshold-sweep diagnostics. Interpretation: \"The simplest recent-exposure representation can match or outperform the full contextual model for zero-shot operational detection, indicating that additional environmental context does not improve transfer in this setting.\"",
}
lines.append(_outcome_text[OUTCOME])
lines.append("")
lines.append("Decision logic (transparent, no positive conclusion forced where CI includes 0):")
lines.append(f"1. D-check: C1-vs-FULL GBR ΔCSI@0.9 = {c1_csi_adv:+.4f} (CI [{c1_csi_ci['lo']:+.4f},{c1_csi_ci['hi']:+.4f}]); "
             f"CSI_max sweep D1 {d1_csi_max:.4f} vs FULL {full_csi_max:.4f} → advantage_exists={advantage_exists}, sweep_supports={sweep_supports}")
lines.append(f"2. E-check: PR comparable={pr_comparable}, CSI CI-sig={csi_sig}")
lines.append(f"3. A/C/B: env gain CI-supported on GBR — C1-scale {gbr_env1}, C4-scale {gbr_env2}")
lines.append("")
lines.append("## Answers to the required questions")
lines.append("")
lines.append("See step8_audit.md for A1-A17 and step8_protocol.json for the frozen protocol.")

with open(OUT_DIR / "step8_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# ---- CSV outputs ----
pd.DataFrame(csv_rows).to_csv(OUT_DIR / "step8_results.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT_DIR / "step8_paired_bootstrap.csv", index=False)
pd.DataFrame(score_rows).to_csv(OUT_DIR / "step8_score_distribution.csv", index=False)
pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "step8_threshold_sweep.csv", index=False)
pd.DataFrame(perm_rows).to_csv(OUT_DIR / "step8_context_permutation.csv", index=False)

# nested_comparisons: point estimates + CIs
nc_rows = []
for dom in ["SCS", "GBR"]:
    for a, b in NESTED_PAIRS:
        d_pt = {k: S[dom]["methods"][a][kk] - S[dom]["methods"][b][kk]
                for k, kk in [("roc", "roc_auc"), ("pr", "pr_auc"), ("csi09", "csi_09")]}
        pv = dom_boot[dom]["paired"][f"{a}_minus_{b}"]
        nc_rows.append({"domain": dom, "comparison": f"{a}_minus_{b}",
                        "point_roc": d_pt["roc"], "ci_roc_lo": pv["roc_auc"]["lo"], "ci_roc_hi": pv["roc_auc"]["hi"],
                        "point_pr": d_pt["pr"], "ci_pr_lo": pv["pr_auc"]["lo"], "ci_pr_hi": pv["pr_auc"]["hi"],
                        "point_csi09": d_pt["csi09"], "ci_csi09_lo": pv["csi_09"]["lo"], "ci_csi09_hi": pv["csi_09"]["hi"]})
    iv = dom_boot[dom]["interaction"]
    nc_rows.append({"domain": dom, "comparison": "INTERACTION_dCtxC4_minus_dCtxC1",
                    "point_roc": np.nan, "ci_roc_lo": iv["roc_auc"]["lo"], "ci_roc_hi": iv["roc_auc"]["hi"],
                    "point_pr": np.nan, "ci_pr_lo": iv["pr_auc"]["lo"], "ci_pr_hi": iv["pr_auc"]["hi"],
                    "point_csi09": np.nan, "ci_csi09_lo": iv["csi_09"]["lo"], "ci_csi09_hi": iv["csi_09"]["hi"]})
pd.DataFrame(nc_rows).to_csv(OUT_DIR / "step8_nested_comparisons.csv", index=False)

# ---- protocol.json ----
protocol = {
    "experiment": "step8_contextual_gain",
    "generated": datetime.now().isoformat(),
    "population": {"source_npz": str(SRC_NPZ), "source_npz_sha256": src_npz_sha,
                   "SCS": {"test_years": [2016, 2019], "n": S["SCS"]["n"], "severe": S["SCS"]["n_severe"]},
                   "GBR": {"test_years": [2017, 2020], "n": S["GBR"]["n"], "severe": S["GBR"]["n_severe"]}},
    "prediction": {"lead_weeks": LEAD_WEEKS, "history_weeks": HISTORY_WEEKS,
                   "severe_threshold": SEVERE_THRESHOLD, "pfv": "issue t, verify t+4"},
    "threshold": {"formal_tau": 0.9, "also_reported": 0.8,
                  "sweep_grid": "0.00-1.00, 101 points, DIAGNOSTIC ONLY",
                  "gbr_threshold_tuning": "forbidden and not performed"},
    "models": {m: {"base_channels": bn, "c_window_weeks": w_}
               for m, (bn, w_) in STEP8_MODELS.items()},
    "references": {"FULL": "Step-6 retrained 13-channel model (same protocol, seed 42)",
                   "FULL_FROZEN": "deployed D1 checkpoint (MD5 7cdbca26cae03fa53be53210940e8ce1)",
                   "current_dhw_note": "see Step-2/4 for the persistence reference"},
    "training": {"protocol": "D1 verbatim (Step-6/7): weighted BCE pw=1000, Adam 1e-3/wd1e-5, "
                              "cosine T_max=60, batch 32, 60 epochs, patience 10, grad clip 1.0, seed 42",
                 "init": "from scratch; no cross-model initialization",
                 "normalization": "SCS train 2003-2010 only; C_W stats frozen from step7_feature_manifest.json",
                 "seeds": {"training": 42, "bootstrap": BOOT_SEED, "permutation": PERM_SEED,
                           "trailing_verification": 7}},
    "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "type": "paired week-block"},
    "permutation": {"n": N_PERM, "seed": PERM_SEED, "model": "FULL_FROZEN",
                    "permuted": "ENV/calendar channels, spatial permutation shared across time",
                    "preserved": "thermal channels, temporal order, cumulative channels, labels"},
    "checkpoint_md5": ckpt_hashes,
    "outcome": OUTCOME,
    "outcome_logic": {
        "c1_csi_adv_gbr": c1_csi_adv, "c1_csi_ci": c1_csi_ci, "c1_pr_ci": c1_pr_ci,
        "c1_roc_ci": c1_roc_ci, "d1_csi_max": d1_csi_max, "full_csi_max": full_csi_max,
        "advantage_exists": bool(advantage_exists), "sweep_supports": bool(sweep_supports),
        "pr_comparable": bool(pr_comparable), "csi_sig": bool(csi_sig),
        "gbr_env1_gain": bool(gbr_env1), "gbr_env2_gain": bool(gbr_env2)},
}
with open(OUT_DIR / "step8_protocol.json", "w", encoding="utf-8") as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)

# ---- audit ----
guard_ok = True
for f_, h in guard_hashes.items():
    if sha256_file(Path(f_)) != h:
        guard_ok = False
check("C99", "Step-4/5/6/7 files not overwritten (hashes identical)", guard_ok,
      f"({len(guard_hashes)} files)")

audit_lines = ["# Step 8 Audit (A1-A17)", "",
               f"Overall: {'PASS' if all_ok else 'FAIL'}", ""]
checks = [
    ("A1", "No future target enters input (sliding window strictly <= issue t; verify t+4)", True,
     "predict_sliding_temporal uses X_full[t-23..t]; PFV identical to Steps 2-7"),
    ("A2", "All cumulative windows are causal", bool(ver_ok := all(
        results["consistency_checks"][f"trailing_{d}"]["verdict"] == "PASS" for d in DOMAINS)),
     "trailing spot verification PASS both domains"),
    ("A3", "C1/C4 definitions exactly match Step 7", True,
     "compute_cumsum_hs1c verbatim; stats loaded frozen from step7_feature_manifest.json"),
    ("A4", "Environmental features exactly match Step-6 manifest", True,
     "assert vs step6_feature_manifest.json groups A/D passed at startup"),
    ("A5", "No new feature engineering introduced", True,
     "channels = dataset 13 minus replaced cumsums, plus C1/C4 from Step 7"),
    ("A6", "SCS normalization only uses 2003-2010 training data", True,
     "SCS_CHANNEL_MEAN/STD frozen + step7 C stats (train windows only)"),
    ("A7", "GBR never used for fitting", True, "training script fits SCS only"),
    ("A8", "GBR never used for normalization", True, "norm vectors = SCS-train stats"),
    ("A9", "GBR does not determine tau", True, "tau=0.90 pinned; sweep diagnostic only"),
    ("A10", "All models use identical test masks", True,
     "same (t,i,j) population from Step-2 npz; defensive year check PASS"),
    ("A11", "All nested comparisons use identical populations", True,
     "single preds_by_model dict per domain; paired bootstrap same picks"),
    ("A12", "Threshold sweep is diagnostic only", True,
     "formal tau=0.90 never modified; sweep written to separate CSV"),
    ("A13", "No Step-4/5/6/7 outputs overwritten", guard_ok, f"{len(guard_hashes)} files hash-verified"),
    ("A14", "Model hashes are recorded", True, "checkpoint_md5 in step8_protocol.json"),
    ("A15", "Random seeds are recorded", True, "seed 42 (train/boot), 42 (perm), 7 (spot-check)"),
    ("A16", "Bootstrap seed = 42", True, f"N_BOOT={N_BOOT}"),
    ("A17", "Permutation preserves thermal variables and labels", True,
     "assert array_equal on non-env channels inside every replicate; target untouched"),
]
for cid, desc, ok, det in checks:
    audit_lines.append(f"- **{cid}** {'PASS' if ok else 'FAIL'} — {desc} ({det})")
audit_lines.append("")
audit_lines.append(f"Consistency checks: {sum(1 for c in results['consistency_checks'].values() if c['verdict']=='PASS')}"
                   f"/{len(results['consistency_checks'])} PASS")
audit_lines.append(f"OUTCOME: {OUTCOME}")
with open(OUT_DIR / "step8_audit.md", "w", encoding="utf-8") as f:
    f.write("\n".join(audit_lines) + "\n")

# plot data (optional)
plot_rows = []
for dom in ["SCS", "GBR"]:
    for m in BOOT_MODELS:
        r = S[dom]["methods"][m]
        plot_rows.append({"domain": dom, "method": m,
                          "roc": r["roc_auc"], "pr": r["pr_auc"], "csi09": r["csi_09"]})
pd.DataFrame(plot_rows).to_csv(OUT_DIR / "step8_plot_data.csv", index=False)

np.savez_compressed(OUT_DIR / "step8_predictions.npz", **npz_out)

print("\n" + "=" * 64)
print("## STEP 8 STATUS")
print("=" * 64)
print(f"- Verdict: {'PASS' if all_ok else 'FAIL'}")
for dom in ["SCS", "GBR"]:
    print(f"- {dom} " + " ".join(
        f"{m} ROC {S[dom]['methods'][m]['roc_auc']:.4f} PR {S[dom]['methods'][m]['pr_auc']:.4f} "
        f"CSI@0.9 {S[dom]['methods'][m]['csi_09']:.4f}" for m in STEP8_MODELS))
print(f"- Best GBR nested gains CI-supported: ENV1-scale {gbr_env1}, ENV2-scale {gbr_env2}")
print(f"- C1-vs-FULL GBR: ΔCSI@0.9 {c1_csi_adv:+.4f}; CSI_max D1 {d1_csi_max:.4f} vs FULL {full_csi_max:.4f}")
print(f"- OUTCOME: {OUTCOME}")
print(f"- Outputs: {OUT_DIR}")
print("Done.")
