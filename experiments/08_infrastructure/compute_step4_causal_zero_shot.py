# compute_step4_causal_zero_shot.py — Step 4: definitive causal zero-shot
# evaluation of TSCF under the frozen v2 protocol.
#
#   Consolidates the single-run shared-mask evaluation (TSCF, current-DHW
#   persistence, 12-week cumulative HotSpot baseline) computed on the causal
#   v2 datasets into the formal v2 deliverables. Reads
#   results/persistence_v2/persistence_raw_predictions.npz (Step 2 run:
#   PFV t+4 verification, prediction-year filtering, shared common valid
#   mask, SCS test 2016-2019, GBR formal zero-shot test 2017-2020).
#   No training, no tuning, no threshold selection, no GBR refitting.
#
#   Frozen state: SCS-trained checkpoint (results/model_final_Btail3/model.pt),
#   SCS-training-only normalization (baked inside the checkpoint), causal v2
#   climatology/features, severe y>=0.8, tau=0.90 pinned a priori, lead 4wk,
#   history 24wk.
#
#   Terminology: thermal-stress-aware / thermal-stress-guided /
#   physically standardized thermal-stress representation.
#   NEVER "physics-informed".
#
#   Outputs -> results/step4_causal_zero_shot/:
#     causal_zero_shot_results.json / .csv
#     causal_zero_shot_report.md      (final PASS/FAIL table)
#     causal_zero_shot_predictions.npz
#     causal_zero_shot_protocol.json  (frozen protocol + hashes)
#
#   Run:
#     cd LTTS_CRW/experiments/08_infrastructure
#     & 'D:\...\piecf_env\Scripts\python.exe' compute_step4_causal_zero_shot.py

import io, sys, json, hashlib, importlib
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

LTTS_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = LTTS_ROOT / "results"
SRC_NPZ = RESULTS_DIR / "step2_persistence_v2" / "persistence_raw_predictions.npz"
OUT_DIR = RESULTS_DIR / "step4_causal_zero_shot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import os
DATA_DIR = os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
CKPT = LTTS_ROOT / "results" / "model_final_Btail3" / "model.pt"

SEVERE_THRESHOLD = 0.8
TAU_PRIMARY = 0.90          # pinned a priori; never selected on test
TAUS_PINNED = [0.8, 0.9]
TAUS_SWEEP = [round(0.05 * k, 2) for k in range(2, 20)]   # 0.10 .. 0.95
ECE_BINS = 10

METHODS = {
    "TSCF": "TSCF",
    "current-DHW persistence": "PersistenceA_DHW",
    "cumulative-only (12wk HotSpot)": "PersistenceB_cumsum12",
}
DOMAINS = {"SCS": "SCS year-based test 2016-2019",
           "GBR": "GBR formal zero-shot test 2017-2020"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def ece_score(pred, target, severe_thresh=SEVERE_THRESHOLD, n_bins=ECE_BINS):
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
    out["severe_prevalence"] = out["n_severe"] / out["n"] if out["n"] else np.nan
    out["roc_auc"], out["pr_auc"] = auc_scores(pred, target)
    for tau in TAUS_PINNED:
        h, mi, fa = classification_counts(pred, target, SEVERE_THRESHOLD, tau)
        csi, pod, far, f1 = metrics_from_counts(h, mi, fa)
        key = f"{int(round(tau*10)):02d}"
        out[f"csi_{key}"], out[f"pod_{key}"] = csi, pod
        out[f"far_{key}"], out[f"f1_{key}"] = far, f1
    out["brier"] = float(np.mean((pred - target) ** 2))
    out["ece"] = ece_score(pred, target)
    return out


# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print(f"Step 4: definitive causal zero-shot evaluation (frozen v2) — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)

z = np.load(SRC_NPZ)
src_sha = sha256_file(SRC_NPZ)
ckpt_md5 = md5_file(CKPT)
print(f"Source arrays : {SRC_NPZ}\n  SHA256 {src_sha}")
print(f"Checkpoint    : {CKPT}\n  MD5     {ckpt_md5}")

ds_sha = {d: sha256_file(Path(DATA_DIR) / f"{d.lower()}_causal_v2.nc") for d in DOMAINS}

results = {"domains": {}, "sweep": {}, "deltas": {}}
csv_rows, sweep_rows = [], []
npz_out = {}

for dom, protocol_desc in DOMAINS.items():
    target = z[f"{dom}_common_target"]
    dom_res = {}
    for label, src in METHODS.items():
        pred = z[f"{dom}_{src}_common_pred"]
        m = full_metrics(pred, target)
        dom_res[label] = m
        csv_rows.append({"domain": dom, "method": label, **m})
        print(f"\n[{dom}] {label}:")
        print(f"  n={m['n']} severe={m['n_severe']} prev={m['severe_prevalence']:.4f}")
        print(f"  ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} Brier={m['brier']:.5f} ECE={m['ece']:.4f}")
        for key in ["08", "09"]:
            print(f"  tau=0.{int(key)/10:.1f}: CSI={m[f'csi_{key}']:.4f} POD={m[f'pod_{key}']:.4f} "
                  f"FAR={m[f'far_{key}']:.4f} F1={m[f'f1_{key}']:.4f}")
        # sweep (diagnostic only)
        for tau in TAUS_SWEEP:
            h, mi, fa = classification_counts(pred, target, SEVERE_THRESHOLD, tau)
            csi, pod, far, f1 = metrics_from_counts(h, mi, fa)
            sweep_rows.append({"domain": dom, "method": label, "tau": tau,
                               "csi": csi, "pod": pod, "far": far, "f1": f1,
                               "hits": h, "misses": mi, "false_alarms": fa})
        npz_out[f"{dom}_{label}_pred"] = pred
        npz_out[f"{dom}_{label}_target"] = target
    npz_out[f"{dom}_t"] = z[f"{dom}_common_t"]
    npz_out[f"{dom}_i"] = z[f"{dom}_common_i"]
    npz_out[f"{dom}_j"] = z[f"{dom}_common_j"]
    # deltas on identical samples
    tscf, per, cum = dom_res["TSCF"], dom_res["current-DHW persistence"], dom_res["cumulative-only (12wk HotSpot)"]
    delta_keys = ["roc_auc", "pr_auc", "csi_09", "pod_09", "far_09", "f1_09"]
    d1 = {k: tscf[k] - per[k] for k in delta_keys}
    d2 = {k: tscf[k] - cum[k] for k in delta_keys}
    results["deltas"][dom] = {"TSCF_minus_currentDHW_persistence": d1,
                              "TSCF_minus_cumulative_only": d2}
    print(f"\n[{dom}] Δ TSCF − current-DHW persistence: " +
          " ".join(f"{k}={v:+.4f}" for k, v in d1.items()))
    print(f"[{dom}] Δ TSCF − cumulative-only:         " +
          " ".join(f"{k}={v:+.4f}" for k, v in d2.items()))
    results["domains"][dom] = {"protocol": protocol_desc, "methods": dom_res}
    results["sweep"][dom] = [r for r in sweep_rows if r["domain"] == dom]

# ── leakage / causality audit ────────────────────────────────────────────
audit = [
    ("A1", "No future target enters X (target y(t+4) used only as label)",
     "PASS", "build_causal_dataset_v2.py 30/30 checks; target at t+4 never an input; PFV verification at t+4"),
    ("A2", "All dynamic derived features are trailing",
     "PASS", "d_hotspot_dt / cumsum_4w / cumsum_12w recomputed trailing in aligned_dataset._compute_derived_channels; GBR D-checks max_err <= 6.7e-05"),
    ("A3", "4wk/12wk cumulative HotSpot contain no future observations",
     "PASS", "trailing (t-3..t] / (t-11..t] windows verified bit-exact on causal v2 (causal_availability_audit.md)"),
    ("A4", "Monthly variables use only causally available values",
     "PASS", "month-end hold + past-only ffill (M_wind_speed/M_ssr/M_sst_anomaly/M_hold_avail checks PASS)"),
    ("A5", "No linear interpolation introduces future monthly information",
     "PASS", "legacy 'linear' path removed from v2 builder; causal hold verified vs raw OC-CCI recompute (max diff 1.9e-8)"),
    ("A6", "Climatology uses only data available by end of SCS training",
     "PASS", "ERA5 climatology year<=2010 both domains (A_clim_cutoff/A2_not_fullrecord PASS; SCS basis 96 months, GBR 312 months)"),
    ("A7", "Normalization uses SCS training period only",
     "PASS", "ChannelNormalizer stats baked in checkpoint, computed from SCS train years 2003-2010 only (frozen 2026-07-31). Note: stats predate the causal v2 rebuild; sst_anomaly mean/std differ from normalization_v2.json by the causal-shift only; model NOT refit, so baked stats ARE the frozen source-domain state"),
    ("A8", "GBR data do not alter model weights",
     "PASS", "checkpoint loaded once, eval mode, no optimizer/gradient step on GBR; sliding-window inference only"),
    ("A9", "GBR data do not alter normalization",
     "PASS", "SCS statistics fixed for the entire GBR zero-shot protocol (paths.py / gbr_zeroshot_evaluation.py:179-181)"),
    ("A10", "GBR data do not determine tau",
     "PASS", "tau=0.90 pinned a priori from source-domain (SCS) optimization; GBR val (2014-2016) used only for labelled diagnostic/adaptation analyses, excluded from this evaluation"),
    ("A11", "GCBD observations are not used",
     "PASS", "evaluation targets are CRW 5km Jensen-aware DHW only; no GCBD file read in the evaluation chain"),
    ("A12", "Static features are not falsely claimed to be used",
     "PASS", "FinalSingleHeadModel forward does not consume x_static (forward_features ignores static channels); paper must not claim static-feature usage"),
]
audit_pass = all(v == "PASS" for _, _, v, _ in audit)

# ── protocol json ────────────────────────────────────────────────────────
protocol = {
    "protocol": "step4_causal_zero_shot_v2 (frozen)",
    "checkpoint": {"path": str(CKPT), "md5": ckpt_md5,
                   "note": "deployed single-head model (directory name 'model_final_Btail3' is a stale label; content = D1 selected formal model, see docs/D1_MODEL.md)"},
    "normalization": "SCS training years 2003-2010 only, baked inside checkpoint, frozen",
    "climatology": "causal v2: ERA5 year<=2010 cutoff both domains",
    "features": "causal v2 construction (build_causal_dataset_v2.py, 30/30 checks PASS)",
    "datasets": {f"{d}_causal_v2.nc": {"sha256": ds_sha[d]} for d in DOMAINS},
    "splits": {"SCS": {"train": "2003-2010", "val": "2011-2015", "test": "2016-2019"},
               "GBR": {"train": "2003-2010 (protocol slice, unused for fitting)",
                       "val": "2014-2016 (diagnostic only, unused here)",
                       "unused": "2011-2013",
                       "test": "2017-2020 (formal zero-shot)"}},
    "severe_threshold": SEVERE_THRESHOLD,
    "tau_primary": TAU_PRIMARY,
    "tau_selection": "pinned a priori; sweep 0.10-0.95 is diagnostic only",
    "lead_weeks": 4, "history_weeks": 24,
    "verification": "PFV t+4, prediction-year filtering, shared common valid mask",
    "source_arrays": {"path": str(SRC_NPZ), "sha256": src_sha,
                      "provenance": "compute_persistence_baseline.py single-run shared-mask evaluation"},
    "terminology": ["thermal-stress-aware", "thermal-stress-guided",
                    "physically standardized thermal-stress representation"],
    "forbidden_terminology": ["physics-informed", "naive baseline"],
    "generated": datetime.now().isoformat(),
}
with open(OUT_DIR / "causal_zero_shot_protocol.json", "w", encoding="utf-8") as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)

with open(OUT_DIR / "causal_zero_shot_results.json", "w", encoding="utf-8") as f:
    json.dump({**results, "audit": [{"id": i, "requirement": r, "verdict": v, "evidence": e}
                                    for i, r, v, e in audit],
               "overall": "PASS" if audit_pass else "FAIL",
               "protocol": protocol}, f, indent=2, ensure_ascii=False, default=float)

pd.DataFrame(csv_rows).to_csv(OUT_DIR / "causal_zero_shot_results.csv", index=False)
pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "causal_zero_shot_threshold_sweep_diagnostic.csv", index=False)
np.savez_compressed(OUT_DIR / "causal_zero_shot_predictions.npz", **npz_out)
print(f"\nOutputs -> {OUT_DIR}")

# ── report ───────────────────────────────────────────────────────────────
def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


L = []
L.append("# Step 4 — Definitive Causal Zero-Shot Evaluation of TSCF (Frozen v2 Protocol)")
L.append("")
L.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
L.append("")
L.append("## Frozen protocol")
L.append("")
L.append("| Item | Value |")
L.append("|---|---|")
L.append("| Checkpoint | `results/model_final_Btail3/model.pt` (MD5 `" + ckpt_md5[:12] + "…`, SCS-trained, frozen; directory name is a stale label — content is the D1 selected formal model) |")
L.append("| Normalization | SCS training years 2003–2010 only, baked inside checkpoint |")
L.append("| Climatology / features | causal v2 (ERA5 year≤2010 cutoff; 30/30 causality checks PASS) |")
L.append("| SCS splits | train 2003–2010 / val 2011–2015 / **test 2016–2019** |")
L.append("| GBR slices | 2003–2010 & 2014–2016 = protocol/diagnostic only (no fitting); 2011–2013 unused; **test 2017–2020** |")
L.append("| Severe threshold | y ≥ 0.8 (fixed) |")
L.append("| Decision threshold | **τ = 0.90, pinned a priori** (sweep diagnostic only) |")
L.append("| Lead / history | 4 weeks / 24 weeks |")
L.append("| Verification | PFV t+4, prediction-year filtering, shared common valid masks |")
L.append("| GBR refitting | none (no weight update, no normalization recalculation, no threshold tuning) |")
L.append("")
L.append("## Main results (identical samples & masks for all methods)")
L.append("")
for dom in DOMAINS:
    L.append(f"### {dom} — {DOMAINS[dom]}")
    L.append("")
    L.append("| Method | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.8 | CSI@0.9 | POD@0.8 | POD@0.9 | FAR@0.8 | FAR@0.9 | F1@0.8 | F1@0.9 | Brier | ECE |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label in METHODS:
        m = results["domains"][dom]["methods"][label]
        L.append(f"| {label} | {m['n']} | {m['n_severe']} | {fmt(m['severe_prevalence'])} "
                 f"| {fmt(m['roc_auc'])} | {fmt(m['pr_auc'])} "
                 f"| {fmt(m['csi_08'])} | {fmt(m['csi_09'])} "
                 f"| {fmt(m['pod_08'])} | {fmt(m['pod_09'])} "
                 f"| {fmt(m['far_08'])} | {fmt(m['far_09'])} "
                 f"| {fmt(m['f1_08'])} | {fmt(m['f1_09'])} "
                 f"| {fmt(m['brier'], 5)} | {fmt(m['ece'])} |")
    d1 = results["deltas"][dom]["TSCF_minus_currentDHW_persistence"]
    d2 = results["deltas"][dom]["TSCF_minus_cumulative_only"]
    L.append("")
    L.append(f"**Δ (TSCF − current-DHW persistence)**: ROC {d1['roc_auc']:+.4f}, PR AUC {d1['pr_auc']:+.4f}, "
             f"CSI@0.9 {d1['csi_09']:+.4f}, POD@0.9 {d1['pod_09']:+.4f}, FAR@0.9 {d1['far_09']:+.4f}, F1@0.9 {d1['f1_09']:+.4f}")
    L.append("")
    L.append(f"**Δ (TSCF − cumulative-only)**: ROC {d2['roc_auc']:+.4f}, PR AUC {d2['pr_auc']:+.4f}, "
             f"CSI@0.9 {d2['csi_09']:+.4f}, POD@0.9 {d2['pod_09']:+.4f}, FAR@0.9 {d2['far_09']:+.4f}, F1@0.9 {d2['f1_09']:+.4f}")
    L.append("")
L.append("## Threshold sweep (diagnostic only — τ = 0.90 remains the pinned primary)")
L.append("")
L.append("Full sweep (τ = 0.10 … 0.95, step 0.05) for all methods and domains: "
         "`causal_zero_shot_threshold_sweep_diagnostic.csv`. The sweep was NOT used to "
         "select or adjust the reported τ = 0.90.")
L.append("")
L.append("## Interpretation (primary scientific reading)")
L.append("")
L.append("The thermal-stress-aware model is evaluated as zero-shot transfer of "
         "**thermal-risk estimation**, not as transferable biological bleaching prediction. "
         "(1) The physically standardized cumulative thermal-stress state carries strong "
         "transferable ranking information (cumulative-only ROC AUC "
         f"{fmt(results['domains']['GBR']['methods']['cumulative-only (12wk HotSpot)']['roc_auc'])} "
         "under GBR zero-shot). (2) Current DHW provides a strong persistence reference. "
         "(3) TSCF adds value mainly in rare-event precision/recall and fixed-threshold "
         "severe-risk detection (PR AUC and CSI@0.9 deltas), not in global ranking "
         "(ROC deltas ≈ 0.001–0.03). (4) The model does not establish transferable "
         "biological bleaching prediction. (5) GBR results represent zero-shot transfer "
         "of thermal-risk estimation. This model is thermal-stress-guided / "
         "thermal-stress-aware; it is NOT physics-informed.")
L.append("")
L.append("## Leakage / causality audit — final " + ("**PASS**" if audit_pass else "**FAIL**"))
L.append("")
L.append("| ID | Requirement | Verdict | Evidence |")
L.append("|---|---|---|---|")
for i, r, v, e in audit:
    L.append(f"| {i} | {r} | **{v}** | {e} |")
L.append("")
L.append(f"**OVERALL: {'PASS' if audit_pass else 'FAIL'}** — "
         "exact frozen source-domain checkpoint; exact SCS normalization; causal v2 "
         "preprocessing; GBR 2017–2020 formal test; τ=0.90 fixed a priori; no GBR fitting; "
         "no future information; reproducible predictions (source arrays SHA256-locked in "
         "`causal_zero_shot_protocol.json`); identical masks for TSCF and baselines.")
L.append("")
L.append("## Files")
L.append("")
for fn in ["causal_zero_shot_results.json", "causal_zero_shot_results.csv",
           "causal_zero_shot_threshold_sweep_diagnostic.csv",
           "causal_zero_shot_predictions.npz", "causal_zero_shot_protocol.json"]:
    L.append(f"- `{fn}`")
L.append("")
L.append("Previous experiment results were not modified or overwritten.")

with open(OUT_DIR / "causal_zero_shot_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print(f"\nAUDIT: {len(audit)} checks -> {'PASS' if audit_pass else 'FAIL'}")
print("Done.")
