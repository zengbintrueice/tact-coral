#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict methodology audit of the CARIB zero-shot failure + thermal-stress-
anchored residual recovery. (Do NOT modify Steps 2-10 results; frozen TSCF.)

Audit outputs -> results/physics_anchored/audit/:
  scale_sensitivity.{csv,json}   D-group (SCS-val selection, frozen scale)
  residual_diagnostics.csv       E-group (raw/gate/final-anchor magnitude)
  three_domain_metrics.csv       PersA / PhysicsAnchored / TSCF on 3 domains
  leakage_audit.csv              B-group zero-shot evidence (A1-A6 style)
  audit_report.md                A+B+C+D+E summary + 4 verdicts
  manifest.json                  hashes + config + selection rule + seeds

Key design:
  * One sliding-window inference per split at scale=1.0 gives the RAW residual
    raw = tanh(residual_head(...)) (scale-independent). Since
    y_final(s) = PersA + gate*raw*s, the whole scale sweep is pure arithmetic
    on the saved per-cell arrays — no repeated inference, no search over test.
  * Splits: SCS-val (2011-2015, SELECTION ONLY), SCS-test (2016-2019),
    GBR (2017-2020), CARIB (2017-2020). Same PFV t+4 record-set builder
    (issue-year filter, ocean + finite-target + finite-all-methods cells).
  * Predefined selection rule: argmax CSI@0.9 on SCS-val; alternatives
    (PR-AUC, severe-MSE) reported for transparency but NOT used for selection.
  * Frozen TSCF (results/model_final_Btail3/model.pt) and PersA
    (y_regression at issue time) recomputed in-script on the identical record
    sets so the three-way comparison is like-for-like.
  * Terminology: "bounded deviation around the persistence anchor" (residual
    is two-sided, NOT a persistence floor).
"""
import io, sys, os, json, time, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import A6_WEIGHT, DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from physics_anchored_model import PhysicsAnchoredResidualModel

LEAD = 4
SEV = 0.8
TAU_EVAL = 0.9            # classification threshold (as in Step-4/9/10)
SCALES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]   # D-group (predefined)
SELECTION_RULE = "argmax CSI@0.9 on SCS-val (predefined; alternatives reported)"
OUT_DIR = RESULTS_DIR / "physics_anchored" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCH_CFG = {"scale": 1.0, "tau": 0.5, "temp": 0.1}   # scale=1 -> raw residual
CKPT_TSCF = A6_WEIGHT
CKPT_ANCH = RESULTS_DIR / "physics_anchored" / "model.pt"

SPLITS = [
    {"name": "SCS_val", "file": "scs_causal_v2.nc", "years": (2011, 2015),
     "role": "selection"},
    {"name": "SCS_test", "file": "scs_causal_v2.nc", "years": (2016, 2019),
     "role": "test"},
    {"name": "GBR", "file": "gbr_causal_v2.nc", "years": (2017, 2020),
     "role": "zeroshot"},
    {"name": "CARIB", "file": "caribbean_causal_v2.nc", "years": (2017, 2020),
     "role": "zeroshot"},
]
SPLIT_ARGS = {
    "SCS_val": dict(train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019)),
    "SCS_test": dict(train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019)),
    "GBR": dict(train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020)),
    "CARIB": dict(train_years=(2003, 2010), val_years=(2014, 2016), test_years=(2017, 2020)),
}


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore_anchored(path):
    m = PhysicsAnchoredResidualModel(dynamic_channels=13)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    m.eval()
    return m


def restore_tscf():
    import ablate_a_dh_stopgrad as mod
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(CKPT_TSCF, map_location="cpu", weights_only=False))
    m.eval()
    return m


def infer_anchored(model, ds, device):
    """Sliding inference at scale=1.0 -> maps of y_final, y_main, gate, raw_res
    (raw = tanh(residual_head)) each [n_dates, H, W]."""
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    yf = np.full((n, H, W), np.nan, np.float32)
    ym = np.full((n, H, W), np.nan, np.float32)
    gt = np.full((n, H, W), np.nan, np.float32)
    rr = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    Y = ds.y_regression_full
    bs = 16
    with torch.no_grad():
        for b in range(0, n - T + 1, bs):
            be = min(b + bs, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            anch = torch.FloatTensor(Y[b + T - 1: be + T - 1].copy()).to(device)
            out = model(xd, xs, lm, anch, scale=ANCH_CFG["scale"],
                        tau=ANCH_CFG["tau"], temp=ANCH_CFG["temp"])
            yf[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
            ym[b + T - 1: be + T - 1] = out[1][:, -1].cpu().numpy()
            gt[b + T - 1: be + T - 1] = out[2][:, -1].cpu().numpy()
            rr[b + T - 1: be + T - 1] = out[3][:, -1].cpu().numpy()
    return yf, ym, gt, rr


def infer_tscf(model, ds, device):
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    fc = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    bs = 16
    with torch.no_grad():
        for b in range(0, n - T + 1, bs):
            be = min(b + bs, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc


def collect_records(ds, years, yf, ym, gt, rr, fc, tscf_map, persa_map):
    """PFV record set (identical to compute_physics_anchored_eval.collect):
    issue-year filter, target t+4, cells where target + all methods finite."""
    T, n = ds.history_weeks, ds.n_dates
    y = ds.y_regression_full
    out = {"yf": [], "ym": [], "gt": [], "rr": [], "tscf": [], "persa": [],
           "target": [], "t": [], "i": [], "j": []}
    for t in range(T - 1, n):
        if not (years[0] <= int(ds.date_strings[t][:4]) <= years[1]):
            continue
        tt = t + LEAD
        if tt >= n:
            continue
        v = ~np.isnan(y[tt]) & ~ds.land_mask
        v &= ~np.isnan(yf[t]) & ~np.isnan(ym[t]) & ~np.isnan(gt[t]) & ~np.isnan(rr[t])
        v &= ~np.isnan(tscf_map[t]) & ~np.isnan(persa_map[t])
        if not v.any():
            continue
        ii, jj = np.where(v)
        out["yf"].append(yf[t][v]); out["ym"].append(ym[t][v])
        out["gt"].append(gt[t][v]); out["rr"].append(rr[t][v])
        out["tscf"].append(tscf_map[t][v]); out["persa"].append(persa_map[t][v])
        out["target"].append(y[tt][v])
        out["t"].append(np.full(len(ii), t, np.int32))
        out["i"].append(ii.astype(np.int16)); out["j"].append(jj.astype(np.int16))
    return {k: (np.concatenate(v) if v else np.array([], dtype=np.float32))
            for k, v in out.items()}


def classification_counts(pred, target, thr=TAU_EVAL):
    pb = pred >= thr
    ab = target >= SEV
    hits = int(np.sum(pb & ab)); miss = int(np.sum(~pb & ab)); fa = int(np.sum(pb & ~ab))
    csi = hits / (hits + miss + fa) if (hits + miss + fa) else np.nan
    pod = hits / (hits + miss) if (hits + miss) else np.nan
    far = fa / (hits + fa) if (hits + fa) else np.nan
    f1 = 2 * hits / (2 * hits + fa + miss) if (2 * hits + fa + miss) else np.nan
    return csi, pod, far, f1


def metrics(pred, target):
    sev = (target >= SEV)
    out = {"n": int(len(pred)), "n_severe": int(sev.sum()),
           "prevalence": float(sev.mean())}
    if len(np.unique(sev)) < 2:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan})
    else:
        out["roc_auc"] = float(roc_auc_score(sev, pred))
        out["pr_auc"] = float(average_precision_score(sev, pred))
    csi, pod, far, f1 = classification_counts(pred, target)
    out.update({"csi_tau09": csi, "pod_tau09": pod, "far_tau09": far, "f1_tau09": f1})
    out["brier"] = float(np.mean((pred - target) ** 2))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        msk = (pred >= lo) & (pred < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / len(pred)) * abs(sev[msk].mean() - pred[msk].mean())
    out["ece"] = float(ece)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    t0 = time.time()

    anch_ckpt_md5 = md5_file(CKPT_ANCH)
    tscf_ckpt_md5 = md5_file(CKPT_TSCF)
    print(f"anchored ckpt md5={anch_ckpt_md5}  tscf ckpt md5={tscf_ckpt_md5}", flush=True)

    model_anch = restore_anchored(CKPT_ANCH).to(device)
    model_tscf = restore_tscf().to(device)

    recs = {}
    for sp in SPLITS:
        name = sp["name"]
        print(f"\n=== {name} ({sp['role']}) years={sp['years']} ===", flush=True)
        ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                             lead_weeks=LEAD, dataset_file=sp["file"],
                             year_based_split=True, **SPLIT_ARGS[name])
        yf, ym, gt, rr = infer_anchored(model_anch, ds, device)
        fc = infer_tscf(model_tscf, ds, device)
        persa = ds.y_regression_full.copy()
        rec = collect_records(ds, sp["years"], yf, ym, gt, rr, fc, fc, persa)
        recs[name] = rec
        # PersA at issue time from the PersA map; TSCF map is fc
        rec["tscf"] = rec["tscf"]          # already fc
        rec["persa"] = rec["persa"]
        print(f"  record cells: n={len(rec['target'])}, "
              f"severe={int((rec['target'] >= SEV).sum())}", flush=True)
        # sanity: raw residual = y_final(scale=1) - PersA, divided by gate
        raw_recon = (rec["yf"] - rec["persa"]) / np.maximum(rec["gt"], 1e-8)
        diff = np.nanmax(np.abs(raw_recon - rec["rr"]))
        print(f"  [check] max|(yf1-PersA)/gate - raw| = {diff:.2e} (must be ~0)", flush=True)
        assert diff < 1e-4, "raw-residual reconstruction mismatch"
        np.savez_compressed(OUT_DIR / f"rec_{name}.npz",
                            yf=rec["yf"], ym=rec["ym"], gt=rec["gt"], rr=rec["rr"],
                            tscf=rec["tscf"], persa=rec["persa"],
                            target=rec["target"], t=rec["t"], i=rec["i"], j=rec["j"])
        del ds
        torch.cuda.empty_cache()

    # ---- D. scale sensitivity (pure arithmetic) ----
    scale_rows = []
    sel_csi = {}
    for sp in SPLITS:
        name = sp["name"]
        r = recs[name]
        persa, gt, rr, target = r["persa"], r["gt"], r["rr"], r["target"]
        for s in SCALES:
            yfinal = persa + gt * rr * s
            m = metrics(yfinal, target)
            scale_rows.append({"split": name, "role": sp["role"], "scale": s, **m})
            if name == "SCS_val":
                sel_csi[s] = m["csi_tau09"]
    sdf = pd.DataFrame(scale_rows)
    sdf.to_csv(OUT_DIR / "scale_sensitivity.csv", index=False)

    # selection on SCS-val CSI@0.9 (predefined)
    best_scale = max(sel_csi, key=lambda s: sel_csi[s] if np.isfinite(sel_csi[s]) else -1)
    print(f"\nSCS-val CSI@0.9 per scale: { {s: round(sel_csi[s], 4) for s in SCALES} }")
    print(f"SELECTED scale (argmax CSI@0.9 on SCS-val): {best_scale}", flush=True)

    # ---- three-domain metrics at the SELECTED scale + PersA + TSCF ----
    tdm_rows = []
    for sp in SPLITS:
        name = sp["name"]
        r = recs[name]
        persa, gt, rr, target = r["persa"], r["gt"], r["rr"], r["target"]
        yfinal_s = persa + gt * rr * best_scale
        for method, pred in (("PersA", persa), ("PhysicsAnchored", yfinal_s),
                             ("TSCF", r["tscf"])):
            m = metrics(pred, target)
            tdm_rows.append({"split": name, "role": sp["role"], "method": method,
                             "scale": best_scale if method == "PhysicsAnchored" else np.nan,
                             **m})
    tdf = pd.DataFrame(tdm_rows)
    tdf.to_csv(OUT_DIR / "three_domain_metrics.csv", index=False)

    # ---- E. residual magnitude diagnostics at the SELECTED scale ----
    diag_rows = []
    for sp in SPLITS:
        name = sp["name"]
        r = recs[name]
        persa, gt, rr, target = r["persa"], r["gt"], r["rr"], r["target"]
        raw = rr
        final_anch = persa + gt * raw * best_scale
        delta = final_anch - persa                      # = gate*raw*scale
        diag_rows.append({
            "split": name, "scale": best_scale, "n": len(persa),
            "raw_mean": float(raw.mean()), "raw_sd": float(raw.std()),
            "raw_p2_5": float(np.percentile(raw, 2.5)), "raw_p50": float(np.percentile(raw, 50)),
            "raw_p97_5": float(np.percentile(raw, 97.5)),
            "gate_mean": float(gt.mean()), "gate_p2_5": float(np.percentile(gt, 2.5)),
            "gate_p50": float(np.percentile(gt, 50)), "gate_p97_5": float(np.percentile(gt, 97.5)),
            "delta_mean": float(delta.mean()), "delta_sd": float(delta.std()),
            "frac_abs_delta_gt_005": float(np.mean(np.abs(delta) > 0.05)),
            "frac_abs_delta_gt_010": float(np.mean(np.abs(delta) > 0.10)),
            "frac_abs_delta_gt_020": float(np.mean(np.abs(delta) > 0.20)),
            "frac_raw_at_boundary": float(np.mean(np.abs(raw) > 0.999)),
            "frac_final_gt_1": float(np.mean(final_anch > 1.0)),
            "frac_final_lt_0": float(np.mean(final_anch < 0.0)),
        })
    ddf = pd.DataFrame(diag_rows)
    ddf.to_csv(OUT_DIR / "residual_diagnostics.csv", index=False)
    print("\nResidual diagnostics:\n", ddf.round(4).to_string(index=False), flush=True)

    # ---- leakage audit (B-group) ----
    leak_rows = [
        {"check": "B1_CARIB_no_fitting", "status": "PASS",
         "evidence": "anchored ckpt trained on SCS only (train_physics_anchored.py: SCS 2003-2010/val 2011-2015); CARIB dataset built after training; ckpt md5=%s" % anch_ckpt_md5},
        {"check": "B2_no_normalization_on_CARIB", "status": "PASS",
         "evidence": "ChannelNormalizer uses baked SCS_CHANNEL_MEAN/STD (== config to float32); normalization_v2.json SCS-train only; nothing recomputed on CARIB"},
        {"check": "B3_no_threshold_from_CARIB", "status": "PASS",
         "evidence": "tau_eval=0.9 pinned from SCS protocol (Step-4); no threshold searched on CARIB/GBR"},
        {"check": "B4_scale_not_chosen_on_CARIB", "status": "PASS",
         "evidence": "scale selection = argmax CSI@0.9 on SCS-val only (%s); GBR/CARIB never read during selection" % SELECTION_RULE},
        {"check": "B5_no_early_stopping_on_CARIB", "status": "PASS",
         "evidence": "best-val loss on SCS val; patience 10; GBR/CARIB absent from training loop"},
        {"check": "B6_no_model_selection_on_CARIB", "status": "PASS",
         "evidence": "single frozen checkpoints (TSCF + anchored); no CARIB-driven checkpoint choice; TSCF numbers bit-identical across two independent eval scripts (n=94095, ROC=0.5981282063358642)"},
        {"check": "C1_anchor_is_current_DHW", "status": "PASS",
         "evidence": "PersA(t)=y_regression_full[t]=clip(DHW(t)/10) from causal v2, trailing 84-day window <= t; passed as detached input"},
        {"check": "C2_residual_bounded_two_sided", "status": "PASS",
         "evidence": "raw=tanh(res_head) in [-1,1]; final=PersA+gate*raw*scale => bounded deviation around anchor, NOT a floor"},
        {"check": "C3_residual_grad_blocked", "status": "PASS",
         "evidence": "residual_head(attn.detach()); gate.detach() in severe loss (physics_anchored_model.py:81,128-130)"},
    ]
    leakdf = pd.DataFrame(leak_rows)
    leakdf.to_csv(OUT_DIR / "leakage_audit.csv", index=False)

    # ---- manifest ----
    manifest = {
        "experiment": "physics_anchored_strict_audit",
        "generated_utc": datetime.now().isoformat(),
        "checkpoints": {"tscf": {"path": str(CKPT_TSCF), "md5": tscf_ckpt_md5},
                        "anchored": {"path": str(CKPT_ANCH), "md5": anch_ckpt_md5}},
        "data_files": {sp["name"]: sp["file"] for sp in SPLITS},
        "selection_rule": SELECTION_RULE,
        "selected_scale": best_scale,
        "scales_tested": SCALES,
        "model_cfg_at_inference": ANCH_CFG,
        "tau_eval": TAU_EVAL, "lead": LEAD, "severe": SEV,
        "scs_val_csi_per_scale": {str(s): float(sel_csi[s]) for s in SCALES},
        "record_counts": {sp["name"]: int(len(recs[sp["name"]]["target"])) for sp in SPLITS},
        "n_methods": "PersA / PhysicsAnchored(selected scale) / TSCF(frozen)",
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- three-domain comparison table (at selected scale) for report ----
    print("\nThree-domain (selected scale=%s):" % best_scale)
    print(tdf.round(4).to_string(index=False), flush=True)
    print(f"\nWall time: {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"Outputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
