# compute_transfer_matrix_seed_scan.py
#
# Multi-seed robustness of the six-direction transfer matrix (reviewer
# concern: the GBR/CARIB source rows of the headline matrix are seed-42
# only, so the Limitations claim "unlikely to be a seed artifact" needs
# direct evidence).
#
# Design (mirrors compute_transfer_coordinate_train_reverse.py VERBATIM,
# only the seed changes):
#   - sources GBR and CARIB; additional seeds {0, 1, 2} (seed-42 models
#     already exist and are NOT retrained or modified)
#   - per (source, seed): train TSCF (D1 weighted BCE pos_weight=1000,
#     Adam 1e-3, cosine 60, batch 32, patience 10) and TSAR (focal gamma
#     0.5 + lambda_sev 20 MSE, scale 0.4, same optimiser) on the source
#     basin 2003-2010 train / 2011-2015 val split; evaluate zero-shot on
#     every basin test set including its own
#   - outputs per-seed matrix rows + ΔROC = ROC(target) − ROC(own source)
#
# PRE-SPECIFIED verdict rule (stated before running): a directed transfer
# counts as a *collapse* if ΔROC <= −0.10 (seed-42 observed collapses are
# −0.23 / −0.26; TSAR |ΔROC| <= 0.021 — the threshold separates the two
# regimes by an order of magnitude).  The seed-42 headline pattern is
# "GBR-sourced TSCF collapses to EVERY target; CARIB-sourced TSCF
# transfers to EVERY target".  Stability = the same qualitative pattern
# holds in N/3 additional seeds.  Per-seed CIs are NOT recomputed (the
# seed-42 paired bootstrap CIs are already published; the seed scan asks
# whether the point-estimate pattern reproduces).
#
# Weights -> results/transfer_coordinate/weights/seed_scan/{src}_seed{seed}/
# Outputs -> results/transfer_coordinate/seed_scan/
#
# Run (piecf_env, GPU):
#   cd LTTS_CRW && python experiments/08_infrastructure/compute_transfer_matrix_seed_scan.py

import os, sys, io, json, time, hashlib, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import gc
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "04_structural_ablation"))
from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import create_scs_dataloaders
import ablate_a_dh_stopgrad as tscf_trainer
from physics_anchored_model import PhysicsAnchoredResidualModel, anchored_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60; BATCH = 32; LR = 1e-3; SEV = 0.8
SEEDS = [0, 1, 2]
SOURCES = ["GBR", "CARIB"]
SRC_YEARS = dict(train_years=(2003, 2010), val_years=(2011, 2015))
TEST_YEARS = {"SCS": (2016, 2019), "GBR": (2017, 2020), "CARIB": (2017, 2020)}
FILE = {"SCS": "scs_causal_v2.nc", "GBR": "gbr_causal_v2.nc",
        "CARIB": "caribbean_causal_v2.nc"}
OUT = RESULTS_DIR / "transfer_coordinate" / "seed_scan"
WT = RESULTS_DIR / "transfer_coordinate" / "weights" / "seed_scan"
OUT.mkdir(parents=True, exist_ok=True)

COLLAPSE_THR = -0.10          # pre-specified


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def make_loaders(source, seed):
    return create_scs_dataloaders(
        data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=seed, lead_weeks=4,
        year_based_split=True, train_years=SRC_YEARS["train_years"],
        val_years=SRC_YEARS["val_years"], test_years=TEST_YEARS[source],
        dataset_file=FILE[source])


def train_tscf(source, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, _ = make_loaders(source, seed)
    model = tscf_trainer.FinalSingleHeadModel(dynamic_channels=13).to(DEVICE)
    pos_weight = torch.tensor(1000.0, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best = float("inf"); best_state = None; patience = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in tr:
            xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE)
            lm = b["land_mask"].to(DEVICE)
            nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
            if nm is not None: nm = nm.to(DEVICE)
            if lab is not None: lab = lab.to(DEVICE)
            y = b["y_lead"].to(DEVICE)
            v = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
            p = model(xd, xs, lm, nan_mask=nm)[0][:, -1]
            y_sev = (y >= SEV).float(); eps = 1e-7
            loss = -(pos_weight * y_sev * torch.log(p.clamp(eps, 1 - eps))
                     + (1 - y_sev) * torch.log((1 - p).clamp(eps, 1 - eps)))
            loss = (loss * v).sum() / (v.sum() + 1e-8)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0.0; nv = 0
        with torch.no_grad():
            for b in va:
                xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE)
                lm = b["land_mask"].to(DEVICE)
                nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
                if nm is not None: nm = nm.to(DEVICE)
                if lab is not None: lab = lab.to(DEVICE)
                y = b["y_lead"].to(DEVICE)
                v = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
                p = model(xd, xs, lm, nan_mask=nm)[0][:, -1]
                y_sev = (y >= SEV).float(); eps = 1e-7
                loss = -(pos_weight * y_sev * torch.log(p.clamp(eps, 1 - eps))
                         + (1 - y_sev) * torch.log((1 - p).clamp(eps, 1 - eps)))
                loss = (loss * v).sum() / (v.sum() + 1e-8)
                vl += loss.item(); nv += 1
        sched.step(); vl /= max(nv, 1)
        if vl < best:
            best = vl; patience = 0
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 10:
            break
    model.load_state_dict(best_state); model.eval()
    d = WT / f"{source}_seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "tscf_model.pt")
    json.dump({"source": source, "seed": seed, "epochs": ep, "best_val": float(best),
               "md5": md5(d / "tscf_model.pt"),
               "timestamp": datetime.now().isoformat()},
              open(d / "tscf_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"  [TSCF {source} seed{seed}] best_val={best:.6f} ep={ep}", flush=True)
    return model


def train_tsar(source, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, _ = make_loaders(source, seed)
    CFG = {"gamma": 0.5, "scale": 0.4, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}
    model = PhysicsAnchoredResidualModel(dynamic_channels=13).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best = float("inf"); best_state = None; patience = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in tr:
            xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE)
            lm = b["land_mask"].to(DEVICE)
            nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
            if nm is not None: nm = nm.to(DEVICE)
            if lab is not None: lab = lab.to(DEVICE)
            y = b["y_lead"].to(DEVICE); anch = b["y_regression"][:, -1].to(DEVICE)
            opt.zero_grad()
            out = model(xd, xs, lm, anch, nan_mask=nm, scale=CFG["scale"],
                        tau=CFG["tau"], temp=CFG["temp"])
            loss, _ = anchored_loss(out, y, lab, CFG)
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval(); vl = 0.0; nv = 0
        with torch.no_grad():
            for b in va:
                xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE)
                lm = b["land_mask"].to(DEVICE)
                nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
                if nm is not None: nm = nm.to(DEVICE)
                if lab is not None: lab = lab.to(DEVICE)
                y = b["y_lead"].to(DEVICE); anch = b["y_regression"][:, -1].to(DEVICE)
                out = model(xd, xs, lm, anch, nan_mask=nm, scale=CFG["scale"],
                            tau=CFG["tau"], temp=CFG["temp"])
                loss, _ = anchored_loss(out, y, lab, CFG)
                vl += loss.item(); nv += 1
        sched.step(); vl /= max(nv, 1)
        if vl < best:
            best = vl; patience = 0
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 10:
            break
    model.load_state_dict(best_state); model.eval()
    d = WT / f"{source}_seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "tsar_model.pt")
    json.dump({"source": source, "seed": seed, "epochs": ep, "best_val": float(best),
               "md5": md5(d / "tsar_model.pt"), "cfg": CFG,
               "timestamp": datetime.now().isoformat()},
              open(d / "tsar_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"  [TSAR {source} seed{seed}] best_val={best:.6f} ep={ep}", flush=True)
    return model


def eval_target(model, kind, target, seed):
    _, _, te = create_scs_dataloaders(
        data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=seed, lead_weeks=4,
        year_based_split=True, train_years=SRC_YEARS["train_years"],
        val_years=SRC_YEARS["val_years"], test_years=TEST_YEARS[target],
        dataset_file=FILE[target])
    lm = torch.BoolTensor(te.dataset.land_mask).to(DEVICE)
    model.eval()
    p_all, t_all = [], []
    with torch.no_grad():
        for b in te:
            xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE)
            lm_ = b["land_mask"].to(DEVICE)
            nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
            if nm is not None: nm = nm.to(DEVICE)
            if lab is not None: lab = lab.to(DEVICE)
            y = b["y_lead"].to(DEVICE)
            v = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm_).float())
            vmask = v > 0.5
            if kind == "tscf":
                p = model(xd, xs, lm, nan_mask=nm)[0][:, -1]
            else:
                anch = b["y_regression"][:, -1].to(DEVICE)
                p = model(xd, xs, lm, anch, nan_mask=nm, scale=0.4, tau=0.5,
                          temp=0.1)[0][:, -1]
            p_all.append(p[vmask].cpu().numpy()); t_all.append(y[vmask].cpu().numpy())
    return (np.concatenate(p_all).astype(float), np.concatenate(t_all).astype(float))


def metrics(pred, target):
    y = (target >= SEV).astype(int)
    roc = float(roc_auc_score(y, pred)) if len(np.unique(y)) >= 2 else np.nan
    pr = float(average_precision_score(y, pred)) if len(np.unique(y)) >= 2 else np.nan
    return {"n": len(pred), "n_severe": int(y.sum()), "roc_auc": roc, "pr_auc": pr}


if __name__ == "__main__":
    print("=" * 66)
    print(f"Transfer-matrix seed scan (sources {SOURCES}, seeds {SEEDS}) — "
          f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Device:", DEVICE)
    print("=" * 66)

    rows = []
    t0 = time.time()
    for src in SOURCES:
        for seed in SEEDS:
            d = WT / f"{src}_seed{seed}"
            print(f"\n--- source {src} seed {seed} ---", flush=True)
            if (d / "tscf_model.pt").exists():
                mT = tscf_trainer.FinalSingleHeadModel(dynamic_channels=13).to(DEVICE)
                mT.load_state_dict(torch.load(d / "tscf_model.pt", map_location=DEVICE,
                                              weights_only=False)); mT.eval()
            else:
                mT = train_tscf(src, seed)
            if (d / "tsar_model.pt").exists():
                mR = PhysicsAnchoredResidualModel(dynamic_channels=13).to(DEVICE)
                mR.load_state_dict(torch.load(d / "tsar_model.pt", map_location=DEVICE,
                                              weights_only=False)); mR.eval()
            else:
                mR = train_tsar(src, seed)
            src_preds = {}
            for kind, m in (("TSCF", mT), ("TSAR", mR)):
                src_preds[kind] = eval_target(m, kind.lower(), src, seed)
                mm = metrics(*src_preds[kind])
                rows.append({"source": src, "seed": seed, "target": src,
                             "model": kind, "role": "in_source", **mm})
                print(f"  {src}->{src} {kind} ROC={mm['roc_auc']:.4f}", flush=True)
            for tgt in ["SCS", "GBR", "CARIB"]:
                if tgt == src:
                    continue
                for kind, m in (("TSCF", mT), ("TSAR", mR)):
                    pred, tlab = eval_target(m, kind.lower(), tgt, seed)
                    mm = metrics(pred, tlab)
                    sP, sT = src_preds[kind]
                    droc = mm["roc_auc"] - metrics(sP, sT)["roc_auc"]
                    rows.append({"source": src, "seed": seed, "target": tgt,
                                 "model": kind, "role": "zero_shot", **mm,
                                 "d_roc_vs_own_source": droc,
                                 "collapse_flag": int(droc <= COLLAPSE_THR)})
                    print(f"  {src}->{tgt} {kind} ROC={mm['roc_auc']:.4f} "
                          f"dROC={droc:+.4f}"
                          f"{'  [COLLAPSE]' if droc <= COLLAPSE_THR else ''}",
                          flush=True)
            del mT, mR, src_preds
            gc.collect()
            torch.cuda.empty_cache()

            # staged flush: persist partial results after EVERY (source, seed)
            # block so a later crash cannot lose completed evaluations
            _df = pd.DataFrame(rows)
            _df.to_csv(OUT / "seed_scan_matrix.csv", index=False)
            with open(OUT / "seed_scan_matrix.csv", "a", encoding="utf-8") as _f:
                _f.flush()
            print(f"  [flush] {len(_df)} rows written after {src} seed {seed}",
                  flush=True)

    # free CUDA context + cached datasets BEFORE aggregation (segfault fix:
    # aggregation previously ran while eval windows/datasets still resided on
    # device, causing an intermittent native crash)
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seed_scan_matrix.csv", index=False)

    # ---- pre-specified pattern verdicts ----
    verdicts = {}
    for src in SOURCES:
        for kind in ("TSCF", "TSAR"):
            sub = df[(df["source"] == src) & (df["model"] == kind)
                     & (df["role"] == "zero_shot")]
            per_seed = {}
            for seed in SEEDS:
                s = sub[sub["seed"] == seed]
                per_seed[seed] = {t: float(s[s["target"] == t]["d_roc_vs_own_source"]
                                           .iloc[0]) for t in s["target"].unique()}
            if src == "GBR" and kind == "TSCF":
                pattern_ok = all(all(v <= COLLAPSE_THR for v in per_seed[s].values())
                                 for s in SEEDS)
                desc = "collapses to every target (dROC <= -0.10)"
            elif src == "CARIB" and kind == "TSCF":
                pattern_ok = all(all(v > COLLAPSE_THR for v in per_seed[s].values())
                                 for s in SEEDS)
                desc = "transfers to every target (dROC > -0.10)"
            elif kind == "TSAR":
                pattern_ok = all(abs(v) <= 0.05 for s in SEEDS
                                 for v in per_seed[s].values())
                desc = "|dROC| <= 0.05 to every target"
            verdicts[f"{src}_{kind}"] = {
                "expectation": desc, "per_seed_droc": per_seed,
                "pattern_holds_in_all_seeds": bool(pattern_ok)}
    summary = {"experiment": "transfer_matrix_seed_scan",
               "generated": datetime.now().isoformat(),
               "device": str(DEVICE),
               "sources": SOURCES, "seeds": SEEDS,
               "collapse_threshold_droc": COLLAPSE_THR,
               "note": ("seed-42 models already exist and are untouched; the "
                        "seed scan asks whether the qualitative seed-42 pattern "
                        "reproduces. Per-seed CIs are not recomputed; the seed-42 "
                        "paired bootstrap CIs are published in "
                        "reverse_matrix_ci.csv."),
               "verdicts": verdicts,
               "runtime_min": round((time.time() - t0) / 60, 1)}
    with open(OUT / "seed_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    print("\n===== VERDICTS (pre-specified rule) =====")
    for k, v in verdicts.items():
        print(f"  {k:16s} {v['expectation']:48s} -> "
              f"{'HOLDS in all seeds' if v['pattern_holds_in_all_seeds'] else 'DOES NOT HOLD'}")
        for s, dd in v["per_seed_droc"].items():
            print(f"      seed {s}: " + "  ".join(f"{t} {x:+.3f}" for t, x in dd.items()))
    print(f"\nOutputs -> {OUT}")
    print("DONE")
