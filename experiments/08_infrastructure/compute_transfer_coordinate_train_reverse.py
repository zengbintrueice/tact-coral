# compute_transfer_coordinate_train_reverse.py
#
# Reverse-source training tier of the transfer-coordinate experiment: train
# TSCF and TSAR on GBR and on CARIB as SOURCE basins (seed 42), then evaluate
# each source model zero-shot on all three basin tests, completing the 6-direction
# transfer matrix. PersA is source-independent and already covered.
#
# Reuses VERBATIM the D1/a6 training loops (train_severe_classifier.py,
# train_physics_anchored.py), retargeted only by dataset_file + split years.
# Normalization is the hardcoded SCS_CHANNEL_MEAN/STD inside every model
# (src/models/piecf_model.py:193), so zero-shot inputs use the same frozen
# normalizer whatever the source basin -> source-only scaling is respected.
#
# Zero-shot: each model is trained on the SOURCE basin's 2003-2010 train only;
# no target is trained/tuned/normalized by its own statistics; tau=0.90 pinned.
#
# Weights/new results -> results/transfer_coordinate/ (never modify locked results)
#
# Run (piecf_env, background): python experiments/08_infrastructure/compute_transfer_coordinate_train_reverse.py

import os, sys, io, json, time, warnings, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import xarray as xr
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "04_structural_ablation"))
from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import create_scs_dataloaders
import ablate_a_dh_stopgrad as tscf_trainer
from physics_anchored_model import PhysicsAnchoredResidualModel, anchored_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60; BATCH = 32; LR = 1e-3; SEED = 42; SEV = 0.8
SRC_YEARS = dict(train_years=(2003, 2010), val_years=(2011, 2015))
TEST_YEARS = {"SCS": (2016, 2019), "GBR": (2017, 2020), "CARIB": (2017, 2020)}
FILE = {"SCS": "scs_causal_v2.nc", "GBR": "gbr_causal_v2.nc", "CARIB": "caribbean_causal_v2.nc"}
OUT = RESULTS_DIR / "transfer_coordinate"
WT = OUT / "weights"

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# ---------- TSCF (Weighted BCE) — verbatim loop, parametrized ----------
def train_tscf(source):
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, va, te = create_scs_dataloaders(data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=SEED, lead_weeks=4, year_based_split=True,
        train_years=SRC_YEARS["train_years"], val_years=SRC_YEARS["val_years"],
        test_years=TEST_YEARS[source], dataset_file=FILE[source])
    model = tscf_trainer.FinalSingleHeadModel(dynamic_channels=13).to(DEVICE)
    pos_weight = torch.tensor(1000.0, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best = float("inf"); best_state = None; patience = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in tr:
            xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE); lm = b["land_mask"].to(DEVICE)
            nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
            if nm is not None: nm = nm.to(DEVICE)
            if lab is not None: lab = lab.to(DEVICE)
            y = b["y_lead"].to(DEVICE)
            v = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
            p = model(xd, xs, lm, nan_mask=nm)[0][:, -1]
            y_sev = (y >= SEV).float(); eps = 1e-7
            loss = -(pos_weight * y_sev * torch.log(p.clamp(eps, 1 - eps)) + (1 - y_sev) * torch.log((1 - p).clamp(eps, 1 - eps)))
            loss = (loss * v).sum() / (v.sum() + 1e-8)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0.0; nv = 0
        with torch.no_grad():
            for b in va:
                xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE); lm = b["land_mask"].to(DEVICE)
                nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
                if nm is not None: nm = nm.to(DEVICE)
                if lab is not None: lab = lab.to(DEVICE)
                y = b["y_lead"].to(DEVICE)
                v = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
                p = model(xd, xs, lm, nan_mask=nm)[0][:, -1]
                y_sev = (y >= SEV).float(); eps = 1e-7
                loss = -(pos_weight * y_sev * torch.log(p.clamp(eps, 1 - eps)) + (1 - y_sev) * torch.log((1 - p).clamp(eps, 1 - eps)))
                vl += ((loss * v).sum() / (v.sum() + 1e-8)).item(); nv += 1
        sched.step(); vl /= max(nv, 1)
        if vl < best: best = vl; patience = 0; best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else: patience += 1
        if patience >= 10: break
    model.load_state_dict(best_state); model.eval()
    d = WT / source; d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "tscf_model.pt")
    json.dump({"experiment": "reverse_tscf", "source": source, "seed": SEED, "epochs": ep,
               "best_val": float(best), "md5": md5(d / "tscf_model.pt"),
               "protocol": "D1/a6 weighted BCE pos_weight=1000; Adam1e-3 cosine60 batch32 pat10",
               "dataset": FILE[source], "split": {"test_years": TEST_YEARS[source], **SRC_YEARS},
               "timestamp": datetime.now().isoformat()},
              open(d / "tscf_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"  [TSCF] trained on {source}: best_val={best:.6f} ep={ep}")
    return model

# ---------- TSAR (physics-anchored residual) — verbatim loop, parametrized ----------
def train_tsar(source):
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, va, te = create_scs_dataloaders(data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=SEED, lead_weeks=4, year_based_split=True,
        train_years=SRC_YEARS["train_years"], val_years=SRC_YEARS["val_years"],
        test_years=TEST_YEARS[source], dataset_file=FILE[source])
    CFG = {"gamma": 0.5, "scale": 0.4, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}
    model = PhysicsAnchoredResidualModel(dynamic_channels=13).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best = float("inf"); best_state = None; patience = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in tr:
            xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE); lm = b["land_mask"].to(DEVICE)
            nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
            if nm is not None: nm = nm.to(DEVICE)
            if lab is not None: lab = lab.to(DEVICE)
            y = b["y_lead"].to(DEVICE); anch = b["y_regression"][:, -1].to(DEVICE)
            opt.zero_grad()
            out = model(xd, xs, lm, anch, nan_mask=nm, scale=CFG["scale"], tau=CFG["tau"], temp=CFG["temp"])
            loss, _ = anchored_loss(out, y, lab, CFG)
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0.0; nv = 0
        with torch.no_grad():
            for b in va:
                xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE); lm = b["land_mask"].to(DEVICE)
                nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
                if nm is not None: nm = nm.to(DEVICE)
                if lab is not None: lab = lab.to(DEVICE)
                y = b["y_lead"].to(DEVICE); anch = b["y_regression"][:, -1].to(DEVICE)
                out = model(xd, xs, lm, anch, nan_mask=nm, scale=CFG["scale"], tau=CFG["tau"], temp=CFG["temp"])
                loss, _ = anchored_loss(out, y, lab, CFG)
                vl += loss.item(); nv += 1
        sched.step(); vl /= max(nv, 1)
        if vl < best: best = vl; patience = 0; best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else: patience += 1
        if patience >= 10: break
    model.load_state_dict(best_state); model.eval()
    d = WT / source; d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "tsar_model.pt")
    json.dump({"experiment": "reverse_tsar", "source": source, "seed": SEED, "epochs": ep,
               "best_val": float(best), "md5": md5(d / "tsar_model.pt"), "cfg": CFG,
               "protocol": "focal gamma0.5 + lambda_sev20 MSE; Adam1e-3 cosine60 batch32 pat10; scale0.4",
               "dataset": FILE[source], "split": {"test_years": TEST_YEARS[source], **SRC_YEARS},
               "timestamp": datetime.now().isoformat()},
              open(d / "tsar_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"  [TSAR] trained on {source}: best_val={best:.6f} ep={ep}")
    return model

# ---------- zero-shot eval of a source model on a target basin test ----------
def eval_target(model, kind, target):
    _, _, te = create_scs_dataloaders(data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=SEED, lead_weeks=4, year_based_split=True,
        train_years=SRC_YEARS["train_years"], val_years=SRC_YEARS["val_years"],
        test_years=TEST_YEARS[target], dataset_file=FILE[target])
    lm = torch.BoolTensor(te.dataset.land_mask).to(DEVICE)
    model.eval()
    p_all, t_all = [], []
    with torch.no_grad():
        for b in te:
            xd = b["x_dynamic"].to(DEVICE); xs = b["x_static"].to(DEVICE)
            nm = b.get("nan_mask_dyn"); lab = b.get("label_mask")
            if nm is not None: nm = nm.to(DEVICE)
            if lab is not None: lab = lab.to(DEVICE)
            y = b["y_lead"].to(DEVICE)
            v = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
            vmask = v > 0.5
            if kind == "tscf":
                p = model(xd, xs, lm, nan_mask=nm)[0][:, -1]
            else:
                anch = b["y_regression"][:, -1].to(DEVICE)
                p = model(xd, xs, lm, anch, nan_mask=nm, scale=0.4, tau=0.5, temp=0.1)[0][:, -1]
            p_all.append(p[vmask].cpu().numpy()); t_all.append(y[vmask].cpu().numpy())
    return np.concatenate(p_all).astype(float), np.concatenate(t_all).astype(float)

def metrics(pred, target):
    y = (target >= SEV).astype(int)
    roc = float(roc_auc_score(y, pred)) if len(np.unique(y)) >= 2 else np.nan
    pr = float(average_precision_score(y, pred)) if len(np.unique(y)) >= 2 else np.nan
    h = int(np.sum((pred >= 0.9) & (target >= SEV))); m = int(np.sum((pred < 0.9) & (target >= SEV)))
    fa = int(np.sum((pred >= 0.9) & (target < SEV)))
    csi = h / (h + m + fa) if (h + m + fa) else np.nan
    pod = h / (h + m) if (h + m) else np.nan; far = fa / (h + fa) if (h + fa) else np.nan
    f1 = 2 * h / (2 * h + fa + m) if (2 * h + fa + m) else np.nan
    return dict(n=len(pred), n_severe=int(y.sum()), severe_prevalence=float(y.mean()),
                roc_auc=roc, pr_auc=pr, csi_tau09=csi, pod_tau09=pod, far_tau09=far, f1_tau09=f1,
                brier=float(np.mean((pred - target) ** 2)))

def diff_boot(srcP, srcT, tgtP, tgtT, seed=SEED, n=1000):
    """Bootstrap 95% CI of (metric on target) - (metric on source), record-level
    resampling (seed 42). Target and source are disjoint basins, so this is a
    difference-of-independent bootstrap, matching the degradation definition."""
    rng = np.random.default_rng(seed)
    droc = metrics(tgtP, tgtT)["roc_auc"] - metrics(srcP, srcT)["roc_auc"]
    dpr = metrics(tgtP, tgtT)["pr_auc"] - metrics(srcP, srcT)["pr_auc"]
    br, bp = [], []
    for _ in range(n):
        iS = rng.integers(0, len(srcP), len(srcP)); iT = rng.integers(0, len(tgtP), len(tgtP))
        br.append(metrics(tgtP[iT], tgtT[iT])["roc_auc"] - metrics(srcP[iS], srcT[iS])["roc_auc"])
        bp.append(metrics(tgtP[iT], tgtT[iT])["pr_auc"] - metrics(srcP[iS], srcT[iS])["pr_auc"])
    br = np.array([x for x in br if np.isfinite(x)]); bp = np.array([x for x in bp if np.isfinite(x)])
    roc_ci = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]
    pr_ci = [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))]
    return float(droc), roc_ci, float(dpr), pr_ci

def _mk_tscf(weights):
    m = tscf_trainer.FinalSingleHeadModel(dynamic_channels=13).to(DEVICE)
    m.load_state_dict(weights); m.eval(); return m

def _mk_tsar(weights):
    m = PhysicsAnchoredResidualModel(dynamic_channels=13).to(DEVICE)
    m.load_state_dict(weights); m.eval(); return m

if __name__ == "__main__":
    print("=" * 66); print("Reverse-source training + 6-direction matrix"); print("Device:", DEVICE); print("=" * 66)
    # --- train reverse-source checkpoints if absent ---
    for src in ["GBR", "CARIB"]:
        if not (OUT / "weights" / src / "tscf_model.pt").exists():
            print(f"\n--- train SOURCE {src} TSCF ---", flush=True); train_tscf(src)
        if not (OUT / "weights" / src / "tsar_model.pt").exists():
            print(f"\n--- train SOURCE {src} TSAR ---", flush=True); train_tsar(src)
    # --- reload trained weights and build the reverse rows ---
    import csv
    rev_rows, deg_rows = [], []
    for src in ["GBR", "CARIB"]:
        wt = torch.load(OUT / "weights" / src / "tscf_model.pt", map_location=DEVICE, weights_only=False)
        wr = torch.load(OUT / "weights" / src / "tsar_model.pt", map_location=DEVICE, weights_only=False)
        mT = _mk_tscf(wt); mR = _mk_tsar(wr)
        srcP_T, srcT_T = eval_target(mT, "tscf", src); srcP_R, srcT_R = eval_target(mR, "tsar", src)
        print(f"source {src}: TSCF ROC={metrics(srcP_T,srcT_T)['roc_auc']:.4f}  TSAR ROC={metrics(srcP_R,srcT_R)['roc_auc']:.4f}", flush=True)
        for tgt in ["SCS", "GBR", "CARIB"]:
            for kind, m, sP, sT in [("TSCF", mT, srcP_T, srcT_T), ("TSAR", mR, srcP_R, srcT_R)]:
                pred, tlab = eval_target(m, kind.lower(), tgt)
                mm = metrics(pred, tlab)
                rev_rows.append({"source": src, "target": tgt, "model": kind, **mm})
                droc, roc_ci, dpr, pr_ci = diff_boot(sP, sT, pred, tlab)
                deg_rows.append({"source": src, "target": tgt, "model": kind,
                                 "d_roc": mm["roc_auc"] - metrics(sP, sT)["roc_auc"],
                                 "d_roc_lo": roc_ci[0], "d_roc_hi": roc_ci[1],
                                 "d_pr": mm["pr_auc"] - metrics(sP, sT)["pr_auc"],
                                 "d_pr_lo": pr_ci[0], "d_pr_hi": pr_ci[1]})
                print(f"  {src}->{tgt} {kind} ROC={mm['roc_auc']:.4f}  dROC={mm['roc_auc']-metrics(sP,sT)['roc_auc']:+.4f}", flush=True)
    with open(OUT / "reverse_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rev_rows[0].keys())); w.writeheader(); w.writerows(rev_rows)
    with open(OUT / "reverse_matrix_ci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(deg_rows[0].keys())); w.writeheader(); w.writerows(deg_rows)
    print("reverse outputs ->", OUT / "reverse_matrix.csv", OUT / "reverse_matrix_ci.csv")
    print("DONE")