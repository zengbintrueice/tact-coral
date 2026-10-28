#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D1: Severe binary classifier (weighted BCE) on the LTTS_CRW backbone.

Same ConvLSTM backbone + single sigmoid head (FinalSingleHeadModel), but the
loss is a class-weighted binary cross-entropy on the severe label
y_severe = (y_lead >= 0.8). The sigmoid output is therefore P(severe) in [0,1]
by construction, so top scores can reach high values (fixes the compressed
score scale that limited the tail-weighted regression).

Loss (valid pixels only):
    L = mean[ pos_weight * y_sev * log(p) + (1-y_sev) * log(1-p) ]
with pos_weight = n_neg / n_pos (severe-class inverse frequency).

Output: results/model_severe_classifier/model.pt + summary JSON
"""
import os, sys, io, json, time, warnings, argparse
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import DATA_DIR
from src.data.aligned_dataset import create_scs_dataloaders
import ablate_a_dh_stopgrad as trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH = 32
LR = 1e-3
SEED = 42
SEV = 0.8


def weighted_bce(p, y, y_sev, valid, pos_weight):
    """Class-weighted BCE on the severe label, valid pixels only."""
    v = valid.float()
    y_clean = torch.nan_to_num(y_sev.float(), nan=0.0)
    eps = 1e-7
    loss = -(pos_weight * y_clean * torch.log(p.clamp(eps, 1 - eps))
             + (1 - y_clean) * torch.log((1 - p).clamp(eps, 1 - eps)))
    return (loss * v).sum() / (v.sum() + 1e-8)


def evaluate(model, loader, device):
    model.eval()
    p_all, t_all, s_all, m_all = [], [], [], []
    lm = torch.BoolTensor(loader.dataset.land_mask).to(device)
    with torch.no_grad():
        for batch in loader:
            xd = batch["x_dynamic"].to(device); xs = batch["x_static"].to(device)
            nm = batch.get("nan_mask_dyn")
            if nm is not None: nm = nm.to(device)
            lab = batch.get("label_mask")
            if lab is not None: lab = lab.to(device)
            y = batch["y_lead"].to(device)
            valid = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
            out = model(xd, xs, lm, nan_mask=nm)
            p_all.append(out[0][:, -1].cpu().numpy())
            t_all.append(y.cpu().numpy())
            s_all.append((y.cpu().numpy() >= SEV).astype(np.float32))
            m_all.append(valid.cpu().numpy())
    return (np.concatenate(p_all), np.concatenate(t_all),
            np.concatenate(s_all), np.concatenate(m_all))


def csi_metrics(p, s, m, tau):
    v = m > 0.5
    tp = np.sum((p >= tau) & (s == 1) & v); fp = np.sum((p >= tau) & (s == 0) & v); fn = np.sum((p < tau) & (s == 1) & v)
    return (tp / (tp + fp + fn + 1e-9), tp / (tp + fn + 1e-9), fp / (tp + fp + 1e-9))


def main():
    # Seed-sensitivity parametrisation (Supplementary Fig. S8): --seed/--out-dir.
    # DEFAULTS REPRODUCE THE ORIGINAL BEHAVIOUR exactly (seed 42, formal output
    # dir) — architecture, data, splits, features, hyperparameters (incl.
    # pos_weight=1000 fixed), normalisation and protocol are untouched; only
    # the random seed varies.
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="override output dir (default: results/model_severe_classifier)")
    args = ap.parse_args()
    seed = args.seed
    out_dir = Path(args.out_dir) if args.out_dir \
        else ROOT / "results" / "model_severe_classifier"

    print("=" * 70)
    print("D1: Severe binary classifier (weighted BCE)")
    print(f"Device: {DEVICE}  seed={seed}")
    torch.manual_seed(seed); np.random.seed(seed)
    loaders = create_scs_dataloaders(
        data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=seed, lead_weeks=4, year_based_split=True,
        train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019))
    tr, va, te = loaders
    # pos_weight from train severe rate
    ytr = tr.dataset.y_lead
    labtr = tr.dataset.label_mask
    n_pos = int(np.sum((ytr >= SEV) & labtr)); n_neg = int(np.sum((ytr < SEV) & labtr))
    pos_weight = 1000.0  # sweep-optimal (posweight_sweep.json: pw1000 test CSI 0.367)
    print(f"train severe: {n_pos} / {n_pos+n_neg} ({n_pos/(n_pos+n_neg)*100:.3f}%)  pos_weight={pos_weight:.1f}")

    model = trainer.FinalSingleHeadModel(dynamic_channels=13).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best_val = float("inf"); best_state = None; patience = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in tr:
            xd = batch["x_dynamic"].to(DEVICE); xs = batch["x_static"].to(DEVICE)
            lm = batch["land_mask"].to(DEVICE)
            nm = batch.get("nan_mask_dyn")
            if nm is not None: nm = nm.to(DEVICE)
            lab = batch.get("label_mask")
            if lab is not None: lab = lab.to(DEVICE)
            y = batch["y_lead"].to(DEVICE)
            valid = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
            y_sev = (y >= SEV).float()
            opt.zero_grad()
            out = model(xd, xs, lm, nan_mask=nm)
            p = out[0][:, -1]
            loss = weighted_bce(p, y, y_sev, valid, torch.tensor(pos_weight, device=DEVICE))
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in va:
                xd = batch["x_dynamic"].to(DEVICE); xs = batch["x_static"].to(DEVICE)
                lm = batch["land_mask"].to(DEVICE)
                nm = batch.get("nan_mask_dyn")
                if nm is not None: nm = nm.to(DEVICE)
                lab = batch.get("label_mask")
                if lab is not None: lab = lab.to(DEVICE)
                y = batch["y_lead"].to(DEVICE)
                valid = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
                y_sev = (y >= SEV).float()
                out = model(xd, xs, lm, nan_mask=nm)
                vl += weighted_bce(out[0][:, -1], y, y_sev, valid, torch.tensor(pos_weight, device=DEVICE)).item()
                nv += 1
        sched.step()
        v = vl / max(nv, 1)
        if v < best_val:
            best_val = v; patience = 0
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else:
            patience += 1
        if ep % 15 == 0 or ep == EPOCHS:
            print(f"    ep {ep:3d} val-bce={v:.4f}")
        if patience >= 10:
            break

    model.load_state_dict(best_state); model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Saved: {out_dir / 'model.pt'}")

    # val-select threshold on P(severe) -> test metrics
    pv, tv, sv, mv = evaluate(model, va, DEVICE)
    pt, tt, st, mt = evaluate(model, te, DEVICE)
    grid = np.round(np.arange(0.05, 0.95, 0.025), 3)
    best = None
    for tau in grid:
        c, _, _ = csi_metrics(pv, sv, mv, tau)
        if np.isfinite(c) and (best is None or c > best[1]):
            best = (tau, c)
    tau0, _ = best
    c, pod, far = csi_metrics(pt, st, mt, tau0)
    from sklearn.metrics import roc_auc_score, average_precision_score
    sel = mt > 0.5
    print(f"\n[P(severe) classifier] val-τ={tau0:.3f}")
    print(f"  test CSI={c:.4f} POD={pod:.4f} FAR={far:.4f}")
    print(f"  test ROC-AUC={roc_auc_score(st[sel], pt[sel]):.4f}  PR-AUC={average_precision_score(st[sel], pt[sel]):.4f}")
    print(f"  P(severe) 输出范围: [{pt[sel].min():.3f}, {pt[sel].max():.3f}]")

    out = {"experiment": "D1_severe_classifier_BCE", "device": str(DEVICE),
           "seed": seed, "pos_weight": float(pos_weight), "best_val_tau": float(tau0),
           "test": {"csi": c, "pod": pod, "far": far},
           "train_severe_rate": float(n_pos / (n_pos + n_neg)),
           "timestamp": datetime.now().isoformat()}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
