#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the physics-anchored residual head on SCS (2003-2010) under the exact
D1/a6 protocol.

  y_final(t+4) = PersA(t) + gate(t) * residual(t)
  PersA(t) = clip(DHW(t)/10,0,1)  (physics persistence anchor, no grads)
  residual = tanh(residual_head(attn.detach())) * scale  (bounded, ±scale)
  gate     = sigmoid((y_main - tau)/temp)

Protocol (identical to train_severe_classifier.py / a6):
  Adam lr=1e-3 wd=1e-5, cosine T_max=60 eta_min=1e-5, batch 32, patience 10,
  grad clip 1.0, seed 42, best-val selection. SCS train 2003-2010, val
  2011-2015, test 2016-2019, lead 4wk, history 24wk.

Strict CRW / no leakage:
  - inputs <= issue t (causal v2), anchor = y(t) available at t,
  - target y(t+4) only in the loss,
  - SCS-train-only normalization (inside backbone channel_norm, frozen).

Output: results/physics_anchored/model.pt + summary.json

Run:
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' experiments/01_training/train_physics_anchored.py
"""
import os, sys, io, json, time, warnings, argparse
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "04_structural_ablation"))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import create_scs_dataloaders
from physics_anchored_model import PhysicsAnchoredResidualModel, anchored_loss

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=0.4,
                help="residual bound scale (frozen config; SCS-val argmax CSI@0.9 = 0.4)")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--seed", type=int, default=42,
                help="random seed (seed-sensitivity audit: 0,1,2,3,42)")
ap.add_argument("--tag", type=str, default=None,
                help="suffix for model file (default: overwrite results/physics_anchored/model.pt)")
args = ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs
BATCH = 32
LR = 1e-3
SEED = args.seed
CFG = {"gamma": 0.5, "scale": args.scale, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}
OUT_DIR = RESULTS_DIR / "physics_anchored"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FN = "model.pt" if args.tag is None else f"model_{args.tag}.pt"


def main():
    print("=" * 70)
    print("Physics-anchored residual head — SCS training (D1/a6 protocol)")
    print(f"Device: {DEVICE}  cfg={CFG}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    loaders = create_scs_dataloaders(
        data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=SEED, lead_weeks=4,
        year_based_split=True, train_years=(2003, 2010),
        val_years=(2011, 2015), test_years=(2016, 2019))
    tr, va, te = loaders

    model = PhysicsAnchoredResidualModel(dynamic_channels=13).to(DEVICE)
    n_res = sum(p.numel() for p in model.residual_head.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  residual_head params={n_res}, total={n_total}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_val = float("inf")
    patience = 0
    best_state = None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl_sum, sl_sum, nb = 0.0, 0.0, 0
        for batch in tr:
            xd = batch["x_dynamic"].to(DEVICE)
            xs = batch["x_static"].to(DEVICE)
            lm = batch["land_mask"].to(DEVICE)
            nm = batch.get("nan_mask_dyn")
            if nm is not None:
                nm = nm.to(DEVICE)
            label = batch.get("label_mask")
            if label is not None:
                label = label.to(DEVICE)
            y = batch["y_lead"].to(DEVICE)
            anchor = batch["y_regression"][:, -1].to(DEVICE)   # PersA(t), no grads
            opt.zero_grad()
            out = model(xd, xs, lm, anchor, nan_mask=nm,
                        scale=CFG["scale"], tau=CFG["tau"], temp=CFG["temp"])
            loss, sev = anchored_loss(out, y, label, CFG)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            tl_sum += loss.item()
            sl_sum += sev.item()
            nb += 1

        model.eval()
        vl_sum, nv = 0.0, 0
        with torch.no_grad():
            for batch in va:
                xd = batch["x_dynamic"].to(DEVICE)
                xs = batch["x_static"].to(DEVICE)
                lm = batch["land_mask"].to(DEVICE)
                nm = batch.get("nan_mask_dyn")
                if nm is not None:
                    nm = nm.to(DEVICE)
                label = batch.get("label_mask")
                if label is not None:
                    label = label.to(DEVICE)
                y = batch["y_lead"].to(DEVICE)
                anchor = batch["y_regression"][:, -1].to(DEVICE)
                out = model(xd, xs, lm, anchor, nan_mask=nm,
                            scale=CFG["scale"], tau=CFG["tau"], temp=CFG["temp"])
                loss, _ = anchored_loss(out, y, label, CFG)
                vl_sum += loss.item()
                nv += 1
        sched.step()
        vl = vl_sum / max(nv, 1)
        if vl < best_val:
            best_val = vl
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep % 10 == 0 or ep == EPOCHS:
            print(f"    ep {ep:3d} train={tl_sum/max(nb,1):.6f} val={vl:.6f} "
                  f"sev={sl_sum/max(nb,1):.6f} t={time.time()-t0:.0f}s", flush=True)
        if patience >= 10:
            print(f"    Early stop at epoch {ep}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), OUT_DIR / MODEL_FN)
    print(f"\n  Best val = {best_val:.6f}  saved {OUT_DIR / MODEL_FN}")

    summary = {
        "experiment": "physics_anchored_residual_head",
        "device": str(DEVICE),
        "cfg": CFG,
        "model_file": MODEL_FN,
        "model_md5": __import__("hashlib").md5(
            (OUT_DIR / MODEL_FN).read_bytes()).hexdigest(),
        "residual_head_params": n_res,
        "total_params": n_total,
        "best_val": float(best_val),
        "epochs_run": ep,
        "seed": SEED,
        "protocol": "D1/a6: Adam 1e-3/wd1e-5, cosine 60ep, batch 32, patience 10, "
                    "SCS train 2003-2010/val 2011-2015/test 2016-2019, lead 4wk, history 24wk",
        "scale_selection": "SCS-val argmax CSI@0.9 (audit D-group; scale 0.4 selected)",
        "leakage": "causal v2 inputs<=t; anchor=y(t); target y(t+4) loss-only; "
                   "SCS-train normalization frozen",
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
