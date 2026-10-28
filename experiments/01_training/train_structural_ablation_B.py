#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Model B (Un-Gated Bounded Residual) on SCS 2003-2010 for the
structural ablation. EXACTLY mirrors train_physics_anchored.py (Model C)
except the loss/forward use gate ≡ 1. scale is the frozen value 0.4 (Work 1,
SCS-val argmax CSI@0.9). seed 42, D1/a6 protocol.

Output: results/physics_anchored/structural_ablation/model_B.pt + summary_B.json

Run:
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' experiments/01_training/train_structural_ablation_B.py
"""
import os, sys, io, json, time, warnings
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
from structural_ablation_models import UnGatedAnchoredResidualModel, ungated_anchored_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH = 32
LR = 1e-3
SEED = 42
# scale = frozen value (Work 1: SCS-val argmax CSI@0.9 = 0.4); tau/temp unused
CFG = {"gamma": 0.5, "scale": 0.4, "lambda_sev": 20}
OUT_DIR = RESULTS_DIR / "physics_anchored" / "structural_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Structural ablation — Model B (un-gated bounded residual)")
    print(f"Device: {DEVICE}  cfg={CFG}  (scale frozen = 0.4)")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    loaders = create_scs_dataloaders(
        data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=SEED, lead_weeks=4,
        year_based_split=True, train_years=(2003, 2010),
        val_years=(2011, 2015), test_years=(2016, 2019))
    tr, va, _ = loaders

    model = UnGatedAnchoredResidualModel(dynamic_channels=13).to(DEVICE)
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
            anchor = batch["y_regression"][:, -1].to(DEVICE)
            opt.zero_grad()
            out = model(xd, xs, lm, anchor, nan_mask=nm, scale=CFG["scale"])
            loss, sev = ungated_anchored_loss(out, y, label, CFG)
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
                out = model(xd, xs, lm, anchor, nan_mask=nm, scale=CFG["scale"])
                loss, _ = ungated_anchored_loss(out, y, label, CFG)
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
    ckpt = OUT_DIR / "model_B.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\n  Best val = {best_val:.6f}  saved {ckpt}")

    summary = {
        "experiment": "structural_ablation_model_B",
        "structure": "y = PersA + tanh(R(attn.detach()))*scale  (gate = 1, un-gated)",
        "device": str(DEVICE),
        "cfg": CFG,
        "model_file": ckpt.name,
        "model_md5": __import__("hashlib").md5(ckpt.read_bytes()).hexdigest(),
        "residual_head_params": n_res,
        "total_params": n_total,
        "best_val": float(best_val),
        "epochs_run": ep,
        "seed": SEED,
        "protocol": "D1/a6; Adam 1e-3/wd1e-5, cosine 60ep, batch 32, patience 10, "
                    "SCS train 2003-2010/val 2011-2015, lead 4wk, history 24wk",
        "scale": "frozen 0.4 (Work 1 SCS-val argmax CSI@0.9); no target-domain selection",
        "leakage": "causal v2 inputs<=t; anchor=y(t); target y(t+4) loss-only; "
                   "SCS-train normalization frozen",
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUT_DIR / "summary_B.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {OUT_DIR / 'summary_B.json'}")


if __name__ == "__main__":
    main()
