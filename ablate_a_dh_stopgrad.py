# -*- coding: utf-8 -*-
"""LEGACY / ABLATION SCRIPT — NOT the formal-model trainer.

This is the a5/a6 stop-gradient dual-head ablation. Its CONFIGS use
80 epochs / patience 20 and residual scale 0.3, which do NOT match the
formal protocol recorded in the paper (60 epochs / patience 10; TSAR
scale 0.4). The formal models are trained by
experiments/01_training/train_severe_classifier.py (TSCF, D1) and
experiments/01_training/train_physics_anchored.py (TSAR). This file is
shipped only because FinalSingleHeadModel (the single-head architecture
used by the frozen TSCF) is defined here.

Ablation (a5): stop-gradient dual-head.

Key fix vs a3 (dual-head): severe gradient is blocked from backbone via
detach on (1) severe_head input features and (2) y_main/gate in severe term.
Each head receives a clean, decoupled loss signal.

  total = focal_MSE(y_main, y)                                # backbone only
        + lambda_sev * MSE( y_main.d + gate.d * res, y, sev) # sev head only

Note: physics soft constraints (thermodynamic upper bound, recursive decay)
were REMOVED from the loss — they showed no training benefit (weights were 0.0
in the deployed a6 "nophys" model) and are dead code in this script.

Reference: fl_gamma05 -> test=0.9521, mild=0.608, severe=-2.255
"""
import os, sys, io, json, time, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent
# DATA_DIR is imported from paths.py below (env -> ./data -> author path).
OUT_ROOT = PROJECT_ROOT / "results" / "ablate_a_dh_stopgrad_20260911_lead4"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

from paths import DATA_DIR  # noqa: E402  (self-contained resolution)
from src.data.aligned_dataset import create_scs_dataloaders, SCSPIECFDataset
from src.models import PIECFModel

BACKBONE_EPOCHS = 80
BACKBONE_BATCH = 32
BACKBONE_LR = 1e-3
EARLY_STOP_PATIENCE = 20
RANDOM_SEED = 42
N_TOTAL_CHANNELS = 13
FOCAL_CAP = 10.0

ZONES = {"General": (0.0, 0.8), "Severe": (0.8, np.finfo(float).max)}

CONFIGS = {
    # CRW-standard DHW makes severe events (y>=0.8 = DHW>=8) much rarer
    # (SCS ~0.29%), so the sweep covers HIGHER lambda_sev than the old
    # distribution. sg_lam20 is kept as the paper-reference value.
    "sg_lam20": {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20},
    "sg_lam40": {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 40},
    "sg_lam80": {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 80},
    "sg_lam160": {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 160},
}


class StopGradDualHeadPIECFModel(nn.Module):
    """Dual-head where severe gradient cannot reach backbone (detach)."""

    def __init__(self, dynamic_channels=N_TOTAL_CHANNELS):
        super().__init__()
        self.backbone = PIECFModel(dynamic_channels=dynamic_channels)
        hid = self.backbone.regression_head[0].in_features
        self.severe_head = nn.Sequential(
            nn.Linear(hid, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1))

    def forward(self, x_dyn, x_stat, land_mask, nan_mask=None,
                scale=0.3, tau=0.5, temp=0.1):
        feat = self.backbone.forward_features(x_dyn, x_stat, land_mask, nan_mask)
        attn = feat["attn_out_reshaped"]                 # [B,H,W,T,C]
        y_main = self.backbone.regression_head(attn).squeeze(-1)   # [B,H,W,T]
        # detach input so severe gradient cannot update backbone
        res = torch.tanh(self.severe_head(attn.detach())).squeeze(-1) * scale
        gate = torch.sigmoid((y_main - tau) / temp)                 # [B,H,W,T]
        # permute to [B,T,H,W] to match target layout
        y_main_t = y_main.permute(0, 3, 1, 2)
        res_t   = res.permute(0, 3, 1, 2)
        gate_t  = gate.permute(0, 3, 1, 2)
        y_final = y_main_t + gate_t * res_t
        return y_final, y_main_t, gate_t, res_t          # all [B,T,H,W]


class FinalSingleHeadModel(PIECFModel):
    """MAIN MODEL for LTTS_CRW: single-head severe-event binary classifier.

    Architecture only — this class defines the single sigmoid head that
    the formal checkpoint uses. The formal checkpoint
    (results/model_final_Btail3/model.pt, MD5
    7cdbca26cae03fa53be53210940e8ce1) was trained with a class-weighted
    binary cross-entropy on the severe label y(t+4) >= 0.8 with
    pos_weight = 1000, so its output is P(severe) in [0, 1]. Training entry
    point: experiments/01_training/train_severe_classifier.py (pos_weight
    sweep: results/model_severe_classifier/posweight_sweep.json).

    NOTE: an earlier version of this docstring described the model as a
    "tail-weighted regression B(lambda=3, gamma=2)". That description was
    wrong: it referred to a superseded candidate (see
    experiments/01_training/train_tail_aware_regression.py) that is NOT the
    shipped checkpoint. The shipped file is byte-identical to
    results/model_severe_classifier/model.pt.

    Subclasses PIECFModel so the checkpoint loads directly (same state_dict
    keys). forward() re-emits the dual-head 4-tuple (y_final, y_main, gate,
    res) with gate=1, res=0, so existing evaluation scripts work unchanged:
    y_final == y_main == the single head's prediction.
    """

    def forward(self, x_dyn, x_stat, land_mask, nan_mask=None,
                scale=0.3, tau=0.5, temp=0.1):
        feat = super().forward_features(x_dyn, x_stat, land_mask, nan_mask)
        attn = feat["attn_out_reshaped"]                       # [B,H,W,T,C]
        y_main = self.regression_head(attn).squeeze(-1).permute(0, 3, 1, 2)  # [B,T,H,W]
        B, T, H, W = y_main.shape
        gate = torch.ones_like(y_main)   # gate=1, res=0 -> y_final == y_main
        res = torch.zeros_like(y_main)
        return y_main, y_main, gate, res


def focal_mse_loss(pred, y_clean, gamma, cap):
    abs_err = (pred - y_clean).abs()
    scale = abs_err.mean().detach() + 1e-8
    if gamma == 0:
        focal_w = torch.ones_like(pred)
    else:
        focal_w = (abs_err / scale).clamp(max=cap).detach() ** gamma
    return focal_w * (pred - y_clean) ** 2


def stopgrad_loss(out, y_lead, label_mask, cfg):
    """Decoupled: main loss -> backbone; severe loss -> severe_head only.

    Uses ONLY the last temporal position (y[:, -1, :, :]) to predict the
    lead-time target y_lead (single value at week window_end + lead_weeks).
    This converts the model from autoencoding (lead_weeks=0) to true
    forecasting (lead_weeks>0).
    """
    y_final, y_main, gate, res = out                   # [B,T,H,W]
    # Last position only — the forecast for the lead-time target
    y_final_last = y_final[:, -1, :, :]                 # [B,H,W]
    y_main_last  = y_main[:, -1, :, :]                  # [B,H,W]
    gate_last    = gate[:, -1, :, :]                    # [B,H,W]
    res_last     = res[:, -1, :, :]                     # [B,H,W]

    # valid = finite future target & labelled ocean cell. label_mask already
    # encodes isfinite(y_lead)&ocean from the dataset; the isfinite guard is a
    # belt-and-suspenders fix against any NaN y_lead reaching the loss.
    valid = (torch.isfinite(y_lead).float()
             * torch.nan_to_num(label_mask.float(), nan=0.0))
    y_clean = torch.nan_to_num(y_lead, nan=0.0)
    n_valid = valid.sum() + 1e-8

    # --- main focal loss (backbone only) ---
    err_main = focal_mse_loss(y_main_last, y_clean, cfg["gamma"], FOCAL_CAP)
    loss_main = (err_main * valid).sum() / n_valid

    # --- severe loss: detach y_main+gate so grad -> severe_head only ---
    sev_pred = y_main_last.detach() + gate_last.detach() * res_last
    sev_mask = (y_clean >= 0.8).float() * valid
    n_sev = sev_mask.sum() + 1e-8
    err_sev = (sev_pred - y_clean) ** 2
    sev_loss = (err_sev * sev_mask).sum() / n_sev

    loss_main = loss_main + cfg["lambda_sev"] * sev_loss
    return (torch.nan_to_num(loss_main, nan=1.0, posinf=1.0),
            torch.nan_to_num(sev_loss, nan=0.0))


def r2_calc(p, t, m):
    m = m > 0.5
    if m.sum() < 5:
        return float("nan")
    ss_res = np.sum((t[m] - p[m]) ** 2)
    ss_tot = np.sum((t[m] - t[m].mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0


def evaluate(model, loader, device, cfg):
    p_fused, p_main, targets, masks = [], [], [], []
    lm = torch.BoolTensor(loader.dataset.land_mask).to(device)
    with torch.no_grad():
        for batch in loader:
            x_dyn = batch["x_dynamic"].to(device)
            x_stat = batch["x_static"].to(device)
            nm = batch.get("nan_mask_dyn")
            if nm is not None:
                nm = nm.to(device)
            label = batch.get("label_mask")
            if label is not None:
                label = label.to(device)
            y = batch["y_lead"].to(device)                     # [B,H,W]
            # Only finite future targets are scored (label_mask already encodes
            # isfinite(y_lead)&ocean; the isfinite guard is defensive).
            valid = ((label.float() if label is not None else (~lm).float())
                     * torch.isfinite(y).float())
            if valid.ndim == 2:
                valid = valid.unsqueeze(0)
            out = model(x_dyn, x_stat, lm, nan_mask=nm,
                        scale=cfg["scale"], tau=cfg["tau"], temp=cfg["temp"])
            y_final, y_main, _, _ = out                        # [B,T,H,W]
            p_fused.append(y_final[:, -1].cpu().numpy())
            p_main.append(y_main[:, -1].cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(valid.cpu().numpy())
    return (np.concatenate(p_fused), np.concatenate(p_main),
            np.concatenate(targets), np.concatenate(masks))


def zone_report(P, T, M):
    out = {"global": {"r2": r2_calc(P, T, M), "n": int((M > 0.5).sum())}}
    for zname, (lo, hi) in ZONES.items():
        zm = (T >= lo) & (T < hi) & (M > 0.5)
        out[zname] = {"r2": r2_calc(P, T, zm.astype(float)),
                      "n": int(zm.sum()),
                      "y_mean": float(T[zm].mean()) if zm.sum() > 0 else None,
                      "yhat_mean": float(P[zm].mean()) if zm.sum() > 0 else None}
    return out

def train_one(cfg_name, cfg, loaders, device):
    train_loader, val_loader, test_loader = loaders
    out_dir = OUT_ROOT / f"exp_{cfg_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    model = StopGradDualHeadPIECFModel(dynamic_channels=N_TOTAL_CHANNELS).to(device)
    n_sev = sum(p.numel() for p in model.severe_head.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=BACKBONE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=BACKBONE_EPOCHS, eta_min=1e-5)
    best_val = float("inf")
    patience = 0
    best_state = None
    train_start = time.time()

    for epoch in range(1, BACKBONE_EPOCHS + 1):
        model.train()
        tl_sum, sl_sum, nb = 0.0, 0.0, 0
        for batch in train_loader:
            x_dyn = batch["x_dynamic"].to(device)
            x_stat = batch["x_static"].to(device)
            lm = batch["land_mask"].to(device)
            nm = batch.get("nan_mask_dyn")
            if nm is not None:
                nm = nm.to(device)
            label = batch.get("label_mask")
            if label is not None:
                label = label.to(device)
            y = batch["y_lead"].to(device)                     # [B,H,W]
            optimizer.zero_grad()
            out = model(x_dyn, x_stat, lm, nan_mask=nm,
                        scale=cfg["scale"], tau=cfg["tau"], temp=cfg["temp"])
            loss, sev = stopgrad_loss(out, y, label, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(loss):
                optimizer.step()
            tl_sum += loss.item(); sl_sum += sev.item(); nb += 1

        model.eval()
        vl_sum, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x_dyn = batch["x_dynamic"].to(device)
                x_stat = batch["x_static"].to(device)
                lm = batch["land_mask"].to(device)
                nm = batch.get("nan_mask_dyn")
                if nm is not None:
                    nm = nm.to(device)
                label = batch.get("label_mask")
                if label is not None:
                    label = label.to(device)
                y = batch["y_lead"].to(device)                     # [B,H,W]
                out = model(x_dyn, x_stat, lm, nan_mask=nm,
                            scale=cfg["scale"], tau=cfg["tau"], temp=cfg["temp"])
                loss, _ = stopgrad_loss(out, y, label, cfg)
                vl_sum += loss.item(); nv += 1
        scheduler.step()
        tl = tl_sum / max(nb, 1)
        vl = vl_sum / max(nv, 1)
        if vl < best_val:
            best_val = vl; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}  train={tl:.6f}  val={vl:.6f}  "
                  f"sev={sl_sum/max(nb,1):.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"t={time.time()-train_start:.0f}s", flush=True)
        if patience >= EARLY_STOP_PATIENCE:
            print(f"    Early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"    Best val = {best_val:.6f}  total = {time.time()-train_start:.0f}s  "
          f"[severe_head params={n_sev} / total={n_total}]", flush=True)

    tr_f, tr_m, tr_t, tr_mk = evaluate(model, train_loader, device, cfg)
    va_f, va_m, va_t, va_mk = evaluate(model, val_loader, device, cfg)
    te_f, te_m, te_t, te_mk = evaluate(model, test_loader, device, cfg)

    return {
        "cfg": cfg,
        "severe_head_params": n_sev,
        "total_params": n_total,
        "train_r2": r2_calc(tr_f, tr_t, tr_mk),
        "val_r2":   r2_calc(va_f, va_t, va_mk),
        "test_r2":  r2_calc(te_f, te_t, te_mk),
        "test_r2_mainhead": r2_calc(te_m, te_t, te_mk),
        "test_zone": zone_report(te_f, te_t, te_mk),
        "test_zone_mainhead": zone_report(te_m, te_t, te_mk),
        "out_dir": str(out_dir),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Ablation (a5): stop-gradient dual-head")
    print("=" * 70)
    print(f"Device: {device}   CONFIGS: {list(CONFIGS)}")
    t0 = time.time()
    loaders = create_scs_dataloaders(
        data_dir=DATA_DIR, batch_size=BACKBONE_BATCH, num_workers=0,
        random_split=False, random_seed=RANDOM_SEED, lead_weeks=4,
        year_based_split=True, train_years=(2003, 2010),
        val_years=(2011, 2015), test_years=(2016, 2019))
    print(f"Data loaders: {time.time()-t0:.1f}s  "
          f"train={len(loaders[0])} val={len(loaders[1])} test={len(loaders[2])}")

    all_results = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n[Config {cfg_name}] gamma={cfg['gamma']} lam={cfg['lambda_sev']}",
              flush=True)
        r = train_one(cfg_name, cfg, loaders, device)
        all_results[cfg_name] = {k: v for k, v in r.items() if k != "out_dir"}
        print(f"  => fused R2={r['test_r2']:.4f}  main-only R2={r['test_r2_mainhead']:.4f}")
        print(f"      fused zones " + " ".join(
            f"{z}={r['test_zone'][z]['r2']:+.3f}" for z in ZONES), flush=True)
        print(f"      main   zones " + " ".join(
            f"{z}={r['test_zone_mainhead'][z]['r2']:+.3f}" for z in ZONES), flush=True)

    summary = {
        "ablation": {k: v for k, v in all_results.items()},
        "best_test": max(all_results.items(), key=lambda kv: kv[1]["test_r2"])[0],
        "best_severe": max(
            ((k, v["test_zone"]["Severe"]["r2"]) for k, v in all_results.items()
             if not np.isnan(v["test_zone"]["Severe"]["r2"])),
            key=lambda kv: kv[1], default=(None, None)),
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUT_ROOT / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    ref_path = PROJECT_ROOT / "results" / "ablate_a_focal_20260715" / "ablation_summary.json"
    ref_g05 = None
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        ref_g05 = ref.get("ablation", {}).get("fl_gamma05")

    print("\n" + "=" * 70)
    print("SUMMARY - stop-gradient dual-head")
    print("=" * 70)
    hdr = f"{'Config':<20}{'train':>8}{'val':>8}{'fused':>8}{'m.head':>8}"
    for z in ZONES:
        hdr += f"{z:>10}"
    print(hdr)
    print("-" * 86)
    for k, v in all_results.items():
        row = (f"{k:<20}{v['train_r2']:>8.4f}{v['val_r2']:>8.4f}"
               f"{v['test_r2']:>8.4f}{v['test_r2_mainhead']:>8.4f}")
        for z in ZONES:
            zr = v["test_zone"][z]["r2"]
            row += f"{'+' if zr >= 0 else ''}{zr:>9.3f}" if not np.isnan(zr) else f"{'nan':>10}"
        print(row)
    if ref_g05 is not None:
        row = (f"{'<ref> fl_gamma05':<20}{ref_g05['train_r2']:>8.4f}"
               f"{ref_g05['val_r2']:>8.4f}{ref_g05['test_r2']:>8.4f}{'—':>8}")
        for z in ZONES:
            zr = ref_g05["test_zone"][z]["r2"]
            row += f"{'+' if zr >= 0 else ''}{zr:>9.3f}" if not np.isnan(zr) else f"{'nan':>10}"
        print(row)
    print(f"\nBest test R2   : {summary['best_test']}")
    print(f"Best severe R2 : {summary['best_severe']}")
    print(f"Saved: {OUT_ROOT / 'ablation_summary.json'}")
    print(f"Total wall time: {time.time()-t0:.0f}s")
    print("DONE")


if __name__ == "__main__":
    main()
