# -*- coding: utf-8 -*-
"""Physics-anchored residual head (proposed for cross-basin generalization).

Design (mirrors a6 stop-gradient dual-head, with the ANCHOR replaced):

  a6:   y_final = y_main + gate * residual          (anchor = model's own head)
  here: y_final = PersA  + gate * residual          (anchor = PHYSICS baseline)

where
  PersA(t)  = clip(DHW(t)/10, 0, 1)  = y_regression at ISSUE time t  [B,H,W]
              (current-DHW persistence, basin-invariant BY CONSTRUCTION:
               CRW HotSpot = SST - local climatological maximum; 84-day
               trailing accumulation) — passed in as a fixed input, no grads.
  residual  = tanh(residual_head(attn.detach())) * scale    ∈ [-scale, +scale]
              (bounded correction learned by the residual head only)
  gate      = sigmoid((y_main - tau) / temp)                ∈ [0, 1]
              (trust weight from the backbone's own prediction)

CRW / no-leakage guarantees:
  * All inputs are observations timestamped <= issue time t (causal v2 data:
    trailing 24-week window, month-end hold for monthly products, climatology
    year<=2010, SCS-train-only normalization inside the backbone).
  * PersA(t) uses only DHW available at t (84-day trailing window) — it is the
    persistence baseline, NOT a future observation.
  * Target y_lead = y(t+4) is used ONLY in the loss / evaluation, never as input.
  * residual head input is detached so its gradients cannot alter the backbone
    (identical detach discipline to a6).

Loss (mirrors a6 stopgrad_loss with the anchor swapped):
    total = focal_MSE(y_main_last, y)                      # backbone only
          + lambda_sev * MSE(PersA + gate.detach()*res_last, y, sev_mask)
                                                           # residual head only
  -> on SCS the residual learns to correct PersA toward the true y(t+4);
  -> on an unseen basin the bounded residual + persistence anchor keep the
     output within ±scale of PersA (no unbounded collapse).

The 'floor = persistence' property is APPROXIMATE here (two-sided bounded
residual, |res|<=scale): worst case degrades PersA by <= scale*gate.
"""
import sys, io, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import PIECFModel

N_TOTAL_CHANNELS = 13
FOCAL_CAP = 10.0


class PhysicsAnchoredResidualModel(nn.Module):
    """y_final = PersA + gate * bounded-residual; residual grad blocked from
    backbone; anchor is a fixed (no-grad) physics input."""

    def __init__(self, dynamic_channels=N_TOTAL_CHANNELS):
        super().__init__()
        self.backbone = PIECFModel(dynamic_channels=dynamic_channels)
        hid = self.backbone.regression_head[0].in_features
        self.residual_head = nn.Sequential(
            nn.Linear(hid, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1))

    def forward(self, x_dyn, x_stat, land_mask, pers_anchor, nan_mask=None,
                scale=0.3, tau=0.5, temp=0.1):
        """pers_anchor: [B,H,W] = y_regression at the window's last input step
        (issue time t), i.e. clip(DHW(t)/10,0,1). Detached internally."""
        B, T, H, W, _ = x_dyn.shape
        if pers_anchor is None:
            # fallback: zero anchor (no physics anchoring) — not for production
            pers_anchor = torch.zeros(B, H, W, device=x_dyn.device)
        pers_anchor = torch.nan_to_num(pers_anchor.detach(), nan=0.0)  # [B,H,W]
        feat = self.backbone.forward_features(x_dyn, x_stat, land_mask, nan_mask)
        attn = feat["attn_out_reshaped"]                 # [B,H,W,T,C]
        y_main = self.backbone.regression_head(attn).squeeze(-1)   # [B,H,W,T]
        # bounded residual, input detached -> grads go to residual_head only
        res = torch.tanh(self.residual_head(attn.detach())).squeeze(-1) * scale
        gate = torch.sigmoid((y_main - tau) / temp)               # [B,H,W,T]
        # permute to [B,T,H,W]
        y_main_t = y_main.permute(0, 3, 1, 2)
        res_t = res.permute(0, 3, 1, 2)
        gate_t = gate.permute(0, 3, 1, 2)
        anchor = pers_anchor.detach().unsqueeze(1).expand(B, T, H, W)  # [B,T,H,W]
        y_final = anchor + gate_t * res_t
        return y_final, y_main_t, gate_t, res_t          # all [B,T,H,W]


def focal_mse_loss(pred, y_clean, gamma, cap=FOCAL_CAP):
    abs_err = (pred - y_clean).abs()
    scale = abs_err.mean().detach() + 1e-8
    if gamma == 0:
        focal_w = torch.ones_like(pred)
    else:
        focal_w = (abs_err / scale).clamp(max=cap).detach() ** gamma
    return focal_w * (pred - y_clean) ** 2


def anchored_loss(out, y_lead, label_mask, cfg):
    """Decoupled loss with the physics anchor.

    - main term (backbone only): focal MSE of y_main_last vs y_lead.
    - severe term (residual head only): MSE of
        PersA + gate_last.detach()*res_last  vs  y_lead  on the severe mask.
    """
    y_final, y_main, gate, res = out              # [B,T,H,W]
    y_final_last = y_final[:, -1, :, :]           # [B,H,W]
    y_main_last = y_main[:, -1, :, :]
    gate_last = gate[:, -1, :, :]
    res_last = res[:, -1, :, :]

    valid = (torch.isfinite(y_lead).float()
             * torch.nan_to_num(label_mask.float(), nan=0.0))
    y_clean = torch.nan_to_num(y_lead, nan=0.0)
    n_valid = valid.sum() + 1e-8

    # main focal loss (backbone only)
    err_main = focal_mse_loss(y_main_last, y_clean, cfg["gamma"], FOCAL_CAP)
    loss_main = (err_main * valid).sum() / n_valid

    # residual term: gate + res detached so grads -> residual_head only.
    # reconstruct the physics anchor (no grads) from y_final = anchor + gate*res;
    # NaN anchor (land / no-current-DHW cells) filled with 0 — those cells are
    # excluded by label_mask anyway (NaN*0 would otherwise poison the sum).
    anchor_det = torch.nan_to_num((y_final_last - gate_last * res_last).detach(),
                                  nan=0.0)   # = PersA(t), no grads
    sev_pred = anchor_det + gate_last.detach() * res_last
    sev_mask = (y_clean >= 0.8).float() * valid
    n_sev = sev_mask.sum() + 1e-8
    err_sev = (sev_pred - y_clean) ** 2
    sev_loss = (err_sev * sev_mask).sum() / n_sev

    loss_total = loss_main + cfg["lambda_sev"] * sev_loss
    return (torch.nan_to_num(loss_total, nan=1.0, posinf=1.0),
            torch.nan_to_num(sev_loss, nan=0.0))
