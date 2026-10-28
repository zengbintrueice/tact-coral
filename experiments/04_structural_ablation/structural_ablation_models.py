# -*- coding: utf-8 -*-
"""Structural ablation models for Thermal-Stress-Anchored Residual Learning.

Four structures compared on the SAME PFV record set (SCS/GBR/CARIB):

  A  PersA_DHW                      y_hat = PersA(t)                       (no model)
  B  Un-gated bounded residual      y_hat = PersA(t) + tanh(R(attn))·scale (gate ≡ 1)
  C  Formal anchored residual       y_hat = PersA(t) + sigmoid((y_main−τ)/T)
                                                       · tanh(R(attn))·scale
  D  Frozen TSCF                    (original single-head model, no anchor)

B and C share the EXACT same backbone (PIECFModel), residual head
(Linear(hid,8)-ReLU-Dropout-Linear(8,1)), stop-gradient discipline
(R consumes attn.detach()), loss structure (focal y_main -> backbone,
lambda_sev·MSE(PersA+res, y, severe) -> residual head), and scale. The ONLY
difference is C's gate multiplier (sigmoid((y_main−τ)/T)) on the residual.

Loss (B, ungated — the clean ablation of C):
    total = focal_MSE(y_main_last, y)                       # backbone only
          + lambda_sev·MSE(PersA + res_last, y, sev_mask)   # residual head only

Strict / no leakage: causal v2 inputs <= issue t; anchor = y_regression at t
(available at t); target y(t+4) loss-only; SCS-train-only normalization
(baked in backbone channel_norm); no target-domain fitting.
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import PIECFModel

N_TOTAL_CHANNELS = 13
FOCAL_CAP = 10.0


class UnGatedAnchoredResidualModel(nn.Module):
    """Model B: y_final = PersA + tanh(R(attn.detach()))·scale. gate ≡ 1.

    Identical to PhysicsAnchoredResidualModel except the residual is applied
    unconditionally (no gate). Used to test whether the gate contributes any
    methodological value beyond a bounded PersA-correction."""

    def __init__(self, dynamic_channels=N_TOTAL_CHANNELS):
        super().__init__()
        self.backbone = PIECFModel(dynamic_channels=dynamic_channels)
        hid = self.backbone.regression_head[0].in_features
        self.residual_head = nn.Sequential(
            nn.Linear(hid, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1))

    def forward(self, x_dyn, x_stat, land_mask, pers_anchor, nan_mask=None,
                scale=0.4):
        B, T, H, W, _ = x_dyn.shape
        if pers_anchor is None:
            pers_anchor = torch.zeros(B, H, W, device=x_dyn.device)
        pers_anchor = torch.nan_to_num(pers_anchor.detach(), nan=0.0)   # [B,H,W]
        feat = self.backbone.forward_features(x_dyn, x_stat, land_mask, nan_mask)
        attn = feat["attn_out_reshaped"]                 # [B,H,W,T,C]
        y_main = self.backbone.regression_head(attn).squeeze(-1)   # [B,H,W,T]
        res = torch.tanh(self.residual_head(attn.detach())).squeeze(-1) * scale
        y_main_t = y_main.permute(0, 3, 1, 2)            # [B,T,H,W]
        res_t = res.permute(0, 3, 1, 2)
        anchor = pers_anchor.detach().unsqueeze(1).expand(B, T, H, W)
        y_final = anchor + res_t                         # gate ≡ 1
        return y_final, y_main_t, res_t


def focal_mse_loss(pred, y_clean, gamma, cap=FOCAL_CAP):
    abs_err = (pred - y_clean).abs()
    scale = abs_err.mean().detach() + 1e-8
    if gamma == 0:
        focal_w = torch.ones_like(pred)
    else:
        focal_w = (abs_err / scale).clamp(max=cap).detach() ** gamma
    return focal_w * (pred - y_clean) ** 2


def ungated_anchored_loss(out, y_lead, label_mask, cfg):
    """Model B loss: backbone from focal y_main; residual head from
    MSE(PersA + res, y, severe). gate ≡ 1."""
    y_final, y_main, res = out                    # [B,T,H,W]
    y_final_last = y_final[:, -1, :, :]
    y_main_last = y_main[:, -1, :, :]
    res_last = res[:, -1, :, :]

    valid = (torch.isfinite(y_lead).float()
             * torch.nan_to_num(label_mask.float(), nan=0.0))
    y_clean = torch.nan_to_num(y_lead, nan=0.0)
    n_valid = valid.sum() + 1e-8

    err_main = focal_mse_loss(y_main_last, y_clean, cfg["gamma"], FOCAL_CAP)
    loss_main = (err_main * valid).sum() / n_valid

    anchor_det = torch.nan_to_num((y_final_last - res_last).detach(), nan=0.0)  # PersA
    sev_pred = anchor_det + res_last
    sev_mask = (y_clean >= 0.8).float() * valid
    n_sev = sev_mask.sum() + 1e-8
    err_sev = (sev_pred - y_clean) ** 2
    sev_loss = (err_sev * sev_mask).sum() / n_sev

    loss_total = loss_main + cfg["lambda_sev"] * sev_loss
    return (torch.nan_to_num(loss_total, nan=1.0, posinf=1.0),
            torch.nan_to_num(sev_loss, nan=0.0))
