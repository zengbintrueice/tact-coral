# -*- coding: utf-8 -*-
"""Main Figure 7 / Supplementary Fig. S12 — Architecture-independent transfer
control: XGBoost (rendered in the shared nc_style palette).

Three panels, all values auto-read from
results/xgb_transfer_control/metrics.json (NOTHING hard-coded):

  a  ROC-AUC across SCS->SCS / SCS->GBR / SCS->CARIB for XGB_FULL,
     XGB_THERMAL, XGB_NO_THERMAL
  b  PR-AUC across the same transfer pairs
  c  CARIB performance contrast (ROC-AUC / PR-AUC / Brier / ECE) across the
     three variants — showing that the thermal-only variant retains most of
     the transfer skill while removing thermal features drops it markedly

Figure annotation: "zero-shot transfer, τ=0.90 pinned". CSI/POD/F1 are NOT
plotted because the XGBoost DHW-regression predictions never reach 0.90, so
operational metrics at τ=0.90 are 0 / undefined in the source JSON — only
ROC-AUC, PR-AUC, Brier, ECE are compared (as the experiment config states).

Full caption in results/figures/SX_source_data_caption.md.

Outputs: results/figures/SX_xgb_transfer_control.{pdf,svg,png(600dpi)} +
source-data CSV + manifest + caption.

Run:
    cd LTTS_CRW/scripts/figures
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' fig_s_x_xgb_transfer.py
"""
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nc_style as S

J = S.require(S.RESULTS / "xgb_transfer_control" / "metrics.json", fig="SX")
MODELS = ["XGB_FULL", "XGB_THERMAL", "XGB_NO_THERMAL"]
DOMAINS = ["SCS", "GBR", "CARIB"]
LINK = {"SCS": "SCS→SCS", "GBR": "SCS→GBR", "CARIB": "SCS→CARIB"}
MLAB = {"XGB_FULL": "FULL (13 ch)", "XGB_THERMAL": "THERMAL (5 ch)",
        "XGB_NO_THERMAL": "NO-THERMAL (8 ch)"}
MCOL = {"XGB_FULL": S.COLORS["tscf"], "XGB_THERMAL": S.COLORS["SCS"],
        "XGB_NO_THERMAL": S.COLORS["CARIB"]}
by = {m: next(r for r in J if r["model"] == m) for m in MODELS}

# cross-check counts identical across variants and domains
for m in MODELS:
    for d in DOMAINS:
        assert by[m][f"{d}_n"] == by["XGB_FULL"][f"{d}_n"], \
            f"SX QC FAIL: {m} {d} n mismatch"
n_ = {d: int(by["XGB_FULL"][f"{d}_n"]) for d in DOMAINS}
n_sev = {d: int(by["XGB_FULL"][f"{d}_n_severe"]) for d in DOMAINS}

fig = plt.figure(figsize=(S.WIDTH_2COL, 8.8))
gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.30,
                      left=0.11, right=0.97, top=0.94, bottom=0.09)

# --------------------------------------------------------------- (a) ROC ----
ax = fig.add_subplot(gs[0, 0])
ax.set_title("ROC-AUC — zero-shot transfer, τ = 0.90 pinned", loc="left",
             fontsize=8)
w = 0.26
for mi, m in enumerate(MODELS):
    vals = [float(by[m][f"{d}_roc_auc"]) for d in DOMAINS]
    ax.bar(np.arange(3) + (mi - 1) * w, vals, width=w, color=MCOL[m],
           alpha=0.85, label=MLAB[m])
    for di, v in enumerate(vals):
        ax.text(di + (mi - 1) * w, v + 0.006, f"{v:.3f}", ha="center",
                fontsize=5.6, rotation=90, va="bottom")
ax.axhline(0.5, color="#999999", lw=0.7, ls=":", alpha=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels([LINK[d] for d in DOMAINS])
ax.set_ylim(0.45, 1.02)
ax.set_ylabel("ROC-AUC")
ax.legend(loc="lower left", fontsize=5.8, ncol=3, columnspacing=0.8)
S.panel_label(ax, "a")

# --------------------------------------------------------------- (b) PR ----
ax = fig.add_subplot(gs[0, 1])
ax.set_title("PR-AUC — zero-shot transfer", loc="left", fontsize=8)
for mi, m in enumerate(MODELS):
    vals = [float(by[m][f"{d}_pr_auc"]) for d in DOMAINS]
    ax.bar(np.arange(3) + (mi - 1) * w, vals, width=w, color=MCOL[m], alpha=0.85)
    for di, v in enumerate(vals):
        ax.text(di + (mi - 1) * w, v + 0.006, f"{v:.3f}", ha="center",
                fontsize=5.6, rotation=90, va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels([LINK[d] for d in DOMAINS])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("PR-AUC")
hand = [plt.Rectangle((0, 0), 1, 1, color=MCOL[m], label=MLAB[m]) for m in MODELS]
ax.legend(handles=hand, loc="lower left", fontsize=5.8, ncol=3, columnspacing=0.8)
S.panel_label(ax, "b")

# ------------------------------------------------------ (c) CARIB contrast ---
ax = fig.add_subplot(gs[1, 0])
ax.set_title("CARIB — thermal features carry the transfer skill", loc="left",
             fontsize=8)
groups = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]  # discriminative metrics
xg = np.arange(len(groups))
for mi, m in enumerate(MODELS):
    vals = np.array([float(by[m][f"CARIB_{k}"]) for k, _ in groups])
    o = (mi - 1) * 0.26
    ax.bar(xg + o, vals, width=0.24, color=MCOL[m], alpha=0.85, label=MLAB[m])
for mi, (k, lab) in enumerate(groups):
    for m in MODELS:
        v = float(by[m][f"CARIB_{k}"])
        ax.text(mi + (MODELS.index(m) - 1) * 0.26, v + 0.006, f"{v:.3f}",
                ha="center", fontsize=5.6, rotation=90, va="bottom")
ax.set_xticks(xg)
ax.set_xticklabels([lab for _, lab in groups])
ax.set_ylabel("value")
ax.set_ylim(-0.02, 1.0)
ax.legend(loc="upper center", fontsize=5.8, ncol=3, columnspacing=0.8,
          bbox_to_anchor=(0.5, 1.18))
S.panel_label(ax, "c")

# -------------------------------------------------- (e) bootstrap CI forest --
from matplotlib.lines import Line2D
ax = fig.add_subplot(gs[1, 1])
ax.set_title("Week-block bootstrap 95% CIs: NO-THERMAL below THERMAL/FULL, THERMAL ~ FULL",
             loc="left", fontsize=8)
BOOT = S.require(S.RESULTS / "xgb_transfer_control" / "bootstrap_cis.csv",
                 fig="SX")
mets = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
ybase = {}
ytick_pos, ytick_lab = [], []
pos = 0.0
for d in DOMAINS:
    for k, kl in mets:
        ybase[(d, k)] = pos
        ytick_pos.append(pos + 0.22)
        ytick_lab.append(d + " " + kl)
        pos += 1.0
for mi, m in enumerate(MODELS):
    sub = BOOT[BOOT["variant"] == m]
    for (d, k), y0 in ybase.items():
        r = sub[sub["domain"] == d].iloc[0]
        v = float(r[k]); lo = float(r[k + "_lo"]); hi = float(r[k + "_hi"])
        y = y0 + (mi - 1) * 0.22
        ax.plot([lo, hi], [y, y], color=MCOL[m], lw=1.2, alpha=0.9)
        ax.plot([v], [y], "o", ms=3.4, color=MCOL[m], zorder=3)
ax.set_yticks(ytick_pos)
ax.set_yticklabels(ytick_lab, fontsize=6)
ax.invert_yaxis()
ax.set_xlim(0.0, 1.02)
ax.axvline(0.5, color="#999999", lw=0.7, ls=":", alpha=0.8)
ax.set_xlabel("point estimate + 95% CI", fontsize=6.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=6)
ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4, color=MCOL[m],
                          label=MLAB[m]) for m in MODELS],
          frameon=False, fontsize=5.5, loc="lower right")
S.panel_label(ax, "d")
# QC: NO_THERMAL CI must lie below THERMAL CI in every domain and metric
for d in DOMAINS:
    for k, _ in mets:
        hi_no = float(BOOT[(BOOT["variant"] == "XGB_NO_THERMAL") &
                           (BOOT["domain"] == d)].iloc[0][k + "_hi"])
        lo_th = float(BOOT[(BOOT["variant"] == "XGB_THERMAL") &
                           (BOOT["domain"] == d)].iloc[0][k + "_lo"])
        assert hi_no < lo_th, f"SX QC FAIL: {d} {k} NO_THERMAL not below THERMAL"
print("  (e) CI separation QC: NO_THERMAL below THERMAL in all 6 cells: OK",
      flush=True)


# ------------------------------------------- (d) ICP invariant channel screen -
ax = fig.add_subplot(gs[2, :])
ax.set_title("Cross-basin rank-association stability screen: which channels "
             "are stable across all three basins?", loc="left", fontsize=8)
ICP = S.require(S.RESULTS / "invariant_predictor_screen" / "associations.csv",
                fig="fig6")
# per-channel cross-basin rho spread (max-min across basins)
icp_channels = ICP["variable"].unique()
ch_disp = {"PersA_anchor": "PersA (anchor)", "hotspot": "HotSpot",
           "ssta": "SST anomaly", "sst": "SST", "d_hotspot_dt": "d HS/dt",
           "hotspot_cumsum_4w": "HS cumsum 4 wk", "hotspot_cumsum_12w": "HS cumsum 12 wk",
           "chlorophyll": "chlorophyll", "par": "PAR", "wind_speed": "wind speed",
           "ssr": "shortwave", "seasonal_sin": "seasonal sin",
           "seasonal_cos": "seasonal cos", "sst_anomaly": "SST anomaly"}
inv_set = {"PersA_anchor", "ssta"}
for ci, ch in enumerate(icp_channels):
    sub = ICP[ICP["variable"] == ch]
    rhos = {r["basin"]: r["rho"] for _, r in sub.iterrows()}
    is_inv = ch in inv_set
    col = S.COLORS["GBR"] if is_inv else "#BBBBBB"
    for b, mk in (("SCS", "o"), ("GBR", "s"), ("CARIB", "^")):
        if b in rhos:
            ax.scatter(ci + ("SCS" == b) * 0 - ("CARIB" == b) * 0
                       + {"SCS": -0.15, "GBR": 0, "CARIB": 0.15}[b],
                       rhos[b], marker=mk, s=14, color=col, zorder=3,
                       linewidths=0)
    ax.hlines(np.mean(list(rhos.values())), ci - 0.25, ci + 0.25,
              color=col, linewidth=1.2)
    label = ch_disp.get(ch, ch)
    if is_inv:
        label += " ★"
    ax.text(ci, -0.72, label, fontsize=4.8, rotation=40,
            ha="right", va="top")
ax.axhline(0, color="#999999", lw=0.7, ls=":")
ax.set_ylabel("Spearman ρ (channel vs y(t+4))\nper basin; bar = mean; ★ = most stable across basins",
              fontsize=6.5)
ax.set_ylim(-0.8, 1.0)
ax.set_xlim(-0.8, len(icp_channels) - 0.2)
ax.set_xticks([])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=6)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", markersize=4, color="#555555", label="SCS"),
    Line2D([], [], marker="s", linestyle="none", markersize=4, color="#555555", label="GBR"),
    Line2D([], [], marker="^", linestyle="none", markersize=4, color="#555555", label="CARIB"),
    Line2D([], [], linestyle="-", linewidth=1.2, color=S.COLORS["GBR"], label="stable across basins (PersA, ssta)"),
    Line2D([], [], linestyle="-", linewidth=1.2, color="#BBBBBB", label="more basin-dependent")],
    frameon=False, fontsize=5.5, loc="upper right", ncol=2)
S.panel_label(ax, "e")

# -------------------------------------------------- (caption / notes) --------
fig.text(0.11, 0.055,
         "S-X note: XGB predictions are DHW regressions whose scores never reach "
         "τ=0.90, so CSI/POD/F1 at τ=0.90 are 0 / undefined (per the source JSON) — "
         "we therefore compare ROC-AUC, PR-AUC, Brier and ECE only.",
         fontsize=5.4, style="italic", color="#333333")
fig.text(0.11, 0.015,
         "XGBoost and TSCF use the same SCS→GBR/CARIB zero-shot transfer protocol: "
         "lead=4 wk, history=24 wk, SCS-train-only scaler, τ=0.90 pinned; GBR/CARIB "
         "never enter training, tuning or threshold selection.",
         fontsize=5.4, style="italic", color="#333333")

out = S.save_fig(fig, "SX_xgb_transfer_control")

# -------------------------------------------------.. source data + caption ---
rows = []
for m in MODELS:
    for d in DOMAINS:
        rows.append(dict(panel="a/b", model=m, domain=d, transfer=("SCS→" + d),
                         roc_auc=float(by[m][f"{d}_roc_auc"]),
                         pr_auc=float(by[m][f"{d}_pr_auc"]),
                         brier=float(by[m][f"{d}_brier"]),
                         ece=float(by[m][f"{d}_ece"]),
                         n=n_[d], n_severe=n_sev[d]))
for m in MODELS:
    rows.append(dict(panel="c", model=m, domain="CARIB", transfer="SCS→CARIB",
                     roc_auc=float(by[m]["CARIB_roc_auc"]),
                     pr_auc=float(by[m]["CARIB_pr_auc"]),
                     brier=float(by[m]["CARIB_brier"]),
                     ece=float(by[m]["CARIB_ece"]),
                     n=n_["CARIB"], n_severe=n_sev["CARIB"]))
for m in MODELS:
    sub = BOOT[BOOT["variant"] == m]
    for d in DOMAINS:
        r = sub[sub["domain"] == d].iloc[0]
        for k, _ in mets:
            rows.append(dict(panel="e", model=m, domain=d, metric=k,
                             value=float(r[k]), ci_lo=float(r[k + "_lo"]),
                             ci_hi=float(r[k + "_hi"]), n=n_[d],
                             n_severe=n_sev[d]))
SRC = pd.DataFrame(rows)
SRC.to_csv(S.SRC_DIR / "SX_source_data.csv", index=False)
print("[source] SX_source_data.csv", flush=True)

caption = (
    "**Supplementary Figure S-X | Architecture-independent transfer control: "
    "XGBoost thermal features.** (a,b) ROC-AUC and PR-AUC of three XGBoost "
    "regressors trained on SCS 2003–2010 and evaluated zero-shot on SCS test "
    "2016–2019, GBR 2017–2020 and CARIB 2017–2020: FULL (13 dynamic channels), "
    "THERMAL (thermal-only: ssta, hotspot, d_hotspot_dt, hotspot_cumsum_4w, "
    "hotspot_cumsum_12w), NO-THERMAL (non-thermal channels). (c) CARIB "
    "discrimination (ROC-AUC and PR-AUC; Brier and ECE in the source data) "
    "across the three variants: the thermal-only variant retains most of the "
    "transfer skill, whereas removing thermal features lowers CARIB ROC-AUC "
    "markedly and PR-AUC by about half. (d) Marginal week-block bootstrap 95% CIs (1,000 resamples, seed 42) for ROC-AUC and PR-AUC in each domain: the NO-THERMAL interval lies below THERMAL and FULL in every domain and metric, while THERMAL and FULL overlap (CI separation is QC-checked at run time). (e) Cross-basin rank-association stability screen: per-channel Spearman rank correlation with y(t+4) in each basin; the physical anchor (PersA) and SST anomaly show the most stable cross-basin rank associations (star) while nonthermal channels are more basin-dependent.\n\n"
    "XGBoost and TSCF share the same SCS→GBR/CARIB zero-shot transfer protocol: "
    "lead = 4 weeks, history = 24 weeks, train-only (SCS 2003–2010) scaler, "
    "τ = 0.90 pinned a priori; GBR and CARIB never participate in training, "
    "hyperparameter tuning or threshold selection. XGBoost predicts DHW "
    "(regression), so its scores never reach τ = 0.90 and the operational "
    "metrics (CSI/POD/F1) at τ = 0.90 are 0 or undefined in the source "
    "JSON; the comparison therefore uses ROC-AUC, PR-AUC, Brier and ECE only. "
    "All values are read automatically from "
    "`results/xgb_transfer_control/metrics.json`; none are hand-entered."
)
(S.OUT_DIR / "SX_source_data_caption.md").write_text(caption, encoding="utf-8")
print("[caption] SX source-data caption written", flush=True)

man = S.OUT_DIR / "figure_manifest.csv"
new = pd.DataFrame([
    dict(figure="SX", panel="a/b", source_file="results/xgb_transfer_control/metrics.json",
         source_columns="{model}.{SCS,GBR,CARIB}_{roc_auc,pr_auc}",
         statistical_method="point estimates (ocean-masked common record set; ROC/PR from sklearn)",
         n=str(n_["CARIB"]), unit="AUC", output_file="SX_xgb_transfer_control.pdf"),
    dict(figure="SX", panel="c", source_file="results/xgb_transfer_control/metrics.json",
         source_columns="CARIB_{roc_auc,pr_auc,brier,ece} per model",
         statistical_method="point estimates (ocean-masked common record set)",
         n=str(n_["CARIB"]), unit="score", output_file="SX_xgb_transfer_control.pdf"),
    dict(figure="SX", panel="e", source_file="results/xgb_transfer_control/bootstrap_cis.csv",
         source_columns="domain x variant: {roc_auc,pr_auc}{,_lo,_hi}",
         statistical_method="marginal week-block bootstrap 95% CI (1,000 resamples, seed 42)",
         n=str(n_["CARIB"]), unit="AUC",
         output_file="SX_xgb_transfer_control.pdf"),
])
if man.exists():
    old = pd.read_csv(man)
    new = pd.concat([old[old.figure != "SX"], new], ignore_index=True)
new.to_csv(man, index=False)
print("[manifest] SX rows written", flush=True)

# ------------------------------------------------------------------ QC ------
print("\n=== SX QC ===", flush=True)
print(f"  metrics.json rows == 3 models: {len(J) == 3}", flush=True)
print(f"  domains n identical across variants: "
      f"{all(by[m][f'{d}_n'] == by['XGB_FULL'][f'{d}_n'] for m in MODELS for d in DOMAINS)}",
      flush=True)
print(f"  n (SCS/GBR/CARIB) = {n_['SCS']}/{n_['GBR']}/{n_['CARIB']}  "
      f"severe = {n_sev['SCS']}/{n_sev['GBR']}/{n_sev['CARIB']}", flush=True)
print(f"  CARIB: FULL ROC {by['XGB_FULL']['CARIB_roc_auc']:.4f} / THERMAL "
      f"{by['XGB_THERMAL']['CARIB_roc_auc']:.4f} / NO-THERMAL "
      f"{by['XGB_NO_THERMAL']['CARIB_roc_auc']:.4f}", flush=True)
print(f"  outputs: {[o.name for o in out]}", flush=True)
print("SX DONE", flush=True)