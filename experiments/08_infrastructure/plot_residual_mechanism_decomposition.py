# plot_residual_mechanism_decomposition.py
#
# Main-figure candidate panels for the residual mechanism decomposition.
# ALL values are read from results/residual_mechanism_decomposition/
# (associations.csv, bootstrap.csv, mechanism_contribution.csv,
#  panel_d_scores.npz) — nothing hand-entered.
#
#   Panel a: partial Spearman (controlling current DHW) of R with candidate
#            mechanism variables, per basin, 95% week-block bootstrap CI
#   Panel b: ΔR² of the nested increments M0→M1→M2 with 95% CI
#   Panel c: mechanism-contribution heatmap (basins × variables, partial rho)
#   Panel d: incremental ROC (and PR) for PersA → +thermal → +context → TSAR
#
# Run (piecf_env):
#   & 'D:\...\piecf_env\Scripts\python.exe' plot_residual_mechanism_decomposition.py

import io, sys, json, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "residual_mechanism_decomposition"

BASIN_ORDER = ["SCS", "GBR", "CARIB"]
BASIN_COLOR = {"SCS": "#3B6FB6", "GBR": "#7B5EA7", "CARIB": "#C0504D"}
VAR_LABEL = {"current_DHW": "current DHW", "dDHW_4w": "ΔDHW (4 wk)",
             "d_hotspot_dt": "d hotspot/dt", "cumsum_4w": "HS cumsum 4 wk",
             "cumsum_12w": "HS cumsum 12 wk", "ssta": "SST anomaly",
             "hotspot": "HotSpot", "chl": "chlorophyll", "par": "PAR",
             "wind": "wind speed"}
NONTH = {"chl", "par", "wind"}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


# ── load ──────────────────────────────────────────────────────────────
assoc = pd.read_csv(RES / "associations.csv")
boot = pd.read_csv(RES / "bootstrap.csv")
mech = pd.read_csv(RES / "mechanism_contribution.csv")
scores = np.load(RES / "panel_d_scores.npz")

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.patch.set_facecolor("white")

# ── Panel a: partial Spearman controlling current DHW ────────────────
ax = axes[0, 0]
part = assoc[(assoc["type"] == "partial_ctrl_DHW")
             & (assoc["variable"] != "current_DHW")]
vars_order = ["dDHW_4w", "d_hotspot_dt", "cumsum_4w", "cumsum_12w",
              "ssta", "hotspot", "chl", "par", "wind"]
nb = len(BASIN_ORDER)
w = 0.8 / nb
for k, b in enumerate(BASIN_ORDER):
    sub = part[part["basin"] == b].set_index("variable").reindex(vars_order)
    xs = np.arange(len(vars_order)) + (k - (nb - 1) / 2) * w
    colors = ["#B03A2E" if v in NONTH else "#2E4053" for v in vars_order]
    ax.bar(xs, sub["rho"], width=w * 0.9, color=[c for c in colors],
           alpha=0.55 if b == "SCS" else 1.0,
           edgecolor="white", linewidth=0.4,
           label=b if k >= 0 else None)
    ax.errorbar(xs, sub["rho"],
                yerr=[sub["rho"] - sub["ci_lo"], sub["ci_hi"] - sub["rho"]],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=1.5, alpha=0.7)
    # FDR significance markers
    for x, rho, q in zip(xs, sub["rho"], sub["p_fdr"]):
        if q < 0.05:
            ax.text(x, rho + (0.012 if rho >= 0 else -0.02), "*", ha="center",
                    fontsize=9, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(vars_order)))
ax.set_xticklabels([VAR_LABEL[v] for v in vars_order], rotation=35, ha="right",
                   fontsize=8.5)
ax.set_ylabel("partial Spearman ρ (controlling current DHW)", fontsize=9.5)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.set_title("(a) Residual R vs mechanism variables\n(partial | current DHW; "
             "* FDR q<0.05; dark=thermal, red=nonthermal)", fontsize=10,
             fontweight="bold", loc="left")
style_ax(ax)

# ── Panel b: nested ΔR² increments ───────────────────────────────────
ax = axes[0, 1]
steps = [("deltaR2_M1_minus_M0", "+ΔDHW + exposure\n(thermal trajectory)"),
         ("deltaR2_M2_minus_M1", "+chl +PAR +wind\n(nonthermal context)")]
w = 0.8 / nb
for k, b in enumerate(BASIN_ORDER):
    pts, los, his = [], [], []
    for st, _ in steps:
        row = boot[(boot["basin"] == b) & (boot["stat"] == st)]
        pts.append(row["point"].iloc[0]); los.append(row["ci_lo"].iloc[0])
        his.append(row["ci_hi"].iloc[0])
    xs = np.arange(len(steps)) + (k - (nb - 1) / 2) * w
    ax.bar(xs, pts, width=w * 0.9, color=BASIN_COLOR[b], alpha=0.85,
           edgecolor="white", linewidth=0.5, label=b)
    ax.errorbar(xs, pts, yerr=[np.array(pts) - np.array(los),
                               np.array(his) - np.array(pts)],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(steps)))
ax.set_xticklabels([s[1] for s in steps], fontsize=8.5)
ax.set_ylabel("ΔR² of R (in-sample, week-block bootstrap 95% CI)", fontsize=9.5)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("(b) Nested incremental explanation of R\n"
             "M0: DHW → M1: +thermal trajectory → M2: +nonthermal",
             fontsize=10, fontweight="bold", loc="left")
style_ax(ax)

# ── Panel c: mechanism-contribution heatmap ──────────────────────────
ax = axes[1, 0]
heat_vars = ["dDHW_4w", "d_hotspot_dt", "cumsum_4w", "cumsum_12w",
             "ssta", "hotspot", "chl", "par", "wind"]
M = np.full((len(BASIN_ORDER), len(heat_vars)), np.nan)
for bi, b in enumerate(BASIN_ORDER):
    sub = mech[mech["basin"] == b].set_index("variable")
    for vi, v in enumerate(heat_vars):
        if v in sub.index:
            M[bi, vi] = sub.loc[v, "partial_rho"]
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(heat_vars)))
ax.set_xticklabels([VAR_LABEL[v] for v in heat_vars], rotation=35, ha="right",
                   fontsize=8.5)
ax.set_yticks(range(len(BASIN_ORDER)))
ax.set_yticklabels(BASIN_ORDER, fontsize=9)
for bi in range(len(BASIN_ORDER)):
    for vi in range(len(heat_vars)):
        if np.isfinite(M[bi, vi]):
            ax.text(vi, bi, f"{M[bi, vi]:+.2f}", ha="center", va="center",
                    fontsize=8, color="black")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("partial Spearman ρ (| current DHW)", fontsize=8.5)
ax.set_title("(c) Mechanism contributions by basin\n(partial association with R)",
             fontsize=10, fontweight="bold", loc="left")

# ── Panel d: incremental ROC (PersA → +thermal → +context → TSAR) ────
ax = axes[1, 1]
STEPS = [("PersA", "#8C8C8C", "-"), ("PersA+thermal", "#2E86C1", "-"),
         ("PersA+thermal+context", "#7B5EA7", "--"), ("TSAR", "#B03A2E", "-")]
handles, labels = [], []
for b in BASIN_ORDER:
    tgt = scores[f"{b}_target"]
    sev = (tgt >= 0.8).astype(int)
    from sklearn.metrics import roc_curve
    for name, color, ls in STEPS:
        s = scores[f"{b}_{name}"]
        fpr, tpr, _ = roc_curve(sev, s)
        auc = float(np.trapezoid(tpr, fpr))
        ln, = ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.6 if b != "SCS" else 1.0,
                      alpha=1.0 if b != "SCS" else 0.45)
        handles.append(ln)
        labels.append(f"{b} {name} (AUC {auc:.3f})")
ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5)
ax.set_xlabel("False positive rate", fontsize=9.5)
ax.set_ylabel("True positive rate", fontsize=9.5)
ax.set_title("(d) Incremental severe-risk ranking\nPersA → +thermal → +context → TSAR "
             "(thin = SCS)", fontsize=10, fontweight="bold", loc="left")
ax.legend(handles, labels, frameon=False, fontsize=6.8, loc="lower right")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
style_ax(ax)

plt.tight_layout()
out_png = RES / "fig_residual_mechanism_panels.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(RES / "fig_residual_mechanism_panels.pdf", bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"Saved {out_png}")
