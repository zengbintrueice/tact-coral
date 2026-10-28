# plot_cross_basin_residual_mechanism.py — 3-panel figure, all values read
# from results/cross_basin_residual_mechanism/ (no hand-entered numbers).
#
#   Panel a: GBR/CARIB residual–mechanism partial Spearman (| current DHW) + CI
#   Panel b: GBR−CARIB effect difference + 95% CI
#   Panel c: standardized (rank-scale) mechanism coefficient heatmap

import io, sys, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "cross_basin_residual_mechanism"

CANDIDATES = ["dDHW_4w", "cumsum_4w", "cumsum_12w", "ssta", "hotspot",
              "chl", "par", "wind"]
NONTHERMAL = {"chl", "par", "wind"}
LABEL = {"dDHW_4w": "ΔDHW (4 wk)", "cumsum_4w": "HS cumsum 4 wk",
         "cumsum_12w": "HS cumsum 12 wk", "ssta": "SST anomaly",
         "hotspot": "HotSpot", "chl": "chlorophyll", "par": "PAR",
         "wind": "wind speed"}
BASIN_COLOR = {"GBR": "#7B5EA7", "CARIB": "#C0504D"}

assoc = pd.read_csv(RES / "mechanism_associations.csv")
coef = pd.read_csv(RES / "coefficient_comparison.csv")
summary = __import__("json").load(open(RES / "summary.json", encoding="utf-8"))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.patch.set_facecolor("white")


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.6)


# ── Panel a ──────────────────────────────────────────────────────────
ax = axes[0]
part = assoc[(assoc["type"] == "partial_ctrl_DHW")
             & (assoc["basin"].isin(["GBR", "CARIB"]))]
w = 0.8 / 2
for k, b in enumerate(["GBR", "CARIB"]):
    sub = part[part["basin"] == b].set_index("variable").reindex(CANDIDATES)
    xs = np.arange(len(CANDIDATES)) + (k - 0.5) * w
    ax.bar(xs, sub["rho"], width=w * 0.9, color=BASIN_COLOR[b], alpha=0.85,
           edgecolor="white", linewidth=0.5, label=b)
    ax.errorbar(xs, sub["rho"],
                yerr=[sub["rho"] - sub["ci_lo"], sub["ci_hi"] - sub["rho"]],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2, alpha=0.7)
    for x, q in zip(xs, sub["p_fdr"]):
        if q < 0.05:
            ax.text(x, 0.02, "*", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(CANDIDATES)))
ax.set_xticklabels([LABEL[v] for v in CANDIDATES], rotation=35, ha="right",
                   fontsize=8.5)
ax.set_ylabel("partial Spearman ρ (| current DHW)", fontsize=9.5)
ax.legend(frameon=False, fontsize=9)
ax.set_title("(a) Residual–mechanism association\n(* FDR q<0.05; red labels = "
             "nonthermal)", fontsize=10, fontweight="bold", loc="left")
style_ax(ax)

# ── Panel b: GBR−CARIB difference ────────────────────────────────────
ax = axes[1]
diff = assoc[assoc["basin"] == "GBR_minus_CARIB"].set_index("variable") \
    .reindex(CANDIDATES)
xs = np.arange(len(CANDIDATES))
colors = ["#B03A2E" if v in NONTHERMAL else "#2E4053" for v in CANDIDATES]
ax.bar(xs, diff["rho"], width=0.6, color=colors, alpha=0.8,
       edgecolor="white", linewidth=0.5)
ax.errorbar(xs, diff["rho"],
            yerr=[diff["rho"] - diff["ci_lo"], diff["ci_hi"] - diff["rho"]],
            fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([LABEL[v] for v in CANDIDATES], rotation=35, ha="right",
                   fontsize=8.5)
ax.set_ylabel("GBR − CARIB partial ρ difference", fontsize=9.5)
ax.set_title("(b) Cross-basin effect difference\n(CI incl. 0 = homogeneous "
             "effect size)", fontsize=10, fontweight="bold", loc="left")
style_ax(ax)

# ── Panel c: rank-coefficient heatmap ────────────────────────────────
ax = axes[2]
basins = ["SCS", "GBR", "CARIB"]
M = np.full((len(basins), len(CANDIDATES)), np.nan)
for bi, b in enumerate(basins):
    sub = coef[coef["basin"] == b].set_index("variable")
    for vi, v in enumerate(CANDIDATES):
        if v in sub.index:
            M[bi, vi] = sub.loc[v, "rank_coef"]
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(CANDIDATES)))
ax.set_xticklabels([LABEL[v] for v in CANDIDATES], rotation=35, ha="right",
                   fontsize=8.5)
ax.set_yticks(range(len(basins)))
ax.set_yticklabels(basins, fontsize=9)
for bi in range(len(basins)):
    for vi in range(len(CANDIDATES)):
        if np.isfinite(M[bi, vi]):
            ax.text(vi, bi, f"{M[bi, vi]:+.2f}", ha="center", va="center",
                    fontsize=8, color="black")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("rank-OLS coefficient (R ~ 8 candidates)", fontsize=8.5)
ax.set_title("(c) Standardized mechanism coefficients\n(rank scale; "
             f"OOS R²: " + ", ".join(f"{b} {summary['basins'][b]['r2_oos_rank_ols']:.3f}"
                                     for b in basins) + ")",
             fontsize=10, fontweight="bold", loc="left")

plt.tight_layout()
out = RES / "fig_cross_basin_mechanism.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(RES / "fig_cross_basin_mechanism.pdf", bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"Saved {out}")
