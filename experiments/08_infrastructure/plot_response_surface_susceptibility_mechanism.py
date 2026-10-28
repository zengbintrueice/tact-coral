# plot_response_surface_susceptibility_mechanism.py — 4-panel main-figure
# candidate. ALL data read from results/response_surface_susceptibility_mechanism/.
#
#   Panel a: three-basin conditional response surfaces (DHW → future severe rate)
#   Panel b: matched-DHW R-low vs R-high severe risk (paired contrast)
#   Panel c: basin-specific susceptibility ΔROC/ΔPR with 95% CI
#   Panel d: residual mechanism hierarchy ΔR² (M0→M1→M2→M3)
#
# Run (piecf_env):
#   & '...\piecf_env\Scripts\python.exe' plot_response_surface_susceptibility_mechanism.py

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
RES = HERE.parents[1] / "results" / "response_surface_susceptibility_mechanism"

BASIN_ORDER = ["SCS", "GBR", "CARIB"]
BASIN_COLOR = {"SCS": "#0072B2", "GBR": "#009E73", "CARIB": "#D55E00"}

surface = pd.read_csv(RES / "response_surface.csv")
suscept = pd.read_csv(RES / "susceptibility.csv")
mech = pd.read_csv(RES / "mechanism_decomposition.csv")
nested = pd.read_csv(RES / "nested_models.csv")
S_json = json.load(open(RES / "summary.json", encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.patch.set_facecolor("white")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


# ── Panel a: conditional response surfaces ────────────────────────────
ax = axes[0, 0]
for b in BASIN_ORDER:
    sub = surface[surface["basin"] == b].sort_values("bin_center")
    ax.plot(sub["bin_center"], sub["severe_rate"] * 100, "-o", markersize=3.5,
            color=BASIN_COLOR[b], label=b, linewidth=1.6)
    ax.fill_between(sub["bin_center"], sub["ci_lo"] * 100, sub["ci_hi"] * 100,
                    color=BASIN_COLOR[b], alpha=0.12, linewidth=0)
ax.set_xlabel("current DHW (°C-weeks)", fontsize=9.5)
ax.set_ylabel("future severe rate (%)", fontsize=9.5)
ax.set_title("(a) Conditional thermal-response surface\n(week-block bootstrap 95% CI)",
             fontsize=10, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
style_ax(ax)

# ── Panel b: R-tertile contrast at matched DHW ─────────────────────────
ax = axes[0, 1]
for b in BASIN_ORDER:
    sub = surface[(surface["basin"] == b) & np.isfinite(surface["R_Q3_minus_Q1"])]
    sub = sub.sort_values("bin_center")
    ax.scatter(sub["bin_center"], sub["R_Q3_minus_Q1"] * 100, s=18,
               color=BASIN_COLOR[b], label=b, zorder=3, alpha=0.8)
    # CI bars where available
    for _, r in sub.iterrows():
        if np.isfinite(r["R_contrast_ci_lo"]):
            lo_ci = (r["R_Q3_minus_Q1"] - r["R_contrast_ci_lo"]) * 100
            hi_ci = (r["R_contrast_ci_hi"] - r["R_Q3_minus_Q1"]) * 100
            ax.errorbar(r["bin_center"], r["R_Q3_minus_Q1"] * 100,
                        yerr=[[max(0, lo_ci)], [max(0, hi_ci)]],
                        fmt="none", ecolor=BASIN_COLOR[b], elinewidth=0.7,
                        capsize=1.5, alpha=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("current DHW bin center (°C-weeks)", fontsize=9.5)
ax.set_ylabel("Δ severe rate (%)  R-Q3 minus R-Q1", fontsize=9)
ax.set_title("(b) Residual R identifies susceptibility at matched DHW\n"
             "(CI-excluded points = basin-specific)", fontsize=10,
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
style_ax(ax)

# ── Panel c: susceptibility ΔROC/ΔPR ──────────────────────────────────
ax = axes[1, 0]
if len(suscept):
    basins_present = [b for b in BASIN_ORDER if b in suscept["basin"].values]
    x = np.arange(len(basins_present))
    w = 0.35
    for mi, (met, lab) in enumerate((("delta_roc", "ΔROC-AUC"),
                                      ("delta_pr", "ΔPR-AUC"))):
        for bi, b in enumerate(basins_present):
            row = suscept[suscept["basin"] == b].iloc[0]
            v = row[f"delta_{met.split('_')[1]}"]
            lo = row[f"delta_{met.split('_')[1]}_ci_lo"]
            hi = row[f"delta_{met.split('_')[1]}_ci_hi"]
            xx = x[bi] + (mi - 0.5) * w
            col = S_COLOR = {"SCS": "#0072B2", "GBR": "#009E73",
                             "CARIB": "#D55E00"}[b]
            ax.bar(xx, v, width=w * 0.9, color=col, alpha=0.5 + 0.4 * mi,
                   edgecolor="white", linewidth=0.5)
            ax.errorbar(xx, v, yerr=[[max(0, v - lo)], [max(0, hi - v)]],
                        fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
            sig = "✓" if lo > 0 else "n.s."
            ax.text(xx, max(v, 0) + 0.004, sig, ha="center", fontsize=6.5,
                    fontweight="bold" if lo > 0 else "normal",
                    color=S_COLOR if lo > 0 else "#999999")
    ax.set_xticks(x)
    ax.set_xticklabels(basins_present, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ (TSAR − PersA)", fontsize=9.5)
    ax.set_title("(c) Basin-specific susceptibility (ΔROC / ΔPR)\n"
                 "✓ = CI excludes zero; faded = ΔPR", fontsize=10,
                 fontweight="bold", loc="left")
else:
    ax.text(0.5, 0.5, "Gate ①→② not passed\nsusceptibility not established",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
style_ax(ax)

# ── Panel d: mechanism hierarchy ΔR² ──────────────────────────────────
ax = axes[1, 1]
steps = ["M1_minus_M0", "M2_minus_M1", "M3_minus_M2"]
slabs = ["+trajectory\n+persistence", "+thermal state\n(HS, SSTa)",
         "+nonthermal\n(chl, PAR, wind)"]
basins_present = [b for b in BASIN_ORDER
                  if b in nested[nested["model"].str.startswith("deltaR2")]["basin"].values]
w = 0.8 / max(len(basins_present), 1)
for k, b in enumerate(basins_present):
    pts, los, his = [], [], []
    for st in steps:
        row = nested[(nested["basin"] == b) & (nested["model"] == f"deltaR2_{st}")]
        if len(row):
            pts.append(row["r2_oos"].iloc[0])
        else:
            pts.append(np.nan); los.append(np.nan); his.append(np.nan)
    # find CI from bootstrap or nested
    xs = np.arange(len(steps)) + (k - (len(basins_present) - 1) / 2) * w
    ax.bar(xs, pts, width=w * 0.9, color=BASIN_COLOR[b], alpha=0.85,
           edgecolor="white", linewidth=0.5, label=b)
    for xi, v in zip(xs, pts):
        if np.isfinite(v):
            ax.text(xi, v + 0.003, f"{v:+.4f}", ha="center", fontsize=5.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(steps)))
ax.set_xticklabels(slabs, fontsize=7.5)
ax.set_ylabel("ΔR² (out-of-sample, rank scale)", fontsize=9.5)
ax.set_title("(d) Residual mechanism hierarchy (OOS ΔR²)\n"
             "M0: DHW → +trajectory → +state → +context", fontsize=10,
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
style_ax(ax)

plt.tight_layout()
out_png = RES / "fig_response_surface_susceptibility_mechanism.png"
plt.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
plt.savefig(RES / "fig_response_surface_susceptibility_mechanism.pdf",
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out_png}")
