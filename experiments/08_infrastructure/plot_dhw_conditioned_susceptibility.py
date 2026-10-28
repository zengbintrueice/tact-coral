# plot_dhw_conditioned_susceptibility.py — main-text candidate figure (4 panels).
# ALL data read from results/dhw_conditioned_susceptibility/ outputs.
#
#   Panel a: DHW → future severe-rate response curves per basin (95% CI bands)
#   Panel b: basin-pair risk difference at matched DHW bins + 95% CI
#   Panel c: within-bin R-tertile (Q1/Q2/Q3) severe-rate gradients
#   Panel d: DHW-only (A) vs DHW+R (B) incremental ROC (+ ΔROC/ΔPR in legend)

import io, sys, json, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "figures"))
import nc_style as S  # unified NC palette/typography (Okabe-Ito)

RES = HERE.parents[1] / "results" / "dhw_conditioned_susceptibility"

BASINS = ["SCS", "GBR", "CARIB"]
BASIN_COLOR = {"SCS": S.COLORS["SCS"], "GBR": S.COLORS["GBR"],
               "CARIB": S.COLORS["CARIB"]}
PAIR_COLOR = {"CARIB_minus_GBR": "#B03A2E", "GBR_minus_SCS": "#3B6FB6",
              "CARIB_minus_SCS": "#2E8B57"}
QT_COLOR = {1: "#8fbde1", 2: "#4f93c8", 3: "#173F5F"}

surv = pd.read_csv(RES / "susceptibility_surface.csv")
pair = pd.read_csv(RES / "basin_pair_risk.csv")
cond = pd.read_csv(RES / "conditional_residual_effect.csv")
boot = pd.read_csv(RES / "bootstrap.csv")
SJSON = json.load(open(RES / "summary.json", encoding="utf-8"))  # noqa: keep nc_style module unshadowed
scores = np.load(RES / "panel_scores.npz")

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.patch.set_facecolor("white")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


# ── Panel a: response curves ─────────────────────────────────────────
ax = axes[0, 0]
for b in BASINS:
    sub = surv[surv["basin"] == b].sort_values("bin_center")
    ax.plot(sub["bin_center"], sub["severe_rate"], "-o", markersize=3.5,
            color=BASIN_COLOR[b], label=b, linewidth=1.6)
    ax.fill_between(sub["bin_center"], sub["ci_lo"], sub["ci_hi"],
                    color=BASIN_COLOR[b], alpha=0.18, linewidth=0)
ax.set_xlabel("current DHW bin center (°C-weeks)", fontsize=9.5)
ax.set_ylabel("future severe rate  P[y(t+4) ≥ 0.8]", fontsize=9.5)
ax.set_title("(a) DHW-conditioned future severe rate\n(week-block bootstrap 95% CI)",
             fontsize=10, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
style_ax(ax)

# ── Panel b: basin-pair risk differences ─────────────────────────────
ax = axes[0, 1]
for p, sub in pair.groupby("pair"):
    sub = sub.sort_values("bin_center")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.errorbar(sub["bin_center"] + 0.06 * list(PAIR_COLOR).index(p),
                sub["risk_diff"],
                yerr=[sub["risk_diff"] - sub["rd_lo"], sub["rd_hi"] - sub["risk_diff"]],
                fmt="-o", markersize=3, color=PAIR_COLOR[p], elinewidth=0.9,
                capsize=1.8, label=p.replace("_minus_", " − "), linewidth=1.3)
ax.set_xlabel("current DHW bin center (°C-weeks)", fontsize=9.5)
ax.set_ylabel("risk difference at matched DHW", fontsize=9.5)
ax.set_title("(b) Basin susceptibility difference at matched DHW\n"
             "(CI excl. 0 = basin-specific)", fontsize=10, fontweight="bold",
             loc="left")
ax.legend(frameon=False, fontsize=8.5)
style_ax(ax)

# ── Panel c: R-tertile gradients within bins ─────────────────────────
ax = axes[1, 0]
for b in BASINS:
    sub = cond[cond["basin"] == b]
    for q in (1, 2, 3):
        s = sub[sub["R_quantile"] == q].sort_values("bin_center")
        ax.plot(s["bin_center"], s["severe_rate"], "-o", markersize=3,
                color=BASIN_COLOR[b],
                alpha=0.45 if q == 1 else (0.7 if q == 2 else 1.0),
                linewidth=2.0 if q == 3 else 1.1,
                linestyle="-" if q == 3 else "--",
                label=f"{b} R-Q{q}" if q >= 2 or b == "SCS" else None)
ax.set_xlabel("current DHW bin center (°C-weeks)", fontsize=9.5)
ax.set_ylabel("future severe rate within DHW bin", fontsize=9.5)
ax.set_title("(c) Residual R tertiles within matched DHW bins\n"
             "(light→dark = R Q1→Q3; bold = Q3)", fontsize=10,
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=6.5, ncol=2)
style_ax(ax)

# ── Panel d: A vs B ROC ──────────────────────────────────────────────
ax = axes[1, 1]
leg_rows = []
for b in BASINS:
    sev = scores[f"{b}_sev"]
    for name, key, ls, lw, a0 in (("A (DHW)", "A", "--", 1.3, 0.55),
                                  ("B (DHW+R)", "B", "-", 2.0, 1.0),
                                  ("TSAR (ref)", "tsar_rank", ":", 1.2, 0.6)):
        s = scores[f"{b}_{key}"]
        fpr, tpr, _ = roc_curve(sev, s)
        a = auc(fpr, tpr)
        ap = average_precision_score(sev, s)
        ax.plot(fpr, tpr, color=BASIN_COLOR[b], linestyle=ls, linewidth=lw,
                alpha=a0, label=f"{b} {name} AUC {a:.3f} / AP {ap:.3f}")
    # delta annotations from bootstrap.csv
    dr = boot[(boot["basin"] == b) & (boot["stat"] == "delta_roc_B_minus_A")].iloc[0]
    leg_rows.append(f"{b}: ΔROC {dr['point']:+.4f} [{dr['ci_lo']:+.4f}, {dr['ci_hi']:+.4f}]")
ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5)
ax.set_xlabel("False positive rate", fontsize=9.5)
ax.set_ylabel("True positive rate", fontsize=9.5)
ax.set_title("(d) DHW-only (A) vs DHW+R (B) — matched-strata gain\n"
             "solid bold = B; dashed = A; dotted = TSAR reference",
             fontsize=10, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=6.2, loc="lower right")
ax.text(0.02, 0.98, "\n".join(leg_rows), transform=ax.transAxes, fontsize=7.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
style_ax(ax)

plt.tight_layout()
out = RES / "figure_dhw_conditioned_susceptibility.png"
plt.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
plt.savefig(RES / "figure_dhw_conditioned_susceptibility.pdf",
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out}")
