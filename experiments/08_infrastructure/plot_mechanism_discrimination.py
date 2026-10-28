# plot_mechanism_discrimination.py — final mechanism figure, all values read
# from results/mechanism_discrimination/ (no hand-entered numbers).
#
#   Panel a: DHW × R conditional response surface (severe rate heat map,
#            basins × R-tertile rows, DHW-bin columns)
#   Panel b: pooled DHW-matched Q1→Q3 severe-risk difference + 95% CI
#   Panel c: incremental ladder M0→M1→M2→TSAR (+M2+R) ΔROC/ΔPR with CI
#   Panel d: evidence matrix (H1 vs H2 criteria × basins)

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
RES = HERE.parents[1] / "results" / "mechanism_discrimination"

BASIN_ORDER = ["SCS", "GBR", "CARIB"]
BASIN_COLOR = {"SCS": "#3B6FB6", "GBR": "#7B5EA7", "CARIB": "#C0504D"}

surf = pd.read_csv(RES / "dhw_matched_surface.csv")
deltas = pd.read_csv(RES / "paired_deltas.csv")
boot = pd.read_csv(RES / "bootstrap.csv")
evid = pd.read_csv(RES / "evidence_matrix.csv")
S = json.load(open(RES / "summary.json", encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
fig.patch.set_facecolor("white")


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.6)


# ── Panel a: DHW × R response surface ────────────────────────────────
ax = axes[0, 0]
rows, ylabels = [], []
for b in BASIN_ORDER:
    sub = surf[surf["basin"] == b].sort_values("bin_lo")
    binned = sub["bin_lo"].astype(int).tolist()
    all_bins = sorted(set(binned))
    for q, qlab in ((0, "R-Q1"), (1, "R-Q2"), (2, "R-Q3")):
        row = np.full(len(all_bins), np.nan)
        for bi, bl in enumerate(all_bins):
            r = sub[sub["bin_lo"].astype(int) == bl]
            if len(r):
                # reconstruct rate by quantile from stored Q1/Q3 only for Q1/Q3
                if qlab == "R-Q1":
                    row[bi] = r["rate_Q1"].iloc[0]
                elif qlab == "R-Q3":
                    row[bi] = r["rate_Q3"].iloc[0]
        rows.append(row)
        ylabels.append(f"{b} {qlab}")
# Q2 is not stored; build the matrix from available rows only
M = np.vstack([r for r in rows if not np.all(np.isnan(r))])
yl = [l for l, r in zip(ylabels, rows) if not np.all(np.isnan(r))]
vmax = np.nanmax(M)
im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
ax.set_yticks(range(len(yl)))
ax.set_yticklabels(yl, fontsize=8)
bin_lab = sorted(set(surf["bin_lo"].astype(int)))
ax.set_xticks(range(len(bin_lab)))
ax.set_xticklabels([f"{v/2:.0f}–{(v+1)/2:.1f}" for v in bin_lab], fontsize=7.5)
ax.set_xlabel("current DHW bin (degree-weeks)", fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i, j]*100:.1f}", ha="center", va="center",
                    fontsize=6.8, color="black")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("severe rate y(t+4)≥0.8", fontsize=8.5)
ax.set_title("(a) DHW-matched response surface\n(severe rate %, R tertiles "
             "within DHW bin)", fontsize=10, fontweight="bold", loc="left")

# ── Panel b: pooled DHW-matched gradient ─────────────────────────────
ax = axes[0, 1]
grad = boot[boot["stat"] == "pooled_risk_diff_Q3_Q1"]
per_bin = surf.dropna(subset=["risk_diff_Q3_Q1"])
for bi, b in enumerate(BASIN_ORDER):
    xs = np.full(len(per_bin[per_bin["basin"] == b]), bi,
                 dtype=float) + np.linspace(-0.15, 0.15,
                                            len(per_bin[per_bin["basin"] == b]))
    ax.scatter(xs, per_bin[per_bin["basin"] == b]["risk_diff_Q3_Q1"],
               s=14, color=BASIN_COLOR[b], alpha=0.45, zorder=2)
row = grad[grad["basin"].isin(BASIN_ORDER)].set_index("basin").reindex(BASIN_ORDER)
xs = np.arange(3)
ax.errorbar(xs, row["point"],
            yerr=[row["point"] - row["ci_lo"], row["ci_hi"] - row["point"]],
            fmt="D", markersize=9, color="black", ecolor="black",
            elinewidth=1.4, capsize=5, zorder=3, label="pooled (95% CI)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(BASIN_ORDER, fontsize=9.5)
ax.set_ylabel("severe rate Q3 − Q1 (DHW-matched)", fontsize=9.5)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("(b) DHW-matched risk gradient of R\n(small dots = per-bin; "
             "diamond = record-weighted pooled)", fontsize=10,
             fontweight="bold", loc="left")
style_ax(ax)

# ── Panel c: incremental ladder ──────────────────────────────────────
ax = axes[1, 0]
contrasts = ["M1_minus_M0", "M2_minus_M1", "M2+R_minus_M2", "TSAR_minus_M2"]
clabels = ["M1−M0\n+traj", "M2−M1\n+context", "M2+R−M2\n+R", "TSAR−M2\nfrozen"]
w = 0.8 / 3
for k, b in enumerate(BASIN_ORDER):
    for mi, (met, mk) in enumerate((("roc", "o"), ("pr", "s"))):
        sub = deltas[(deltas["basin"] == b) & (deltas["metric"] == met)] \
            .set_index("contrast").reindex(contrasts)
        xs = np.arange(len(contrasts)) + (k - 1) * w + (0.12 * (mi - 0.5))
        ax.errorbar(xs, sub["point"],
                    yerr=[sub["point"] - sub["ci_lo"],
                          sub["ci_hi"] - sub["point"]],
                    fmt=mk, markersize=4.5, color=BASIN_COLOR[b],
                    ecolor=BASIN_COLOR[b], elinewidth=1.1, capsize=2.5,
                    alpha=1.0 if mi == 0 else 0.55,
                    label=f"{b} {'ΔROC' if mi == 0 else 'ΔPR'}")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(contrasts)))
ax.set_xticklabels(clabels, fontsize=8.5)
ax.set_ylabel("paired Δ (week-block bootstrap 95% CI)", fontsize=9.5)
ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
ax.set_title("(c) Mechanism ladder increments\n(filled=ΔROC, faded=ΔPR; "
             "training-free rank scores)", fontsize=10, fontweight="bold",
             loc="left")
style_ax(ax)

# ── Panel d: evidence matrix ─────────────────────────────────────────
ax = axes[1, 1]
crit = ["E1_partial_assoc_excl_0", "E4_oos_R2_of_R", "E5_M2plusR_gain"]
crit_lab = ["E1: partial assoc(R,y|DHW)\nCI excl 0",
            "E4: explicit vars leave R\nmostly unexplained (OOS R²<0.5)",
            "E5: M2+R beats M2\n(CI-supported)"]
Mv = np.full((len(crit), len(BASIN_ORDER)), np.nan)
for ci_, c in enumerate(crit):
    for bi_, b in enumerate(BASIN_ORDER):
        r = evid[(evid["evidence"] == c) & (evid["basin"] == b)]
        if len(r):
            v = r["value"].iloc[0]
            Mv[ci_, bi_] = v if c != "E4_oos_R2_of_R" else (1.0 if v < 0.5 else 0.0)
im = ax.imshow(Mv, cmap="RdYlGn", vmin=-0.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(BASIN_ORDER)))
ax.set_xticklabels(BASIN_ORDER, fontsize=9.5)
ax.set_yticks(range(len(crit)))
ax.set_yticklabels(crit_lab, fontsize=8)
for i in range(len(crit)):
    for j in range(len(BASIN_ORDER)):
        if np.isfinite(Mv[i, j]):
            txt = ("YES" if Mv[i, j] > 0.5 else
                   "no" if Mv[i, j] < -0.2 else f"{Mv[i, j]:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    fontweight="bold")
ax.set_title("(d) Evidence matrix for basin-specific susceptibility (H2)\n"
             f"GBR−CARIB gradient diff CI [{S['evidence']['GBR_vs_CARIB_gradient_diff_ci'][0]:+.3f}, "
             f"{S['evidence']['GBR_vs_CARIB_gradient_diff_ci'][1]:+.3f}]",
             fontsize=10, fontweight="bold", loc="left")

plt.tight_layout()
out = RES / "fig_mechanism_discrimination.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(RES / "fig_mechanism_discrimination.pdf", bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"Saved {out}")
