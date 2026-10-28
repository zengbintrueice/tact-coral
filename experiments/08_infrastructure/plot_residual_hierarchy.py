# plot_residual_hierarchy.py — main-figure candidate for the residual
# hierarchy test. ALL values read from results/residual_hierarchy/ outputs.
#
#   Panel a: ROC curves H0→H1→H2→H3 (+ frozen TSAR) per basin
#   Panel b: PR curves H0→H1→H2→H3 (+ frozen TSAR) per basin
#   Panel c: paired adjacent-level ΔROC / ΔPR with 95% CI
#   Panel d: level overview (ROC bars + PR markers)
#   Panel e: out-of-sample R² of the residual-explanation models
#   Panel f: heatmap of ROC-AUC over basins × hierarchy levels

import io, sys, json, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "residual_hierarchy"

BASIN_ORDER = ["SCS", "GBR", "CARIB"]
BASIN_COLOR = {"SCS": "#3B6FB6", "GBR": "#7B5EA7", "CARIB": "#C0504D"}
LEVELS = ["H0", "H1", "H2", "H3", "TSAR"]
LEVEL_LABEL = {"H0": "H0\nPersA", "H1": "H1\n+thermal traj",
               "H2": "H2\n+context", "H3": "H3\n+R", "TSAR": "TSAR\n(frozen)"}
LEVEL_STYLE = {"H0": ("-", 1.4), "H1": ("-", 1.4), "H2": ("--", 1.2),
               "H3": ("-", 1.8), "TSAR": (":", 1.4)}
LEVEL_SHADE = {"H0": 0.45, "H1": 0.65, "H2": 0.45, "H3": 1.0, "TSAR": 0.8}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


hier = pd.read_csv(RES / "hierarchy_metrics.csv")
deltas = pd.read_csv(RES / "paired_deltas.csv")
expl = pd.read_csv(RES / "residual_explanation.csv")
scores = np.load(RES / "panel_scores.npz")

fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
fig.patch.set_facecolor("white")

CURVE_STEPS = [("H0", "#8C8C8C"), ("H1", "#2E86C1"), ("H2", "#7B5EA7"),
               ("H3", "#B03A2E"), ("TSAR", "#117A65")]

# ── Panel a: ROC curves ──────────────────────────────────────────────
ax = axes[0, 0]
for b in BASIN_ORDER:
    sev = (scores[f"{b}_target"] >= 0.8).astype(int)
    for lv, color in CURVE_STEPS:
        s = scores[f"{b}_score_{lv}"]
        fpr, tpr, _ = roc_curve(sev, s)
        a = auc(fpr, tpr)
        ls, lw = LEVEL_STYLE[lv]
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                alpha=LEVEL_SHADE[lv] if b != "SCS" else 0.35,
                label=f"{b} {lv} (AUC {a:.3f})" if b != "SCS" else None)
ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5)
ax.set_xlabel("False positive rate", fontsize=9.5)
ax.set_ylabel("True positive rate", fontsize=9.5)
ax.set_title("(a) ROC — information hierarchy vs severe y(t+4)\n"
             "(thin/faded = SCS reference; H3 bold)", fontsize=10,
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=6.2, loc="lower right")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
style_ax(ax)

# ── Panel b: PR curves ───────────────────────────────────────────────
ax = axes[0, 1]
for b in BASIN_ORDER:
    sev = (scores[f"{b}_target"] >= 0.8).astype(int)
    for lv, color in CURVE_STEPS:
        s = scores[f"{b}_score_{lv}"]
        prec, rec, _ = precision_recall_curve(sev, s)
        a = auc(rec, prec)
        ls, lw = LEVEL_STYLE[lv]
        ax.plot(rec, prec, color=color, linestyle=ls, linewidth=lw,
                alpha=LEVEL_SHADE[lv] if b != "SCS" else 0.35,
                label=f"{b} {lv} (AP {a:.3f})" if b != "SCS" else None)
ax.set_xlabel("Recall", fontsize=9.5)
ax.set_ylabel("Precision", fontsize=9.5)
ax.set_title("(b) Precision–Recall — information hierarchy\n"
             "(H1 improves AP markedly; H2 degrades)", fontsize=10,
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=6.2, loc="upper right")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
style_ax(ax)

# ── Panel c: paired adjacent-level deltas with CI ─────────────────────
ax = axes[0, 2]
contrasts = ["H1_minus_H0", "H2_minus_H1", "H3_minus_H2", "H3_minus_TSAR"]
clabels = ["H1−H0\nthermal traj", "H2−H1\n+context", "H3−H2\n+R",
           "H3−TSAR\nconsistency"]
w = 0.8 / len(BASIN_ORDER)
for k, b in enumerate(BASIN_ORDER):
    for mi, (met, mk) in enumerate((("roc", "o"), ("pr", "s"))):
        sub = deltas[(deltas["basin"] == b) & (deltas["metric"] == met)] \
            .set_index("contrast").reindex(contrasts)
        xs = np.arange(len(contrasts)) + (k - 1) * w + (0.12 * (mi - 0.5))
        ax.errorbar(xs, sub["point"],
                    yerr=[sub["point"] - sub["ci_lo"], sub["ci_hi"] - sub["point"]],
                    fmt=mk, markersize=4.5, color=BASIN_COLOR[b],
                    ecolor=BASIN_COLOR[b], elinewidth=1.1, capsize=2.5,
                    alpha=1.0 if mi == 0 else 0.55,
                    label=f"{b} {'ΔROC' if mi == 0 else 'ΔPR'}")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(contrasts)))
ax.set_xticklabels(clabels, fontsize=8.5)
ax.set_ylabel("paired Δ (week-block bootstrap 95% CI)", fontsize=9.5)
ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="lower left")
ax.set_title("(c) Paired incremental gains\n(filled=ΔROC, faded=ΔPR; "
             "H3−TSAR = consistency check)", fontsize=10, fontweight="bold",
             loc="left")
style_ax(ax)

# ── Panel d: level overview (ROC bars + PR markers) ──────────────────
ax = axes[1, 0]
ax2 = ax.twinx()
x = np.arange(len(LEVELS))
w = 0.8 / len(BASIN_ORDER)
for k, b in enumerate(BASIN_ORDER):
    sub = hier[hier["basin"] == b].set_index("level").reindex(LEVELS)
    xs = x + (k - 1) * w
    ax.bar(xs, sub["roc"], width=w * 0.9, color=BASIN_COLOR[b], alpha=0.85,
           edgecolor="white", linewidth=0.5, label=b)
    ax2.plot(xs, sub["pr"], marker="o", markersize=5, linestyle="-",
             color=BASIN_COLOR[b], linewidth=1.4, alpha=0.9)
ax.set_ylim(0.5, 1.0)
ax.set_xticks(x)
ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS], fontsize=8.5)
ax.set_ylabel("ROC-AUC", fontsize=9.5)
ax2.set_ylabel("PR-AUC (lines + markers)", fontsize=9)
ax2.set_ylim(0, max(hier["pr"]) * 1.25)
ax2.spines["top"].set_visible(False)
h_, l_ = ax.get_legend_handles_labels()
ax.legend(h_, l_, frameon=False, fontsize=8.5, loc="lower left")
ax.set_title("(d) Level overview\nbars: ROC-AUC; lines: PR-AUC",
             fontsize=10, fontweight="bold", loc="left")
style_ax(ax)

# ── Panel e: residual explanation out-of-sample R² ───────────────────
ax = axes[1, 1]
models = ["R~H1", "R~H1+H2"]
w = 0.8 / len(BASIN_ORDER)
for k, b in enumerate(BASIN_ORDER):
    sub = expl[(expl["basin"] == b) & (expl["model"].isin(models))] \
        .set_index("model").reindex(models)
    xs = np.arange(len(models)) + (k - 1) * w
    ax.bar(xs, sub["r2_oos"], width=w * 0.9, color=BASIN_COLOR[b], alpha=0.85,
           edgecolor="white", linewidth=0.5, label=b)
    lo = np.clip(sub["r2_oos"] - sub["ci_lo"], 0, None)
    hi = np.clip(sub["ci_hi"] - sub["r2_oos"], 0, None)
    ax.errorbar(xs, sub["r2_oos"], yerr=[lo, hi], fmt="none",
                ecolor="black", elinewidth=0.9, capsize=2.5)
ax.set_xticks(np.arange(len(models)))
ax.set_xticklabels(["R ~ H1 vars\n(thermal traj)",
                    "R ~ H1+H2 vars\n(+context)"], fontsize=8.5)
ax.set_ylabel("out-of-sample R² of R (week-grouped 5-fold CV, 95% CI)",
              fontsize=9)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("(e) How much of R do explicit candidate variables explain?\n"
             "(out-of-sample, week-grouped CV)", fontsize=10,
             fontweight="bold", loc="left")
style_ax(ax)

# ── Panel f: heatmap ROC over basins × levels ────────────────────────
ax = axes[1, 2]
M = np.full((len(BASIN_ORDER), len(LEVELS)), np.nan)
for bi, b in enumerate(BASIN_ORDER):
    sub = hier[hier["basin"] == b].set_index("level")
    for li, l in enumerate(LEVELS):
        M[bi, li] = sub.loc[l, "roc"]
im = ax.imshow(M, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(LEVELS)))
ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS], fontsize=8)
ax.set_yticks(range(len(BASIN_ORDER)))
ax.set_yticklabels(BASIN_ORDER, fontsize=9)
for bi in range(len(BASIN_ORDER)):
    for li in range(len(LEVELS)):
        ax.text(li, bi, f"{M[bi, li]:.3f}", ha="center", va="center",
                fontsize=8.5, color="white" if M[bi, li] < M.max() * 0.9 else "black")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("ROC-AUC (severe y(t+4))", fontsize=8.5)
ax.set_title("(f) Hierarchy ROC-AUC by basin", fontsize=10,
             fontweight="bold", loc="left")

plt.tight_layout()
out_png = RES / "fig_residual_hierarchy_panels.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(RES / "fig_residual_hierarchy_panels.pdf", bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"Saved {out_png}")
