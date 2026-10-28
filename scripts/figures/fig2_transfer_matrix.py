# fig2_transfer_matrix.py — MAIN FIGURE 2: fully connected six-direction
# transfer matrix. All values read from results/transfer_coordinate/ CSVs
# (transfer_matrix.csv, transfer_matrix_ci.csv, reverse_matrix.csv,
# reverse_matrix_ci.csv) — nothing hand-entered.
#
#   Panel a: zero-shot ROC-AUC heatmap, frozen TSCF (rows = source basin,
#            cols = target basin; diagonal = in-source reference)
#   Panel b: same for TSAR (anchored residual)
#   Panel c: ΔROC (target − source) per direction with paired week-block
#            bootstrap 95% CI — the "collapse follows the target basin" panel
#   Panel d: seed scan (seeds 0/1/2 + the seed-42 reference) per reverse-
#            source direction — the "seed lottery vs deterministic anchoring"
#            panel (unanchored TSCF spans wide, TSAR is seed-invariant)
#
# Run (piecf_env):
#   cd LTTS_CRW && & '...\piecf_env\Scripts\python.exe' scripts/figures/fig2_transfer_matrix.py

import io, sys, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nc_style as S

RES = HERE.parents[1] / "results" / "transfer_coordinate"
FIGDIR = HERE.parents[1] / "results" / "figures"
FIGDIR.mkdir(exist_ok=True)

BASINS = ["SCS", "GBR", "CARIB"]

tm = pd.read_csv(RES / "transfer_matrix.csv")            # SCS-source (tier 1)
tci = pd.read_csv(RES / "transfer_matrix_ci.csv")
rm = pd.read_csv(RES / "reverse_matrix.csv")             # GBR/CARIB-source
rci = pd.read_csv(RES / "reverse_matrix_ci.csv")
scan = pd.read_csv(RES / "seed_scan" / "seed_scan_results.csv")

# ---- assemble ROC matrices (source × target) + in-source diagonals ----
roc = {m: pd.DataFrame(np.nan, index=BASINS, columns=BASINS)
       for m in ("TSCF", "TSAR", "PersA")}
for _, r in tm.iterrows():
    roc[r["model"]].loc[r["source"], r["target"]] = r["roc_auc"]
for _, r in rm.iterrows():
    roc[r["model"]].loc[r["source"], r["target"]] = r["roc_auc"]
# in-source SCS reference from the frozen three-domain audit (seed-42)
aud = pd.read_csv(HERE.parents[1] / "results" / "physics_anchored" / "audit"
                  / "three_domain_metrics.csv")
aud = aud[aud["split"] == "SCS_test"]
name_map = {"TSCF": "TSCF", "TSAR": "PhysicsAnchored", "PersA": "PersA"}
for m, col in name_map.items():
    v = aud[aud["method"] == col]["roc_auc"]
    if len(v):
        roc[m].loc["SCS", "SCS"] = float(v.iloc[0])

# ---- ΔROC vs source (target − source) with 95% CI per direction ----
# transfer_matrix_ci.csv is a MIXED file: it contains the six SCS-source
# rows AND the eight reverse-source (GBR/CARIB) rows. Label every row with
# its own recorded source (never a hard-coded "SCS"), then de-duplicate on
# (source, target, model) because reverse rows also appear in
# reverse_matrix_ci.csv. Diagonals are dropped (ΔROC = 0 by design).
delta_map = {}
for _, r in pd.concat([tci, rci], ignore_index=True).iterrows():
    if r["source"] == r["target"]:
        continue
    delta_map[(r["source"], r["target"], r["model"])] = {
        "source": r["source"], "target": r["target"], "model": r["model"],
        "d": r["d_roc"], "lo": r["d_roc_lo"], "hi": r["d_roc_hi"]}
delta_rows = [delta_map[k] for k in sorted(delta_map)]
dd = pd.DataFrame(delta_rows)
DIRS = [(s, t) for s in BASINS for t in BASINS if s != t]

fig, axes = plt.subplots(2, 2, figsize=(S.WIDTH_2COL, 5.4))
fig.patch.set_facecolor("white")

def heat(ax, M, title, vmin=0.55, vmax=1.0):
    Z = M.loc[BASINS, BASINS].values.astype(float)
    # colour-blind-safe diverging map (RdBu family): red = collapse,
    # blue = stable, centred at 0.80 so the collapse/success split is visible
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.80, vmax=vmax)
    im = ax.imshow(Z, cmap=S.diverging_cmap(), norm=norm)
    ax.set_xticks(range(3)); ax.set_xticklabels(BASINS, fontsize=6.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(BASINS, fontsize=6.5)
    ax.set_xlabel("Target basin (zero-shot)", fontsize=7)
    ax.set_ylabel("Source basin (training)", fontsize=7)
    for i in range(3):
        for j in range(3):
            v = Z[i, j]
            diag = (i == j)
            ax.text(j, i, f"{v:.3f}" + ("\n(in-source)" if diag else ""),
                    ha="center", va="center", fontsize=6.0 if not diag else 5.2,
                    fontweight="bold" if not diag else "normal",
                    color="black",
                    linespacing=0.95,
                    bbox=dict(boxstyle="round,pad=0.12",
                              fc="white", alpha=0.55 if not diag else 0.85,
                              ec="none"))
    ax.set_title(title, fontsize=8, fontweight="bold", loc="left", pad=3)
    return im

im_a = heat(axes[0, 0], roc["TSCF"], "(a) Frozen contextual TSCF — ROC AUC")
heat(axes[0, 1], roc["TSAR"], "(b) Anchored residual TSAR — ROC AUC")
cb = fig.colorbar(im_a, ax=axes[0, 1], shrink=0.88, pad=0.03)
cb.set_label("ROC AUC (severe y(t+4))", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.5)

# ---- Panel c: ΔROC per direction ----
ax = axes[1, 0]
w = 0.38
for k, (m, color, mk) in enumerate((("TSCF", S.COLORS["tscf"], "o"),
                                    ("TSAR", S.COLORS["gated"], "s"))):
    xs, ys, los, his = [], [], [], []
    for di, (s, t) in enumerate(DIRS):
        row = dd[(dd["model"] == m) & (dd["source"] == s) & (dd["target"] == t)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        xs.append(di + (k - 0.5) * w)
        ys.append(row["d"]); los.append(row["d"] - row["lo"]); his.append(row["hi"] - row["d"])
    ax.errorbar(xs, ys, yerr=[los, his], fmt=mk, markersize=3.5, color=color,
                ecolor=color, elinewidth=0.9, capsize=1.8, label=m,
                linestyle="none")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xticks(range(len(DIRS)))
ax.set_xticklabels([f"{s}→{t}" for s, t in DIRS], rotation=40, ha="right",
                   fontsize=6.2)
ax.set_ylabel("ΔROC AUC (target − source)\nweek-block bootstrap 95% CI",
              fontsize=7)
ax.set_title("(c) Transfer degradation vs source:\ncollapse follows the TARGET basin",
             fontsize=8, fontweight="bold", loc="left", pad=3)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=6.5)

# ---- Panel d: seed scan per reverse-source direction ----
ax = axes[1, 1]
DIRS_D = [("GBR", "GBR"), ("GBR", "SCS"), ("GBR", "CARIB"),
          ("CARIB", "SCS"), ("CARIB", "GBR"), ("CARIB", "CARIB")]
w = 0.30
for k, (m, color, mk) in enumerate((("TSCF", S.COLORS["tscf"], "o"),
                                    ("TSAR", S.COLORS["gated"], "s"))):
    for di, (s, t) in enumerate(DIRS_D):
        vals = scan[(scan["source"] == s) & (scan["target"] == t)
                    & (scan["model"] == m)]["roc"].values
        ref = rm[(rm["source"] == s) & (rm["target"] == t)
                 & (rm["model"] == m)]["roc_auc"].values
        x0 = di + (k - 0.5) * w
        if len(vals):
            ax.hlines(np.min(vals), x0 - w * 0.30, x0 + w * 0.30,
                      color=color, linewidth=0.9, alpha=0.6)
            ax.scatter([x0] * len(vals), vals, s=9, color=color, alpha=0.85,
                       marker=mk, linewidths=0)
        if len(ref):
            ax.scatter([x0], ref, s=30, color=color, marker="D",
                       edgecolor="white", linewidth=0.6, zorder=5)
ax.axhline(0.80, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
ax.text(5.42, 0.80, "0.80", fontsize=5.5, va="bottom", ha="right", alpha=0.7)
ax.set_xticks(range(len(DIRS_D)))
ax.set_xticklabels([f"{s}→{t}" for s, t in DIRS_D], rotation=40, ha="right",
                   fontsize=6.2)
ax.set_ylabel("ROC AUC (severe y(t+4))", fontsize=7)
ax.set_ylim(0.55, 1.0)
ax.set_title("(d) Seed scan (seeds 0/1/2 + seed-42 ref.):\n"
             "unanchored transfer is a seed lottery, anchoring is not",
             fontsize=8, fontweight="bold", loc="left", pad=3)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", markersize=4,
           color=S.COLORS["tscf"], label="TSCF seeds 0/1/2"),
    Line2D([], [], marker="D", linestyle="none", markersize=5.5,
           markerfacecolor=S.COLORS["tscf"], markeredgecolor="white",
           label="TSCF seed 42 (ref.)"),
    Line2D([], [], marker="s", linestyle="none", markersize=4,
           color=S.COLORS["gated"], label="TSAR seeds 0/1/2"),
    Line2D([], [], marker="D", linestyle="none", markersize=5.5,
           markerfacecolor=S.COLORS["gated"], markeredgecolor="white",
           label="TSAR seed 42 (ref.)")],
    frameon=False, fontsize=5.8, loc="lower left", ncol=1)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=6.5)

plt.tight_layout(w_pad=1.0)
S.save_fig(fig, "fig2_transfer_matrix")

# source data
dd.to_csv(FIGDIR / "source_data" / "fig2_source_data_transfer_matrix.csv",
          index=False) if (FIGDIR / "source_data").exists() else (
    (FIGDIR / "source_data").mkdir(exist_ok=True),
    dd.to_csv(FIGDIR / "source_data" / "fig2_source_data_transfer_matrix.csv",
              index=False))
print(f"Saved {FIGDIR / 'fig2_transfer_matrix.png'}")
print("\nΔROC table:")
print(dd.pivot_table(index=["source", "target"], columns="model", values="d")
      .round(4).to_string())
