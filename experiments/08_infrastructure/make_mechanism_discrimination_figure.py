# make_mechanism_discrimination_figure.py
# Final mechanism figure — ALL values read back from results/mechanism_discrimination/.
#   Panel a: dhw_matched_surface.csv  -> DHW x R conditional response (Q1 vs Q3) per basin
#   Panel b: dhw_matched_surface.csv  -> per-bin Q3-Q1 risk difference + 95% CI
#   Panel c: paired_deltas.csv        -> incremental ROC/PR ladder M0->M1->M2->TSAR (+M2+R)
#   Panel d: summary.json evidence    -> H1 vs H2 evidence matrix
import io, sys, json, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LTTS_ROOT = Path(__file__).resolve().parents[2]
OUT = LTTS_ROOT / "results" / "mechanism_discrimination"
COL = {"SCS": "#2166AC", "GBR": "#4393C3", "CARIB": "#B2182B"}
DOMS = ["SCS", "GBR", "CARIB"]

surf = pd.read_csv(OUT / "dhw_matched_surface.csv")
deltas = pd.read_csv(OUT / "paired_deltas.csv")
summary = json.load(open(OUT / "summary.json", encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))

# ---- Panel a: DHW x R conditional response ----
ax = axes[0, 0]
for dom in DOMS:
    d = surf[surf["basin"] == dom].assign(
        center=(surf["bin_lo"] + surf["bin_hi"]) / 2).sort_values("center")
    ax.plot(d["center"], d["rate_Q1"] * 100, "s--", color=COL[dom], ms=4, lw=1.1,
            alpha=0.75, label=f"{dom} R-Q1")
    ax.plot(d["center"], d["rate_Q3"] * 100, "o-", color=COL[dom], ms=4.5, lw=1.5,
            label=f"{dom} R-Q3")
ax.set_xlabel("current DHW (degC-weeks, matched bin centre)")
ax.set_ylabel("future severe probability y(t+4)≥0.8 [%]")
ax.set_title("a  DHW-matched response: R-Q1 vs R-Q3", loc="left",
             fontsize=10, fontweight="bold")
ax.legend(frameon=False, fontsize=6, ncol=2)
ax.grid(alpha=0.25)

# ---- Panel b: per-bin Q3-Q1 risk difference + CI ----
ax = axes[0, 1]
for dom in DOMS:
    d = surf[surf["basin"] == dom].assign(
        center=(surf["bin_lo"] + surf["bin_hi"]) / 2).sort_values("center")
    y = d["risk_diff_Q3_Q1"] * 100
    ax.errorbar(d["center"], y,
                yerr=[y - d["ci_lo"] * 100, d["ci_hi"] * 100 - y],
                fmt="o", color=COL[dom], capsize=2.5, lw=1.2, ms=4.5, label=dom)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("current DHW (degC-weeks, matched bin centre)")
ax.set_ylabel("severe-rate difference Q3−Q1 [pp]")
ax.set_title("b  Within-DHW residual risk gradient (95% CI)", loc="left",
             fontsize=10, fontweight="bold")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25)

# ---- Panel c: incremental ladder ROC/PR ----
ax = axes[1, 0]
order = ["M1_minus_M0", "M2_minus_M1", "M2+R_minus_M2", "TSAR_minus_M2"]
labels = ["M1−M0\n(+trajectory)", "M2−M1\n(+context)", "M2+R−M2\n(+residual R)", "TSAR−M2\n(frozen model)"]
w = 0.38
for mi, (metric, col, hatch) in enumerate((("roc", "#2166AC", ""), ("pr", "#B2182B", "//"))):
    for di, dom in enumerate(DOMS):
        vals, los, his = [], [], []
        for c in order:
            r = deltas[(deltas["basin"] == dom) & (deltas["contrast"] == c)
                       & (deltas["metric"] == metric)]
            vals.append(r["point"].iloc[0] if len(r) else np.nan)
            los.append(r["ci_lo"].iloc[0] if len(r) else np.nan)
            his.append(r["ci_hi"].iloc[0] if len(r) else np.nan)
        xs = np.arange(len(order)) + (di - 1) * 0.22 + (mi - 0.5) * 0.42
        ax.bar(xs, vals, width=0.19, color=col, alpha=0.55 + 0.15 * di,
               hatch=hatch, edgecolor="white", lw=0.4,
               label=f"{metric.upper()} {dom}" if di == 1 else None)
        ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(los),
                                     np.array(his) - np.array(vals)],
                    fmt="none", ecolor="#333333", capsize=1.5, lw=0.7)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("paired ΔAUC (95% CI, week-block)")
ax.set_title("c  Incremental ladder M0→M1→M2, +R, →TSAR", loc="left",
             fontsize=10, fontweight="bold")
ax.legend(frameon=False, fontsize=6.5, ncol=2)
ax.grid(axis="y", alpha=0.25)

# ---- Panel d: evidence matrix ----
ax = axes[1, 1]
ax.axis("off")
ev = summary.get("evidence", {})
rows = []
for dom in DOMS:
    e = ev.get(dom, {})
    rows.append((dom,
                 "YES" if e.get("E1_partial_assoc_excl_0") else "no",
                 f"{e.get('TSAR_minus_M2_roc', {}).get('point', float('nan')):+.4f}",
                 "YES" if e.get("E5_M2plusR_gain") else "no"))
gci = ev.get("GBR_vs_CARIB_gradient_diff_ci", [float("nan")] * 2)
e3 = ev.get("E3_effect_size_differs", False)
cell = [["Basin", "E1 partial ρ\nCI excl 0", "TSAR−M2 ΔROC", "E5 M2+R\ngain"]]
for r in rows:
    cell.append(list(r))
tbl = ax.table(cellText=cell[1:], colLabels=cell[0], loc="center",
               cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.5)
ax.set_title("d  Evidence matrix (target basins GBR / CARIB; E3 gradient-diff "
             f"CI [{gci[0]:+.3f}, {gci[1]:+.3f}] "
             f"{'excl 0' if e3 else 'incl 0'})",
             loc="left", fontsize=9, fontweight="bold")
fig.text(0.01, 0.005,
         "Frozen records; R = TSAR(seed-42) − PersA; severe y(t+4)≥0.8; week-block bootstrap "
         "1000 seed 42; associative / representation-level only — no causal claim.",
         fontsize=6.2, style="italic")
fig.tight_layout(rect=[0, 0.02, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"mechanism_figure.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {OUT / 'mechanism_figure.png'} (+pdf)")