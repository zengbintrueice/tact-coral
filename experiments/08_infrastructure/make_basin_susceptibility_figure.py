# make_basin_susceptibility_figure.py
# Renders the core 4-panel figure for the Basin-Specific Susceptibility /
# Thermal-Response Heterogeneity test. ALL values are read back from the
# experiment's written result files (no hard-coded numbers):
#   Panel a: response_surface.csv   (Part A curves + Wilson CI)
#   Panel b: interaction_model.csv  (Part B slopes / interaction contrasts + CI)
#   Panel c: pairwise_contrasts.csv (Part C per-bin basin differences + CI)
#   Panel d: summary.json partD     (matched-DHW R-tertile severe rate + CI)
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
OUT = LTTS_ROOT / "results" / "basin_susceptibility"

COL = {"SCS": "#2166AC", "GBR": "#4393C3", "CARIB": "#B2182B"}
DOMS = ["SCS", "GBR", "CARIB"]

resp = pd.read_csv(OUT / "response_surface.csv")
inter = pd.read_csv(OUT / "interaction_model.csv")
pair = pd.read_csv(OUT / "pairwise_contrasts.csv")
summary = json.load(open(OUT / "summary.json", encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))

# ---- Panel a: conditional response curves ----
ax = axes[0, 0]
for dom in DOMS:
    d = resp[(resp["part"] == "A") & (resp["domain"] == dom)].sort_values("dhw_bin_lo")
    x = (d["dhw_bin_lo"] + d["dhw_bin_hi"]) / 2
    y = d["severe_rate"] * 100
    ax.errorbar(x, y, yerr=[(y - d["wilson_lo"] * 100), (d["wilson_hi"] * 100 - y)],
                fmt="o-", color=COL[dom], capsize=2.5, lw=1.4, ms=4.5, label=dom)
lo = summary["common_support"]["lo"]; hi = summary["common_support"]["hi"]
ax.axvspan(lo, hi, color="#dddddd", alpha=0.35, zorder=0)
ax.set_xlabel("current DHW / PersA (bin centre)")
ax.set_ylabel("future severe probability  y(t+4)≥0.8  [%]")
ax.set_title("a  Conditional thermal-response curves (common DHW support)", loc="left", fontsize=10, fontweight="bold")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25)

# ---- Panel b: slopes + interaction contrasts (logistic) ----
ax = axes[0, 1]
rows = []
for dom in DOMS:
    r = inter[(inter["part"] == "B") & (inter["model"] == "logistic") & (inter["term"] == f"slope_{dom}")]
    if len(r):
        r = r.iloc[0]
        rows.append((f"slope {dom}", r["value"], r["ci_lo"], r["ci_hi"]))
for name in ["GBR-SCS", "CARIB-SCS", "CARIB-GBR"]:
    r = inter[(inter["part"] == "B") & (inter["model"] == "logistic") & (inter["term"] == f"contrast_{name}")]
    if len(r):
        r = r.iloc[0]
        rows.append((f"DHW×basin {name}", r["value"], r["ci_lo"], r["ci_hi"]))
ypos = np.arange(len(rows))[::-1]
for yp, (lab, v, lo_, hi_) in zip(ypos, rows):
    excl = (lo_ > 0 or hi_ < 0)
    ax.errorbar(v, yp, xerr=[[v - lo_], [hi_ - v]], fmt="D",
                color="#1b7837" if excl else "#999999", capsize=3, ms=5, lw=1.3)
    ax.text(hi_ + 0.012, yp, f"{v:+.3f} [{lo_:+.3f},{hi_:+.3f}]" + ("" if excl else " (n.s.)"),
            va="center", fontsize=6.4, color="#333333")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel("log-odds slope per unit PersA (logistic)")
ax.set_title("b  Basin slopes and DHW×basin interaction (95% CI)", loc="left", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)

# ---- Panel c: per-bin basin risk differences ----
ax = axes[1, 0]
cmap = {"GBR-SCS": "#4393C3", "CARIB-SCS": "#B2182B", "CARIB-GBR": "#7f3b97"}
for name, col in cmap.items():
    d = pair[(pair["part"] == "C") & (pair["contrast"] == f"bin_severe_rate {name}")].copy()
    if not len(d):
        continue
    d["bin"] = d["section"].str.extract(r"\[([\d.]+),")[0].astype(float)
    d = d.sort_values("bin")
    x = d["bin"] + 0.05
    y = d["value"] * 100
    ax.errorbar(x, y, yerr=[(y - d["ci_lo"] * 100), (d["ci_hi"] * 100 - y)],
                fmt="s--", color=col, capsize=2.5, lw=1.2, ms=4, label=name)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("current DHW / PersA bin")
ax.set_ylabel("severe-rate difference [pp]")
ax.set_title("c  Basin risk difference within common DHW bins", loc="left", fontsize=10, fontweight="bold")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25)

# ---- Panel d: matched-DHW R-tertile severe probability ----
ax = axes[1, 1]
w = 0.26
for i, dom in enumerate(DOMS):
    pd_ = summary["partD"][dom]
    qs = [1, 2, 3]
    vals = [pd_["R_tertile_severe_rate"][str(q)] * 100 for q in qs]
    los = [pd_["R_tertile_ci"][str(q)][0] * 100 for q in qs]
    his = [pd_["R_tertile_ci"][str(q)][1] * 100 for q in qs]
    xs = np.arange(3) + (i - 1) * w
    ax.bar(xs, vals, width=w * 0.92, color=COL[dom], label=dom, alpha=0.9)
    ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt="none", ecolor="#333333", capsize=2.5, lw=1.0)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(["R Q1 (low)", "R Q2", "R Q3 (high)"], fontsize=8.5)
ax.set_ylabel("future severe probability [%]")
psp = {d: summary["partD"][d]["partial_spearman_R_future_given_DHW"] for d in DOMS}
ax.set_title("d  Matched-DHW: residual R-tertile vs severe probability\n" +
             "  ".join(f"{d} ρ={psp[d]:+.2f}" for d in DOMS),
             loc="left", fontsize=9.5, fontweight="bold")
ax.legend(frameon=False, fontsize=8, title="basin", title_fontsize=8)
ax.grid(axis="y", alpha=0.25)

fig.text(0.01, 0.005,
         "Frozen records, PFV y(t+4); severe = y≥0.8; common DHW support only; "
         "week-block bootstrap 1000 seed 42. Associative/predictive only — no causal claim.",
         fontsize=6.4, style="italic")
fig.tight_layout(rect=[0, 0.02, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"panel_figure.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {OUT / 'panel_figure.png'} (+pdf) from result files")