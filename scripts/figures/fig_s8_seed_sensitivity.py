# -*- coding: utf-8 -*-
"""Supplementary Fig. S8 — Random-seed sensitivity of the zero-shot transfer
boundary.

Reads results/seed_sensitivity/{seed_metrics.csv, seed_transfer_boundary.csv}
(NOTHING hard-coded) and draws:

  a  per-seed ROC / PR / CSI@0.9 on GBR and CARIB (individual seed points
     emphasised, paired by seed with light lines) + mean ± 95% CI (t-based,
     n=5, descriptive only — NO significance testing)
  b  ΔPR = PR_GBR − PR_CARIB per seed
  c  ΔCSI = CSI_GBR − CSI_CARIB per seed
  + takeaway: the qualitative transfer boundary was preserved across seeds,
    although absolute performance varied substantially with random
    initialization.

Seeds are COMPUTATIONAL replicates, not biological replicates; CIs are
descriptive. No higher-level claim (e.g. "robust across all seeds",
"significant") is made.

Outputs: results/figures/S8_seed_sensitivity.{pdf,svg,png(600dpi)},
         results/figures/source_data/S8_seed_sensitivity.csv,
         manifest rows, printed QC summary (completeness / CI / NaN / duplicates).

Run:
    cd LTTS_CRW/scripts/figures
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' fig_s8_seed_sensitivity.py
"""
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from scipy import stats

import nc_style as S

SEEDS = [0, 1, 2, 3, 42]
DOMAINS = ["GBR", "CARIB"]
METRICS = [("ROC", "ROC-AUC"), ("PR", "PR-AUC"), ("CSI", "CSI@0.9")]

# ------------------------------------------------------------- load + QC ----
metrics_csv = S.RESULTS / "seed_sensitivity" / "seed_metrics.csv"
bound_csv = S.RESULTS / "seed_sensitivity" / "seed_transfer_boundary.csv"
DF = S.require(metrics_csv, cols=["seed", "domain", "protocol", "ROC", "PR",
                                  "CSI", "POD", "FAR", "F1", "Brier", "ECE"],
               fig="S8")
BOR = S.require(bound_csv, cols=["seed", "PR_GBR", "PR_CARIB", "dPR",
                                 "CSI_GBR", "CSI_CARIB", "dCSI"], fig="S8")

std = DF[DF.protocol == "standard"]
sub = std[std.domain.isin(DOMAINS) & std.seed.isin(SEEDS)]

qc = {}
for d in DOMAINS:
    sd = sub[sub.domain == d]
    assert sorted(sd.seed.tolist()) == sorted(SEEDS), \
        f"S8 QC FAIL: {d} seeds incomplete: {sorted(sd.seed.tolist())}"
    assert sd[["ROC", "PR", "CSI"]].notna().all().all(), \
        f"S8 QC FAIL: NaN in plotted metrics for {d}"
qc["completeness"] = "5 seeds × GBR/CARIB complete (standard protocol), no NaN"

dup = sub.duplicated(subset=["seed", "domain"]).any()
assert not dup, "S8 QC FAIL: duplicated (seed,domain) records"
qc["duplicates"] = "none on (seed, domain)"

recomp = []
for seed in SEEDS:
    g = float(sub[(sub.seed == seed) & (sub.domain == "GBR")].PR.iloc[0])
    c = float(sub[(sub.seed == seed) & (sub.domain == "CARIB")].PR.iloc[0])
    g2 = float(sub[(sub.seed == seed) & (sub.domain == "GBR")].CSI.iloc[0])
    c2 = float(sub[(sub.seed == seed) & (sub.domain == "CARIB")].CSI.iloc[0])
    recomp.append((seed, g - c, g2 - c2))
for (seed, dpr, dcsi) in recomp:
    row = BOR[BOR.seed == seed].iloc[0]
    assert abs(dpr - float(row.dPR)) < 1e-9, f"S8 QC FAIL: ΔPR mismatch seed {seed}"
    assert abs(dcsi - float(row.dCSI)) < 1e-9, f"S8 QC FAIL: ΔCSI mismatch seed {seed}"
qc["cross_file_consistency"] = ("ΔPR/ΔCSI recomputed from seed_metrics.csv match "
                                "seed_transfer_boundary.csv (|Δ|<1e-9)")

# --------------------------------------------------------- derived values ---
val = {(m, d): np.array([float(sub[(sub.seed == s) & (sub.domain == d)][m].iloc[0])
                         for s in SEEDS]) for m, _ in METRICS for d in DOMAINS}
T_CRIT = float(stats.t.ppf(0.975, len(SEEDS) - 1))  # t(0.975, 4) — descriptive CI

def mean_ci(v):
    m = float(np.mean(v))
    half = T_CRIT * float(np.std(v, ddof=1)) / np.sqrt(len(v))
    return m, m - half, m + half

dpr = BOR.sort_values("seed").dPR.values
dcsi = BOR.sort_values("seed").dCSI.values
n_nonneg_dpr = int((dpr >= 0).sum())
n_tie_dcsi = int((dcsi == 0).sum())
n_nonneg_dcsi = int((dcsi >= 0).sum())

# ---------------------------------------------------------------- figure ----
fig = plt.figure(figsize=(7.08, 4.3))
fig.suptitle("Random-seed sensitivity of the zero-shot transfer boundary",
             fontsize=9, y=0.985)
gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1.0, 1.0],
                      wspace=0.36, left=0.08, right=0.97, top=0.86, bottom=0.13)

# --------------------------------------------------------------- (a) --------
ax = fig.add_subplot(gs[0, 0])
ax.set_title("Per-seed performance on the two zero-shot basins", loc="left",
             fontsize=8)
rng = np.random.default_rng(7)
for mi, (m, mlab) in enumerate(METRICS):
    # paired seed lines GBR -> CARIB (kept: the per-seed pairing is informative)
    for si, s in enumerate(SEEDS):
        y1 = val[(m, "GBR")][si]
        y2 = val[(m, "CARIB")][si]
        ax.plot([mi - 0.31, mi + 0.31], [y1, y2], color="#D8D8D8", lw=0.55, zorder=0)
    for di, d in enumerate(DOMAINS):
        x = mi + (di - 0.5) * 0.62
        v = val[(m, d)]
        jit = rng.uniform(-0.035, 0.035, len(v))
        # individual seed points EMPHASISED
        ax.plot(x + jit, v, "o", ms=5.2, mfc=S.DOMAIN_COLOR[d],
                mec="white", mew=1.0, alpha=1.0, zorder=4)
        # mean ± 95% CI as auxiliary (smaller, behind the points)
        m_, lo, hi = mean_ci(v)
        xm = x + (0.18 if d == "GBR" else -0.18)
        ax.errorbar(xm, m_, yerr=[[m_ - lo], [hi - m_]], fmt="D", ms=4.0,
                    color="black", mfc="white", mew=1.1, capsize=2.2, lw=1.0,
                    zorder=3)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([m[1] for m in METRICS])
ax.set_xlim(-0.65, 2.65)
ax.set_ylim(-0.03, 1.03)
ax.set_ylabel("score")
seed_handles = [
    plt.Line2D([], [], marker="o", ls="", mfc=S.COLORS["GBR"], mec="white", mew=1.0),
    plt.Line2D([], [], marker="o", ls="", mfc=S.COLORS["CARIB"], mec="white", mew=1.0),
    plt.Line2D([], [], marker="D", ls="", mfc="white", mec="k"),
]
handles = [tuple(seed_handles),
           plt.Line2D([], [], color="#D8D8D8", lw=0.8)]
labels = ["individual seeds; mean ± 95% CI (t, n=5)",
          "paired by seed"]
ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=3)},
          loc="upper right", fontsize=5.4)
S.panel_label(ax, "a")

# --------------------------------------------------------------- (b) --------
ax = fig.add_subplot(gs[0, 1])
ax.set_title("Transfer boundary: ΔPR", loc="left", fontsize=8)
m_, lo, hi = mean_ci(dpr)
ax.axhspan(lo, hi, color=S.COLORS["gated"], alpha=0.14, zorder=0)
ax.axhline(m_, color=S.COLORS["gated"], lw=0.9, ls="--", zorder=1)
ax.axhline(0, color="k", lw=0.7, zorder=2)
ax.plot(np.arange(len(SEEDS)), dpr, "o", ms=5, color=S.COLORS["gated"],
        mec="black", mew=0.5, zorder=3)
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels([str(s) for s in SEEDS])
ax.set_xlabel("random seed")
ax.set_ylabel("ΔPR = PR$_{GBR}$ − PR$_{CARIB}$")
ax.set_ylim(-0.03, max(dpr) * 1.25)
ax.text(0.97, 0.06, f"ΔPR ≥ 0 in {n_nonneg_dpr}/{len(SEEDS)} seeds",
        transform=ax.transAxes, ha="right", fontsize=6.0,
        bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, alpha=0.9, pad=1.5))
S.panel_label(ax, "b")

# --------------------------------------------------------------- (c) --------
ax = fig.add_subplot(gs[0, 2])
ax.set_title("Transfer boundary: ΔCSI", loc="left", fontsize=8)
m_, lo, hi = mean_ci(dcsi)
ax.axhspan(lo, hi, color=S.COLORS["gated"], alpha=0.14, zorder=0)
ax.axhline(m_, color=S.COLORS["gated"], lw=0.9, ls="--", zorder=1)
ax.axhline(0, color="k", lw=0.7, zorder=2)
ax.plot(np.arange(len(SEEDS)), dcsi, "o", ms=5, color=S.COLORS["gated"],
        mec="black", mew=0.5, zorder=3)
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels([str(s) for s in SEEDS])
ax.set_xlabel("random seed")
ax.set_ylabel("ΔCSI = CSI$_{GBR}$ − CSI$_{CARIB}$")
ax.set_ylim(-0.03, max(dcsi) * 1.25)
ax.text(0.97, 0.06,
        f"ΔCSI ≥ 0 in {n_nonneg_dcsi}/{len(SEEDS)} seeds "
        f"(one tie{'s' if n_tie_dcsi > 1 else ''})",
        transform=ax.transAxes, ha="right", fontsize=6.0,
        bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, alpha=0.9, pad=1.5))
S.panel_label(ax, "c")

# takeaway + provenance note
fig.text(0.01, 0.05,
         "The qualitative transfer boundary was preserved across seeds, "
         "although absolute performance varied substantially with random "
         "initialization.",
         fontsize=6.4, style="italic")
fig.text(0.01, 0.012,
         "Seeds are computational replicates (random initialization), not "
         "biological replicates; CIs are descriptive (t-based, n=5); "
         "no significance testing. τ=0.90 pinned; ocean-masked PFV record sets.",
         fontsize=5.4, style="italic", color="#444444")

out = S.save_fig(fig, "S8_seed_sensitivity")

# ------------------------------------------------------ source data CSV -----
rows = []
for (m, mlab) in METRICS:
    for d in DOMAINS:
        v = val[(m, d)]
        m_, lo, hi = mean_ci(v)
        for si, s in enumerate(SEEDS):
            rows.append(dict(panel="a", seed=s, domain=d, metric=mlab,
                             value=float(v[si]), mean=np.nan, ci_lo=np.nan,
                             ci_hi=np.nan))
        rows.append(dict(panel="a", seed="mean", domain=d, metric=mlab,
                         value=m_, mean=m_, ci_lo=lo, ci_hi=hi))
for si, s in enumerate(SEEDS):
    rows.append(dict(panel="b", seed=s, domain="GBR−CARIB", metric="ΔPR",
                     value=float(dpr[si]), mean=np.nan, ci_lo=np.nan, ci_hi=np.nan))
    rows.append(dict(panel="c", seed=s, domain="GBR−CARIB", metric="ΔCSI",
                     value=float(dcsi[si]), mean=np.nan, ci_lo=np.nan, ci_hi=np.nan))
for panel, v, mlab in [("b", dpr, "ΔPR"), ("c", dcsi, "ΔCSI")]:
    m_, lo, hi = mean_ci(v)
    rows.append(dict(panel=panel, seed="mean", domain="GBR−CARIB", metric=mlab,
                     value=m_, mean=m_, ci_lo=lo, ci_hi=hi))
sd = pd.DataFrame(rows)
sd.to_csv(S.SRC_DIR / "S8_seed_sensitivity.csv", index=False)
print("[source] S8_seed_sensitivity.csv", flush=True)

# ------------------------------------------------------------- manifest -----
new = pd.DataFrame([
    dict(figure="S8", panel="a", source_file="results/seed_sensitivity/seed_metrics.csv",
         source_columns="seed,domain,protocol=standard,{ROC,PR,CSI}",
         statistical_method="individual seeds + mean ± 95% CI (t-based, n=5, descriptive)",
         n=len(SEEDS), unit="score", output_file="S8_seed_sensitivity.pdf"),
    dict(figure="S8", panel="b", source_file="results/seed_sensitivity/seed_transfer_boundary.csv",
         source_columns="seed,dPR",
         statistical_method="per-seed ΔPR; mean ± 95% CI (t-based, descriptive)",
         n=len(SEEDS), unit="Δscore", output_file="S8_seed_sensitivity.pdf"),
    dict(figure="S8", panel="c", source_file="results/seed_sensitivity/seed_transfer_boundary.csv",
         source_columns="seed,dCSI",
         statistical_method="per-seed ΔCSI; mean ± 95% CI (t-based, descriptive)",
         n=len(SEEDS), unit="Δscore", output_file="S8_seed_sensitivity.pdf"),
])
man = S.OUT_DIR / "figure_manifest.csv"
if man.exists():
    old = pd.read_csv(man)
    new = pd.concat([old[old.figure != "S8"], new], ignore_index=True)
new.to_csv(man, index=False)
print("[manifest] S8 rows written", flush=True)

# ------------------------------------------------------------------ QC ------
print("\n" + "=" * 60)
print("S8 QC SUMMARY")
print("=" * 60)
print(f"  {qc['completeness']}")
print(f"  duplicates: {qc['duplicates']}")
print(f"  {qc['cross_file_consistency']}")
print(f"  CI method: mean ± t(0.975, df={len(SEEDS)-1})·sd/√n "
      f"(t_crit={T_CRIT:.3f}); descriptive only, no significance testing")
for m, _ in METRICS:
    for d in DOMAINS:
        m_, lo, hi = mean_ci(val[(m, d)])
        print(f"  {d:5s} {m:3s}: mean={m_:.4f}  95%CI [{lo:.4f}, {hi:.4f}]")
print(f"  ΔPR : mean={mean_ci(dpr)[0]:+.4f} 95%CI [{mean_ci(dpr)[1]:+.4f}, "
      f"{mean_ci(dpr)[2]:+.4f}]  (≥0 in {n_nonneg_dpr}/5)")
print(f"  ΔCSI: mean={mean_ci(dcsi)[0]:+.4f} 95%CI [{mean_ci(dcsi)[1]:+.4f}, "
      f"{mean_ci(dcsi)[2]:+.4f}]  (≥0 in {n_nonneg_dcsi}/5; ties={n_tie_dcsi})")
print(f"  outputs: {[o.name for o in out]}")
print("  takeaway: qualitative boundary preserved; absolute performance varied "
      "with initialization (no significance claim).")
print("S8 DONE")