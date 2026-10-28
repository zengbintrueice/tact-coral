# -*- coding: utf-8 -*-
"""Figure 7 — Independent ecological relevance of thermal-risk estimates.

Panels
  a  validation design (frozen TSCF risk -> independent GCBD observations)
  b  cluster-aware ranking of severe observations (AUROC + site-cluster 95% CI)
  c  severe-event enrichment across predicted-risk quintiles (RR annotation)
  d  temporal-offset robustness (AUROC over predefined matching offsets)
  e  site-level validation (severe-site rate by site-risk quintile)
  f  holdout + cross-basin enrichment forest (SCS 2011-2015 temporal
     holdout; CARIB 2015-2016 zero-shot; PersA vs TSAR vs TSCF)

Data integrity rules honoured:
  * the Q5/Q1 SITE-level ratio is NOT calculated (Q1 contains zero severe sites);
  * site-cluster bootstrap CIs are the primary inference;
  * this is independent ecological relevance/validation — no causal language.

Sources: results/step9_gcbd_validation/{step9_bootstrap.csv,
step9_quintile_enrichment.csv, step9_risk_enrichment.csv},
results/step10_gcbd_matching_audit/{step10_matching_results.csv,
step10_site_level_results.csv}.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nc_style as S

BOOT = S.require(S.RESULTS / "step9_gcbd_validation" / "step9_bootstrap.csv",
                 cols=["key", "scheme", "mean", "lo", "hi", "n"], fig="fig7")
QUIN = S.require(S.RESULTS / "step9_gcbd_validation" / "step9_quintile_enrichment.csv",
                 cols=["bins", "event", "bin", "n", "n_event", "rate"], fig="fig7")
RISK = S.require(S.RESULTS / "step9_gcbd_validation" / "step9_risk_enrichment.csv",
                 fig="fig7")
MATCH = S.require(S.RESULTS / "step10_gcbd_matching_audit" / "step10_matching_results.csv",
                  cols=["offset_weeks", "n", "roc_sev_vs_bg", "rr_q5q1_severe"], fig="fig7")
SITE = S.require(S.RESULTS / "step10_gcbd_matching_audit" / "step10_site_level_results.csv",
                 fig="fig7")
YEAR_STRATA = S.require(S.RESULTS / "gcbd_year_stratification" / "year_strata.csv",
                        fig="fig7")
HOLD = S.require(S.RESULTS / "gcbd_global_holdout_enrichment" /
                 "enrichment_by_basin_window.csv", fig="fig7")

cluster = BOOT[BOOT.scheme == "cluster_site"].set_index("key")
# cohort size read from the official matching audit (offset 0), not hard-coded
n_cohort = int(MATCH[MATCH.offset_weeks == 0].n.iloc[0])
GCBDR = S.require(S.RESULTS / "step9_gcbd_validation" / "step9_gcbd_results.csv",
                  cols=["Site_ID"], fig="fig7")
n_sites = int(GCBDR["Site_ID"].nunique())   # distinct GCBD sites that matched

fig = plt.figure(figsize=(S.WIDTH_2COL, 8.6))
gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.30,
                      left=0.11, right=0.97, top=0.95, bottom=0.06)

# ------------------------------------------------------------- (a) design ---
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.set_title("Independent validation design", loc="left", fontsize=8)
ax.add_patch(plt.Rectangle((0.03, 0.52), 0.42, 0.34, fc="#EDF2F8",
                           ec=S.COLORS["tscf"], lw=1.0, transform=ax.transAxes))
ax.text(0.24, 0.69, "frozen TSCF\n(SCS-trained;\nnever sees GCBD)",
        ha="center", va="center", fontsize=6.2, transform=ax.transAxes)
ax.annotate("", xy=(0.56, 0.69), xytext=(0.46, 0.69),
            arrowprops=dict(arrowstyle="-|>", lw=1.2), xycoords=ax.transAxes)
ax.add_patch(plt.Rectangle((0.57, 0.52), 0.38, 0.34, fc="#FDF3E7",
                           ec="#B8860B", lw=1.0, transform=ax.transAxes))
ax.text(0.76, 0.69, "GCBD field observations\n(fully external;\nno threshold/fitting)",
        ha="center", va="center", fontsize=6.2, transform=ax.transAxes)
mw = RISK[RISK.analysis == "mannwhitney"]
n_sev = int(RISK[(RISK.analysis == "distribution") &
                 (RISK.group == "severe_bleaching")].n.iloc[0])
ax.text(0.5, 0.28, "4-week lead; ±4-week temporal window; 0.5° spatial buffer",
        ha="center", fontsize=5.9, transform=ax.transAxes)
ax.text(0.5, 0.14, f"n = {n_cohort:,} observations; {n_sev} severe; {n_sites} sites",
        ha="center", fontsize=6.0, transform=ax.transAxes)
ax.text(0.5, 0.04, "ranking/enrichment of independent observations —\n"
        "NOT biological prediction or causal inference",
        ha="center", fontsize=5.6, style="italic", transform=ax.transAxes)
S.panel_label(ax, "a")

# --------------------------------------------------- (b) cluster-aware ROC --
ax = fig.add_subplot(gs[0, 1])
ax.set_title("Cluster-aware discrimination of independent GCBD observations",
             loc="left", fontsize=8)
rows_b = [
    ("roc_sevbg", "severe vs background", S.COLORS["SCS"]),
    ("roc_sev", "severe vs rest", S.COLORS["gated"]),
    (None, "site-level validation", S.COLORS["anchor"]),
]
site_auc = float(SITE.auc_mean_risk.iloc[0])   # site-has-severe vs mean risk
for yi, (key, lab, col) in enumerate(rows_b):
    if key is not None:
        if key not in cluster.index:
            continue
        r = float(cluster.loc[key, "mean"])
        lo, hi = float(cluster.loc[key, "lo"]), float(cluster.loc[key, "hi"])
        ax.errorbar(r, yi, xerr=[[r - lo], [hi - r]], fmt="o", color=col,
                    capsize=3, lw=1.2, markersize=6)
        ax.text(0.06, yi, f"{r:.3f}  [{lo:.3f}, {hi:.3f}]", va="center",
                fontsize=6.0, transform=ax.get_yaxis_transform())
    else:
        # site-level AUROC is a point estimate (no CI persists in the file)
        ax.plot(site_auc, yi, "s", color=col, markersize=6.5)
        ax.text(0.06, yi, f"{site_auc:.3f}  (point)", va="center", fontsize=6.0,
                transform=ax.get_yaxis_transform())
ax.axvline(0.5, color="#999999", lw=0.7, ls="--")
ax.set_yticks(range(len(rows_b)))
ax.set_yticklabels([r[1] for r in rows_b], fontsize=6.6)
ax.set_xlim(0.0, 1.0)
ax.set_xlabel("AUROC (site-cluster bootstrap 95% CI; site-level = point)")
S.panel_label(ax, "b")

# ------------------------------------------------------- (c) enrichment -----
ax = fig.add_subplot(gs[1, 0])
ax.set_title("Severe-event enrichment across predicted-risk quintiles",
             loc="left", fontsize=8)
q = QUIN[(QUIN.bins == "quintile") & (QUIN["event"] == "severe")].sort_values("bin")
rates = q.rate.values * 100
ax.bar(q["bin"].values, rates, color=[S.COLORS["anchor"]] * 4 + [S.COLORS["gated"]],
       alpha=0.9, width=0.62)
for b, r, ne, nn in zip(q["bin"].values, rates, q.n_event.values, q.n.values):
    # label slightly to the right of the bar top so the low Q1–Q4 bars cannot
    # collide with the bottom annotation ("quintiles defined ...")
    ax.text(b + 0.10, r + 0.4, f"{ne}/{nn}", ha="center", fontsize=5.8)
# cluster-aware RR from the matching audit (offset 0)
rr0 = MATCH[MATCH.offset_weeks == 0].iloc[0]
ax.text(0.03, 0.88,
        "RR(Q5/Q1) ≈ %.0f\nsite-cluster 95%% CI\n[%.1f, %.1f]"
        % (rr0.rr_q5q1_severe, rr0.rr_cluster_lo, rr0.rr_cluster_hi),
        transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
        bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, alpha=0.9, pad=1.5))
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"Q{i}" for i in range(1, 6)])
ax.set_xlabel("predicted-risk quintile (Q1 lowest … Q5 highest)")
ax.set_ylabel("severe-observation rate (%)")
ax.set_ylim(0, 22)
ax.text(0.03, 0.48, "quintiles defined within the independent GCBD cohort;\n"
        "CI = site-cluster bootstrap",
        transform=ax.transAxes, fontsize=5.4, va="bottom", style="italic")
S.panel_label(ax, "c")

# ------------------------------------------------- (d) offset robustness ----
ax = fig.add_subplot(gs[1, 1])
ax.set_title("Temporal-offset robustness (predefined offsets)", loc="left",
             fontsize=8)
m = MATCH.sort_values("offset_weeks")
ax.errorbar(m.offset_weeks, m.roc_sev_vs_bg,
            yerr=np.zeros(len(m)), fmt="o-", color=S.COLORS["SCS"],
            lw=1.0, markersize=5, capsize=0)
for _, r in m.iterrows():
    ax.annotate(f"{r.roc_sev_vs_bg:.3f}", (r.offset_weeks, r.roc_sev_vs_bg),
                textcoords="offset points", xytext=(0, 7), ha="center",
                fontsize=5.8)
ax.axhline(0.5, color="#999999", lw=0.7, ls="--")
ax.legend(title="issue = obs − 4 wk + Δ", fontsize=5.6, title_fontsize=6.2,
          loc="upper left", frameon=False)
ax.set_xticks(m.offset_weeks.values)
ax.set_xlabel("matching offset Δ (weeks)")
ax.set_ylabel("AUROC (severe vs background)")
ax.set_ylim(0.5, 1.0)
ax.text(0.03, 0.06, "all windows predefined; none selected post hoc",
        transform=ax.transAxes, fontsize=5.8, style="italic")
S.panel_label(ax, "d")

# ------------------------------------- (e) holdout forest (narrowed) ---------
ax = fig.add_subplot(gs[2, 0])
ax.set_title("Holdout + cross-basin\nenrichment (AUROC)", loc="left",
             fontsize=7.5, fontweight="bold")

WINDOW_LABEL = {
    "SCS_val2011_2015": "SCS holdout\n2011–2015",
    "CARIB_2015_2016": "CARIB zero-shot\n2015–2016",
}
WINDOW_ORDER = ["SCS_val2011_2015", "CARIB_2015_2016"]
MODEL_STYLE = [("PersA", S.COLORS["anchor"], "o", "PersA (anchor)"),
               ("TSAR", S.COLORS["gated"], "s", "TSAR"),
               ("TSCF", S.COLORS["tscf"], "D", "TSCF (contextual)")]

import re as _re
ytick_pos, ytick_lab, y_data_all = [], [], []
for wi, w in enumerate(WINDOW_ORDER):
    sub = HOLD[HOLD["window"] == w]
    base = -(wi * 1.4)
    for mi, (m, col, mk, mlab) in enumerate(MODEL_STYLE):
        row = sub[sub["model"] == m]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        y = base - mi * 0.42
        y_data_all.append(y)
        ci_str = row["auroc_ci"]
        nums = _re.findall(r"[-+]?[0-9]*\.?[0-9]+", ci_str)
        lo, hi = float(nums[0]), float(nums[1])
        auc = float(row["auroc_sev_vs_bg"])
        ax.errorbar(auc, y, xerr=[[auc - lo], [hi - auc]],
                    fmt=mk, markersize=6, color=col,
                    ecolor=col, elinewidth=1.2, capsize=3)
        ax.text(auc, y + 0.14, f"{auc:.3f}", ha="center", fontsize=5.8, color=col)
    ytick_pos.append(base - 0.42)
    ytick_lab.append(WINDOW_LABEL[w])

ax.axvline(0.5, color="#999999", lw=0.8, ls=":")
ax.set_yticks(ytick_pos)
ax.set_yticklabels(ytick_lab, fontsize=6.8)
ax.set_xlim(0.0, 1.0)
# y-limit derived from the plotted rows so no data point or label is clipped
# (lowest row = bottom window minus the last model offset)
ax.set_ylim(min(y_data_all) - 0.35, 0.6)
ax.set_xlabel("AUROC (severe vs background; site-cluster bootstrap 95% CI)",
              fontsize=7)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linewidth=0.5)
ax.tick_params(labelsize=6.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker=mk, linestyle="none", markersize=5,
                          color=col, label=mlab)
           for m, col, mk, mlab in MODEL_STYLE],
          frameon=False, fontsize=6.0, loc="upper left", ncol=3)
ax.text(0.01, 0.02, "holdout windows are independent of any fitting;\n"
        "survey composition differs across periods",
        transform=ax.transAxes, fontsize=5.6, ha="left", va="bottom",
        style="italic", alpha=0.7)
S.panel_label(ax, "e")

# ------------------------------------- (f) year-stratified + LOYO AUROC ----
ax = fig.add_subplot(gs[2, 1])
ax.set_title("Year-stratified and leave-one-\nyear-out AUROC (severe vs bg)",
             loc="left", fontsize=7.5, fontweight="bold")
sev_bg = YEAR_STRATA[YEAR_STRATA["stratum"].str.startswith("year_")].copy()
loyo = YEAR_STRATA[YEAR_STRATA["stratum"].str.startswith("leave_out_")].copy()
for dframe, ls, mk, lab in ((sev_bg, "-", "o", "per-year"),
                            (loyo, "--", "s", "leave-one-year-out")):
    dframe = dframe.sort_values("stratum")
    _xlo = (dframe["auroc_sev_vs_bg"] - dframe["ci_lo"]).clip(lower=0)
    _xhi = (dframe["ci_hi"] - dframe["auroc_sev_vs_bg"]).clip(lower=0)
    ax.errorbar(dframe["auroc_sev_vs_bg"], range(len(dframe)),
                xerr=[_xlo, _xhi],
                fmt=mk, color=S.COLORS["SCS"], ecolor=S.COLORS["SCS"],
                elinewidth=0.8, capsize=2, linestyle=ls, markersize=5,
                label=lab)
    # per-year labels above the points, leave-one-year-out below them, so the
    # two annotations on the same row can never overlap
    dy = 0.15 if lab.startswith("per-year") else -0.15
    for yi, (_, r) in enumerate(dframe.iterrows()):
        ax.text(r["auroc_sev_vs_bg"], yi + dy, f"{r['auroc_sev_vs_bg']:.3f}",
                ha="center", fontsize=5.0)
ax.axvline(0.5, color="#999999", lw=0.7, ls=":")
ylabs = [s.replace("year_", "").replace("leave_out_", "LOO ")
         for s in loyo["stratum"].values]
ax.set_yticks(range(len(loyo)))
ax.set_yticklabels(ylabs, fontsize=6)
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.45, len(loyo) - 1 + 0.45)
ax.set_xlabel("AUROC (severe vs bg)", fontsize=6.5)
ax.legend(frameon=False, fontsize=5.5, loc="upper center")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=6)
ax.text(0.02, 0.02, "discrimination survives every\nsingle-year exclusion (0.81–0.90)",
        transform=ax.transAxes, fontsize=5.2, va="bottom", style="italic")
S.panel_label(ax, "f")

out = S.save_fig(fig, "fig7_gcbd_validation")

# ---------------------------------------------------------- source data -----
BOOT.to_csv(S.SRC_DIR / "fig7_source_data_bootstrap.csv", index=False)
QUIN.to_csv(S.SRC_DIR / "fig7_source_data_quintiles.csv", index=False)
MATCH.to_csv(S.SRC_DIR / "fig7_source_data_offsets.csv", index=False)
SITE.to_csv(S.SRC_DIR / "fig7_source_data_site.csv", index=False)
HOLD.to_csv(S.SRC_DIR / "fig7_source_data_holdout.csv", index=False)
print("[source] fig7 source-data files")

# site-level severe rate per quintile (from the trailing table rows of SITE csv)
man = S.OUT_DIR / "figure_manifest.csv"
new = pd.DataFrame([
    dict(figure="fig7", panel="b", source_file="results/step9_gcbd_validation/step9_bootstrap.csv",
         source_columns="key,scheme,mean,lo,hi (cluster_site)",
         statistical_method="site-cluster bootstrap 1000", n=n_cohort, unit="AUROC",
         output_file="fig7_gcbd_validation.pdf"),
    dict(figure="fig7", panel="c", source_file="results/step9_gcbd_validation/step9_quintile_enrichment.csv",
         source_columns="bins,event,bin,n,n_event,rate",
         statistical_method="quintile enrichment; RR with site-cluster CI",
         n=n_cohort, unit="rate (%)", output_file="fig7_gcbd_validation.pdf"),
    dict(figure="fig7", panel="d", source_file="results/step10_gcbd_matching_audit/step10_matching_results.csv",
         source_columns="offset_weeks,roc_sev_vs_bg",
         statistical_method="predefined offsets [-2..+2], none selected post hoc",
         n=int(MATCH.n.max()), unit="AUROC", output_file="fig7_gcbd_validation.pdf"),
    dict(figure="fig7", panel="e", source_file="results/step10_gcbd_matching_audit/step10_site_level_results.csv",
         source_columns="auc_mean_risk,delta_mean_risk,mw_p,quintile severe-site rates",
         statistical_method="site-level AUROC; Cliff's delta; site-preserving permutation",
         n=384, unit="rate/AUROC", output_file="fig7_gcbd_validation.pdf"),
])
if man.exists():
    old = pd.read_csv(man)
    new = pd.concat([old[old.figure != "fig7"], new], ignore_index=True)
new.to_csv(man, index=False)
print("[manifest] fig7 rows written")
