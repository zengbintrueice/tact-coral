# -*- coding: utf-8 -*-
"""Figure 3 — Transferability has a cross-basin boundary.

Panels
  a  three-domain metric heatmap for the frozen TSCF
     (ROC / PR / CSI@0.9 / ECE (probability); ocean-masked record sets)
  b  ROC curves per domain
  c  precision-recall curves per domain
  d  predicted-risk decile vs observed severe frequency per domain

Sources: results/physics_anchored/structural_ablation/{structural_ablation.json,
rec_structural_ablation.npz} (D_TSCF shares the identical ocean-masked PFV
record set as the other deliverables).

Scientific check before saving: CARIB TSCF PR and CSI must be dramatically
lower than GBR (transfer collapse), while GBR tracks SCS.
"""
import matplotlib.pyplot as plt
import numpy as np

import nc_style as S

SAB = S.require(S.RESULTS / "physics_anchored" / "structural_ablation" /
                "structural_ablation.json", fig="fig3")
Z = np.load(S.RESULTS / "physics_anchored" / "structural_ablation" /
            "rec_structural_ablation.npz")
DOMAINS = ["SCS", "GBR", "CARIB"]

tscf = {d: dict(SAB["domains"][d]["models"]["D_TSCF"]) for d in DOMAINS}
# Canonical semantics: ECE here is a genuine probability-calibration error
# (TSCF is a sigmoid probability output), taken from metrics_v2.
MV2 = S.require(S.RESULTS / "metrics" / "metrics_v2.json",
                fig="fig3")
MV = {(r["domain"], r["model"]): r for r in MV2
      if r["source"] == "structural_ablation"}
for d in DOMAINS:
    tscf[d]["ece"] = MV[(d, "D_TSCF")]["ece_probability"]
    tscf[d]["brier_binary"] = MV[(d, "D_TSCF")]["brier_binary"]

# ---- scientific pre-save check -------------------------------------------
if not (tscf["CARIB"]["pr_auc"] < 0.5 * tscf["GBR"]["pr_auc"]
        and tscf["CARIB"]["csi_tau09"] < 0.5 * tscf["GBR"]["csi_tau09"]):
    raise RuntimeError(
        "fig3 QC FAILED: CARIB TSCF PR/CSI are not dramatically lower than GBR — "
        f"PR {tscf['CARIB']['pr_auc']:.3f} vs {tscf['GBR']['pr_auc']:.3f}, "
        f"CSI {tscf['CARIB']['csi_tau09']:.3f} vs {tscf['GBR']['csi_tau09']:.3f}")

fig = plt.figure(figsize=(S.WIDTH_2COL, 5.4))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                      left=0.10, right=0.97, top=0.93, bottom=0.08)

# --------------------------------------------------------- (a) heatmap ------
ax = fig.add_subplot(gs[0, 0])
ax.set_title("Three-domain performance of frozen TSCF", loc="left", fontsize=8)
mkeys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
         ("csi_tau09", "CSI@0.9"), ("ece", "ECE (probability)")]
M = np.array([[tscf[d][k] for d in DOMAINS] for k, _ in mkeys])
im = ax.imshow(M, cmap="Purples", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3))
ax.set_xticklabels(["SCS\n(source)", "GBR\n(zero-shot)", "CARIB\n(zero-shot)"])
ax.set_yticks(range(4))
ax.set_yticklabels([lab for _, lab in mkeys])
for i in range(4):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                color="white" if M[i, j] > 0.7 else "#1a1a1a")
for j in range(3):
    ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 4, fill=False,
                               edgecolor=S.DOMAIN_COLOR[DOMAINS[j]], lw=1.6))
S.panel_label(ax, "a")

# ---------------------------------------------------------- (b) ROC ---------
ax = fig.add_subplot(gs[0, 1])
ax.set_title("ROC — frozen TSCF", loc="left", fontsize=8)
ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--")
roc_src = []
for d in DOMAINS:
    p = Z[f"{d}_D_TSCF_pred"]
    t = Z[f"{d}_target"]
    fpr, tpr, _, _ = S.roc_pr_curve(p, t)
    ax.plot(fpr, tpr, color=S.DOMAIN_COLOR[d], lw=1.2,
            label=f"{S.DOMAIN_LABEL[d]}  (AUC {tscf[d]['roc_auc']:.3f})")
    roc_src.append(dict(domain=d) | {f"fpr_{i}": v for i, v in enumerate(fpr)} |
                   {f"tpr_{i}": v for i, v in enumerate(tpr)})
ax.set_xlabel("false-positive rate")
ax.set_ylabel("true-positive rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
S.panel_label(ax, "b")

# ----------------------------------------------------------- (c) PR ---------
ax = fig.add_subplot(gs[1, 0])
ax.set_title("Precision–recall — frozen TSCF", loc="left", fontsize=8)
pr_src = []
for d in DOMAINS:
    p = Z[f"{d}_D_TSCF_pred"]
    t = Z[f"{d}_target"]
    _, _, rec, prec = S.roc_pr_curve(p, t)
    ax.plot(rec, prec, color=S.DOMAIN_COLOR[d], lw=1.2,
            label=f"{S.DOMAIN_LABEL[d]}  (AUPRC {tscf[d]['pr_auc']:.3f})")
    pr_src.append(dict(domain=d) | {f"rec_{i}": v for i, v in enumerate(rec)} |
                  {f"prec_{i}": v for i, v in enumerate(prec)})
# prevalence baseline per domain: y = n_severe / n (read from the official
# structural_ablation.json, no hard-coding)
for d in DOMAINS:
    prev = float(SAB["domains"][d]["n_severe"]) / float(SAB["domains"][d]["n"])
    ax.plot([0, 1], [prev, prev], ls=":", lw=0.9, color=S.DOMAIN_COLOR[d], alpha=0.5)
    pr_src.append(dict(domain=d, baseline=True, prevalence=prev,
                       rec="", prec=""))
ax.plot([0, 1], [np.nan, np.nan], ls=":", lw=0.9, color="#666666", label="prevalence baseline")
ax.set_xlabel("recall (POD)")
ax.set_ylabel("precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=5.8)
S.panel_label(ax, "c")

# -------------------------------------------- (d) prediction vs observation -
ax = fig.add_subplot(gs[1, 1])
ax.set_title("Risk stratification across predicted-risk deciles", loc="left",
             fontsize=8)
w = 0.26
YLOW0 = 1.6e-4          # display floor marker position for TRUE zero (log axis)
pred_obs_src = []
for di, d in enumerate(DOMAINS):
    p = Z[f"{d}_D_TSCF_pred"]
    t = Z[f"{d}_target"]
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    cents, freqs, zeros = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            cents.append(np.nan)
            freqs.append(np.nan)
            zeros.append(False)
            continue
        f = float((t[m] >= 0.8).mean())
        cents.append(float(p[m].mean()))
        freqs.append(f)
        zeros.append(f == 0.0)     # TRUE zero, never epsilon-substituted
    xcent = np.arange(10) + (di - 1) * w
    for b in range(10):
        if not np.isfinite(freqs[b]):
            continue
        if freqs[b] > 0.0:
            ax.bar(xcent[b], freqs[b], width=w, color=S.DOMAIN_COLOR[d], alpha=0.85)
        else:
            # TRUE zero: no bar on a log axis; mark it explicitly at the floor
            ax.plot(xcent[b], YLOW0, "v", ms=3.5, color=S.DOMAIN_COLOR[d],
                    mec=S.DOMAIN_COLOR[d])
    pred_obs_src.append(dict(domain=d) |
                        {f"decile{k}_pred": v for k, v in enumerate(cents)} |
                        {f"decile{k}_sevfreq": v for k, v in enumerate(freqs)} |
                        {f"decile{k}_true_zero": v for k, v in enumerate(zeros)})
ax.set_xticks(np.arange(10))
ax.set_xticklabels([f"D{i+1}" for i in range(10)])
ax.set_xlabel("predicted-risk decile (D1 lowest … D10 highest)")
ax.set_ylabel("observed severe frequency (log scale)")
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
handles = [plt.Rectangle((0, 0), 1, 1, fc=S.DOMAIN_COLOR[dd], alpha=0.85,
                         label=S.DOMAIN_LABEL[dd]) for dd in DOMAINS]
handles.append(plt.Line2D([], [], marker="v", ls="", color="#666666", ms=4,
                          label="▼ 0 = no severe events (true zero)"))
ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=5.8,
          columnspacing=1.0, handletextpad=0.4)
S.panel_label(ax, "d")

out = S.save_fig(fig, "fig3_transfer_boundary")

# ---------------------------------------------------------- source data -----
import pandas as pd
rows = []
for ki, (k, lab) in enumerate(mkeys):
    for di, d in enumerate(DOMAINS):
        rows.append(dict(panel="a", domain=d, metric=lab, value=tscf[d][k],
                         n=SAB["domains"][d]["n"], n_severe=SAB["domains"][d]["n_severe"],
                         unit="score"))
pd.DataFrame(rows).to_csv(S.SRC_DIR / "fig3_source_data_metrics.csv", index=False)
pd.DataFrame(roc_src).to_csv(S.SRC_DIR / "fig3_source_data_roc.csv", index=False)
pd.DataFrame(pr_src).to_csv(S.SRC_DIR / "fig3_source_data_pr.csv", index=False)
pd.DataFrame(pred_obs_src).to_csv(S.SRC_DIR / "fig3_source_data_predobs.csv", index=False)
print("[source] fig3 source-data files (metrics, roc, pr, predobs)")

man = S.OUT_DIR / "figure_manifest.csv"
new = pd.DataFrame([
    dict(figure="fig3", panel="a", source_file="results/physics_anchored/structural_ablation/structural_ablation.json",
         source_columns="D_TSCF{roc_auc,pr_auc,csi_tau09} (structural_ablation.json) + {ece_probability} (metrics_v2.json)",
         statistical_method="point estimates on ocean-masked PFV record set",
         n="49322/19065/83025", unit="score", output_file="fig3_transfer_boundary.pdf"),
    dict(figure="fig3", panel="b", source_file="results/physics_anchored/structural_ablation/rec_structural_ablation.npz",
         source_columns="{SCS,GBR,CARIB}_D_TSCF_pred + _target",
         statistical_method="ROC curve (sklearn)", n="as panel a", unit="rate",
         output_file="fig3_transfer_boundary.pdf"),
    dict(figure="fig3", panel="c", source_file="results/physics_anchored/structural_ablation/rec_structural_ablation.npz",
         source_columns="{SCS,GBR,CARIB}_D_TSCF_pred + _target",
         statistical_method="precision-recall curve (sklearn)", n="as panel a", unit="rate",
         output_file="fig3_transfer_boundary.pdf"),
    dict(figure="fig3", panel="d", source_file="results/physics_anchored/structural_ablation/rec_structural_ablation.npz",
         source_columns="{SCS,GBR,CARIB}_D_TSCF_pred + _target",
         statistical_method="predicted-risk deciles vs observed severe frequency (log scale)",
         n="as panel a", unit="frequency", output_file="fig3_transfer_boundary.pdf"),
])
if man.exists():
    old = pd.read_csv(man)
    new = pd.concat([old[old.figure != "fig3"], new], ignore_index=True)
new.to_csv(man, index=False)
print("[manifest] fig3 rows written")
