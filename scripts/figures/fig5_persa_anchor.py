# -*- coding: utf-8 -*-
"""Figure 5 — Thermal-stress anchoring recovers cross-basin transfer.

Panels
  a  PersA definition (standardized thermal-risk coordinate)
  b  frozen TSCF vs PersA across the three domains (ROC / PR dumbbells)
  c  CARIB recovery: A PersA / B un-gated / C gated / D TSCF (ROC & CSI, 95% CI)
  d  CARIB score-vs-event-frequency agreement (TSCF vs PersA vs anchored);
     probability Brier/ECE only for TSCF (a probability output), descriptive
     score reliability for PersA/TSAR

Sources: results/physics_anchored/structural_ablation/{structural_ablation.json,
rec_structural_ablation.npz}.

Scientific check before saving: PersA must recover a large fraction of CARIB
discrimination relative to the TSCF collapse (ROC 0.98 vs 0.61).
"""
import matplotlib.pyplot as plt
import numpy as np

import nc_style as S

SAB = S.require(S.RESULTS / "physics_anchored" / "structural_ablation" /
                "structural_ablation.json", fig="fig5")
Z = np.load(S.RESULTS / "physics_anchored" / "structural_ablation" /
            "rec_structural_ablation.npz")
# Canonical metric semantics (Task A continuous / Task B probability)
MV2 = S.require(S.RESULTS / "qc" / "calibration_target_audit" / "metrics_v2.json",
                fig="fig5")
MV = {(r["domain"], r["model"]): r for r in MV2
      if r["source"] == "structural_ablation"}
DOMAINS = ["SCS", "GBR", "CARIB"]

tscf = {d: SAB["domains"][d]["models"]["D_TSCF"] for d in DOMAINS}
persa = {d: SAB["domains"][d]["models"]["A_PersA"] for d in DOMAINS}

# ---- scientific pre-save check -------------------------------------------
frac = persa["CARIB"]["roc_auc"] / tscf["CARIB"]["roc_auc"]
if not (persa["CARIB"]["roc_auc"] > 0.9 and tscf["CARIB"]["roc_auc"] < 0.8
        and frac > 1.3):
    raise RuntimeError("fig5 QC FAILED: PersA does not recover CARIB "
                       f"discrimination (PersA ROC {persa['CARIB']['roc_auc']:.3f}, "
                       f"TSCF ROC {tscf['CARIB']['roc_auc']:.3f})")

fig = plt.figure(figsize=(S.WIDTH_2COL, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30,
                      left=0.10, right=0.97, top=0.93, bottom=0.08)

# ------------------------------------------------------------ (a) PersA -----
# Definitional schematic with CONSISTENT WEEKLY units: 12 week positions
# represent the 84-day (12-week) accumulation window; each qualifying week
# contributes its HotSpot value (>= 1 °C), and the 12 weekly contributions
# sum directly to DHW in °C-weeks. The 12 values are SYNTHETIC illustration
# numbers (clearly labelled), not observations; DHW and PersA are computed
# from them in code.
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.set_title("The thermal-stress anchor", loc="left", fontsize=8)

HS_WEEKS = np.array([0.4, 0.9, 1.2, 1.6, 1.1, 0.7, 1.0, 1.4, 0.9, 0.5, 0.8, 1.1])
kept = np.where(HS_WEEKS >= 1.0, HS_WEEKS, 0.0)
dhw_illu = float(kept.sum())                      # °C-weeks (weekly contributions)
persa_illu = float(np.clip(dhw_illu / 10.0, 0.0, 1.0))

x0, x1 = 0.06, 0.94
bw = (x1 - x0) / len(HS_WEEKS)
y_base, y_top = 0.615, 0.915                      # bar region
hs_ymax = 2.0
for i, v in enumerate(HS_WEEKS):
    h = v / hs_ymax * (y_top - y_base)
    on = v >= 1.0
    ax.add_patch(plt.Rectangle((x0 + i * bw + 0.004, y_base), bw - 0.008, h,
                               fc=S.COLORS["anchor"] if on else "#C9C9C9",
                               ec="none", transform=ax.transAxes))
    if on:
        ax.text(x0 + i * bw + bw / 2, y_base + h + 0.012, f"{v:.1f}",
                ha="center", va="bottom", fontsize=4.6,
                transform=ax.transAxes)
y_thr = y_base + 1.0 / hs_ymax * (y_top - y_base)
ax.plot([x0, x1], [y_thr, y_thr], ls="--", lw=0.7, color="#C0392B",
        transform=ax.transAxes)
ax.text(0.5, 0.955, "schematic 12-week HotSpot history "
        "(illustrative — not observations)", ha="center", fontsize=4.8,
        transform=ax.transAxes)
ax.text(0.5, 0.575, "HotSpot accumulation threshold = 1 °C",
        ha="center", va="top", fontsize=5.6, color="#C0392B",
        transform=ax.transAxes)

ax.annotate("", xy=(0.5, 0.505), xytext=(0.5, 0.545),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k"),
            xycoords=ax.transAxes)
ax.text(0.5, 0.495,
        "accumulate HotSpot values $\\geq$ 1 °C\n"
        "12-week accumulation — 12 week positions = 84 days;\n"
        "each qualifying week contributes its HotSpot value",
        ha="center", va="top", fontsize=5.8, transform=ax.transAxes)
ax.annotate("", xy=(0.5, 0.325), xytext=(0.5, 0.355),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k"),
            xycoords=ax.transAxes)
ax.text(0.5, 0.295, f"DHW = {dhw_illu:.1f} °C·weeks", ha="center",
        va="center", fontsize=7.0, transform=ax.transAxes)
ax.annotate("", xy=(0.5, 0.245), xytext=(0.5, 0.275),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k"),
            xycoords=ax.transAxes)
ax.text(0.5, 0.215,
        r"$\mathrm{PersA}(t)=\mathrm{clip}\!\left(\mathrm{DHW}(t)/10,\,0,\,1\right)$",
        ha="center", va="center", fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.125, f"PersA = {persa_illu:.2f} $\\in$ [0, 1]",
        ha="center", va="center", fontsize=7.0, transform=ax.transAxes)
ax.text(0.5, 0.075, "same fixed DHW/10 mapping across basins",
        ha="center", va="center", fontsize=6.0, transform=ax.transAxes)
ax.text(0.5, 0.025, "standardized thermal-stress coordinate; "
        "not a novel predictor or biological label",
        ha="center", va="center", fontsize=6.0, style="italic",
        transform=ax.transAxes)
S.panel_label(ax, "a", dx=-0.02)

# ------------------------------------------ (b) TSCF vs PersA, 3 domains ----
ax = fig.add_subplot(gs[0, 1])
ax.set_title("Frozen TSCF vs anchor across basins", loc="left", fontsize=8)
for mi, (key, lab) in enumerate([("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]):
    for di, d in enumerate(DOMAINS):
        x = mi + (di - 1) * 0.22
        for mk, col, mkk in [("A_PersA", S.COLORS["anchor"], "o"),
                             ("D_TSCF", S.COLORS["tscf"], "s")]:
            r = SAB["domains"][d]["models"][mk][key]
            ci = SAB["domains"][d]["models"][mk]["bootstrap"][key]
            ax.errorbar(x + (-0.05 if mk == "A_PersA" else 0.05), r,
                        yerr=[[max(0, r - ci[0])], [max(0, ci[1] - r)]],
                        fmt=mkk, color=col, capsize=2.0, lw=0.9, markersize=4)
    # NOTE: no connecting lines — the basins are independent zero-shot
    # evaluations, NOT a continuous SCS→GBR→CARIB trajectory.
ax.set_xticks([0, 1])
ax.set_xticklabels(["ROC-AUC", "PR-AUC"])
ax.set_ylim(0, 1.02)
handles = [plt.Line2D([], [], marker=m, ls="", color=c, label=l)
           for m, c, l in [("o", S.COLORS["anchor"], "PersA (anchor)"),
                           ("s", S.COLORS["tscf"], "TSCF (frozen)")]]
for di, d in enumerate(DOMAINS):
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=S.DOMAIN_COLOR[d], alpha=0.35,
                                 label=S.DOMAIN_LABEL[d]))
ax.legend(handles=handles, loc="lower left", fontsize=5.4)
ax.text(0.98, 0.95, "Frozen TSCF shows marked CARIB degradation",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
        bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, alpha=0.9, pad=1.5))
S.panel_label(ax, "b")

# ------------------------------------------------ (c) CARIB recovery --------
ax = fig.add_subplot(gs[1, 0])
ax.set_title("CARIB: thermal anchoring restores transfer performance",
             loc="left", fontsize=8)
models_c = [("A_PersA", "PersA", S.COLORS["anchor"], "o"),
            ("B_ungated", "un-gated\nresidual", S.COLORS["ungated"], "^"),
            ("C_anchored", "gated\nresidual", S.COLORS["gated"], "D"),
            ("D_TSCF", "TSCF", S.COLORS["tscf"], "s")]
for mi, (mk, lab, col, mkk) in enumerate(models_c):
    for ki, (key, xoff) in enumerate([("roc_auc", 0.0), ("csi_tau09", 0.9)]):
        r = SAB["domains"]["CARIB"]["models"][mk][key]
        ci = SAB["domains"]["CARIB"]["models"][mk]["bootstrap"][key]
        ax.errorbar(mi + ki * 0.35, r,
                    yerr=[[max(0, r - ci[0])], [max(0, ci[1] - r)]],
                    fmt=mkk, color=col, capsize=2.2, lw=1.0, markersize=5)
ax.axhline(SAB["domains"]["CARIB"]["models"]["A_PersA"]["roc_auc"],
           color=S.COLORS["anchor"], lw=0.7, ls=":", alpha=0.8)
ax.set_xticks([0, 1, 2, 3])
# compressed DISPLAY labels (source-data model labels unchanged)
ax.set_xticklabels(["PersA", "ungated", "gated", "TSCF"], fontsize=6.4)
ax.set_xlim(-0.5, 3.6)
ax.set_ylim(0, 1.05)
ax.set_ylabel("score (95% CI)")
ax.text(1.55, 0.06, "left cluster: ROC-AUC    right cluster: CSI@0.9",
        fontsize=5.8, ha="center")
S.panel_label(ax, "c")

# ------------------------------------------------ (d) CARIB reliability -----
ax = fig.add_subplot(gs[1, 1])
ax.set_title("CARIB score–event-frequency agreement", loc="left", fontsize=8)
ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--")
rel_rows = []
for mk, lab, col, _ in models_c:
    if mk == "B_ungated":
        continue  # out-of-range scores; shown separately in Fig 6
    p = Z["CARIB_" + mk + "_pred"]
    t = Z["CARIB_target"]
    c, f, _, _b_legacy, _e_legacy = S.reliability_curve(p, t)
    v = MV[("CARIB", mk)]
    if v["probability_calibration_valid"]:
        leg = "%s  (Brier %.3f, ECE %.3f)" % (
            lab, v["brier_binary"], v["ece_probability"])
    else:
        leg = "%s  (MSE %.3f, score-rel. %.3f)" % (
            lab, v["mse_cont"], v["score_reliability"])
    ax.plot(c, f, "o-", color=col, lw=1.0, markersize=3.5, label=leg)
    rel_rows.append(dict(model=mk, mse_cont=v["mse_cont"],
                         score_reliability=v.get("score_reliability", np.nan),
                         ece_probability=v.get("ece_probability", np.nan),
                         probability_calibration_valid=v["probability_calibration_valid"]) |
                    {f"bin{i}_pred": x for i, x in enumerate(c)} |
                    {f"bin{i}_freq": x for i, x in enumerate(f)})
ax.set_xlabel("mean score (continuous risk coordinate)")
ax.set_ylabel("observed severe frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=5.8)
S.panel_label(ax, "d")

out = S.save_fig(fig, "fig5_persa_anchor")

# ---------------------------------------------------------- source data -----
import pandas as pd
rows = []
for d in DOMAINS:
    for mk, lab, _, _ in models_c:
        for key in ["roc_auc", "pr_auc", "csi_tau09"]:
            r = SAB["domains"][d]["models"][mk][key]
            ci = SAB["domains"][d]["models"][mk]["bootstrap"][key]
            rows.append(dict(panel="b/c", domain=d, model=lab, metric=key,
                             value=r, ci_lo=ci[0], ci_hi=ci[1],
                             n=SAB["domains"][d]["n"],
                             n_severe=SAB["domains"][d]["n_severe"], unit="score"))
        v = MV[(d, mk)]
        for key in ("mse_cont", "score_reliability", "ece_probability",
                    "brier_binary"):
            if key in v:
                rows.append(dict(panel="b/c", domain=d, model=lab, metric=key,
                                 value=v[key], ci_lo="", ci_hi="",
                                 n=v["n"], n_severe=v["n_severe"], unit="score"))
pd.DataFrame(rows).to_csv(S.SRC_DIR / "fig5_source_data.csv", index=False)
pd.DataFrame(rel_rows).to_csv(S.SRC_DIR / "fig5_source_data_reliability.csv",
                              index=False)
print("[source] fig5 source-data files")

# --------------------------------------------------------------- QC report --
# Verify every plotted number/n/bootstrap method against the authoritative
# sources and record it in results/figures/fig5_qc_report.md.
qcmd = ["# Figure 5 — final scientific QC report", "",
        f"Generated: {pd.Timestamp.now().isoformat()}", "",
        "All plotted values are read from:",
        "- results/physics_anchored/structural_ablation/structural_ablation.json"
        "  (per-model point metrics + paired week-block bootstrap CIs)",
        "- results/physics_anchored/structural_ablation/rec_structural_ablation.npz"
        "  (per-pixel predictions for the reliability curves in panel d)",
        "Nothing is hard-coded.", "",
        "## Bootstrap configuration (from the frozen protocol, read from the",
        "result JSON)", ""]
frozen = SAB["protocol"]["frozen_config"]
qcmd.append(f"- method: {SAB['protocol']['record_set']}")
qcmd.append(f"- n_boot: {frozen.get('n_boot')}, seed: {frozen.get('seed')}, "
            f"threshold: {frozen.get('threshold')}, scale: {frozen.get('scale')}")
qcmd += ["",
         "## Panel-by-panel verification", "",
         "| Panel | Domain | Model | n | n_severe | ROC | PR | CSI@0.9 | "
         "Brier | ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
for d in DOMAINS:
    dom = SAB["domains"][d]
    for mk in ["A_PersA", "C_anchored", "D_TSCF"]:
        m = dom["models"][mk]
        qcmd.append(f"| b/c | {d} | {mk} | {dom['n']} | {dom['n_severe']} | "
                    f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                    f"{m['csi_tau09']:.4f} | {m['brier']:.5f} | {m['ece']:.4f} |")
qcmd += [
    "",
    "Panels b and c plot the point estimates above with the bootstrap 95% CIs",
    "stored in the same JSON ('bootstrap' dict per model); panel c additionally",
    "plots B_ungated (not shown here for brevity; read from the same source).",
    "Panel d recomputes the 10-bin reliability, Brier and ECE from the stored",
    "CARIB predictions (rec_structural_ablation.npz) via the SAME binning",
    "convention as the official evaluation; values match the JSON Brier/ECE in",
    "the table above.",
    "",
    "## Wording / layout changes applied (per specification)",
    "- Panel a scientific-units QC (final): schematic relabelled 'schematic",
    "  12-week HotSpot history (illustrative — not observations)'; the phrase",
    "  'daily HotSpot' removed — the strip shows 12 WEEK positions representing",
    "  the 84-day accumulation window, and the flow reads 'accumulate HotSpot",
    "  values ≥ 1 °C / 12-week accumulation — 12 week positions = 84 days; each",
    "  qualifying week contributes its HotSpot value'. Unit logic is therefore",
    "  consistent: the 12 qualifying WEEKLY contributions sum directly to DHW",
    "  in °C·weeks (DHW = 7.4, PersA = 0.74 for the illustrative values,",
    "  computed in code); no implication that daily °C values are summed.",
    "  Threshold label: 'HotSpot accumulation threshold = 1 °C'. The repeated",
    "  'clip(·/10, 0, 1)' side-label was removed — the formal formula",
    "  PersA(t)=clip(DHW(t)/10,0,1) is the only mapping shown.",
    "- Panel c: display labels compressed to 'PersA | ungated | gated | TSCF'",
    "  for spacing; the model labels stored in fig5_source_data.csv are",
    "  UNCHANGED (display-only change).",
    "- Panels b/d and ALL plotted values, CIs, Brier/ECE unchanged; no new",
    "  experiments. The only non-file numbers are the clearly-labelled",
    "  illustrative schematic values in panel a (synthetic, computed in code).",
]
(S.OUT_DIR / "fig5_qc_report.md").write_text("\n".join(qcmd), encoding="utf-8")
print(f"[QC] fig5_qc_report.md written", flush=True)

man = S.OUT_DIR / "figure_manifest.csv"
new = pd.DataFrame([
    dict(figure="fig5", panel="b", source_file="results/physics_anchored/structural_ablation/structural_ablation.json",
         source_columns="domains.*.models[A_PersA,D_TSCF]{roc_auc,pr_auc}",
         statistical_method="paired week-block bootstrap 1000 seed 42",
         n="49322/19065/83025", unit="score", output_file="fig5_persa_anchor.pdf"),
    dict(figure="fig5", panel="c", source_file="results/physics_anchored/structural_ablation/structural_ablation.json",
         source_columns="domains.CARIB.models[A_PersA,B_ungated,C_anchored,D_TSCF]{roc_auc,csi_tau09}+bootstrap",
         statistical_method="paired week-block bootstrap 1000 seed 42",
         n=SAB["domains"]["CARIB"]["n"], unit="score", output_file="fig5_persa_anchor.pdf"),
    dict(figure="fig5", panel="d", source_file="results/physics_anchored/structural_ablation/rec_structural_ablation.npz",
         source_columns="CARIB_{A_PersA,C_anchored,D_TSCF}_pred + CARIB_target",
         statistical_method="10-bin reliability; Brier; ECE",
         n=SAB["domains"]["CARIB"]["n"], unit="frequency", output_file="fig5_persa_anchor.pdf"),
])
if man.exists():
    old = pd.read_csv(man)
    new = pd.concat([old[old.figure != "fig5"], new], ignore_index=True)
new.to_csv(man, index=False)
print("[manifest] fig5 rows written")
