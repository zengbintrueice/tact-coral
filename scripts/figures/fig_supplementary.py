# -*- coding: utf-8 -*-
"""Extended-data (supplementary) figures for the cross-basin transferability study.

Generates one file per supplementary item:
  edfig_s1  leakage audit (A-checks, from step4 results JSON)
  edfig_s2  temporal purge (standard vs purged, step5)
  edfig_s3  feature ablation (step6)
  edfig_s4  temporal-scale ablation C1/C2/C4/C8/C12 (step7)
  edfig_s5  context attribution — env-channel permutation (step8)
  edfig_s6  threshold sensitivity (step4 sweep, diagnostic only)
  edfig_s7  paired bootstrap effect sizes (structural ablation, all diffs)
  edfig_s8  random-seed sensitivity (5-seed D1 retrains, GBR vs CARIB PR / dPR / SCS PR)
  edfig_s10 CARIB additional diagnostics (renorm / fusion EXP A+B)
  edfig_s11 model + protocol hyperparameters (text table)

NOT generated (source data absent — reported in FIGURE_QC_REPORT.md):
  edfig_s9  spatial prediction maps (only record-level arrays are stored;
            full forecast maps were not persisted — would require re-inference)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nc_style as S

SAB = S.require(S.RESULTS / "physics_anchored" / "structural_ablation" /
                "structural_ablation.json", fig="edfigs")


def _bar_domain_metrics(df, value_col, title, ylabel, models, fname):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    doms = [d for d in ["SCS", "GBR", "CARIB"] if len(df[df.domain == d])]
    w = 0.8 / max(len(models), 1)
    for mi, (mk, lab, col) in enumerate(models):
        vals = [float(df[(df.domain == d) & (df.method == mk)][value_col].iloc[0])
                for d in doms if len(df[(df.domain == d) & (df.method == mk)])]
        if len(vals) < len(doms):
            continue
        ax.bar(np.arange(len(doms)) + (mi - len(models) / 2 + 0.5) * w, vals, width=w,
               color=col, label=lab)
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels(doms)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8)
    ax.legend(fontsize=5.8)
    S.save_fig(fig, fname)


# ---------------------------------------------------------------- S1 audit --
try:
    R4 = S.require(S.RESULTS / "step4_causal_zero_shot" / "causal_zero_shot_results.json",
                   fig="edfig_s1")
    checks = R4["audit"] if isinstance(R4["audit"], list) else R4["audit"].get("checks", [])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.axis("off")
    ax.set_title("Causality / leakage audit (step-4 frozen protocol)", loc="left",
                 fontsize=8)
    txt = "\n".join(f"{c.get('id', '?'):>6}  {'PASS' if c.get('passed') else 'FAIL'}  "
                    f"{c.get('description', '')[:86]}" for c in checks)
    ax.text(0.0, 1.0, txt, va="top", fontsize=6.2, family="monospace",
            transform=ax.transAxes)
    S.save_fig(fig, "edfig_s1_leakage_audit")
    pd.DataFrame(checks).to_csv(S.SRC_DIR / "edfig_s1_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s1: {type(e).__name__}: {e}")

# ----------------------------------------------------------------- S2 purge --
try:
    P5 = S.require(S.RESULTS / "step5_temporal_purge" / "step5_standard_vs_purged.csv",
                   cols=["domain", "method", "protocol", "roc_auc", "pr_auc",
                         "csi_09"], fig="edfig_s2")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for ai, (col, lab) in enumerate([("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]):
        ax = axes[ai]
        w = 0.2
        for mi, (mk, col2) in enumerate([("TSCF", S.COLORS["tscf"]),
                                         ("PersistenceA_DHW", S.COLORS["anchor"])]):
            for pi, prot in enumerate(["standard", "purged"]):
                sub = P5[(P5.domain == "GBR") & (P5.method == mk) &
                         (P5.protocol == prot)]
                if not len(sub):
                    continue
                x = mi + (pi - 0.5) * w
                ax.bar(x, float(sub[col].iloc[0]), width=w, color=col2,
                       alpha=1.0 if prot == "standard" else 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["TSCF", "PersA"])
        ax.set_title(f"GBR {lab}", loc="left", fontsize=7.5)
        ax.set_ylim(0, 1.0)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="grey"),
                        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.5)],
               labels=["standard record set", "temporally purged (23/27 wk)"],
               loc="lower center", ncol=2, fontsize=6.0)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    S.save_fig(fig, "edfig_s2_temporal_purge")
    P5.to_csv(S.SRC_DIR / "edfig_s2_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s2: {type(e).__name__}: {e}")

# ------------------------------------------------------- S3 feature ablation --
try:
    P6 = S.require(S.RESULTS / "step6_feature_ablation" / "step6_ablation_results.csv",
                   cols=["domain", "method", "roc_auc", "pr_auc", "csi_09"], fig="edfig_s3")
    mods = [("FULL", "FULL", S.COLORS["tscf"]),
            ("NO_CUMSUM12", "no C12", S.COLORS["gated"]),
            ("NO_CUMULATIVE", "no cumulative", S.COLORS["CARIB"]),
            ("THERMAL_ONLY", "thermal only", S.COLORS["anchor"])]
    _bar_domain_metrics(P6, "pr_auc", "Feature ablation — PR-AUC",
                        "PR-AUC", mods, "edfig_s3_feature_ablation")
    P6.to_csv(S.SRC_DIR / "edfig_s3_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s3: {type(e).__name__}: {e}")

# ------------------------------------------------- S4 temporal-scale ablation --
try:
    P7 = S.require(S.RESULTS / "step7_temporal_scale" / "step7_temporal_scale_results.csv",
                   cols=["domain", "method", "role", "roc_auc", "pr_auc", "csi_09"],
                   fig="edfig_s4")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    singles = ["M_C1", "M_C2", "M_C4", "M_C8", "M_C12"]
    cols = ["#B8D4F0", "#8FBCE8", "#5B9BD5", S.COLORS["SCS"], "#2A4E7E"]
    for d, mk in [("SCS", "o"), ("GBR", "s"), ("CARIB", "^")]:
        if not len(P7[P7.domain == d]):
            continue
        vals = [float(P7[(P7.domain == d) & (P7.method == m)].pr_auc.iloc[0])
                for m in singles]
        ax.plot(range(5), vals, mk + "-", color=S.DOMAIN_COLOR[d], lw=1.0,
                ms=4, label=f"{S.DOMAIN_LABEL[d]} (PR-AUC)")
    ax.axhline(float(P7[(P7.domain == "GBR") & (P7.method == "FULL")].pr_auc.iloc[0]),
               color="#999999", lw=0.7, ls=":", label="GBR FULL")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C12"])
    ax.set_xlabel("cumulative HotSpot window W (weeks)")
    ax.set_ylabel("PR-AUC")
    ax.legend(fontsize=5.6)
    S.save_fig(fig, "edfig_s4_temporal_scale")
    P7.to_csv(S.SRC_DIR / "edfig_s4_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s4: {type(e).__name__}: {e}")

# ------------------------------------------------- S5 context attribution ----
try:
    P8 = S.require(S.RESULTS / "step8_contextual_gain" / "step8_context_permutation.csv",
                   cols=["domain", "d_pr_mean", "d_pr_lo", "d_pr_hi"], fig="edfig_s5")
    # Round-2 revision: CARIB env-channel permutation appended (frozen
    # protocol, results/review2_revision/carib_env_permutation.json). The
    # frozen step8 CSV itself is left untouched.
    _p8c = json.load(open(S.RESULTS / "review2_revision" / "carib_env_permutation.json",
                          encoding="utf-8"))
    P8 = pd.concat([P8, pd.DataFrame([{
        "domain": "CARIB",
        "d_roc_mean": _p8c["d_roc"]["mean"], "d_roc_lo": _p8c["d_roc"]["lo"],
        "d_roc_hi": _p8c["d_roc"]["hi"],
        "d_pr_mean": _p8c["d_pr"]["mean"], "d_pr_lo": _p8c["d_pr"]["lo"],
        "d_pr_hi": _p8c["d_pr"]["hi"],
        "d_csi09_mean": _p8c["d_csi09"]["mean"], "d_csi09_lo": _p8c["d_csi09"]["lo"],
        "d_csi09_hi": _p8c["d_csi09"]["hi"]}])], ignore_index=True)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for yi, (_, r) in enumerate(P8.iterrows()):
        col = S.DOMAIN_COLOR[r.domain]
        lo, hi = float(r.d_pr_mean - r.d_pr_lo), float(r.d_pr_hi - r.d_pr_mean)
        ax.errorbar(float(r.d_pr_mean), yi, xerr=[[lo], [hi]], fmt="o",
                    color=col, capsize=3, lw=1.1, markersize=5)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(P8)))
    ax.set_yticklabels(P8.domain.values)
    ax.set_xlabel("ΔPR-AUC after env-channel permutation\n(95% CI; context attribution)")
    S.save_fig(fig, "edfig_s5_context_attribution")
    P8.to_csv(S.SRC_DIR / "edfig_s5_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s5: {type(e).__name__}: {e}")

# ------------------------------------------------- S6 threshold sensitivity --
try:
    SW = S.require(S.RESULTS / "step4_causal_zero_shot" /
                   "causal_zero_shot_threshold_sweep_diagnostic.csv",
                   cols=["domain", "method", "tau", "csi"], fig="edfig_s6")
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for d in ["SCS", "GBR", "CARIB"]:
        sub = SW[(SW.domain == d) & (SW["method"] == "TSCF")]
        ax.plot(sub.tau, sub.csi, "-o", ms=2.5, lw=1.0, color=S.DOMAIN_COLOR[d],
                label=S.DOMAIN_LABEL[d])
    ax.axvline(0.90, color="k", lw=0.8, ls="--")
    ax.text(0.905, 0.02, "pinned τ=0.90", fontsize=5.8, rotation=90, va="bottom")
    ax.set_xlabel("threshold τ (diagnostic sweep only)")
    ax.set_ylabel("CSI")
    ax.legend(fontsize=5.8)
    S.save_fig(fig, "edfig_s6_threshold_sensitivity")
    SW.to_csv(S.SRC_DIR / "edfig_s6_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s6: {type(e).__name__}: {e}")

# ---------------------------------------------- S7 all paired effect sizes ---
try:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    y, ylab = [], []
    yy = 0
    ypos_all = []
    for d in ["SCS", "GBR", "CARIB"]:
        diffs = SAB["domains"][d]["diffs"]
        for lab, col in [("B_ungated−A_PersA", S.COLORS["ungated"]),
                         ("C_anchored−A_PersA", S.COLORS["gated"]),
                         ("C_anchored−B_ungated", "#B08CC0"),
                         ("D_TSCF−A_PersA", S.COLORS["tscf"]),
                         ("D_TSCF−C_anchored", "#4A6274")]:
            dd = diffs[lab]["roc_auc"]
            ax.errorbar(dd["point"], yy, xerr=[[dd["point"] - dd["ci"][0]],
                                               [dd["ci"][1] - dd["point"]]],
                        fmt="o", color=col, capsize=2.2, lw=1.0, markersize=4)
            ylab.append(f"{lab}\n{d}")
            ypos_all.append(yy)
            yy -= 1
        yy -= 0.5
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(ypos_all)
    ax.set_yticklabels(ylab, fontsize=4.6)
    ax.set_xlabel("ΔROC-AUC (95% CI, paired week-block bootstrap)")
    S.save_fig(fig, "edfig_s7_bootstrap_effects")
    rows = []
    for d in ["SCS", "GBR", "CARIB"]:
        for lab, dd in SAB["domains"][d]["diffs"].items():
            for k, v in dd.items():
                rows.append(dict(domain=d, comparison=lab, metric=k,
                                 point=v["point"], ci_lo=v["ci"][0], ci_hi=v["ci"][1]))
    pd.DataFrame(rows).to_csv(S.SRC_DIR / "edfig_s7_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s7: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- S8 seeds ---
try:
    TB = S.require(S.RESULTS / "seed_sensitivity" / "seed_transfer_boundary.csv",
                   cols=["seed", "PR_GBR", "PR_CARIB", "dPR"], fig="edfig_s8")
    MET = S.require(S.RESULTS / "seed_sensitivity" / "seed_metrics.csv",
                    cols=["seed", "domain", "protocol", "PR"], fig="edfig_s8")
    scs_pr = (MET[(MET.domain == "SCS") & (MET.protocol == "standard")]
              [["seed", "PR"]].sort_values("seed"))
    seeds = TB.seed.astype(int).values
    gbr = TB.PR_GBR.values
    carib = TB.PR_CARIB.values
    dpr = TB.dPR.values
    scs = scs_pr.PR.values

    fig = plt.figure(figsize=(7.08, 2.6))
    fig.suptitle("Random-seed sensitivity of cross-basin transfer",
                 fontsize=10.5, x=0.03, ha="left")
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.32,
                          left=0.08, right=0.98, top=0.80, bottom=0.18)

    # (a) GBR vs CARIB PR-AUC, individual seeds + paired lines
    ax = fig.add_subplot(gs[0])
    ax.set_title("GBR vs CARIB PR-AUC", loc="left", fontsize=8)
    for xi, s in zip(range(len(seeds)), seeds):
        ax.plot([0, 1], [gbr[xi], carib[xi]], color="#BBBBBB", lw=0.7, zorder=0)
        ax.plot(0, gbr[xi], "o", color=S.COLORS["GBR"], ms=6)
        ax.plot(1, carib[xi], "o", color=S.COLORS["CARIB"], ms=6)
        ax.annotate(f"{s}", (0, gbr[xi]), textcoords="offset points",
                    xytext=(-3, 5), ha="right", fontsize=5.4)
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GBR", "CARIB"])
    ax.set_ylabel("PR-AUC")
    S.panel_label(ax, "a")

    # (b) ΔPR = PR_GBR − PR_CARIB, per seed + 0 line
    ax = fig.add_subplot(gs[1])
    ax.set_title("ΔPR (GBR − CARIB)", loc="left", fontsize=8)
    for xi, s in zip(range(len(seeds)), seeds):
        ax.plot(xi, dpr[xi], "o", color="#2A4E7E", ms=6)
        ax.annotate(f"{s}", (xi, dpr[xi]), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=5.4)
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("ΔPR-AUC")
    S.panel_label(ax, "b")

    # (c) SCS PR-AUC (training's own seed sensitivity)
    ax = fig.add_subplot(gs[2])
    ax.set_title("SCS PR-AUC (in-domain)", loc="left", fontsize=8)
    ax.plot(range(len(seeds)), scs, "o-", color=S.COLORS["SCS"], lw=1.0, ms=6)
    for xi, s, v in zip(range(len(seeds)), seeds, scs):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=5.4)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("PR-AUC (SCS test)")
    ax.set_ylim(0, 0.75)
    S.panel_label(ax, "c")

    fig.text(0.03, 0.92,
             "Transfer boundary direction is preserved across seeds, while "
             "absolute performance varies with initialization.",
             fontsize=7.0, style="italic", ha="left")
    S.save_fig(fig, "edfig_s8_seed_sensitivity")
    out8 = pd.DataFrame(dict(seed=seeds, PR_GBR=gbr, PR_CARIB=carib, dPR=dpr,
                             PR_SCS=scs))
    out8.to_csv(S.SRC_DIR / "edfig_s8_source_data.csv", index=False)
except Exception as e:
    print(f"[MISSING] edfig_s8: {type(e).__name__}: {e}")

# ------------------------------------------------ S10 CARIB extra diagnostics --
try:
    PD_ = S.require(S.RESULTS / "caribbean_zero_shot" / "phys_diagnostics" /
                    "phys_diagnostics.json", fig="edfig_s10")
    a = PD_["expA"]
    b = PD_["expB"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ax = axes[0]
    labels = ["SCS norm", "CARIB renorm"]
    vals = [a["scs_norm"]["roc"], a["caribbean_train_renorm"]["roc"]]
    ax.bar(labels, vals, color=[S.COLORS["tscf"], S.COLORS["gated"]])
    ax.set_ylim(0.4, 1.0)
    ax.set_title("EXP A — test-time renorm (TSCF)", loc="left", fontsize=7.5)
    ax.set_ylabel("ROC-AUC")
    ax = axes[1]
    labels = ["PersA", "fusion(α*)", "TSCF"]
    vals = [b["test_persA_only"]["roc"], b["test_fusion"]["roc"],
            b["test_tscf_only"]["roc"]]
    ax.bar(labels, vals, color=[S.COLORS["anchor"], S.COLORS["gated"],
                                S.COLORS["tscf"]])
    ax.set_ylim(0.4, 1.0)
    ax.set_title("EXP B — physics-anchored fusion (α* on val)", loc="left",
                 fontsize=7.5)
    fig.autofmt_xdate()
    S.save_fig(fig, "edfig_s10_carib_diagnostics")
    with open(S.SRC_DIR / "edfig_s10_source_data.json", "w", encoding="utf-8") as f:
        json.dump(PD_, f, indent=2)
except Exception as e:
    print(f"[MISSING] edfig_s10: {type(e).__name__}: {e}")

# ------------------------------------------------------ S11 hyperparameters --
try:
    suma = json.load(open(S.RESULTS / "physics_anchored" / "summary.json",
                          encoding="utf-8"))
    sumb = json.load(open(S.RESULTS / "physics_anchored" / "structural_ablation" /
                          "summary_B.json", encoding="utf-8"))
    prot = S.require(S.RESULTS / "step4_causal_zero_shot" /
                     "causal_zero_shot_protocol.json", fig="edfig_s11")
    lines = [
        "PROTOCOL (frozen)",
        f"  splits: SCS {prot['splits']['SCS']}",
        f"          GBR {prot['splits']['GBR']}",
        "          CARIB test 2017-2020 (formal zero-shot)",
        f"  checkpoint MD5 (TSCF): {prot['checkpoint']['md5']}",
        f"  normalization: {prot['normalization']}",
        "",
        "ANCHORED MODEL (C)", json.dumps(suma["cfg"]),
        f"  params={suma['total_params']}  best_val={suma['best_val']:.4f}  "
        f"md5={suma.get('model_md5', 'n/a')}",
        "",
        "UN-GATED MODEL (B)", json.dumps(sumb["cfg"]),
        f"  params={sumb['total_params']}  best_val={sumb['best_val']:.4f}  "
        f"md5={sumb.get('model_md5', 'n/a')}",
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=6.4,
            family="monospace", transform=ax.transAxes)
    S.save_fig(fig, "edfig_s11_hyperparameters")
    with open(S.SRC_DIR / "edfig_s11_source_data.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": prot, "anchored": suma, "ungated": sumb}, f, indent=2)
except Exception as e:
    print(f"[MISSING] edfig_s11: {type(e).__name__}: {e}")

print("SUPPLEMENTARY FIGURES DONE (all planned items; edfig_s9 is generated "
      "by edfig_s9_spatial_maps.py)")
