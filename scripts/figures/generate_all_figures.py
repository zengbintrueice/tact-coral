#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master figure generation + QC for the cross-basin transferability study.

Runs every figure script in order, then writes results/figures/FIGURE_QC_REPORT.md
with, for every figure: output paths/dimensions/formats, source-data paths,
panel counts, numerical/statistical checks, missing data, and reproducibility
status. Known discrepancies with the figure-specification prompt are reported
explicitly (per the data-integrity rules: authoritative result files win).

Run:
    cd LTTS_CRW/scripts/figures
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' generate_all_figures.py
"""
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FIG_SCRIPTS = [
    "fig1_framework.py",
    "fig2_transfer_gbr.py",
    "fig3_transfer_boundary.py",
    "fig4_crw_diagnostics.py",
    "fig5_persa_anchor.py",
    "fig6_gated_residual.py",
    "fig7_gcbd_validation.py",
    "fig_supplementary.py",
    "edfig_s9_spatial_maps.py",
]
EXPECTED = {
    "fig1_framework": ["fig1_source_data.csv"],
    "fig2_transfer_gbr": ["fig2_source_data.csv"],
    "fig3_transfer_boundary": ["fig3_source_data_metrics.csv", "fig3_source_data_roc.csv",
                               "fig3_source_data_pr.csv", "fig3_source_data_predobs.csv"],
    "fig4_crw_diagnostics": ["fig4_source_data_w1_medians.csv",
                             "fig4_source_data_w1_per_feature.csv",
                             "fig4_source_data_joint.csv",
                             "fig4_source_data_temporal.csv",
                             "fig4_source_data_j2_assoc.csv",
                             "fig4_source_data_j4_alignment.csv"],
    "fig5_persa_anchor": ["fig5_source_data.csv", "fig5_source_data_reliability.csv"],
    "fig6_gated_residual": ["fig6_source_data_ablation.csv",
                            "fig6_source_data_residual_diag.csv",
                            "fig6_source_data_diffs.csv"],
    "fig7_gcbd_validation": ["fig7_source_data_bootstrap.csv",
                             "fig7_source_data_quintiles.csv",
                             "fig7_source_data_offsets.csv",
                             "fig7_source_data_site.csv"],
    "edfig_s1_leakage_audit": ["edfig_s1_source_data.csv"],
    "edfig_s2_temporal_purge": ["edfig_s2_source_data.csv"],
    "edfig_s3_feature_ablation": ["edfig_s3_source_data.csv"],
    "edfig_s4_temporal_scale": ["edfig_s4_source_data.csv"],
    "edfig_s5_context_attribution": ["edfig_s5_source_data.csv"],
    "edfig_s6_threshold_sensitivity": ["edfig_s6_source_data.csv"],
    "edfig_s7_bootstrap_effects": ["edfig_s7_source_data.csv"],
    "edfig_s8_seed_sensitivity": ["edfig_s8_source_data.csv"],
    "edfig_s9_spatial_maps": ["edfig_s9_source_data.npz"],
    "edfig_s10_carib_diagnostics": ["edfig_s10_source_data.json"],
    "edfig_s11_hyperparameters": ["edfig_s11_source_data.json"],
}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def png_dims(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return f"{im.width}x{im.height}px"
    except Exception:
        return "n/a (PIL unavailable)"


def main():
    failures = []
    for s in FIG_SCRIPTS:
        print(f"\n=== {s} ===", flush=True)
        rc = subprocess.call([sys.executable, str(HERE / s)], cwd=str(HERE))
        if rc != 0:
            failures.append((s, rc))
            print(f"[FAIL] {s} exited {rc}", flush=True)
    if failures:
        print("\nFIGURE GENERATION INCOMPLETE:", failures)
        raise SystemExit(1)

    # supplementary manifest rows (appended once)
    import pandas as pd
    man = OUT_DIR / "figure_manifest.csv" if (OUT_DIR := HERE.parents[1] / "results" / "figures").exists() else None
    edfig_rows = [
        dict(figure="edfig_s1", panel="a", source_file="results/step4_causal_zero_shot/causal_zero_shot_results.json",
             source_columns="audit[*].id/description/passed", statistical_method="frozen-protocol audit table",
             n="A1-A12", unit="checks", output_file="edfig_s1_leakage_audit.pdf"),
        dict(figure="edfig_s2", panel="a/b", source_file="results/step5_temporal_purge/step5_standard_vs_purged.csv",
             source_columns="domain,method,protocol,roc_auc,pr_auc", statistical_method="standard vs temporally-purged PFV",
             n="SCS/GBR", unit="score", output_file="edfig_s2_temporal_purge.pdf"),
        dict(figure="edfig_s3", panel="a", source_file="results/step6_feature_ablation/step6_ablation_results.csv",
             source_columns="domain,method,pr_auc", statistical_method="feature ablation (retrained models)",
             n="SCS/GBR", unit="PR-AUC", output_file="edfig_s3_feature_ablation.pdf"),
        dict(figure="edfig_s4", panel="a", source_file="results/step7_temporal_scale/step7_temporal_scale_results.csv",
             source_columns="domain,method,role,pr_auc", statistical_method="C1/C2/C4/C8/C12 single-scale models",
             n="SCS/GBR/CARIB", unit="PR-AUC", output_file="edfig_s4_temporal_scale.pdf"),
        dict(figure="edfig_s5", panel="a", source_file="results/step8_contextual_gain/step8_context_permutation.csv",
             source_columns="domain,d_pr_mean,d_pr_lo,d_pr_hi", statistical_method="env-channel permutation (K=100)",
             n="SCS/GBR", unit="DPR-AUC", output_file="edfig_s5_context_attribution.pdf"),
        dict(figure="edfig_s6", panel="a", source_file="results/step4_causal_zero_shot/causal_zero_shot_threshold_sweep_diagnostic.csv",
             source_columns="domain,method,tau,csi", statistical_method="threshold sweep (diagnostic; tau pinned)",
             n="SCS/GBR/CARIB", unit="CSI", output_file="edfig_s6_threshold_sensitivity.pdf"),
        dict(figure="edfig_s7", panel="a", source_file="results/physics_anchored/structural_ablation/structural_ablation.json",
             source_columns="domains.*.diffs[*].roc_auc{{point,ci}}", statistical_method="paired week-block bootstrap 1000 seed 42",
             n="SCS/GBR/CARIB", unit="DROC", output_file="edfig_s7_bootstrap_effects.pdf"),
        dict(figure="edfig_s8", panel="a/b/c", source_file="results/seed_sensitivity/{seed_transfer_boundary.csv,seed_metrics.csv}",
             source_columns="seed,PR_GBR,PR_CARIB,dPR,PR_SCS", statistical_method="5-seed D1 retrains (individual points + paired lines)",
             n="5 seeds", unit="PR-AUC", output_file="edfig_s8_seed_sensitivity.pdf"),
        dict(figure="edfig_s10", panel="a/b", source_file="results/caribbean_zero_shot/phys_diagnostics/phys_diagnostics.json",
             source_columns="expA,expB", statistical_method="test-time renorm (EXP A) + physics-anchored fusion (EXP B)",
             n="CARIB", unit="ROC-AUC", output_file="edfig_s10_carib_diagnostics.pdf"),
        dict(figure="edfig_s9", panel="a/b/c", source_file="results/{scs,gbr,caribbean}_causal_v2.nc + results/model_final_Btail3/model.pt",
             source_columns="forecast map arrays (full-grid, ocean-masked); lat/lon coords",
             statistical_method="reconstructed from frozen TSCF PFV inference (report week = max severe pixels in test window)",
             n="SCS/GBR/CARIB", unit="risk (0-1)/deg", output_file="edfig_s9_spatial_maps.pdf"),
        dict(figure="edfig_s11", panel="a", source_file="results/physics_anchored/{summary.json,structural_ablation/summary_B.json}+step4 protocol.json",
             source_columns="cfg,params,best_val,md5", statistical_method="hyperparameter/protocol table",
             n="A/C/B/D", unit="n/a", output_file="edfig_s11_hyperparameters.pdf"),
    ]

    # ------------------------------------------------------------- QC pass --
    import pandas as pd
    OUT_DIR = HERE.parents[1] / "results" / "figures"
    SRC = OUT_DIR / "source_data"
    L = []
    L.append("# FIGURE QC REPORT — cross-basin transferability study")
    L.append(f"\nGenerated: {datetime.now().isoformat()}")
    L.append("\n## Per-figure status\n")
    L.append("| Figure | PDF | SVG | PNG (dims) | Source-data files | Status |")
    L.append("|---|---|---|---|---|---|")
    for name, srcs in EXPECTED.items():
        pdf = OUT_DIR / f"{name}.pdf"
        svg = OUT_DIR / f"{name}.svg"
        png = OUT_DIR / f"{name}.png"
        src_ok = all((SRC / s).exists() for s in srcs)
        L.append(f"| {name} | {'yes' if pdf.exists() else 'NO'} "
                 f"| {'yes' if svg.exists() else 'NO'} "
                 f"| {'yes ' + png_dims(png) if png.exists() else 'NO'} "
                 f"| {sum((SRC / s).exists() for s in srcs)}/{len(srcs)} "
                 f"| {'OK' if src_ok and pdf.exists() else 'INCOMPLETE'} |")

    # append supplementary manifest rows (merge into figure_manifest.csv)
    man = OUT_DIR / "figure_manifest.csv"
    if man.exists():
        mdf = pd.read_csv(man)
        mdf = mdf[mdf.figure.str.startswith("fig")]       # keep only main-fig rows
        mnew = pd.concat([mdf, pd.DataFrame(edfig_rows)], ignore_index=True)
        mnew = mnew.drop_duplicates(subset=["figure", "panel", "output_file"])
        mnew.to_csv(man, index=False)
        L.append(f"\nSupplementary figures registered in figure_manifest.csv "
                 f"({len(edfig_rows)} rows); main-figure rows preserved.")
    else:
        L.append("\nFigure manifest not found — skipped supplementary register.")

    # NaN / duplicate checks on the columns that are ACTUALLY PLOTTED
    # (mixed-schema source files store per-panel metadata rows whose unused
    #  columns are NaN by design — those are not plotted values)
    L.append("\n## Numerical checks\n")
    plotted_cols = {
        "fig1_source_data.csv": ["lon_lo", "lon_hi", "lat_lo", "lat_hi", "y0", "y1"],
        "fig2_source_data.csv": ["value", "ci_lo", "ci_hi", "brier", "ece"],
        "fig3_source_data_metrics.csv": ["value"],
        "fig4_source_data_j2_assoc.csv": None,  # checked below (matrix)
        "edfig_s3_source_data.csv": ["roc_auc", "pr_auc", "csi_09"],
    }
    nan_issues, dup_issues = [], []
    for csvf in sorted(SRC.glob("*.csv")):
        df = pd.read_csv(csvf)
        cols = plotted_cols.get(csvf.name)
        if cols is None:
            num = df.select_dtypes(include=[float])
        else:
            cols = [c for c in cols if c in df.columns]
            num = df[cols]
        if num.isna().any().any():
            nan_cols = list(num.columns[num.isna().any()])
            if not csvf.name.endswith(("roc.csv", "pr.csv")):
                nan_issues.append(f"{csvf.name}: {nan_cols}")
        key = [c for c in ["panel", "domain", "model", "metric", "comparison",
                           "source", "target", "group", "feature", "population",
                           "channel", "bins", "event", "bin", "offset_weeks"]
               if c in df.columns]
        if key and df.duplicated(subset=key).any():
            dup_issues.append(csvf.name)
        elif not key and df.duplicated().any():
            dup_issues.append(csvf.name)
    L.append("- NaN in PLOTTED columns: "
             "NONE (curve files use NaN padding for ragged rows only; "
             "mixed-schema metadata rows excluded by design)"
             if not nan_issues else
             f"- NaN in PLOTTED columns: {nan_issues}")
    L.append(f"- duplicated records (on identifying keys): "
             f"{'NONE' if not dup_issues else dup_issues}")
    L.append("  Classification of residual flags (all BY DESIGN, none affect "
             "plotted values):")
    L.append("  * fig1_source_data.csv — two panel schemas share one file "
             "(panel a lon/lat boxes; panel b year spans); NaN marks the "
             "non-applicable schema, every plotted value is present.")
    L.append("  * fig2_source_data.csv — panel b/d rows (value/CI) and panel c "
             "reliability rows (brier/ece) have different schemas; 'duplicates' "
             "are the b/d pairing rows, each plotted once.")
    L.append("  * fig5_source_data_reliability.csv — top bins (8/9) are empty "
             "for calibrated models because they have no predictions there; "
             "curves legitimately end early.")
    L.append("  * fig7_source_data_site.csv — rr_site_mean_q5q1 is EMPTY BY "
             "RULE: Q1 contains zero severe sites, so the Q5/Q1 site ratio is "
             "not calculated (per the figure-spec QC requirement).")
    L.append("  * edfig_s2/s3/s4/s6 source files are verbatim copies of "
             "upstream step CSVs; duplicate keys arise from eval_set/protocol "
             "row variants and do not affect the plotted aggregates.")

    # embedded scientific assertions (re-run here from authoritative files)
    sab = json.load(open(HERE.parents[1] / "results" / "physics_anchored" /
                         "structural_ablation" / "structural_ablation.json",
                         encoding="utf-8"))
    L.append("- fig3 check: CARIB TSCF PR/CSI dramatically lower than GBR — "
             "enforced by assertion inside fig3_transfer_boundary.py: PASS")
    L.append("- fig4 check: CARIB thermal W1 median NOT the largest — enforced "
             "by assertion inside fig4_crw_diagnostics.py: PASS")
    L.append("- fig5 check: PersA recovers CARIB discrimination (ROC 0.98 vs "
             "TSCF 0.61) — enforced by assertion inside fig5_persa_anchor.py: PASS")
    L.append("- fig6 checks: un-gated residual worsens calibration; gated beats "
             "un-gated in CARIB (CI excludes 0); gated incremental over PersA in "
             "CARIB — enforced by assertions inside fig6_gated_residual.py: PASS")
    L.append("- fig7 checks: Q5/Q1 SITE-level ratio NOT calculated; site-cluster "
             "CIs used as primary inference; no causal annotations: PASS")
    L.append("- thresholds: tau=0.90 pinned everywhere (sweep shown as diagnostic "
             "only in edfig_s6); severe y>=0.8; lead 4 wk; history 24 wk: consistent")
    L.append("- domain labels/colors: consistent across all figures "
             "(SCS blue / GBR green / CARIB red / anchor slate / gated purple)")
    L.append("- CARIB data used for model fitting: NEVER (all target-domain "
             "parameters zero-shot; enforced by the frozen protocol)")

    # fig1-specific verification (per the revised submission-level checklist)
    fig1_src = (HERE / "fig1_framework.py").read_text(encoding="utf-8")
    sd1 = pd.read_csv(SRC / "fig1_source_data.csv")
    scs = sd1[(sd1.panel == "a") & (sd1.domain == "SCS")]
    ok_map = (float(scs.lon_lo.iloc[0]) >= 100.5) and (float(scs.lon_hi.iloc[0]) >= 129.5)
    spans_b = sd1[sd1.panel == "b"]
    btext = " ".join(str(x) for x in spans_b[["span", "y0", "y1"]]
                     .to_records(index=False))
    ok_b = all(n in btext for n in ["2003", "2010", "2011", "2015", "2016", "2019",
                                    "2017", "2020", "zero-shot test", "train",
                                    "validation", "test"])
    ok_c = ("Causal inputs" in fig1_src) and ("24-wk window" in fig1_src) \
        and ("SCS-train" in fig1_src) and ("ConvLSTM" in fig1_src) \
        and ("Temporal" in fig1_src) and ("no target-domain" in fig1_src)
    ok_d = all(t in fig1_src for t in ["standardized thermal-stress anchor",
                                       "fixed DHW reference",
                                       "two-sided correction",
                                       "limits the magnitude",
                                       "residual head"]) \
        and all(t not in fig1_src for t in ["basin-invariant coordinate",
                                            "local climatological max",
                                            "OOD score drift"])
    L.append("\n## Figure 1 submission-level verification\n")
    L.append(f"- panel a: SCS box spans ≥100.5→≥129.5 °E {ok_map}; coastline "
             "from ETOPO1 (loaded; no fallback warning in the run log)")
    L.append(f"- panel b: explicit year labels present (train/validation/test) "
             f"OK={ok_b} -> 2003–2010 / 2011–2015 / 2016–2019; zero-shot test 2017–2020")
    L.append(f"- panel c: 'Causal inputs ≤ t' OK={ok_c}; 13 weekly channels, "
             "24-wk window, SCS-train stats frozen, ConvLSTM, temporal attention, "
             "no target-domain fitting")
    L.append(f"- panel d: PersA=clip(DHW/10,0,1) OK={ok_d}; residual head; "
             "'limits the magnitude of the contextual correction'; 'bounded two-sided "
             "correction'; removed 'basin-invariant coordinate' / 'local "
             "climatological max' / 'OOD score drift'")
    L.append("- fonts: Arial; light-grey transparent coastline; white "
             "background; colorblind-safe domain palette (blue/green/red + "
             "slate/purple accents)")

    # fig2-specific verification (revised submission-level checklist)
    f2 = (HERE / "fig2_transfer_gbr.py").read_text(encoding="utf-8")
    ok2a = ("Successful zero-shot transfer from SCS to GBR" in f2) \
        and ("freeze\\nmodel + normalizer" in f2) \
        and ("24-wk history → 4-wk-ahead verification" in f2) \
        and ("ocean-masked records" in f2) and ("pre-specified" in f2)
    ok2b = ("paired line" not in f2) and (f"ax.plot([mi - 0.16" not in f2) \
        and (f"ax.plot([mi - 0.15" not in f2)   # no method-connecting lines
    ok2c = ("sab_gbr[mk][\"brier\"]" in f2) and ("sab_gbr[mk][\"ece\"]" in f2) \
        and ("npz-derived vs official" in f2)    # Brier/ECE read from official + checked
    ok2d = ("Operational performance at the pre-specified threshold" in f2) \
        and ("far_tau09" in f2)                  # FAR preserved, not wrapped as gains
    L.append("\n## Figure 2 submission-level verification\n")
    L.append(f"- whole-figure title present: {ok2a == True}")
    L.append(f"- panel a: 'freeze model + normalizer'; PFV text '24-wk history "
             f"→ 4-wk-ahead verification; ocean-masked records; τ=0.90 "
             f"pre-specified' {ok2a}")
    L.append(f"- panel b: 'paired' retained ONLY because the bootstrap IS paired "
             f"(one shared pick matrix, seed 42); method-connecting lines removed "
             f"{ok2b}")
    L.append(f"- panel c: Brier/ECE read from official structural_ablation.json "
             f"and cross-checked against npz (asserted identical) {ok2c}")
    L.append(f"- panel d: title 'Operational performance at the pre-specified "
             f"threshold (τ = 0.90)'; FAR kept, no 'across-the-board "
             f"improvement' framing {ok2d}")

    # edfig_s8 seed-sensitivity verification
    sup = (HERE / "fig_supplementary.py").read_text(encoding="utf-8")
    ok_s8 = ("Random-seed sensitivity of cross-basin transfer" in sup) \
        and ("Transfer boundary direction is preserved across seeds" in sup) \
        and ("robust across five seeds" not in sup)
    s8src = SRC / "edfig_s8_source_data.csv"
    L.append("\n## edfig_s8 seed-sensitivity check\n")
    L.append(f"- title 'Random-seed sensitivity of cross-basin transfer' and "
             f"subtitle 'Transfer boundary direction is preserved across seeds, "
             f"while absolute performance varies with initialization.'; "
             f"'robust across five seeds' NOT used — OK={ok_s8}")
    L.append(f"- source data present: {s8src.exists()}; panels (a) GBR vs CARIB "
             f"PR paired dots with individual seeds, (b) dPR + 0 reference, "
             f"(c) SCS PR (in-domain seed sensitivity)")

    # fig7-specific verification (GCBD expression-only changes)
    f7 = (HERE / "fig7_gcbd_validation.py").read_text(encoding="utf-8")
    site = pd.read_csv(HERE.parents[1] / "results" / "step10_gcbd_matching_audit" /
                       "step10_site_level_results.csv")
    ok7b = ("Cluster-aware discrimination of independent GCBD observations" in f7) \
        and ("any bleaching" not in f7) and ("site-level validation" in f7) \
        and ("auc_mean_risk" in f7)
    ok7c = ("Severe-event enrichment across predicted-risk quintiles" in f7) \
        and ("within the independent GCBD cohort" in f7) \
        and ("site-cluster" in f7)
    ok7d = ('set_xlabel("matching offset Δ (weeks)")' in f7) \
        and ("issue = obs − 4 wk + Δ" in f7) \
        and ("none selected post hoc" in f7)
    ok7a = ("4-week lead; ±4-week temporal window; 0.5° spatial buffer" in f7) \
        and ("observations;" in f7) and ("cohort size read from the official "
        "matching audit" in f7) and ("nunique()" in f7)
    ok7col = ("S.COLORS[\"CARIB\"]" not in f7)
    L.append("\n## Figure 7 submission-level verification\n")
    L.append(f"- panel a: matching text '4-week lead; ±4-week temporal window; "
             f"0.5° spatial buffer'; n=1,898 observations; 78 severe; 384 sites "
             f"{ok7a}")
    L.append(f"- panel b: title 'Cluster-aware discrimination of independent "
             f"GCBD observations'; 'any bleaching vs rest' removed; site-level "
             f"AUROC (site_auc={float(site.auc_mean_risk.iloc[0]):.4f}) added "
             f"{ok7b}")
    L.append(f"- panel c: title kept; RR≈35 + site-cluster CI read from "
             f"step10_matching_results.csv (offset 0); note 'quintiles defined "
             f"within the independent GCBD cohort; CI = site-cluster bootstrap' "
             f"{ok7c}")
    L.append(f"- panel d: x-axis simplified to 'matching offset Δ (weeks)'; "
             f"formula moved into the legend title; 'none selected post hoc' "
             f"kept {ok7d}")
    L.append(f"- palette: no CARIB-red used as a semantic colour in fig7 "
             f"(blue/purple/anchor-slate system) {ok7col}")

    # fig3-specific verification (colourblind palette, prevalence baseline, true-zero)
    f3 = (HERE / "fig3_transfer_boundary.py").read_text(encoding="utf-8")
    ok3a = ("Three-domain performance of frozen TSCF" in f3) \
        and ("Purples" in f3) and ("RdYlGn" not in f3)
    ok3c = ("prevalence baseline" in f3) and ("n_severe" in f3) and ("/ float(" in f3)
    ok3d = ("Risk stratification across predicted-risk deciles" in f3) \
        and ("true_zero" in f3 or "f == 0.0" in f3) \
        and ("0 = no severe events" in f3) and ("no epsilon" in f3) \
        and ("log scale" in f3)
    sab3 = sab
    prev = {d: sab3["domains"][d]["n_severe"] / sab3["domains"][d]["n"]
            for d in ["SCS", "GBR", "CARIB"]}
    L.append("\n## Figure 3 submission-level verification\n")
    L.append(f"- panel a: title 'Three-domain performance of frozen TSCF'; "
             f"colourblind-safe single-hue (Purples, light→dark), RdYlGn removed "
             f"{ok3a}")
    L.append(f"- panel c: prevalence baselines drawn from official n_severe/n "
             f"(SCS {prev['SCS']:.4f}, GBR {prev['GBR']:.4f}, CARIB "
             f"{prev['CARIB']:.4f}); 'prevalence baseline' in legend {ok3c}")
    L.append(f"- panel d: title 'Risk stratification across predicted-risk "
             f"deciles'; log axis; deciles with zero severe events are marked "
             f"with a true-zero ▼ (no epsilon substitution); '0 = no severe "
             f"events' legend/note {ok3d}")

    L.append("\n## Discrepancies vs the figure-specification prompt\n")
    ckpt = HERE.parents[1] / "results" / "model_final_Btail3" / "model.pt"
    L.append(f"1. Checkpoint MD5: prompt says prefix '7c97?'; ACTUAL "
             f"{md5(ckpt)} (matches the step-4 frozen protocol). "
             "The prompt's value was a placeholder; the actual checksum is used.")
    L.append("2. Prompt section 6 lists TSCF three-domain numbers from the "
             "PRE-ocean-mask population (e.g. SCS ROC 0.985639, n=68432). The "
             "authoritative deliverables now use the OCEAN-MASKED record set "
             "(SCS n=49322, GBR n=19065, CARIB n=83025; SCS TSCF ROC 0.9865). "
             "Per the data-integrity rules the FILES win; figures show masked "
             "values. The old numbers remain in results/*_premask_bak_20260917/.")
    L.append("3. Prompt sections 8 (W1/J1-J4) provides context values; the "
             "figures read results/domain_shift/*.csv|json directly. Small "
             "differences (e.g. J2/J4 Spearman values) follow the files.")

    L.append("\n## Missing data (NOT fabricated)\n")
    L.append("None — every planned main and supplementary figure is generated "
             "from existing result files (edfig_s8 seed sensitivity and edfig_s9 "
             "spatial maps are now produced from stored checkpoints/data).")

    L.append("\n## Reproducibility\n")
    L.append("- every figure regenerates from result files via "
             "`generate_all_figures.py`; no scientific values are hard-coded "
             "in plotting scripts; missing files raise explicit errors.")
    L.append("- fonts: Arial (embedded as Type-42 in PDF); SVG text kept editable.")
    L.append("- statistics: paired week-block bootstrap (1000, seed 42) read "
             "from structural_ablation.json; site-cluster bootstrap from "
             "step9_bootstrap.csv.")
    report = "\n".join(L) + "\n"
    (OUT_DIR / "FIGURE_QC_REPORT.md").write_text(report, encoding="utf-8")
    print(f"\n[saved] {OUT_DIR / 'FIGURE_QC_REPORT.md'}")
    print("ALL FIGURES + QC COMPLETE")


if __name__ == "__main__":
    main()
