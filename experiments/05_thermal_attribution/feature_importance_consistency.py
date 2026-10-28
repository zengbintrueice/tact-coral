# ==========================================================================
# Experiment: Feature-importance cross-domain consistency (data only)
# Paper:      Fig. 4d / Sup Fig. 5c rank consistency
# Purpose:    Compute SCS-vs-GBR permutation-importance rank consistency
#             (Spearman rho_rank over all 13 channels + thermal/non-thermal
#             subgroups, Pearson magnitude r) and emit the JSON that backs
#             Fig. 4d / Sup Fig. 5c. The figure is drawn by the canonical
#             figures/supplement/sfig5/sfig5_feature_consistency.py, which reads this JSON.
# Source:     paper/a7_dual_output/plot_feature_importance_consistency.py
#             (data block retained verbatim; plotting removed 2026-08-23)
# Output:     results/thermal_attribution/feature_importance_consistency.json
# ==========================================================================

# -*- coding: utf-8 -*-
"""feature_importance_consistency.py — Feature importance cross-domain
consistency (data generation only).

Reads results/thermal_attribution/permutation_importance_cross_domain.json
(permutation importance ΔAUC per channel in SCS and GBR), computes:

  - per-domain importance ranking (all 13 dynamic channels, not just Top-5)
  - Spearman rank correlation ρ_rank over ALL common features
  - Pearson importance correlation r_importance (magnitude consistency)
  - thermal vs non-thermal subgroup Spearman ρ
  - cross-domain top-10 membership

Saves:
  - results/thermal_attribution/feature_importance_consistency.json   ← plotting data

The figure is drawn by figures/supplement/sfig5/sfig5_feature_consistency.py, which reads this
JSON. Feature grouping marker legend is retained in the JSON for the plotter:
● Thermal  ○ Environmental  △ Seasonal  □ Static
"""
import json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr, pearsonr

CWD = Path(__file__).resolve().parents[2]          # LTTS root
RESULTS_DIR = CWD / "results"
SRC_JSON = RESULTS_DIR / "thermal_attribution" / "permutation_importance_cross_domain.json"
OUT_JSON = RESULTS_DIR / "thermal_attribution" / "feature_importance_consistency.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# Feature grouping for the plotter's marker symbols (kept in JSON)
FEATURE_GROUPS = {
    "sst": "thermal", "ssta": "thermal", "hotspot": "thermal",
    "d_hotspot_dt": "thermal", "hotspot_cumsum_4w": "thermal",
    "hotspot_cumsum_12w": "thermal", "sst_anomaly": "thermal",
    "chlorophyll": "environmental", "par": "environmental",
    "wind_speed": "environmental", "ssr": "environmental",
    "seasonal_sin": "seasonal", "seasonal_cos": "seasonal",
}
GROUP_LABELS = {
    "thermal": "● Thermal",
    "environmental": "○ Environmental",
    "seasonal": "△ Seasonal",
    "static": "□ Static",
}
TOP_N = 10


def rank_features(values):
    """Rank features by importance value (descending; higher = more important).
    Equal values get the average rank. Returns dict feature -> rank (1 = best)."""
    names = list(values.keys())
    vals = np.array([values[n] for n in names])
    order = np.argsort(-vals)              # descending
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(names) + 1)
    # handle ties (average rank)
    # simple version: no ties expected here, but keep robust
    return {n: float(r) for n, r in zip(names, ranks)}


def main():
    # ── Load source data ────────────────────────────────────────
    with open(SRC_JSON, encoding="utf-8") as f:
        src = json.load(f)

    scs_imp = {fc["channel"]: fc["scs_auc_drop"] for fc in src["feature_comparison"]}
    gbr_imp = {fc["channel"]: fc["gbr_auc_drop"] for fc in src["feature_comparison"]}
    all_channels = list(FEATURE_GROUPS.keys())
    assert set(all_channels) == set(scs_imp.keys()) == set(gbr_imp.keys()), \
        "channel set mismatch between config and data"

    # ── Rankings (all common features) ─────────────────────────
    scs_rank = rank_features(scs_imp)
    gbr_rank = rank_features(gbr_imp)

    # ── Top-10 membership (per domain) ─────────────────────────
    scs_top10 = sorted(all_channels, key=lambda c: -scs_imp[c])[:TOP_N]
    gbr_top10 = sorted(all_channels, key=lambda c: -gbr_imp[c])[:TOP_N]
    common_top10 = [c for c in scs_top10 if c in gbr_top10]   # connect lines

    # ── Correlations ───────────────────────────────────────────
    # Spearman rank over ALL common features (the clean method)
    rank_pairs = [(scs_rank[c], gbr_rank[c]) for c in all_channels]
    rho_rank, p_rank = spearmanr([p[0] for p in rank_pairs],
                                 [p[1] for p in rank_pairs])
    # Pearson on importance values (magnitude consistency)
    imp_pairs = [(scs_imp[c], gbr_imp[c]) for c in all_channels]
    r_imp, p_imp = pearsonr([p[0] for p in imp_pairs],
                            [p[1] for p in imp_pairs])

    # Subgroup Spearman
    sub_stats = {}
    for gname in ["thermal", "environmental", "seasonal"]:
        chs = [c for c in all_channels if FEATURE_GROUPS[c] == gname]
        if len(chs) >= 3:
            r, p = spearmanr([scs_rank[c] for c in chs],
                             [gbr_rank[c] for c in chs])
            sub_stats[gname] = {"n": len(chs), "rho": float(r), "p": float(p)}
        else:
            sub_stats[gname] = {"n": len(chs), "rho": None, "p": None}

    # ── Build plotting data ─────────────────────────────────────
    features = []
    for c in all_channels:
        features.append({
            "channel": c,
            "group": FEATURE_GROUPS[c],
            "scs_importance": scs_imp[c],
            "gbr_importance": gbr_imp[c],
            "scs_rank": scs_rank[c],
            "gbr_rank": gbr_rank[c],
            "in_scs_top10": c in scs_top10,
            "in_gbr_top10": c in gbr_top10,
            "connect": c in common_top10,
        })
    # sort by SCS importance (top of left chart = most important)
    features.sort(key=lambda x: -x["scs_importance"])

    data = {
        "experiment": "feature_importance_cross_domain_consistency",
        "description": "SCS vs GBR permutation-importance ranking consistency; "
                       "all 13 dynamic channels, Spearman rank + Pearson magnitude",
        "source_data": SRC_JSON.name,
        "scs_baseline_auc": src["scs_baseline_auc"],
        "gbr_baseline_auc": src["gbr_baseline_auc"],
        "top_n": TOP_N,
        "correlations": {
            "spearman_rho_all_rank": float(rho_rank),
            "spearman_p_all_rank": float(p_rank),
            "pearson_r_importance": float(r_imp),
            "pearson_p_importance": float(p_imp),
            "subgroups": sub_stats,
        },
        "rankings": {
            "scs_top10": scs_top10,
            "gbr_top10": gbr_top10,
            "common_top10": common_top10,
        },
        "features": features,
        "group_labels": GROUP_LABELS,
        "timestamp": "2026-08-07",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUT_JSON}")

    # ── Console summary ─────────────────────────────────────────
    print("\n" + "=" * 66)
    print("Feature importance cross-domain consistency summary")
    print("=" * 66)
    print(f"  Spearman ρ_rank (all {len(all_channels)} features): {rho_rank:.3f}  (p={p_rank:.3g})")
    print(f"  Pearson r_importance (all {len(all_channels)} features): {r_imp:.3f}  (p={p_imp:.3g})")
    for g, s in sub_stats.items():
        rho_str = f"{s['rho']:.3f}" if s["rho"] is not None else "N/A (<3 feats)"
        print(f"  {g:12s} Spearman ρ = {rho_str}  (n={s['n']})")
    print(f"  SCS top-10: {scs_top10}")
    print(f"  GBR top-10: {gbr_top10}")
    print(f"  Common in both top-10 ({len(common_top10)}): {common_top10}")
    print("=" * 66)


if __name__ == "__main__":
    main()
