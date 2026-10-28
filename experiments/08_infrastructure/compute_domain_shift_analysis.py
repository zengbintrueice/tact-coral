#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain shift analysis: SCS train vs SCS/GBR/CARIB test — feature-level
source-target distribution shift for the frozen TSCF input space.

PURE ANALYSIS — no model training, no hyperparameter search, no target-domain
adaptation, no threshold selection, no changes to frozen Step 2-10 outputs.

Populations (ocean cells only, aligned with the label_mask protocol):
  A  SCS train   2003-2010   (scs_causal_v2.nc)
  B  SCS test    2016-2019   (scs_causal_v2.nc)
  C  GBR test    2017-2020   (gbr_causal_v2.nc)
  D  CARIB test  2017-2020   (caribbean_causal_v2.nc)

Features (16, pooled week x ocean-pixel values):
  instantaneous thermal (5): sst, ssta, hotspot, sst_anomaly, d_hotspot_dt
  cumulative thermal    (5): C1, C2, C4, C8, C12 — trailing W-week sums of
      HotSpot>=1 C, Step-7 semantics (NaN weeks contribute 0; NaN where
      hotspot[t] NaN), computed from each file's weekly hotspot series.
  environmental/context (6): chlorophyll, par, wind_speed, ssr,
      seasonal_sin, seasonal_cos

Standardization: ALL populations z-scored with SCS-TRAIN (A) statistics ONLY.
No per-domain normalization.

Metrics per (pair, feature):
  - standardized mean difference  SMD = (mean_target - mean_A) / std_A
  - Wasserstein-1 distance on z-scored values (scipy, exact 1D)
  - energy distance on z-scored values (scipy)

Performance anchor (ocean-masked frozen TSCF, Steps 2/4 re-run 2026-09-17):
  SCS  ROC 0.9865 PR 0.5463 CSI@0.9 0.3374
  GBR  ROC 0.9624 PR 0.5543 CSI@0.9 0.3648
  CARIB ROC 0.6101 PR 0.0321 CSI@0.9 0.0262
Association language only ("associated with" / "consistent with" /
"corresponds to"); NO causal claims.

Outputs -> results/domain_shift/:
  feature_shift.csv, feature_shift_summary.csv, wasserstein.csv,
  energy_distance.csv, analysis.md, manifest.json, ed_fig9_draft.png
"""
import io, sys, os, json, hashlib, warnings
from pathlib import Path
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance, energy_distance

LTTS_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
OUT_DIR = LTTS_ROOT / "results" / "domain_shift"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_N = 200_000     # per-population cap per feature (deterministic stride thinning)

INST_THERMAL = ["sst", "ssta", "hotspot", "sst_anomaly", "d_hotspot_dt"]
CUM_THERMAL = ["C1", "C2", "C4", "C8", "C12"]
CONTEXT = ["chlorophyll", "par", "wind_speed", "ssr", "seasonal_sin", "seasonal_cos"]
FEATURES = INST_THERMAL + CUM_THERMAL + CONTEXT
GROUP = ({f: "instantaneous_thermal" for f in INST_THERMAL} |
         {f: "cumulative_thermal" for f in CUM_THERMAL} |
         {f: "context_calendar" for f in CONTEXT})

POPULATIONS = {
    "SCS_train": dict(file="scs_causal_v2.nc", years=(2003, 2010)),
    "SCS_test": dict(file="scs_causal_v2.nc", years=(2016, 2019)),
    "GBR_test": dict(file="gbr_causal_v2.nc", years=(2017, 2020)),
    "CARIB_test": dict(file="caribbean_causal_v2.nc", years=(2017, 2020)),
}
PAIRS = [("SCS_train", "SCS_test"), ("SCS_train", "GBR_test"), ("SCS_train", "CARIB_test")]

PERF = {  # frozen TSCF, ocean-masked (Steps 2/4 re-run 2026-09-17)
    "SCS_test":  dict(roc=0.9865, pr=0.5463, csi=0.3374, n=49322, sev=419),
    "GBR_test":  dict(roc=0.9624, pr=0.5543, csi=0.3648, n=19065, sev=676),
    "CARIB_test": dict(roc=0.6101, pr=0.0321, csi=0.0262, n=83025, sev=1356),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trailing_cw(hotspot, w):
    """Trailing W-week sum of HotSpot>=1 C (Step-7 semantics):
    NaN weeks contribute 0; NaN where hotspot[t] is NaN."""
    hs = np.asarray(hotspot, dtype=np.float32)
    nanm = np.isnan(hs)
    clipped = np.where(np.isfinite(hs) & (hs >= 1.0), hs, 0.0)
    cum = np.cumsum(clipped, axis=0)
    out = np.empty_like(cum)
    out[:w] = cum[:w]
    if cum.shape[0] > w:
        out[w:] = cum[w:] - cum[:-w]
    out[nanm] = np.nan
    return out.astype(np.float32)


def load_population(pop):
    """Return {feature: 1D array of pooled ocean-cell values} for one population."""
    ds = xr.open_dataset(DATA_DIR / pop["file"])
    times = pd.DatetimeIndex(ds.time.values)
    ocean = ds["ocean_mask"].values.astype(bool)
    yrmask = (times.year >= pop["years"][0]) & (times.year <= pop["years"][1])
    hot_full = ds["hotspot"].values.astype(np.float32)   # [T,H,W]
    cw = {w: trailing_cw(hot_full, w) for w in [1, 2, 4, 8, 12]}
    # d_hotspot_dt (loader semantics: hs(t)-hs(t-1), NaN weeks contribute 0
    # in the differencing, NaN where hotspot[t] is NaN)
    nanm = np.isnan(hot_full)
    hsc = np.where(nanm, 0.0, hot_full)
    d_hs = np.zeros_like(hot_full)
    d_hs[1:] = hsc[1:] - hsc[:-1]
    d_hs[nanm] = np.nan
    # seasonal_cos (deterministic calendar: cos(2*pi*week_of_year/52), full axis)
    woy_all = np.array([t.isocalendar()[1] for t in times], dtype=np.float32)
    seas_cos = np.broadcast_to(
        np.cos(2 * np.pi * woy_all / 52.0)[:, None, None].astype(np.float32),
        hot_full.shape).copy()
    feats = {}
    for f in FEATURES:
        if f in CUM_THERMAL:
            arr = cw[int(f[1:])]
        elif f == "d_hotspot_dt":
            arr = d_hs
        elif f == "seasonal_cos":
            arr = seas_cos
        else:
            arr = ds[f].values.astype(np.float32)
        sub = arr[yrmask][:, ocean]                       # [weeks_in_pop, n_ocean]
        v = sub.ravel()
        v = v[np.isfinite(v)]
        if len(v) > MAX_N:                                # deterministic thinning
            stride = int(np.ceil(len(v) / MAX_N))
            v = v[::stride]
        feats[f] = v
    n_weeks = int(yrmask.sum())
    n_ocean = int(ocean.sum())
    ds.close()
    return feats, n_weeks, n_ocean


def main():
    print("Loading populations...", flush=True)
    pops, meta_pops = {}, {}
    for name, pop in POPULATIONS.items():
        feats, nw, no = load_population(pop)
        pops[name] = feats
        meta_pops[name] = dict(file=pop["file"], years=list(pop["years"]),
                               n_weeks=nw, n_ocean_cells=no,
                               n_per_feature={f: int(len(feats[f])) for f in FEATURES})
        print(f"  {name}: weeks={nw} ocean_cells={no} "
              f"n={min(len(feats[f]) for f in FEATURES)}..{max(len(feats[f]) for f in FEATURES)}",
              flush=True)

    # SCS-train statistics (the ONLY normalization source)
    print("Computing SCS-train statistics...", flush=True)
    stats = {}
    for f in FEATURES:
        a = pops["SCS_train"][f]
        stats[f] = dict(mean=float(np.mean(a)), std=float(np.std(a)))
        print(f"  {f:<14} mean={stats[f]['mean']:+.4f} std={stats[f]['std']:.4f}", flush=True)

    def zscore(name, f):
        return (pops[name][f] - stats[f]["mean"]) / stats[f]["std"]

    # per-pair, per-feature metrics
    rows, w_rows, e_rows = [], [], []
    for s_name, t_name in PAIRS:
        print(f"Pair {s_name} -> {t_name} ...", flush=True)
        for f in FEATURES:
            zs, zt = zscore(s_name, f), zscore(t_name, f)
            mean_s, std_s = stats[f]["mean"], stats[f]["std"]
            mean_t = float(np.mean(pops[t_name][f]))
            smd = (mean_t - mean_s) / std_s if std_s > 0 else np.nan
            w1 = float(wasserstein_distance(zs, zt))
            ed = float(energy_distance(zs, zt))
            rows.append(dict(source=s_name, target=t_name, feature=f,
                             group=GROUP[f],
                             n_source=len(zs), n_target=len(zt),
                             mean_source=mean_s, std_source=std_s, mean_target=mean_t,
                             smd=smd, wasserstein1_z=w1, energy_z=ed))
            w_rows.append(dict(source=s_name, target=t_name, feature=f, group=GROUP[f],
                               wasserstein1_z=w1))
            e_rows.append(dict(source=s_name, target=t_name, feature=f, group=GROUP[f],
                               energy_z=ed))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "feature_shift.csv", index=False)
    pd.DataFrame(w_rows).to_csv(OUT_DIR / "wasserstein.csv", index=False)
    pd.DataFrame(e_rows).to_csv(OUT_DIR / "energy_distance.csv", index=False)

    # group-level summary
    srows = []
    for (s_name, t_name) in PAIRS:
        sub = df[(df.source == s_name) & (df.target == t_name)]
        for g in ["instantaneous_thermal", "cumulative_thermal", "context_calendar"]:
            gs = sub[sub.group == g]
            srows.append(dict(
                source=s_name, target=t_name, group=g, n_features=len(gs),
                smd_mean=float(gs.smd.mean()), smd_median=float(gs.smd.median()),
                smd_absmax=float(gs.smd.abs().max()),
                w1_mean=float(gs.wasserstein1_z.mean()),
                w1_median=float(gs.wasserstein1_z.median()),
                w1_max=float(gs.wasserstein1_z.max()),
                energy_mean=float(gs.energy_z.mean()),
                energy_median=float(gs.energy_z.median()),
                energy_max=float(gs.energy_z.max())))
    # thermal-vs-context summary (pooled thermal = instantaneous + cumulative)
    for (s_name, t_name) in PAIRS:
        sub = df[(df.source == s_name) & (df.target == t_name)]
        th = sub[sub.group.isin(["instantaneous_thermal", "cumulative_thermal"])]
        cx = sub[sub.group == "context_calendar"]
        srows.append(dict(
            source=s_name, target=t_name, group="THERMAL_ALL", n_features=len(th),
            smd_mean=float(th.smd.mean()), smd_median=float(th.smd.median()),
            smd_absmax=float(th.smd.abs().max()),
            w1_mean=float(th.wasserstein1_z.mean()),
            w1_median=float(th.wasserstein1_z.median()),
            w1_max=float(th.wasserstein1_z.max()),
            energy_mean=float(th.energy_z.mean()),
            energy_median=float(th.energy_z.median()),
            energy_max=float(th.energy_z.max())))
        srows.append(dict(
            source=s_name, target=t_name, group="CONTEXT_ALL", n_features=len(cx),
            smd_mean=float(cx.smd.mean()), smd_median=float(cx.smd.median()),
            smd_absmax=float(cx.smd.abs().max()),
            w1_mean=float(cx.wasserstein1_z.mean()),
            w1_median=float(cx.wasserstein1_z.median()),
            w1_max=float(cx.wasserstein1_z.max()),
            energy_mean=float(cx.energy_z.mean()),
            energy_median=float(cx.energy_z.median()),
            energy_max=float(cx.energy_z.max())))
    dfg = pd.DataFrame(srows)
    dfg.to_csv(OUT_DIR / "feature_shift_summary.csv", index=False)

    # draft Extended-Data Fig 9 base figure (base data only; formal figure untouched)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True)
        feat_order = FEATURES
        y = np.arange(len(feat_order))
        for ax, (s_name, t_name) in zip(axes, PAIRS):
            sub = df[(df.source == s_name) & (df.target == t_name)].set_index("feature")
            smd = [sub.loc[f, "smd"] for f in feat_order]
            w1 = [sub.loc[f, "wasserstein1_z"] for f in feat_order]
            colors = ["#1f77b4" if GROUP[f] == "instantaneous_thermal" else
                      "#2ca02c" if GROUP[f] == "cumulative_thermal" else "#d62728"
                      for f in feat_order]
            ax.barh(y + 0.2, smd, height=0.38, color=colors, label="SMD")
            ax.barh(y - 0.2, w1, height=0.38, color=colors, alpha=0.45, label="W1(z)")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{s_name} -> {t_name}", fontsize=10)
            ax.set_xlabel("SMD  /  Wasserstein-1 (z, SCS-train stats)")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(feat_order, fontsize=8)
        axes[0].invert_yaxis()
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=a)
                   for c, a in [("#1f77b4", 1), ("#2ca02c", 1), ("#d62728", 1), ("#555555", 0.45)]]
        axes[2].legend(handles, ["inst. thermal", "cum. thermal (C_W)", "context/calendar",
                                 "lighter = W1(z)"], fontsize=7, loc="lower right")
        fig.suptitle("Source-target feature shift (draft base data for ED Fig 9)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "ed_fig9_draft.png", dpi=180)
        plt.close(fig)
        print("[saved] ed_fig9_draft.png", flush=True)
    except Exception as e:
        print(f"[warn] figure failed: {type(e).__name__}: {e}", flush=True)

    # auto-flagged findings for analysis.md
    lines = ["# Domain Shift Analysis — SCS train vs SCS/GBR/CARIB test", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "Populations (ocean cells only; weeks pooled with ocean pixels):"]
    for name in POPULATIONS:
        m = meta_pops[name]
        lines.append(f"- **{name}**: {m['file']}, years {m['years'][0]}-{m['years'][1]}, "
                     f"{m['n_weeks']} weeks x {m['n_ocean_cells']} ocean cells")
    lines += ["", "Standardization: SCS-train (A) statistics ONLY (no per-domain normalization).",
              "Metrics: SMD = (mean_target − mean_A)/std_A; Wasserstein-1 and energy distance "
              "on z-scored values.", "",
              "## Group-level summary", "",
              "| source | target | group | SMD median | SMD |max| | W1(z) median | W1 max | energy median | energy max |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in dfg.iterrows():
        lines.append(f"| {r.source} | {r.target} | {r.group} | {r.smd_median:+.3f} | "
                     f"{r.smd_absmax:.3f} | {r.w1_median:.3f} | {r.w1_max:.3f} | "
                     f"{r.energy_median:.3f} | {r.energy_max:.3f} |")
    lines += ["", "## Per-feature table", "",
              "| source | target | feature | group | SMD | W1(z) | energy(z) |",
              "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.source} | {r.target} | {r.feature} | {r.group} | "
                     f"{r.smd:+.3f} | {r.wasserstein1_z:.3f} | {r.energy_z:.3f} |")
    lines += ["", "## Auto-flagged extremes (per pair)", ""]
    for (s_name, t_name) in PAIRS:
        sub = df[(df.source == s_name) & (df.target == t_name)]
        top_smd = sub.reindex(sub.smd.abs().sort_values(ascending=False).index).head(5)
        top_w1 = sub.sort_values("wasserstein1_z", ascending=False).head(5)
        lines.append(f"### {s_name} -> {t_name}")
        lines.append("- Largest |SMD|: " + "; ".join(
            f"{r.feature} ({r.smd:+.2f}, {r.group})" for _, r in top_smd.iterrows()))
        lines.append("- Largest W1(z): " + "; ".join(
            f"{r.feature} ({r.wasserstein1_z:.2f})" for _, r in top_w1.iterrows()))
        lines.append("")
    lines += ["## Performance anchor (frozen TSCF, ocean-masked)", "",
              "| target | n | severe | ROC | PR | CSI@0.9 |", "|---|---|---|---|---|---|"]
    for t_name, p in PERF.items():
        lines.append(f"| {t_name} | {p['n']} | {p['sev']} | {p['roc']:.4f} | "
                     f"{p['pr']:.4f} | {p['csi']:.4f} |")
    lines += ["", "NOTE: association language only — this analysis does NOT establish that "
              "distribution shift causes the transfer performance difference.", ""]
    (OUT_DIR / "analysis.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "title": "Domain shift analysis — frozen TSCF input space",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "experiments/08_infrastructure/compute_domain_shift_analysis.py",
        "deterministic": True, "randomness": "none (no subsampling below MAX_N=200000; "
                                               "deterministic stride thinning above cap)",
        "populations": meta_pops,
        "features": FEATURES, "groups": GROUP,
        "standardization": "SCS-train mean/std only",
        "metrics": ["smd", "wasserstein1_z", "energy_z"],
        "pairs": [list(p) for p in PAIRS],
        "performance_anchor": PERF,
        "inputs": {POPULATIONS[k]["file"]: sha256_file(DATA_DIR / POPULATIONS[k]["file"])
                   for k in POPULATIONS},
        "constraints": "no model training; no hyperparameter search; no target adaptation; "
                       "no CARIB-test tuning; Step 2-10 untouched",
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
