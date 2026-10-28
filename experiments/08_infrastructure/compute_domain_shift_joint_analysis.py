#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compute_domain_shift_joint_analysis.py — JOINT-dependence structure shift
across the three TSCF domains (SCS source; GBR / CARIB zero-shot targets).

Follow-up to the marginal analysis (compute_domain_shift_analysis.py), which
showed that per-feature marginal distances do NOT rank-order the transfer
outcome (CARIB shifts LESS than GBR yet fails). This script quantifies the
hypothesis the marginal audit could not test: the failure is associated with a
RELATIONAL / joint-structure mismatch — how channels co-vary, how they map to
the future target, and how they evolve in time.

Sections (all ocean-masked; NO model training, no target-domain fitting):
  J1. Correlation-matrix distance vs SCS-train reference (13 model-input
      channels): Frobenius distance, mean |dcorr|, top departing channel pairs.
  J2. Channel→target association shift: Spearman corr(channel at issue t,
      y(t+4)) per domain on the shared PFV record set — which channel→future
      relationships break on CARIB vs GBR. Also corr(channel, PersA).
  J3. Temporal-dynamics shift: pooled lag-1 and lag-4 autocorrelation of each
      channel per domain (the dynamics a ConvLSTM can exploit).
  J4. Prediction-level: Spearman corr(TSCF_pred, channel) per domain (from the
      saved physics_anchored records — no re-inference).

Outputs -> results/domain_shift/:
  joint_shift.json / joint_corr_distance.csv / joint_assoc_shift.csv
  joint_temporal_dynamics.csv / joint_pred_corr.csv / joint_shift.md

Run:
    cd LTTS_CRW/experiments/08_infrastructure
    & 'D:\\...\\piecf_env\\Scripts\\python.exe' compute_domain_shift_joint_analysis.py
"""
import io, sys, os, json, warnings
from pathlib import Path

# no ablate import here — stdout wrap is safe
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

LTTS_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get(
    "PIECF_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
OUT_DIR = LTTS_ROOT / "results" / "domain_shift"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_NPZ = LTTS_ROOT / "results" / "physics_anchored" / "physics_anchored_records.npz"

RAW_DYN = ['sst', 'ssta', 'hotspot', 'chlorophyll', 'par', 'seasonal_sin',
           'wind_speed', 'ssr', 'sst_anomaly']
DERIVED = ['d_hotspot_dt', 'hotspot_cumsum_4w', 'hotspot_cumsum_12w', 'seasonal_cos']
CHANNELS = RAW_DYN + DERIVED

DOMAINS = [
    {"name": "SCS_train", "file": "scs_causal_v2.nc", "years": (2003, 2010), "is_ref": True},
    {"name": "SCS_test", "file": "scs_causal_v2.nc", "years": (2016, 2019)},
    {"name": "GBR_test", "file": "gbr_causal_v2.nc", "years": (2017, 2020)},
    {"name": "CARIB_test", "file": "caribbean_causal_v2.nc", "years": (2017, 2020)},
]
RECORD_DOMAINS = {"SCS_test": "SCS", "GBR_test": "GBR", "CARIB_test": "CARIB"}


def load_channels_13(path):
    """9 raw dynamic channels + 4 derived recomputed (trailing, NaN-preserving),
    exactly as aligned_dataset does at load time."""
    ds = xr.open_dataset(path)
    times = pd.DatetimeIndex(ds.time.values)
    ocean = ds['ocean_mask'].values.astype(bool)
    arrs = [np.ascontiguousarray(ds[n].values.astype(np.float32)) for n in RAW_DYN]
    ds.close()
    hs = arrs[2]
    nanm = np.isnan(hs)
    hsc = np.where(nanm, 0.0, hs)
    cum = np.cumsum(hsc, axis=0)
    d_hs = np.zeros_like(hs); d_hs[1:] = hsc[1:] - hsc[:-1]; d_hs[nanm] = np.nan
    cs4 = np.zeros_like(hs); cs12 = np.zeros_like(hs)
    for i in range(hs.shape[0]):
        cs4[i] = cum[i] if i < 4 else cum[i] - cum[i - 4]
        cs12[i] = cum[i] if i < 12 else cum[i] - cum[i - 12]
    cs4[nanm] = np.nan; cs12[nanm] = np.nan
    woy = np.array([t.isocalendar()[1] for t in times], dtype=np.float32)
    sc = np.broadcast_to(np.cos(2 * np.pi * woy / 52.0)[:, None, None].astype(np.float32),
                         hs.shape).copy()
    return times, ocean, arrs + [d_hs, cs4, cs12, sc]


def week_pooled(arrs, ocean, week_mask):
    """[n_weeks*n_ocean, 13] pooled channel values over ocean cells of selected weeks."""
    out = []
    for a in arrs:
        sub = a[week_mask][:, ocean]      # [tw, n_ocean]
        out.append(sub.ravel())
    return np.stack(out, axis=1)


def corr_matrix(X):
    """Pairwise-deletion Pearson corr matrix."""
    k = X.shape[1]
    C = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(a, k):
            m = np.isfinite(X[:, a]) & np.isfinite(X[:, b])
            if m.sum() > 100:
                va, vb = X[m, a], X[m, b]
                if va.std() > 0 and vb.std() > 0:
                    C[a, b] = C[b, a] = float(np.corrcoef(va, vb)[0, 1])
    return C


def pooled_lag_corr(arrs, ocean, lag, ci):
    """Pooled Pearson autocorr of channel ci at lag `lag` weeks, over ocean
    cells and all valid (t, t-lag) pairs."""
    a = arrs[ci][:, ocean]                # [T, n_ocean]
    x0 = a[lag:]
    x1 = a[:-lag]
    m = np.isfinite(x0) & np.isfinite(x1)
    if m.sum() < 100 or x0[m].std() == 0 or x1[m].std() == 0:
        return np.nan
    return float(np.corrcoef(x0[m].ravel(), x1[m].ravel())[0, 1])


def spearman_cols(X, y):
    out = np.full(X.shape[1], np.nan)
    for c in range(X.shape[1]):
        v = X[:, c]
        m = np.isfinite(v) & np.isfinite(y)
        if m.sum() > 100 and len(np.unique(y[m])) > 1:
            out[c] = float(spearmanr(v[m], y[m]).statistic)
    return out


def main():
    recs = np.load(RECORDS_NPZ)

    # ---- load all populations ----
    pops = {}
    for dom in DOMAINS:
        times, ocean, arrs = load_channels_13(DATA_DIR / dom["file"])
        wk = (times.year >= dom["years"][0]) & (times.year <= dom["years"][1])
        pops[dom["name"]] = dict(times=times, ocean=ocean, arrs=arrs, wk=wk,
                                 years=dom["years"], is_ref=dom.get("is_ref", False))
        print(f"[load] {dom['name']}: weeks={int(wk.sum())} ocean={int(ocean.sum())}",
              flush=True)

    ref = pops["SCS_train"]
    Xref = week_pooled(ref["arrs"], ref["ocean"], ref["wk"])
    Cref = corr_matrix(Xref)

    rows_J1, rows_J2, rows_J3, rows_J4 = [], [], [], []
    results = {"reference": "SCS train 2003-2010 ocean cells",
               "no_training": True, "channels": CHANNELS, "populations": {}}

    for dom in DOMAINS:
        name = dom["name"]
        p = pops[name]
        X = week_pooled(p["arrs"], p["ocean"], p["wk"])
        Cx = corr_matrix(X)
        frob = float(np.linalg.norm(Cx - Cref, ord="fro"))
        off = ~(np.eye(len(CHANNELS), dtype=bool))
        meand = float(np.nanmean(np.abs(Cx - Cref)[off]))
        results["populations"][name] = {"n_px": int(X.shape[0]),
                                        "corr_frobenius_vs_ref": frob,
                                        "mean_abs_dcorr_offdiag": meand}
        print(f"[{name}] corrFrob={frob:.3f} mean|dcorr|={meand:.3f}", flush=True)

        # top departing channel pairs
        dd = np.abs(Cx - Cref)
        pairs = []
        for a in range(len(CHANNELS)):
            for b in range(a + 1, len(CHANNELS)):
                if np.isfinite(dd[a, b]):
                    pairs.append((float(dd[a, b]), CHANNELS[a], CHANNELS[b],
                                  float(Cref[a, b]), float(Cx[a, b])))
        pairs.sort(reverse=True)
        for dv, ca, cb, rref, rdom in pairs[:5]:
            rows_J1.append({"population": name, "channel_a": ca, "channel_b": cb,
                            "corr_ref": rref, "corr_domain": rdom, "abs_dcorr": dv})
        results["populations"][name]["top_pair_departures"] = [
            {"a": ca, "b": cb, "ref": rref, "dom": rdom, "abs_d": dv}
            for dv, ca, cb, rref, rdom in pairs[:8]]

        # J2 + J4 on the shared PFV record set (test populations only)
        rd = RECORD_DOMAINS.get(name)
        if rd is not None:
            t_idx = recs[f"{rd}_PersA_DHW_t"].astype(int)
            i_idx = recs[f"{rd}_i"].astype(int)
            j_idx = recs[f"{rd}_j"].astype(int)
            target = recs[f"{rd}_target"]
            persA = recs[f"{rd}_PersA_DHW_pred"]
            tscf = recs[f"{rd}_TSCF_frozen_pred"]
            Xrec = np.stack([p["arrs"][ci][t_idx, i_idx, j_idx]
                             for ci in range(len(CHANNELS))], axis=1)
            sp_y = spearman_cols(Xrec, target)
            sp_pA = spearman_cols(Xrec, persA)
            sp_tscf = spearman_cols(Xrec, tscf)
            r_tp = float(spearmanr(tscf, persA).statistic)
            for ci, ch in enumerate(CHANNELS):
                rows_J2.append({"population": name, "channel": ch,
                                "spearman_channel_vs_y(t+4)": float(sp_y[ci]),
                                "spearman_channel_vs_PersA": float(sp_pA[ci])})
                rows_J4.append({"population": name, "channel": ch,
                                "spearman_TSCFpred_vs_channel": float(sp_tscf[ci])})
            rows_J4.append({"population": name, "channel": "(PersA)",
                            "spearman_TSCFpred_vs_channel": r_tp})
            results["populations"][name]["spearman_TSCFpred_vs_PersA"] = r_tp
            print(f"[{name}] corr(TSCF,PersA)={r_tp:.3f}", flush=True)

        # J3 temporal dynamics
        for ci, ch in enumerate(CHANNELS):
            l1 = pooled_lag_corr(p["arrs"], p["ocean"], 1, ci)
            l4 = pooled_lag_corr(p["arrs"], p["ocean"], 4, ci)
            rows_J3.append({"population": name, "channel": ch,
                            "lag1_autocorr": l1, "lag4_autocorr": l4})

    # ---- association-shift contrast table (GBR vs CARIB vs SCS_test) ----
    dfJ2 = pd.DataFrame(rows_J2)
    piv = dfJ2.pivot(index="channel", columns="population",
                     values="spearman_channel_vs_y(t+4)")

    dfJ1 = pd.DataFrame(rows_J1)
    dfJ3 = pd.DataFrame(rows_J3)
    dfJ4 = pd.DataFrame(rows_J4)
    dfJ1.to_csv(OUT_DIR / "joint_corr_distance.csv", index=False)
    dfJ2.to_csv(OUT_DIR / "joint_assoc_shift.csv", index=False)
    dfJ3.to_csv(OUT_DIR / "joint_temporal_dynamics.csv", index=False)
    dfJ4.to_csv(OUT_DIR / "joint_pred_corr.csv", index=False)
    with open(OUT_DIR / "joint_shift.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md = ["# Joint-Dependence Structure Shift (follow-up to the marginal audit)", "",
          "Tests the hypothesis the marginal analysis could not: relational mismatch "
          "(channel co-variation, channel→future-target mapping, temporal dynamics).",
          "Ocean-masked; no model training; reference = SCS train 2003-2010.", "",
          "## J1. Correlation-matrix distance vs SCS-train", "",
          "| Population | n_px | Frobenius vs ref | mean |dcorr| (off-diag) |",
          "|---|---|---|---|"]
    for name in [d["name"] for d in DOMAINS]:
        r = results["populations"][name]
        md.append(f"| {name} | {r['n_px']} | {r['corr_frobenius_vs_ref']:.3f} | "
                  f"{r['mean_abs_dcorr_offdiag']:.3f} |")
    md += ["", "### Largest channel-pair departures (top 5 per population)", "",
           "| Population | Pair | corr(ref) | corr(dom) | |dcorr| |", "|---|---|---|---|---|"]
    for _, r in dfJ1.iterrows():
        md.append(f"| {r['population']} | {r['channel_a']} × {r['channel_b']} | "
                  f"{r['corr_ref']:+.3f} | {r['corr_domain']:+.3f} | {r['abs_dcorr']:.3f} |")

    md += ["", "## J2. Channel → y(t+4) association (Spearman, shared PFV record set)", "",
           "| Channel | SCS_test | GBR_test | CARIB_test |", "|---|---|---|---|"]
    for ch in CHANNELS:
        md.append(f"| {ch} | {piv.loc[ch].get('SCS_test', np.nan):+.3f} | "
                  f"{piv.loc[ch].get('GBR_test', np.nan):+.3f} | "
                  f"{piv.loc[ch].get('CARIB_test', np.nan):+.3f} |")

    md += ["", "## J3. Temporal dynamics (pooled lag-1 / lag-4 autocorrelation)", "",
           "| Channel | SCS_train l1/l4 | SCS_test l1/l4 | GBR_test l1/l4 | CARIB_test l1/l4 |",
           "|---|---|---|---|---|"]
    dfJ3p = dfJ3.pivot(index="channel", columns="population")
    for ch in CHANNELS:
        cells = []
        for name in [d["name"] for d in DOMAINS]:
            try:
                l1 = dfJ3p[("lag1_autocorr", name)][ch]
                l4 = dfJ3p[("lag4_autocorr", name)][ch]
                cells.append(f"{l1:+.3f} / {l4:+.3f}")
            except KeyError:
                cells.append("-")
        md.append(f"| {ch} | " + " | ".join(cells) + " |")

    md += ["", "## J4. What the frozen TSCF prediction actually tracks "
           "(Spearman, shared record set)", "",
           "| Population | ρ(pred, PersA) | ρ(pred, hotspot) | ρ(pred, cumsum12) | "
           "ρ(pred, sst_anomaly) | ρ(pred, seasonal_sin) |", "|---|---|---|---|---|---|"]
    for name in ["SCS_test", "GBR_test", "CARIB_test"]:
        g = dfJ4[dfJ4["population"] == name].set_index("channel")["spearman_TSCFpred_vs_channel"]
        md.append(f"| {name} | {g.get('(PersA)', np.nan):+.3f} | {g.get('hotspot', np.nan):+.3f} | "
                  f"{g.get('hotspot_cumsum_12w', np.nan):+.3f} | {g.get('sst_anomaly', np.nan):+.3f} | "
                  f"{g.get('seasonal_sin', np.nan):+.3f} |")
    md += ["", "Interpretation: if J1-J3 show CARIB departing the reference "
           "joint structure more than GBR (while the marginal audit showed the "
           "opposite ordering), the failure is associated with relational/"
           "representational mismatch rather than marginal distribution distance. "
           "Association language only; no causal claims.", ""]
    (OUT_DIR / "joint_shift.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nOutputs -> {OUT_DIR}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
