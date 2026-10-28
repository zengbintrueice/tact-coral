#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint-structure shift analysis — quantifies the 'relational mismatch'
hypothesis left open by the marginal domain-shift analysis.

PURE ANALYSIS — no training, no tuning, no target adaptation, no changes to
frozen Step 2-10 outputs.

Two complementary quantities, per source->target pair
(SCS_train -> SCS_test / GBR_test / CARIB_test):

  1. CHANNEL CORRELATION-STRUCTURE SHIFT
     Pearson correlation matrix R (16 features, pooled week x ocean-pixel
     rows, listwise-complete) per population. Distance metric:
     Frobenius ||R_target - R_SCS_train||_F.
     Two references make the distances interpretable:
       (a) split-half noise floor: ||R_A_odd - R_A_even||_F (estimation noise);
       (b) in-domain temporal drift: ||R_SCS_test - R_A||_F (same-basin drift
           the model tolerates).

  2. CHANNEL <-> y(t+4) ASSOCIATION DRIFT
     Per feature: Pearson corr(feature(t), y(t+4)) pooled over the PFV-style
     record (issue weeks in the population years, ocean cells, finite target;
     features at the window's last step t). Drift = corr_target - corr_train,
     with paired week-block bootstrap CIs.

Week-block bootstrap (1,000 replicates, seed 42): issue/observation weeks are
resampled with replacement WITHIN each population (one shared pick matrix per
pair), so structure distances and association drifts carry sampling CIs.

Outputs -> results/domain_shift/:
  joint_structure.csv (Frobenius distances + refs + bootstrap CIs)
  y_assoc.csv         (per-feature corr and drift + CIs)
  corr_matrices.npz   (the four R matrices)
  joint_structure.md

Reuses feature definitions from compute_domain_shift_analysis.py.
"""
import io, sys, os, json, warnings
from pathlib import Path
from datetime import datetime, timezone

# NOTE: do NOT wrap sys.stdout here — compute_domain_shift_analysis (imported
# below) re-wraps it at module level; a double wrap raises
# "ValueError: I/O operation on closed file". Console output below is ASCII.
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_domain_shift_analysis import (  # noqa: E402
    FEATURES, GROUP, POPULATIONS, PAIRS, trailing_cw, DATA_DIR,
)

LTTS_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = LTTS_ROOT / "results" / "domain_shift"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
BOOT_SEED = 42


def load_full(pop):
    """Full feature cube [T,H,W,16] + ocean mask + times for one population."""
    import xarray as xr
    ds = xr.open_dataset(DATA_DIR / pop["file"])
    times = pd.DatetimeIndex(ds.time.values)
    ocean = ds["ocean_mask"].values.astype(bool)
    hot = ds["hotspot"].values.astype(np.float32)
    nanm = np.isnan(hot)
    hsc = np.where(nanm, 0.0, hot)
    d_hs = np.zeros_like(hot)
    d_hs[1:] = hsc[1:] - hsc[:-1]
    d_hs[nanm] = np.nan
    woy = np.array([t.isocalendar()[1] for t in times], dtype=np.float32)
    seas_cos = np.broadcast_to(
        np.cos(2 * np.pi * woy / 52.0)[:, None, None].astype(np.float32), hot.shape).copy()
    cubes = []
    for f in FEATURES:
        if f in ("C1", "C2", "C4", "C8", "C12"):
            arr = trailing_cw(hot, int(f[1:]))
        elif f == "d_hotspot_dt":
            arr = d_hs
        elif f == "seasonal_cos":
            arr = seas_cos
        else:
            arr = ds[f].values.astype(np.float32)
        cubes.append(arr[..., None])
    X = np.concatenate(cubes, axis=-1)          # [T,H,W,16]
    y = ds["y_regression"].values.astype(np.float32)
    ds.close()
    return X, y, ocean, times


def rows_for_weeks(X, ocean, week_ids):
    """Gather [n_rows, 16] rows for the given week ids (all ocean pixels)."""
    sub = X[week_ids][:, ocean, :]              # [n_weeks, n_ocean, 16]
    return sub.reshape(-1, sub.shape[-1])


def corr_matrix(rows):
    """Listwise-complete Pearson correlation matrix (16x16)."""
    ok = np.isfinite(rows).all(axis=1)
    r = rows[ok]
    if r.shape[0] < 100:
        return np.full((rows.shape[1], rows.shape[1]), np.nan)
    return np.corrcoef(r, rowvar=False)


def frob(a, b):
    return float(np.linalg.norm(a - b))


def year_week_ids(times, years):
    return np.where((times.year >= years[0]) & (times.year <= years[1]))[0]


def main():
    print("Loading full feature cubes...", flush=True)
    data = {}
    for name, pop in POPULATIONS.items():
        X, y, ocean, times = load_full(pop)
        wids = year_week_ids(times, pop["years"])
        data[name] = dict(X=X, y=y, ocean=ocean, times=times, wids=wids)
        print(f"  {name}: T={X.shape[0]} ocean={int(ocean.sum())} "
              f"pop_weeks={len(wids)}", flush=True)

    results = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "pairs": {}}
    jrows, arows = [], []
    R_store = {}

    # ---- point correlation matrices ----
    print("Correlation matrices (point)...", flush=True)
    R = {}
    for name in POPULATIONS:
        d = data[name]
        rows = rows_for_weeks(d["X"], d["ocean"], d["wids"])
        R[name] = corr_matrix(rows)
        R_store[name] = R[name]
        cov = np.isfinite(rows).all(axis=1).mean()
        print(f"  {name}: listwise-complete frac={cov:.4f}", flush=True)

    # split-half noise floor within SCS_train (odd/even pop-week index)
    wids = data["SCS_train"]["wids"]
    R_half1 = corr_matrix(rows_for_weeks(data["SCS_train"]["X"], data["SCS_train"]["ocean"],
                                         wids[0::2]))
    R_half2 = corr_matrix(rows_for_weeks(data["SCS_train"]["X"], data["SCS_train"]["ocean"],
                                         wids[1::2]))
    noise_floor = frob(R_half1, R_half2)
    R_store["SCS_train_halves_1"] = R_half1
    R_store["SCS_train_halves_2"] = R_half2
    print(f"  split-half noise floor = {noise_floor:.4f}", flush=True)

    # ---- y(t+4) association records ----
    print("Building y(t+4) association records...", flush=True)
    H = {name: dict(X=[], y=[], wids=[]) for name in POPULATIONS}
    for name in POPULATIONS:
        d = data[name]
        X, yy, ocean, times, wids = d["X"], d["y"], d["ocean"], d["times"], d["wids"]
        n = X.shape[0]
        rec_X, rec_y, rec_w = [], [], []
        for t in wids:
            tt = t + 4
            if tt >= n:
                continue
            v = np.isfinite(yy[tt]) & ocean
            if v.sum() == 0:
                continue
            rec_X.append(X[t][ocean][:, :])     # features at issue week t, ocean cells
            rec_y.append(yy[tt][ocean])
            rec_w.append(np.full(int(ocean.sum()), t, np.int32))
        H[name] = dict(X=np.concatenate(rec_X), y=np.concatenate(rec_y),
                       wids=np.concatenate(rec_w))
        # listwise-complete mask against NaN features
        ok = np.isfinite(H[name]["X"]).all(axis=1) & np.isfinite(H[name]["y"])
        H[name]["X"] = H[name]["X"][ok]
        H[name]["y"] = H[name]["y"][ok]
        H[name]["wids"] = H[name]["wids"][ok]
        print(f"  {name}: assoc rows={len(H[name]['y'])} "
              f"weeks={len(np.unique(H[name]['wids']))}", flush=True)

    def pearson_vec(Xm, yv):
        ok = np.isfinite(Xm).all(axis=1) & np.isfinite(yv)
        Xm, yv = Xm[ok], yv[ok]
        if len(yv) < 100:
            return np.full(Xm.shape[1], np.nan)
        Xc = Xm - Xm.mean(0)
        yc = yv - yv.mean()
        num = (Xc * yc[:, None]).sum(0)
        den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
        return num / np.where(den == 0, np.nan, den)

    # ---- point metrics ----
    print("Point metrics...", flush=True)
    assoc = {name: pearson_vec(H[name]["X"], H[name]["y"]) for name in POPULATIONS}
    for s_name, t_name in PAIRS:
        fd = frob(R[t_name], R[s_name])
        jrows.append(dict(source=s_name, target=t_name, metric="frobenius_corr",
                          value=fd, ci_lo=np.nan, ci_hi=np.nan, kind="point"))
        print(f"  ||R_{t_name} - R_{s_name}||_F = {fd:.4f}", flush=True)
        for i, f in enumerate(FEATURES):
            drift = float(assoc[t_name][i] - assoc[s_name][i])
            arows.append(dict(source=s_name, target=t_name, feature=f, group=GROUP[f],
                              corr_source=float(assoc[s_name][i]),
                              corr_target=float(assoc[t_name][i]),
                              drift=drift, ci_lo=np.nan, ci_hi=np.nan))

    # ---- week-block bootstrap ----
    print(f"Week-block bootstrap ({N_BOOT}, seed {BOOT_SEED})...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot_f = {p: [] for p in PAIRS}
    boot_a = {p: [] for p in PAIRS}
    for b in range(N_BOOT):
        Rb = {}
        Hb = {}
        for name in POPULATIONS:
            d = data[name]
            w = d["wids"]
            pick = w[rng.integers(0, len(w), size=len(w))]
            Rb[name] = corr_matrix(rows_for_weeks(d["X"], d["ocean"], pick))
            hh = H[name]
            uw = np.unique(hh["wids"])
            upick = uw[rng.integers(0, len(uw), size=len(uw))]
            # map resampled issue-weeks to rows
            mask = np.isin(hh["wids"], upick)
            Hb[name] = (hh["X"][mask], hh["y"][mask])
        for p in PAIRS:
            boot_f[p].append(frob(Rb[p[1]], Rb[p[0]]))
            av = pearson_vec(Hb[p[1]][0], Hb[p[1]][1])
            asv = pearson_vec(Hb[p[0]][0], Hb[p[0]][1])
            boot_a[p].append(av - asv)
        if (b + 1) % 200 == 0:
            print(f"    boot {b+1}/{N_BOOT}", flush=True)

    for p in PAIRS:
        v = np.array([x for x in boot_f[p] if np.isfinite(x)])
        for r in jrows:
            if r["source"] == p[0] and r["target"] == p[1]:
                r["ci_lo"], r["ci_hi"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        Av = np.array(boot_a[p])                      # [n_boot, 16]
        for i, f in enumerate(FEATURES):
            for r in arows:
                if r["source"] == p[0] and r["target"] == p[1] and r["feature"] == f:
                    col = Av[:, i]
                    col = col[np.isfinite(col)]
                    r["ci_lo"] = float(np.percentile(col, 2.5))
                    r["ci_hi"] = float(np.percentile(col, 97.5))

    # reference rows
    jrows.append(dict(source="SCS_train", target="SCS_train_split_half",
                      metric="frobenius_corr", value=noise_floor,
                      ci_lo=np.nan, ci_hi=np.nan, kind="noise_floor_reference"))
    jrows.append(dict(source="SCS_train", target="SCS_test", metric="frobenius_corr",
                      value=[r["value"] for r in jrows
                             if r["target"] == "SCS_test" and r["kind"] == "point"][0],
                      ci_lo=[r["ci_lo"] for r in jrows
                             if r["target"] == "SCS_test" and r["kind"] == "point"][0],
                      ci_hi=[r["ci_hi"] for r in jrows
                             if r["target"] == "SCS_test" and r["kind"] == "point"][0],
                      kind="in_domain_drift_reference"))

    pd.DataFrame(jrows).to_csv(OUT_DIR / "joint_structure.csv", index=False)
    pd.DataFrame(arows).to_csv(OUT_DIR / "y_assoc.csv", index=False)
    np.savez_compressed(OUT_DIR / "corr_matrices.npz", **R_store)

    # ---- markdown ----
    lines = ["# Joint-Structure Shift — correlation matrices and y(t+4) association drift", "",
             f"Generated: {results['generated_utc']} · bootstrap {N_BOOT} seed {BOOT_SEED} "
             "(week-block, within-population resampling)", "",
             "16 features as in the marginal analysis; Pearson correlations on pooled "
             "week x ocean-pixel rows (listwise-complete); distances are Frobenius norms "
             "of correlation-matrix differences. Association uses features at the issue "
             "week t vs y_regression at t+4 (PFV-style, ocean cells).", "",
             "## Correlation-structure distances", "",
             "| pair | Frobenius | 95% CI | note |", "|---|---|---|---|"]
    for r in jrows:
        note = {"point": "", "noise_floor_reference": "split-half estimation noise floor",
                "in_domain_drift_reference": "in-domain temporal drift"}[r["kind"]]
        lines.append(f"| {r['source']} -> {r['target']} | {r['value']:.4f} | "
                     f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {note} |")
    lines += ["", "## Channel <-> y(t+4) association drift", "",
              "| pair | feature | group | corr_source | corr_target | drift | 95% CI |",
              "|---|---|---|---|---|---|---|"]
    for r in arows:
        lines.append(f"| {r['source']} -> {r['target']} | {r['feature']} | {r['group']} | "
                     f"{r['corr_source']:+.4f} | {r['corr_target']:+.4f} | "
                     f"{r['drift']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |")
    lines += ["", "Association language only; no causal attribution.", ""]
    (OUT_DIR / "joint_structure.md").write_text("\n".join(lines), encoding="utf-8")

    results["frobenius"] = jrows
    results["y_assoc"] = arows
    with open(OUT_DIR / "joint_structure_manifest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
