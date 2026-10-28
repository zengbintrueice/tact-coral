# compute_matched_dhw_spatial_bootstrap.py
#
# Spatial-cluster robustness of the matched-DHW conditional effects
# (companion to results/dhw_conditioned_susceptibility/ and
#  results/conditional_residual/).
#
# The published matched-DHW conditional effects (pooled R-Q3−Q1 severe-rate
# change per basin; basin-pair risk differences) use a WEEK-block bootstrap.
# Weeks do not account for spatial dependence among pixels. This script adds
# a GRID-CELL spatial-cluster bootstrap (robustness, NOT a replacement):
#   - cluster all records by grid cell (i, j);
#   - resample cells with replacement (1,000, seed 42), preserving each
#     cell's whole time series;
#   - recompute per basin: pooled conditional effect (R tertile Q3−Q1 within
#     0.5 °C-week matched-DHW bins, bin-count-weighted; tertile boundaries
#     fixed from the full record set — outcome-free);
#   - recompute basin-pair risk differences at matched DHW (pooled over bins,
#     count-weighted), paired across models by sharing the same resampled
#     cells.
#
# FIXED PROTOCOL: frozen rec_structural_ablation.npz (md5-verified); seed-42
# TSAR/PersA; no retraining, no target fitting/tuning, no threshold
# optimisation; associative / conditional-risk interpretation only.
#
# Outputs -> results/dhw_conditioned_susceptibility/spatial/
#
# Run (piecf_env):
#   & 'D:\...\piecf_env\Scripts\python.exe'
#       compute_matched_dhw_spatial_bootstrap.py

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

LTTS_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(LTTS_ROOT))
from paths import RESULTS_DIR

OUT = RESULTS_DIR / "dhw_conditioned_susceptibility" / "spatial"
OUT.mkdir(parents=True, exist_ok=True)

REC_NPZ = RESULTS_DIR / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
REC_MD5 = "4035a19fd30fa4d14151f4c8055dfe59"

BASINS = ["SCS", "GBR", "CARIB"]
SEV = 0.8
BIN_W = 0.5
MIN_BIN_N = 40
N_BOOT = 1000
BOOT_SEED = 42


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pooled_r2(y, yhat):  # unused placeholder kept for symmetry
    return np.nan


if __name__ == "__main__":
    print("=" * 66)
    print(f"Matched-DHW spatial-cluster bootstrap — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 66)
    assert md5(REC_NPZ) == REC_MD5, "frozen record set hash mismatch"
    print(f"frozen records OK ({REC_MD5})")

    z = np.load(REC_NPZ)
    D = {}
    for b in BASINS:
        persa = z[f"{b}_A_PersA_pred"].astype(np.float64)
        tsar = z[f"{b}_C_anchored_pred"].astype(np.float64)
        tgt = z[f"{b}_target"].astype(np.float64)
        t = z[f"{b}_A_PersA_t"].astype(np.int64)
        i = z[f"{b}_i"].astype(np.int64)
        j = z[f"{b}_j"].astype(np.int64)
        ok = np.isfinite(persa) & np.isfinite(tsar) & np.isfinite(tgt)
        D[b] = {"dhw": persa[ok] * 10.0, "R": (tsar - persa)[ok],
                "sev": (tgt[ok] >= SEV).astype(int), "week": t[ok],
                "cell": np.stack([i[ok], j[ok]], axis=1)}
        print(f"[{b}] n={ok.sum()}")

    # common supported bins (same rule as the week-block analysis)
    edges = np.arange(0, 10.0 + BIN_W / 2, BIN_W)
    counts = {b: np.histogram(D[b]["dhw"], bins=edges)[0] for b in BASINS}
    common = np.where(np.min([counts[b] for b in BASINS], axis=0) >= MIN_BIN_N)[0]
    print(f"common bins: {len(common)}")

    for b in BASINS:
        bidx = np.digitize(D[b]["dhw"], edges) - 1
        D[b]["pos"] = np.searchsorted(common, bidx)
        D[b]["keep"] = np.isin(bidx, common)
        cells, inv = np.unique(D[b]["cell"][D[b]["keep"]], axis=0,
                               return_inverse=True)
        D[b]["inv_cell"] = inv
        D[b]["n_cells"] = len(cells)
        D[b]["k"] = D[b]["keep"]
        print(f"[{b}] spatial cells: {len(cells)}")

    # fixed outcome-free tertile boundaries per bin (full record set)
    for b in BASINS:
        d = D[b]
        tert = np.full(d["keep"].sum(), -1)
        R, pos = d["R"][d["k"]], d["pos"][d["k"]]
        for jj in range(len(common)):
            m = pos == jj
            if m.sum() >= 3:
                q = np.quantile(R[m], [1 / 3, 2 / 3])
                tert[m] = np.digitize(R[m], q)
        d["tert"] = tert

    rows = []
    rate_boot_by = {}     # basin -> [N_BOOT, nb] replicate rate matrix
    bin_n_by = {}
    # ---- per-basin pooled conditional effect under spatial bootstrap ----
    for b in BASINS:
        d = D[b]
        pos, sev, tert = d["pos"][d["k"]], d["sev"][d["k"]], d["tert"]
        inv = d["inv_cell"]
        n_cells = d["n_cells"]
        nb = len(common)
        bin_n_full = np.array([(pos == j).sum() for j in range(nb)], float)
        # eligible bins (both extreme tertiles populated in full data)
        elig = []
        for j in range(nb):
            m0 = (pos == j) & (tert == 0); m2 = (pos == j) & (tert == 2)
            if m0.sum() > 0 and m2.sum() > 0:
                elig.append(j)
        rng = np.random.default_rng(BOOT_SEED)
        pooled_boot = np.full(N_BOOT, np.nan)
        rate_boot = np.full((N_BOOT, nb), np.nan)
        for r in range(N_BOOT):
            pick = rng.integers(0, n_cells, n_cells)
            idx = np.concatenate([np.where(inv == c)[0] for c in pick])
            p_, s_, tq = pos[idx], sev[idx], tert[idx]
            n = np.bincount(p_, minlength=nb)
            ss = np.bincount(p_, weights=s_.astype(float), minlength=nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                rate_boot[r] = np.where(n > 0, ss / n, np.nan)
            d3 = np.array([np.nanmean(s_[(p_ == j) & (tq == 2)]) if
                           ((p_ == j) & (tq == 2)).sum() else np.nan
                           for j in range(nb)])
            d0 = np.array([np.nanmean(s_[(p_ == j) & (tq == 0)]) if
                           ((p_ == j) & (tq == 0)).sum() else np.nan
                           for j in range(nb)])
            okj = [j for j in elig if np.isfinite(d3[j]) and np.isfinite(d0[j])]
            if okj:
                w = np.array([bin_n_full[j] for j in okj], float)
                pooled_boot[r] = np.average([d3[j] - d0[j] for j in okj], weights=w)
        point_rates = np.array([sev[pos == j].mean() for j in range(nb)])
        for j in range(nb):
            rows.append({"basin": b, "stat": f"bin_rate_{round(0.25 + 0.5*common[j], 2)}",
                         "point": point_rates[j],
                         "sp_lo": float(np.nanpercentile(rate_boot[:, j], 2.5)),
                         "sp_hi": float(np.nanpercentile(rate_boot[:, j], 97.5))})
        _w = np.array([bin_n_full[j] for j in elig], float)
        rows.append({"basin": b, "stat": "pooled_conditional_effect_R_Q3_minus_Q1_spatial",
                     "point": float(np.average(
                         [np.nanmean(sev[(pos == j) & (tert == 2)])
                          - np.nanmean(sev[(pos == j) & (tert == 0)])
                          for j in elig], weights=_w)),
                     "sp_lo": float(np.nanpercentile(pooled_boot, 2.5)),
                     "sp_hi": float(np.nanpercentile(pooled_boot, 97.5))})
        rate_boot_by[b] = rate_boot
        bin_n_by[b] = bin_n_full
        pt = rows[-1]["point"]
        print(f"[{b}] pooled conditional effect (spatial) = {pt:+.4f} "
              f"[{rows[-1]['sp_lo']:+.4f}, {rows[-1]['sp_hi']:+.4f}] "
              f"over {len(elig)} bins, {n_cells} cells")

    # ---- basin-pair risk differences under spatial bootstrap ----
    rate_by = {b: np.array([r["point"] for r in rows
                            if r["basin"] == b and r["stat"].startswith("bin_rate")])
               for b in BASINS}
    for bA, bB in (("GBR", "SCS"), ("CARIB", "SCS"), ("CARIB", "GBR")):
        rd_bins = rate_boot_by[bA] - rate_boot_by[bB]   # [N_BOOT, nb], paired draws
        w_pair = 0.5 * (bin_n_by[bA] + bin_n_by[bB])    # pooled bin weights
        rd_pooled = np.average(rd_bins, axis=1, weights=w_pair)
        point = float(np.average(rate_by[bA] - rate_by[bB], weights=w_pair))
        rows.append({"basin": f"{bA}_minus_{bB}",
                     "stat": "pooled_matched_dhw_risk_diff_spatial",
                     "point": point,
                     "sp_lo": float(np.nanpercentile(rd_pooled, 2.5)),
                     "sp_hi": float(np.nanpercentile(rd_pooled, 97.5))})
        print(f"[{bA}-{bB}] pooled matched-DHW RD (spatial) = {point:+.4f} "
              f"[{rows[-1]['sp_lo']:+.4f}, {rows[-1]['sp_hi']:+.4f}]")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "spatial_bootstrap.csv", index=False)
    summary = {"experiment": "matched_dhw_spatial_bootstrap",
               "generated": datetime.now().isoformat(),
               "record_npz_md5": REC_MD5,
               "protocol": {"cluster": "grid cell (i, j)", "n_boot": N_BOOT,
                            "seed": BOOT_SEED, "bin_width": BIN_W,
                            "min_bin_n": MIN_BIN_N, "n_common_bins": len(common),
                            "role": "spatial robustness; week-block bootstrap "
                                    "remains primary",
                            "interpretation": "associative / conditional-risk only"},
               "n_cells": {b: int(D[b]["n_cells"]) for b in BASINS}}
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nOutputs -> {OUT}")
    print("Done.")
