# compute_dhw_conditioned_susceptibility.py
#
# DHW-Conditioned Basin Susceptibility Test.
#
# Question: at MATCHED current-DHW levels, do the three basins differ in
# future thermal susceptibility (severe y(t+4) rate), and does the TSAR
# residual R = TSAR − PersA separate that heterogeneity within matched-DHW
# strata?
#
# FIXED PROTOCOL (frozen records; seed-42 TSAR; NO retraining, target
# fitting/tuning or threshold optimisation; associative / conditional-risk
# language only):
#   1. current DHW = PersA×10 (clipped coordinate), fixed 0.5 °C-week bins,
#      restricted to bins supported by ALL THREE basins (≥40 records each)
#   2. per basin/bin future severe rate + week-block bootstrap 95% CI
#      (1,000, seed 42)
#   3. basin-pair risk difference / risk ratio per bin + bootstrap CI
#      (independent per-basin streams, element-wise difference)
#   4. within each basin/bin: outcome-free fixed tertiles of R (boundaries
#      from the full record set); Q1→Q3 severe-rate change per bin; pooled
#      conditional effect = bin-count-weighted mean of (Q3−Q1), paired
#      week-block bootstrap CI
#   5. A = DHW-only vs B = DHW+R via pre-specified training-free percentile
#      rank aggregation (mean of within-basin percentile ranks); ROC-AUC /
#      PR-AUC + paired ΔROC/ΔPR CI (shared resampled weeks); frozen TSAR
#      reported as reference
#
# Outputs -> results/dhw_conditioned_susceptibility/
#
# Run (piecf_env; pure statistics, no GPU inference needed):
#   cd LTTS_CRW/experiments/08_infrastructure
#   & 'D:\...\piecf_env\Scripts\python.exe' compute_dhw_conditioned_susceptibility.py

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import RESULTS_DIR

OUT = RESULTS_DIR / "dhw_conditioned_susceptibility"
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


# ══════════════════════════════════════════════════════════════════════
print("=" * 66)
print(f"DHW-conditioned basin susceptibility — {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 66)
assert md5(REC_NPZ) == REC_MD5
print(f"frozen records OK ({REC_MD5})")

z = np.load(REC_NPZ)

# ---- load + bin ----
D = {}
for b in BASINS:
    persa = z[f"{b}_A_PersA_pred"].astype(np.float64)
    tsar = z[f"{b}_C_anchored_pred"].astype(np.float64)
    tgt = z[f"{b}_target"].astype(np.float64)
    t = z[f"{b}_A_PersA_t"].astype(np.int64)
    ok = np.isfinite(persa) & np.isfinite(tsar) & np.isfinite(tgt)
    dhw = persa[ok] * 10.0
    D[b] = {"dhw": dhw, "R": (tsar - persa)[ok], "sev": (tgt[ok] >= SEV).astype(int),
            "week": t[ok], "tsar": tsar[ok], "persa": persa[ok]}
    print(f"[{b}] n={ok.sum()} DHW range [{dhw.min():.2f}, {dhw.max():.2f}]")

# common supported bins
edges = np.arange(0, 10.0 + BIN_W / 2, BIN_W)
centers = edges[:-1] + BIN_W / 2
counts = {}
for b in BASINS:
    c, _ = np.histogram(D[b]["dhw"], bins=edges)
    counts[b] = c
common_bins = np.where(np.min([counts[b] for b in BASINS], axis=0) >= MIN_BIN_N)[0]
print(f"common supported bins (≥{MIN_BIN_N} each): "
      f"{[f'{centers[i]:.2f}' for i in common_bins]}")

for b in BASINS:
    bidx = np.digitize(D[b]["dhw"], edges) - 1
    keep = np.isin(bidx, common_bins)
    D[b].update(bidx=bidx, keep=keep)

surv_rows, pair_rows, cond_rows, boot_rows = [], [], [], []
panel = {}
basin_week_boot = {}

# ---- per-basin/bin rates + within-bin R-tertile effects + bootstrap ----
bin_rate_boot = {}     # basin -> replicate matrix [n_boot, n_common_bins]
for b in BASINS:
    d = D[b]
    k = d["keep"]
    dhw, sev, week, R = d["dhw"][k], d["sev"][k], d["week"][k], d["R"][k]
    bidx = d["bidx"][k]
    pos = np.searchsorted(common_bins, bidx)
    nb = len(common_bins)
    # fixed outcome-free tertile boundaries per bin (from full records)
    tert = np.full(len(sev), -1)
    for j in range(nb):
        m = pos == j
        if m.sum() >= 3:
            q = np.quantile(R[m], [1 / 3, 2 / 3])
            tert[m] = np.digitize(R[m], q)          # 0/1/2
    weeks, inv = np.unique(week, return_inverse=True)
    idx_by_week = [np.where(inv == w)[0] for w in range(len(weeks))]
    bin_n_full = np.array([ (pos == j).sum() for j in range(nb)], float)
    bin_n_sev_full = np.array([sev[pos == j].sum() for j in range(nb)], float)
    rates_full = bin_n_sev_full / bin_n_full
    # tertile rates per bin (full)
    tert_rows = {}
    for j in range(nb):
        for q in range(3):
            m = (pos == j) & (tert == q)
            tert_rows[(j, q)] = (sev[m].sum(), m.sum())
    # pooled conditional effect (Q3−Q1), count-weighted over eligible bins
    elig = [j for j in range(nb) if tert_rows[(j, 0)][1] > 0 and tert_rows[(j, 2)][1] > 0]
    wts = np.array([tert_rows[(j, 0)][1] + tert_rows[(j, 2)][1] for j in elig], float)
    diffs_full = np.array([tert_rows[(j, 2)][0] / tert_rows[(j, 2)][1]
                           - tert_rows[(j, 0)][0] / tert_rows[(j, 0)][1]
                           for j in elig])
    pooled_full = float(np.average(diffs_full, weights=wts)) if len(elig) else np.nan
    # bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    rate_boot = np.full((N_BOOT, nb), np.nan)
    pooled_boot = np.full(N_BOOT, np.nan)
    tertq_boot = {q: np.full((N_BOOT, nb), np.nan) for q in range(3)}
    for r in range(N_BOOT):
        pick = rng.integers(0, len(weeks), len(weeks))
        idx = np.concatenate([idx_by_week[w] for w in pick])
        p_, s_, tq = pos[idx], sev[idx], tert[idx]
        n = np.bincount(p_, minlength=nb)
        ssum = np.bincount(p_, weights=s_, minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_boot[r] = np.where(n > 0, ssum / n, np.nan)
        for q in range(3):
            for j in range(nb):
                m = (p_ == j) & (tq == q)
                if m.sum() > 0:
                    tertq_boot[q][r, j] = s_[m].sum() / m.sum()
        # pooled conditional effect on this replicate (fixed tert assignment)
        d3 = np.array([tertq_boot[2][r, j] for j in range(nb)])
        d0 = np.array([tertq_boot[0][r, j] for j in range(nb)])
        okj = [j for j in elig if np.isfinite(d3[j]) and np.isfinite(d0[j])]
        if okj:
            w2 = np.array([bin_n_full[j] for j in okj], float)
            pooled_boot[r] = np.average([d3[j] - d0[j] for j in okj], weights=w2)
    basin_week_boot[b] = rate_boot
    for j in range(nb):
        c = centers[common_bins[j]]
        surv_rows.append({"basin": b, "bin_center": c, "n": int(bin_n_full[j]),
                          "n_severe": int(bin_n_sev_full[j]),
                          "severe_rate": rates_full[j],
                          "ci_lo": float(np.nanpercentile(rate_boot[:, j], 2.5)),
                          "ci_hi": float(np.nanpercentile(rate_boot[:, j], 97.5))})
    for j in elig:
        c = centers[common_bins[j]]
        for q in range(3):
            nq, sq = tert_rows[(j, q)]
            cond_rows.append({"basin": b, "bin_center": c, "R_quantile": q + 1,
                              "n": nq, "n_severe": sq,
                              "severe_rate": (sq / nq) if nq else np.nan})
    boot_rows.append({"basin": b, "stat": "pooled_conditional_effect_R_Q3_minus_Q1",
                      "point": pooled_full,
                      "ci_lo": float(np.nanpercentile(pooled_boot, 2.5)),
                      "ci_hi": float(np.nanpercentile(pooled_boot, 97.5))})
    print(f"[{b}] pooled conditional effect (Q3−Q1 | DHW bin) = {pooled_full:+.4f} "
          f"[{boot_rows[-1]['ci_lo']:+.4f}, {boot_rows[-1]['ci_hi']:+.4f}] "
          f"over {len(elig)} bins")
    panel[b] = {"persa": D[b]["persa"][k], "R": R, "sev": sev, "tsar": d["tsar"][k],
                "target": (D[b]["sev"][k])}

# ---- basin-pair RD/RR per bin ----
pairs = [("CARIB", "GBR"), ("GBR", "SCS"), ("CARIB", "SCS")]
for bA, bB in pairs:
    rateA = basin_week_boot[bA]
    rateB = basin_week_boot[bB]
    for j in range(len(common_bins)):
        c = centers[common_bins[j]]
        pA = surv_rows[ [i for i, r in enumerate(surv_rows)
                         if r["basin"] == bA and r["bin_center"] == c][0]]["severe_rate"]
        pB = surv_rows[ [i for i, r in enumerate(surv_rows)
                         if r["basin"] == bB and r["bin_center"] == c][0]]["severe_rate"]
        rd = rateA[:, j] - rateB[:, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = rateA[:, j] / rateB[:, j]
        pair_rows.append({"pair": f"{bA}_minus_{bB}", "bin_center": c,
                          "risk_diff": pA - pB,
                          "rd_lo": float(np.nanpercentile(rd, 2.5)),
                          "rd_hi": float(np.nanpercentile(rd, 97.5)),
                          "risk_ratio": (pA / pB) if pB > 0 else np.nan,
                          "rr_lo": float(np.nanpercentile(rr, 2.5)),
                          "rr_hi": float(np.nanpercentile(rr, 97.5))})

# ---- Part 5: A = DHW-only vs B = DHW+R (rank aggregation) ----
for b in BASINS:
    d = panel[b]
    sev = d["sev"]
    p_persa = rankdata(d["persa"], method="average") / len(d["persa"])
    p_R = rankdata(d["R"], method="average") / len(d["R"])
    p_tsar = rankdata(d["tsar"], method="average") / len(d["tsar"])
    A = p_persa
    Bscore = (p_persa + p_R) / 2.0
    week = D[b]["week"][D[b]["keep"]]
    weeks, inv = np.unique(week, return_inverse=True)
    idx_by_week = [np.where(inv == w)[0] for w in range(len(weeks))]
    rng = np.random.default_rng(BOOT_SEED)
    bs = {"A": {"roc": [], "pr": []}, "B": {"roc": [], "pr": []},
          "TSAR": {"roc": [], "pr": []}}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(weeks), len(weeks))
        idx = np.concatenate([idx_by_week[w] for w in pick])
        for name, s in (("A", A), ("B", Bscore), ("TSAR", p_tsar)):
            try:
                bs[name]["roc"].append(roc_auc_score(sev[idx], s[idx]))
                bs[name]["pr"].append(average_precision_score(sev[idx], s[idx]))
            except ValueError:
                pass
    def pt(name, s):
        return {"roc": roc_auc_score(sev, s), "pr": average_precision_score(sev, s)}
    for name, s in (("A", A), ("B", Bscore), ("TSAR", p_tsar)):
        m = pt(name, s)
        for met in ("roc", "pr"):
            arr = np.array(bs[name][met])
            boot_rows.append({"basin": b, "stat": f"{met}_{name}", "point": m[met],
                              "ci_lo": float(np.percentile(arr, 2.5)),
                              "ci_hi": float(np.percentile(arr, 97.5))})
    for met in ("roc", "pr"):
        dd = np.array(bs["B"][met]) - np.array(bs["A"][met])
        pm = pt("B", Bscore)[met] - pt("A", A)[met]
        boot_rows.append({"basin": b, "stat": f"delta_{met}_B_minus_A", "point": pm,
                          "ci_lo": float(np.percentile(dd, 2.5)),
                          "ci_hi": float(np.percentile(dd, 97.5))})
    print(f"[{b}] A(DHW) ROC {pt('A',A)['roc']:.4f} PR {pt('A',A)['pr']:.4f} | "
          f"B(DHW+R) ROC {pt('B',Bscore)['roc']:.4f} PR {pt('B',Bscore)['pr']:.4f} | "
          f"ΔROC {boot_rows[-2]['point']:+.4f} [{boot_rows[-2]['ci_lo']:+.4f},"
          f"{boot_rows[-2]['ci_hi']:+.4f}]")
    panel[b].update({"A": A, "B": Bscore, "tsar_rank": p_tsar})

np.savez_compressed(OUT / "panel_scores.npz", **{
    f"{b}_{k}": v for b, d in panel.items() for k, v in d.items()})

# ---- verdicts (pre-specified) ----
verdicts = {}
# (1) matched-DHW basin specificity: bins with CI-excluded RD for CARIB−GBR
cg = [r for r in pair_rows if r["pair"] == "CARIB_minus_GBR"]
n_excl = sum(1 for r in cg if (r["rd_lo"] > 0) or (r["rd_hi"] < 0))
verdicts["basin_specific_susceptibility_matcheddhw"] = {
    "n_common_bins": len(cg), "n_bins_CARIB_GBR_RD_CI_excl0": n_excl,
    "established": bool(n_excl >= max(1, len(cg) // 2))}
# (2) R separates within matched DHW: pooled conditional effect CI > 0
pe = {r["basin"]: r for r in boot_rows
      if r["stat"] == "pooled_conditional_effect_R_Q3_minus_Q1"}
verdicts["R_separates_susceptibility_within_dhw"] = {
    b: {"point": pe[b]["point"], "ci": [pe[b]["ci_lo"], pe[b]["ci_hi"]],
        "established": bool(pe[b]["ci_lo"] > 0)} for b in BASINS}
# (3) GBR/CARIB consistent residual pattern: sign agreement of pooled effects
verdicts["GBR_CARIB_consistent_residual_pattern"] = {
    "GBR_point": pe["GBR"]["point"], "CARIB_point": pe["CARIB"]["point"],
    "sign_consistent": bool(np.sign(pe["GBR"]["point"]) == np.sign(pe["CARIB"]["point"])
                            and pe["GBR"]["point"] != 0)}

summary = {"experiment": "dhw_conditioned_susceptibility",
           "generated": datetime.now().isoformat(),
           "record_npz_md5": REC_MD5,
           "protocol": {"bin_width": BIN_W, "min_bin_n": MIN_BIN_N,
                        "common_bins": [float(centers[i]) for i in common_bins],
                        "bootstrap": {"type": "week-block per basin", "n": N_BOOT,
                                      "seed": BOOT_SEED},
                        "aggregation": "training-free mean percentile rank (DHW, R)",
                        "interpretation": "associative / conditional-risk only"},
           "verdicts": verdicts}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

pd.DataFrame(surv_rows).to_csv(OUT / "susceptibility_surface.csv", index=False)
pd.DataFrame(cond_rows).to_csv(OUT / "conditional_residual_effect.csv", index=False)
pd.DataFrame(pair_rows).to_csv(OUT / "basin_pair_risk.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap.csv", index=False)

# ---- AUDIT.md ----
L = ["# AUDIT — DHW-Conditioned Basin Susceptibility Test", "",
     f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  ",
     f"Frozen records md5 `{REC_MD5}` (verified); seed-42 TSAR / PersA from the "
     "frozen record set. Environment: piecf_env (pure statistics; no GPU "
     "inference — no model is run).", "",
     "## Protocol adherence", "",
     f"- current DHW = PersA×10 (clipped coordinate), fixed {BIN_W} °C-week "
     f"bins; only bins with ≥{MIN_BIN_N} records in EVERY basin are used "
     f"({len(common_bins)} common bins).",
     "- Future severe = y(t+4) ≥ 0.8 from the frozen record targets; no "
     "threshold was optimised, no model retrained, no target fitting.",
     "- All CIs: week-block bootstrap (1,000, seed 42) within basin; "
     "basin-pair differences use independent per-basin streams (no cross-basin "
     "pairing).",
     "- Within-bin R tertiles use boundaries fixed from the full record set "
     "(outcome-free); the pooled conditional effect is the bin-count-weighted "
     "mean of (Q3−Q1) severe-rate changes.",
     "- A vs B comparison uses a pre-specified training-free aggregation "
     "(mean of within-basin percentile ranks of DHW and R); no coefficients "
     "fitted; paired week-block bootstrap on shared resampled weeks.",
     "- Language: associative / conditional risk. NOT causal mechanism.", "",
     "## Pre-specified verdicts", "",
     f"1. Basin-specific susceptibility at matched DHW: "
     f"{verdicts['basin_specific_susceptibility_matcheddhw']['n_bins_CARIB_GBR_RD_CI_excl0']}"
     f"/{len(cg)} common bins have CARIB−GBR risk-difference CIs excluding 0 "
     f"→ **{'established' if verdicts['basin_specific_susceptibility_matcheddhw']['established'] else 'not established'}**.",
     ""]
for b in BASINS:
    v = verdicts["R_separates_susceptibility_within_dhw"][b]
    L.append(f"2{chr(96+BASINS.index(b)+1)}. R separates within matched DHW "
             f"({b}): {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] "
             f"→ **{'established' if v['established'] else 'not established'}**.")
L += ["", f"3. GBR/CARIB residual-pattern sign consistency: "
      f"GBR {verdicts['GBR_CARIB_consistent_residual_pattern']['GBR_point']:+.4f} "
      f"vs CARIB {verdicts['GBR_CARIB_consistent_residual_pattern']['CARIB_point']:+.4f} "
      f"→ {'sign-consistent' if verdicts['GBR_CARIB_consistent_residual_pattern']['sign_consistent'] else 'not consistent'}.",
      "", "## Interpretation guard", "",
      "Rates are conditional-risk descriptions of the frozen record set. "
      "'Susceptibility' means higher observed future severe-thermal-event "
      "frequency at matched current DHW; it is not a causal or biological "
      "claim.", ""]
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print(f"\nOutputs -> {OUT}")
print("Done.")