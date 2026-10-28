# compute_conditional_residual_information_test.py
#
# CONDITIONAL RESIDUAL INFORMATION TEST
#
# Goal: does the TSAR residual R = TSAR - PersA carry CONDITIONAL CONTEXT
# information about the future severe label beyond PersA_DHW, or does it only
# re-encode DHW?
#
# Strictly source-only / zero-shot: uses ONLY the frozen record sets
# (rec_structural_ablation.npz, itself derived from the frozen TSCF/TSAR and
# the causal-v2 PFV protocol). No target-domain training, tuning or threshold
# selection. No training protocol change.
#
# Design (per basin SCS / GBR / CARIB):
#   1. PersA = clip(DHW/10,0,1); R = TSAR - PersA; y = future target y(t+4);
#      severe = y >= 0.8.
#   2. Model comparison:
#        A: y ~ PersA ;  B: y ~ PersA + R (= TSAR)
#      report ROC-AUC, PR-AUC, and PAIRED week-block bootstrap (1000, seed 42,
#      shared resample matrix over weeks) difference DeltaROC / DeltaPR 95% CI.
#   3. DHW-matched test: bin PersA at 0.05; within each bin test Spearman
#      rho(R, y) with a bootstrap CI, and report severe-rate by R-quantile.
#   4. Outputs -> results/conditional_residual/ {summary.json, bootstrap.csv,
#      matched_bins.csv, AUDIT.md}; provenance hashes recorded.
#
# Conclusion rule (strict):
#   residual contains conditional context information ONLY IF
#     (i) the PersA+R (=TSAR) increment paired CI excludes 0 (ROC or PR), AND
#     (ii) in the DHW-matched bins R shows a stable association with y
#          (rho 95% CI excludes 0 with consistent sign in a majority of the
#          bins that have enough severe events).
#   Otherwise: "the TSAR residual is NOT proven to provide information
#   independent of DHW."

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
REC = LTTS_ROOT / "results" / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
OUT = LTTS_ROOT / "results" / "conditional_residual"
OUT.mkdir(parents=True, exist_ok=True)

SEV = 0.8
DHW_BIN = 0.05
MIN_BIN_N = 40          # min records in a bin to report it
BOOT_N = 1000
BOOT_SEED = 42
RHO_N = 1000
RHO_SEED = 42
DOMAINS = ["SCS", "GBR", "CARIB"]


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def roc_pr(sev, score):
    try:
        roc = float(roc_auc_score(sev, score))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(sev, score))
    except Exception:
        pr = float("nan")
    return roc, pr


def week_block_paired(persa, tsar, y, t_idx, n_boot=BOOT_N, seed=BOOT_SEED):
    """Paired week-block bootstrap (shared resample matrix over weeks).
    Returns per-rep DeltaROC/DeltaPR and their 95% CI."""
    sev = (y >= SEV).astype(int)
    rng = np.random.default_rng(seed)
    weeks = np.unique(t_idx)
    n_w = len(weeks)
    bywk = {w: np.where(t_idx == w)[0] for w in weeks}
    d_roc = np.empty(n_boot); d_pr = np.empty(n_boot)
    for r in range(n_boot):
        pick = rng.integers(0, n_w, n_w)
        idx = np.concatenate([bywk[weeks[w]] for w in pick])
        s, y_, p, ts = sev[idx], y[idx], persa[idx], tsar[idx]
        r_a, _ = roc_pr(s, p); r_b, _ = roc_pr(s, ts)
        p_a, _ = (float("nan"), 0.0) if False else (roc_pr(s, p)[1], 0.0)
        _, pr_a = roc_pr(s, p); _, pr_b = roc_pr(s, ts)
        d_roc[r] = r_b - r_a; d_pr[r] = pr_b - pr_a
    def ci(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"lo": float("nan"), "hi": float("nan")}
        return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                "mean": float(v.mean())}
    return {"droc": ci(d_roc), "dpr": ci(d_pr)}


def spearman_boot(R, y, n=RHO_N, seed=RHO_SEED):
    """Bootstrap (observation resample, seed 42) CI for Spearman rho(R, y)."""
    rng = np.random.default_rng(seed)
    m = np.isfinite(R) & np.isfinite(y)
    Rc, yc = R[m], y[m]
    if len(Rc) < 10 or len(np.unique(Rc)) < 2 or len(np.unique(yc)) < 2:
        return None
    base = spearmanr(Rc, yc).statistic
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(Rc), len(Rc))
        if len(np.unique(Rc[idx])) < 2 or len(np.unique(yc[idx])) < 2:
            continue
        vals.append(spearmanr(Rc[idx], yc[idx]).statistic)
    vals = np.array(vals, dtype=float); vals = vals[np.isfinite(vals)]
    if vals.size < 100:
        return None
    return {"rho": float(base),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5))}


def severe_by_R_quantile(R, y, severe, q=3):
    m = np.isfinite(R) & np.isfinite(y)
    Rc, s = R[m], severe[m]
    if len(Rc) < 3 * q:
        return None
    qr = pd.qcut(Rc, q, labels=False, duplicates="drop")
    out = []
    for qi in sorted(np.unique(qr[~np.isnan(qr)])):
        mask = (qr == qi)
        if mask.sum() == 0:
            continue
        out.append({"R_quantile": int(qi) + 1, "n": int(mask.sum()),
                    "n_severe": int(s[mask].sum()),
                    "severe_rate": float(s[mask].mean())})
    return out


def dhw_matched(persa, R, y):
    sev = (y >= SEV).astype(int)
    edges = np.arange(0.0, 1.00001, DHW_BIN)
    bins_out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (persa >= lo) & (persa < hi)
        n = int(m.sum())
        if n < MIN_BIN_N:
            continue
        pbi = {"bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
               "n": n, "n_severe": int(sev[m].sum()),
               "severe_rate": float(sev[m].mean())}
        rb = spearman_boot(R[m], y[m])
        pbi["spearman_rho"] = rb["rho"] if rb else None
        pbi["rho_ci_lo"] = rb["ci_lo"] if rb else None
        pbi["rho_ci_hi"] = rb["ci_hi"] if rb else None
        pbi["R_quantile_severe"] = severe_by_R_quantile(R[m], y[m], sev[m])
        bins_out.append(pbi)
    return bins_out


print("=" * 66)
print("CONDITIONAL RESIDUAL INFORMATION TEST")
print(f"Source: {REC}")
print(f"md5: {md5_file(REC)}")
print(f"Outputs: {OUT}")
print("=" * 66)

z = np.load(REC)
summary = {"experiment": "conditional_residual_information_test",
           "source": str(REC), "source_md5": md5_file(REC),
           "definitions": {"persa": "clip(DHW/10,0,1) at issue time t",
                           "residual R": "TSAR - PersA = gate*raw*scale(0.4)",
                           "y": "future target y(t+4)",
                           "severe": "y >= 0.8"},
           "protocol": {"zero_shot": True, "no_target_fitting": True,
                        "no_threshold_selection": True,
                        "bootstrap": {"type": "paired week-block", "n": BOOT_N, "seed": BOOT_SEED},
                        "matched": {"DHW_bin": DHW_BIN, "min_bin_n": MIN_BIN_N,
                                    "spearman_bootstrap": {"n": RHO_N, "seed": RHO_SEED}}},
           "domains": {}}
boot_rows = []
matched_rows = []
a_b_rows = []

for dom in DOMAINS:
    persa = z[f"{dom}_A_PersA_pred"].astype(float)
    tsar = z[f"{dom}_C_anchored_pred"].astype(float)
    t_idx = z[f"{dom}_A_PersA_t"].astype(np.int64)
    y = z[f"{dom}_target"].astype(float)
    R = tsar - persa
    sev = (y >= SEV).astype(int)

    roc_a, pr_a = roc_pr(sev, persa)
    roc_b, pr_b = roc_pr(sev, tsar)
    dr = week_block_paired(persa, tsar, y, t_idx)
    inc_roc_excl = (dr["droc"]["lo"] > 0) if not np.isnan(dr["droc"]["lo"]) else False
    inc_pr_excl = (dr["dpr"]["lo"] > 0) if not np.isnan(dr["dpr"]["lo"]) else False
    inc_supported = inc_roc_excl or inc_pr_excl

    matched = dhw_matched(persa, R, y)
    # stable matched association: rho CI excludes 0 with consistent sign in a
    # majority of bins that have >= some severe events (power)
    pow_bins = [b for b in matched if b["n_severe"] >= 5]
    if pow_bins:
        signs = []
        for b in pow_bins:
            if b["spearman_rho"] is None:
                continue
            if b["rho_ci_lo"] is not None and b["rho_ci_hi"] is not None:
                if b["rho_ci_lo"] > 0:
                    signs.append(1)
                elif b["rho_ci_hi"] < 0:
                    signs.append(-1)
                else:
                    signs.append(0)
        matched_stable = (len(signs) > 0 and signs.count(1) == len(signs)) or \
                         (len(signs) > 0 and signs.count(-1) == len(signs))
        # require majority (>= ceil(n/2)) of powered bins, all remaining non-conflicting
        pos = signs.count(1); neg = signs.count(-1); zero = signs.count(0)
        majority = max(pos, neg) >= (len(signs) + 1) // 2 and zero == 0 and max(pos, neg) > 0
        matched_stable = majority
    else:
        matched_stable = False

    concl = (inc_supported and matched_stable)
    dom_res = {"n": int(len(y)), "n_severe": int(sev.sum()),
               "A_persa": {"roc": roc_a, "pr": pr_a},
               "B_tsar": {"roc": roc_b, "pr": pr_b},
               "delta_paired": {"droc": dr["droc"], "dpr": dr["dpr"]},
               "increment_ci_excludes_0": {"roc": inc_roc_excl, "pr": inc_pr_excl,
                                           "any": inc_supported},
               "matched": {"powered_bins": len(pow_bins), "n_bins": len(matched),
                           "rho_sign_consistency": matched_stable},
               "verdict_residual_has_conditional_context_information": concl}
    summary["domains"][dom] = dom_res
    boot_rows.append({"domain": dom, "comparison": "TSAR-PersA(=PersA+R)",
                      "d_roc_point": float(roc_b - roc_a),
                      "d_roc_lo": dr["droc"]["lo"], "d_roc_hi": dr["droc"]["hi"],
                      "d_pr_point": pr_b - pr_a,
                      "d_pr_lo": dr["dpr"]["lo"], "d_pr_hi": dr["dpr"]["hi"],
                      "increment_CI_excludes_0": inc_supported})
    a_b_rows.append({"domain": dom, "model": "A_PersA", "roc": roc_a, "pr": pr_a,
                     "n": int(len(y)), "n_severe": int(sev.sum())})
    a_b_rows.append({"domain": dom, "model": "B_TSAR", "roc": roc_b, "pr": pr_b,
                     "n": int(len(y)), "n_severe": int(sev.sum())})
    for b in matched:
        rq = b.pop("R_quantile_severe")
        matched_rows.append({"domain": dom, "bin_lo": b["bin_lo"], "bin_hi": b["bin_hi"],
                             "n": b["n"], "n_severe": b["n_severe"],
                             "severe_rate": b["severe_rate"],
                             "spearman_rho": b["spearman_rho"],
                             "rho_ci_lo": b["rho_ci_lo"], "rho_ci_hi": b["rho_ci_hi"],
                             "R_Q1_rate": rq[0]["severe_rate"] if rq and len(rq) >= 1 else None,
                             "R_Q2_rate": rq[1]["severe_rate"] if rq and len(rq) >= 2 else None,
                             "R_Q3_rate": rq[2]["severe_rate"] if rq and len(rq) >= 3 else None})
    print(f"\n[{dom}] n={len(y)} n_sev={int(sev.sum())}")
    print(f"  A(PersA)  ROC {roc_a:.4f} PR {pr_a:.4f}")
    print(f"  B(TSAR)   ROC {roc_b:.4f} PR {pr_b:.4f}")
    print(f"  paired ΔROC [{dr['droc']['lo']:+.4f},{dr['droc']['hi']:+.4f}] "
          f"ΔPR [{dr['dpr']['lo']:+.4f},{dr['dpr']['hi']:+.4f}]  CI-excl0={inc_supported}")
    print(f"  matched bins reported: {len(matched)} (powered_>=5sev: {len(pow_bins)}), "
          f"rho-sign stable={matched_stable}")

def global_conclusion_summary():
    return ""

# conclusion (strict, global: all three basins must satisfy)
s = summary["domains"]
global_has_cond = all(s[d]["verdict_residual_has_conditional_context_information"] for d in DOMAINS)
summary["conclusion"] = {
    "residual_contains_conditional_context_information": bool(global_has_cond),
    "rule": ("PersA+R increment paired-CI excludes 0 AND matched-DHW R shows stable "
             "association -> residual has conditional context info; otherwise NOT proven "
             "independent of DHW."),
    "text": (
        ("The TSAR residual R provides conditional context information beyond PersA_DHW "
         "only in basins where (i) the PersA+R increment CI excludes 0 and (ii) R remains "
         "stably associated with future severe outcome within matched-DHW bins.") if global_has_cond
        else ("The TSAR residual R is NOT proven to provide conditional context information "
              "independent of DHW: the evidence does not satisfy BOTH (i) a CI-supported "
              "PersA+R increment and (ii) a stable within-DHW association of R with future "
              "severe outcome."))
}

# ---- write outputs ----
pd.DataFrame(a_b_rows).to_csv(OUT / "ab_roc_pr.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap.csv", index=False)
pd.DataFrame(matched_rows).to_csv(OUT / "matched_bins.csv", index=False)
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

audit = []
audit.append("# Conditional Residual Information Test — AUDIT")
audit.append("")
audit.append(f"Generated: {datetime.now().isoformat()}")
audit.append(f"Source records: {REC}")
audit.append(f"Source md5: {md5_file(REC)}")
audit.append("")
audit.append("## Provenance / integrity")
audit.append(f"- REC source md5: {md5_file(REC)} (frozen rec_structural_ablation.npz)")
audit.append("- A_PersA_pred = PersA = clip(DHW/10,0,1) at issue time t")
audit.append("- C_anchored_pred = TSAR = PersA + gate*raw*0.4 (anchor + bounded residual)")
audit.append("- R = TSAR - PersA (the residual correction being tested)")
audit.append("- target = future coordinate y(t+4); severe = y >= 0.8")
audit.append("- Analysis is purely associative diagnostics; associations are not causal")
audit.append("")
audit.append("## Protocol compliance")
audit.append("- [PASS] Source-only / zero-shot: uses only frozen record sets; no fitting")
audit.append("- [PASS] No training-protocol change; no target-domain training/tuning")
audit.append("- [PASS] No threshold selection on any target (severe >= 0.8 is the frozen definition)")
audit.append(f"- [PASS] Bootstrap: paired week-block, n={BOOT_N}, seed={BOOT_SEED}")
audit.append(f"- [PASS] DHW-matched bins: {DHW_BIN} width, min n={MIN_BIN_N}; "
             f"Spearman rho bootstrap n={RHO_N}, seed={RHO_SEED}")
audit.append("")
audit.append("## Conclusion (strict rule)")
for d in DOMAINS:
    v = s[d]
    audit.append(f"- {d}: increment CI excludes 0 = {v['increment_ci_excludes_0']}; "
                 f"matched-DHW rho stable = {v['matched']['rho_sign_consistency']}; "
                 f"VERDICT = residual has conditional info: "
                 f"{v['verdict_residual_has_conditional_context_information']}")
audit.append(f"- GLOBAL conclusion: residual contains conditional context information = "
             f"{bool(global_has_cond)}")
audit.append("- Rule: BOTH (CI-supported PersA+R increment) AND (stable within-DHW R "
             "association) required; otherwise NOT proven independent of DHW.")
with open(OUT / "AUDIT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(audit) + "\n")

print("\n== OUTPUTS ==")
print(f"  {OUT / 'summary.json'}")
print(f"  {OUT / 'bootstrap.csv'}")
print(f"  {OUT / 'matched_bins.csv'}")
print(f"  {OUT / 'AUDIT.md'}")
print(f"\nGLOBAL conclusion: residual contains conditional context information = {bool(global_has_cond)}")
for d in DOMAINS:
    v = s[d]
    print(f"  [{d}] increment CI excl0={v['increment_ci_excludes_0']}  "
          f"matched stable={v['matched']['rho_sign_consistency']}  "
          f"verdict={v['verdict_residual_has_conditional_context_information']}")
print("\nDONE")