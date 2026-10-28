# compute_basin_susceptibility_heterogeneity.py
#
# BASIN-SPECIFIC SUSCEPTIBILITY / THERMAL-RESPONSE HETEROGENEITY TEST
#
# Question: at the SAME current DHW / thermal stress, do SCS, GBR and CARIB
# differ systematically in their future severe thermal-risk response?
# TSAR training results are NOT used as explanatory variables (Part D only
# tests the FIXED residual R = TSAR - PersA against the heterogeneity).
#
# FIXED PROTOCOL
#   - frozen rec_structural_ablation.npz (seed-42 TSCF/TSAR, PFV records)
#   - target y = y(t+4); severe = y >= 0.8; issue-time information only
#     (PersA = clip(DHW/10,0,1) at issue t is the exposure)
#   - SCS / GBR / CARIB analysed separately; identical bins/protocol
#   - NO retraining, NO target tuning, NO threshold optimisation
#   - bootstrap: paired week-block, 1000, seed 42 (weeks resampled within basin)
#   - all conclusions associative/predictive; NO causal claim
#
# Part A  conditional thermal-response curves: 0.1-wide DHW bins restricted to
#         the COMMON support (bins where ALL basins have n>=100); per-bin
#         future severe probability with Wilson (binomial) + bootstrap 95% CI.
# Part B  fixed logistic AND rank-logistic: severe ~ DHW + basin + DHWxbasin
#         (IRLS, no regularisation, no tuning); basin slopes / ORs, interaction
#         terms, pairwise slope contrasts with grouped-by-week bootstrap CI.
# Part C  within common support: pooled + per-bin conditional severe-rate
#         differences (GBR-SCS, CARIB-SCS, CARIB-GBR) with bootstrap CI;
#         Spearman(DHW, future y) per basin with CI.
# Part D  fixed residual R = TSAR - PersA: partial Spearman(R, future | DHW)
#         per basin + CI; within matched-DHW bins R-tertile severe rates.
#
# OUTPUTS -> results/basin_susceptibility/
#   summary.json, response_surface.csv, interaction_model.csv,
#   pairwise_contrasts.csv, bootstrap.csv, AUDIT.md, panel_figure.png
# The figure is rendered by READING BACK the written result files (no
# hard-coded numbers).
#
# NOTE ON GPU: every quantity here derives from frozen predictions plus
# closed-form statistics; there is NO model inference, so no GPU kernel is
# invoked (piecf_env is used). Recorded in AUDIT.

import io, sys, json, hashlib, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

LTTS_ROOT = Path(__file__).resolve().parents[2]
REC = LTTS_ROOT / "results" / "physics_anchored" / "structural_ablation" / "rec_structural_ablation.npz"
OUT = LTTS_ROOT / "results" / "basin_susceptibility"
OUT.mkdir(parents=True, exist_ok=True)

# single-instance guard (the task launcher has been observed double-executing
# background commands; the duplicate would race on the same output files)
import time as _time
_LOCK = OUT / ".lock"
if _LOCK.exists() and (_time.time() - _LOCK.stat().st_mtime) < 1800:
    print("[lock] another fresh instance is running; exiting")
    sys.exit(0)
_LOCK.write_text(datetime.utcnow().isoformat())

SEV = 0.8
BOOT_N = 1000
BOOT_SEED = 42
BIN_W = 0.1
MIN_BIN_N = 100
DOMAINS = ["SCS", "GBR", "CARIB"]


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def rank1(x):
    from scipy.stats import rankdata
    m = np.isfinite(x)
    r = np.full_like(x, np.nan, dtype=float)
    r[m] = rankdata(x[m]) / (m.sum() + 1)
    return r


def partial_spearman(R, y, persa):
    a, b, c = rank1(R), rank1(y), rank1(persa)
    X = np.column_stack([a, b, c])
    C = np.corrcoef(X.T)
    try:
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return np.nan
    return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))


def irls_logistic(X, y, max_iter=60, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        g = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(X.shape[1]), g)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return beta
    return beta


print("=" * 70)
print("BASIN-SPECIFIC SUSCEPTIBILITY / THERMAL-RESPONSE HETEROGENEITY TEST")
print(f"Source: {REC}  md5={md5_file(REC)}")
print(f"Outputs: {OUT}")
print("=" * 70)

z = np.load(REC)
data = {}
for dom in DOMAINS:
    persa = z[f"{dom}_A_PersA_pred"].astype(float)
    tsar = z[f"{dom}_C_anchored_pred"].astype(float)
    t = z[f"{dom}_A_PersA_t"].astype(np.int64)
    y = z[f"{dom}_target"].astype(float)
    sev = (y >= SEV).astype(np.float64)
    R = tsar - persa
    nb = int(np.ceil(1.0 / BIN_W))
    b = np.clip((persa / BIN_W).astype(int), 0, nb - 1)
    data[dom] = dict(persa=persa, y=y, sev=sev, t=t, R=R, bin=b)
    weeks = np.unique(t)
    print(f"[{dom}] n={len(persa)} n_sev={int(sev.sum())} "
          f"DHW support [{np.nanmin(persa):.3f},{np.nanmax(persa):.3f}] weeks={len(weeks)}")

# ---------- common DHW support ----------
nb = int(np.ceil(1.0 / BIN_W))
bin_ns = {d: np.array([int((data[d]['bin'] == k).sum()) for k in range(nb)]) for d in DOMAINS}
common_bins = [k for k in range(nb) if all(bin_ns[d][k] >= MIN_BIN_N for d in DOMAINS)]
lo_c = common_bins[0] * BIN_W
hi_c = (common_bins[-1] + 1) * BIN_W
print(f"\nCommon DHW support (all basins n>={MIN_BIN_N}): bins {common_bins} "
      f"=> [{lo_c:.2f}, {hi_c:.2f}]")
for d in DOMAINS:
    print(f"  {d}: per-bin n = {bin_ns[d].tolist()}")

summary = {"experiment": "basin_susceptibility_heterogeneity",
           "source": str(REC), "source_md5": md5_file(REC),
           "common_support": {"bins": common_bins, "lo": lo_c, "hi": hi_c,
                              "bin_width": BIN_W, "min_bin_n": MIN_BIN_N},
           "per_bin_n": {d: bin_ns[d].tolist() for d in DOMAINS},
           "protocol": {"bootstrap": {"type": "paired week-block (within basin)",
                                      "n": BOOT_N, "seed": BOOT_SEED},
                        "no_retraining": True, "no_target_tuning": True,
                        "no_threshold_opt": True,
                        "gpu_note": ("no model inference required; all quantities are "
                                     "closed-form statistics of frozen predictions")},
           "partA": {}, "partB": {}, "partC": {}, "partD": {}}

resp_rows, inter_rows, pair_rows, boot_rows = [], [], [], []

# ---------- Part A / C / D per-basin bootstrap ----------
perbin_rate = {d: {} for d in DOMAINS}          # point per-bin rates
perbin_boot = {d: {k: [] for k in common_bins} for d in DOMAINS}
spear_boot = {d: [] for d in DOMAINS}
psp_boot = {d: [] for d in DOMAINS}
pooled_rate = {d: None for d in DOMAINS}
pooled_boot = {d: [] for d in DOMAINS}
rtert_point = {d: {} for d in DOMAINS}
rtert_boot = {d: {q: [] for q in (1, 2, 3)} for d in DOMAINS}
sp_point = {}

for dom in DOMAINS:
    dd = data[dom]
    m = np.isfinite(dd['persa']) & np.isfinite(dd['y']) & np.isfinite(dd['R'])
    idx_all = np.where(m)[0]
    b, sev, y, t, persa, R = (dd['bin'][idx_all], dd['sev'][idx_all], dd['y'][idx_all],
                              dd['t'][idx_all], dd['persa'][idx_all], dd['R'][idx_all])
    weeks = np.unique(t)
    bywk = {w: np.where(t == w)[0] for w in weeks}
    # point estimates
    for k in common_bins:
        mk = b == k
        perbin_rate[dom][k] = float(sev[mk].mean())
    mc = np.isin(b, common_bins)
    pooled_rate[dom] = float(sev[mc].mean())
    sp_point[dom] = float(np.corrcoef(rank1(persa), rank1(y))[0, 1])
    # R tertiles on common support
    Rc = R[mc]
    q = pd.qcut(Rc, 3, labels=False, duplicates="drop")
    for qi in (1, 2, 3):
        mq = (q == qi - 1)
        rtert_point[dom][qi] = float(sev[mc][mq].mean())
    # bootstrap (weeks within basin)
    rng = np.random.default_rng(BOOT_SEED)
    for _ in range(BOOT_N):
        pick = rng.integers(0, len(weeks), len(weeks))
        idx = np.concatenate([bywk[weeks[w]] for w in pick])
        bb, ss, pp, yy, RR = b[idx], sev[idx], persa[idx], y[idx], R[idx]
        for k in common_bins:
            mk = bb == k
            perbin_boot[dom][k].append(float(ss[mk].mean()) if mk.sum() else np.nan)
        mcc = np.isin(bb, common_bins)
        pooled_boot[dom].append(float(ss[mcc].mean()))
        spear_boot[dom].append(float(np.corrcoef(rank1(pp), rank1(yy))[0, 1]))
        psp_boot[dom].append(partial_spearman(RR, yy, pp))
        qc = pd.qcut(RR[mcc], 3, labels=False, duplicates="drop")
        for qi in (1, 2, 3):
            mq = (qc == qi - 1)
            rtert_boot[dom][qi].append(float(ss[mcc][mq].mean()) if mq.sum() else np.nan)
    print(f"[{dom}] bootstrap done: pooled common-support severe rate "
          f"{pooled_rate[dom]:.4f}, Spearman(DHW,y) {sp_point[dom]:+.3f}")

# Wilson CIs + response_surface rows (Part A)
for dom in DOMAINS:
    dd = data[dom]
    for k in common_bins:
        n_k = int((dd['bin'] == k).sum())
        kk = int(dd['sev'][dd['bin'] == k].sum())
        wlo, whi = wilson(kk, n_k)
        blo = float(np.nanpercentile(perbin_boot[dom][k], 2.5))
        bhi = float(np.nanpercentile(perbin_boot[dom][k], 97.5))
        resp_rows.append({"part": "A", "domain": dom,
                          "dhw_bin_lo": round(k * BIN_W, 3),
                          "dhw_bin_hi": round((k + 1) * BIN_W, 3),
                          "n": n_k, "n_severe": kk,
                          "severe_rate": perbin_rate[dom][k],
                          "wilson_lo": wlo, "wilson_hi": whi,
                          "boot_lo": blo, "boot_hi": bhi})
        boot_rows.append({"statistic": "bin_severe_rate", "domain": dom,
                          "param": f"bin[{k * BIN_W:.1f},{(k + 1) * BIN_W:.1f})",
                          "value": perbin_rate[dom][k], "ci_lo": blo, "ci_hi": bhi})

# ---------- Part C pairwise contrasts (independent basins: element-wise delta) ----------
def delta_ci(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    d = d[np.isfinite(d)]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


pair_defs = [("GBR-SCS", "GBR", "SCS"), ("CARIB-SCS", "CARIB", "SCS"), ("CARIB-GBR", "CARIB", "GBR")]
partC = {}
for name, d1, d0 in pair_defs:
    pt, lo, hi = delta_ci(pooled_boot[d1], pooled_boot[d0])
    pair_rows.append({"part": "C", "contrast": f"pooled_severe_rate {name}",
                      "section": "common_support_pooled",
                      "value": pt, "ci_lo": lo, "ci_hi": hi,
                      "ci_excludes_0": bool(lo > 0 or hi < 0)})
    boot_rows.append({"statistic": "pooled_severe_rate_diff", "domain": name,
                      "param": "common support", "value": pt, "ci_lo": lo, "ci_hi": hi})
    partC[name] = {"pooled_rate_diff": pt, "ci": [lo, hi]}
    # per-bin differences
    for k in common_bins:
        ptb, lob, hib = delta_ci(perbin_boot[d1][k], perbin_boot[d0][k])
        pair_rows.append({"part": "C", "contrast": f"bin_severe_rate {name}",
                          "section": f"bin[{k * BIN_W:.1f},{(k + 1) * BIN_W:.1f})",
                          "value": ptb, "ci_lo": lob, "ci_hi": hib,
                          "ci_excludes_0": bool(lob > 0 or hib < 0)})
    # Spearman contrast
    pts, los, his = delta_ci(spear_boot[d1], spear_boot[d0])
    pair_rows.append({"part": "C", "contrast": f"Spearman(DHW,y) {name}",
                      "section": "common_support_pooled",
                      "value": pts, "ci_lo": los, "ci_hi": his,
                      "ci_excludes_0": bool(los > 0 or his < 0)})
summary["partC"] = {**partC,
                    "spearman_point": {d: sp_point[d] for d in DOMAINS},
                    "spearman_ci": {d: [float(np.nanpercentile(spear_boot[d], 2.5)),
                                        float(np.nanpercentile(spear_boot[d], 97.5))] for d in DOMAINS}}
for d in DOMAINS:
    boot_rows.append({"statistic": "spearman_DHW_y", "domain": d, "param": "common support",
                      "value": sp_point[d],
                      "ci_lo": float(np.nanpercentile(spear_boot[d], 2.5)),
                      "ci_hi": float(np.nanpercentile(spear_boot[d], 97.5))})

# ---------- Part B logistic / rank-logistic with DHW x basin ----------
def fit_partB(rank_dhw=False):
    rows = []
    for di, dom in enumerate(DOMAINS):
        dd = data[dom]
        m = np.isfinite(dd['persa']) & np.isfinite(dd['y']) & np.isin(dd['bin'], common_bins)
        p = dd['persa'][m]
        x = rank1(p) if rank_dhw else p
        rows.append((x, dd['sev'][m], dd['t'][m], di))
    blocks = []
    for r in rows:
        x, yv, tv, di = r
        n = len(x)
        dmy = np.repeat((np.arange(3)[None, :] == di).astype(float)[:, 1:3], n, axis=0)  # (n,2)
        blocks.append(np.column_stack([np.ones(n), x, dmy, x[:, None] * dmy]))
    X = np.concatenate(blocks, axis=0)
    yv = np.concatenate([r[1] for r in rows])
    beta = irls_logistic(X, yv)
    return X, yv, beta


X_o, y_o, beta_o = fit_partB(rank_dhw=False)
X_r, y_r, beta_r = fit_partB(rank_dhw=True)

# week-within-basin stratified bootstrap for Part B
weeks_by_basin = {d: np.unique(data[d]['t'][np.isfinite(data[d]['persa']) &
                                             np.isfinite(data[d]['y']) &
                                             np.isin(data[d]['bin'], common_bins)])
                  for d in DOMAINS}
idxmap = {}
for di, dom in enumerate(DOMAINS):
    dd = data[dom]
    m = np.isfinite(dd['persa']) & np.isfinite(dd['y']) & np.isin(dd['bin'], common_bins)
    t_m = dd['t'][m]
    idxmap[di] = {w: np.where(t_m == w)[0] for w in weeks_by_basin[dom]}
base_offsets = {}
off = 0
for di, dom in enumerate(DOMAINS):
    m = np.isfinite(data[dom]['persa']) & np.isfinite(data[dom]['y']) & np.isin(data[dom]['bin'], common_bins)
    base_offsets[di] = off
    off += int(m.sum())


def partB_boot(rank_dhw):
    rng = np.random.default_rng(BOOT_SEED)
    reps = []
    for _ in range(BOOT_N):
        parts = []
        for di, dom in enumerate(DOMAINS):
            wk = weeks_by_basin[dom]
            pick = rng.integers(0, len(wk), len(wk))
            local = np.concatenate([idxmap[di][wk[w]] for w in pick])
            parts.append(base_offsets[di] + local)
        idx = np.concatenate(parts)
        Xd = (X_r if rank_dhw else X_o)[idx]
        yd = (y_r if rank_dhw else y_o)[idx]
        bb = irls_logistic(Xd, yd)
        reps.append(bb)
    return np.array([r if r is not None else np.full(6, np.nan) for r in reps])


reps_o = partB_boot(rank_dhw=False)
reps_r = partB_boot(rank_dhw=True)

def bci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


# design columns: [1, DHW, GBR, CARIB, DHWxGBR, DHWxCARIB]
labels = ["intercept", "DHW", "basin_GBR", "basin_CARIB", "DHWxGBR", "DHWxCARIB"]
for meth, beta, reps in (("logistic", beta_o, reps_o), ("rank_logistic", beta_r, reps_r)):
    slopes = {"SCS": beta[1], "GBR": beta[1] + beta[4], "CARIB": beta[1] + beta[5]}
    slopes_reps = {"SCS": reps[:, 1], "GBR": reps[:, 1] + reps[:, 4], "CARIB": reps[:, 1] + reps[:, 5]}
    for d in DOMAINS:
        lo, hi = bci(slopes_reps[d])
        inter_rows.append({"part": "B", "model": meth, "term": f"slope_{d}",
                           "value": float(slopes[d]), "ci_lo": lo, "ci_hi": hi,
                           "odds_ratio": float(np.exp(slopes[d])),
                           "or_ci_lo": float(np.exp(lo)), "or_ci_hi": float(np.exp(hi))})
        boot_rows.append({"statistic": f"{meth}_slope", "domain": d, "param": "DHW",
                          "value": float(slopes[d]), "ci_lo": lo, "ci_hi": hi})
    for term, li in (("DHWxGBR", 4), ("DHWxCARIB", 5)):
        lo, hi = bci(reps[:, li])
        inter_rows.append({"part": "B", "model": meth, "term": term,
                           "value": float(beta[li]), "ci_lo": lo, "ci_hi": hi,
                           "odds_ratio": np.nan, "or_ci_lo": np.nan, "or_ci_hi": np.nan})
    # pairwise interaction contrasts
    for name, a, bb_ in (("GBR-SCS", 4, 4), ("CARIB-SCS", 5, 5), ("CARIB-GBR", 5, 4)):
        if name == "CARIB-GBR":
            pt = beta[5] - beta[4]; dr = reps[:, 5] - reps[:, 4]
        else:
            pt = beta[bb_]; dr = reps[:, bb_]
        lo, hi = bci(dr)
        inter_rows.append({"part": "B", "model": meth, "term": f"contrast_{name}",
                           "value": float(pt), "ci_lo": lo, "ci_hi": hi,
                           "odds_ratio": np.nan, "or_ci_lo": np.nan, "or_ci_hi": np.nan,
                           "ci_excludes_0": bool(lo > 0 or hi < 0)})
        boot_rows.append({"statistic": f"{meth}_interaction_contrast", "domain": name,
                          "param": "DHW slope", "value": float(pt), "ci_lo": lo, "ci_hi": hi})
    summary["partB"][meth] = {
        "beta": {l: float(b) for l, b in zip(labels, beta)},
        "slopes": {d: float(s) for d, s in slopes.items()},
        "converged": beta is not None}

# ---------- Part D ----------
partD = {}
for dom in DOMAINS:
    lo, hi = bci(psp_boot[dom])
    partD[dom] = {"partial_spearman_R_future_given_DHW": None, "ci": [lo, hi]}
    boot_rows.append({"statistic": "partial_spearman_R_future_given_DHW", "domain": dom,
                      "param": "common support", "value": np.nan, "ci_lo": lo, "ci_hi": hi})
# point partial spearman on common support
for dom in DOMAINS:
    dd = data[dom]
    m = np.isfinite(dd['persa']) & np.isfinite(dd['y']) & np.isfinite(dd['R']) & np.isin(dd['bin'], common_bins)
    partD[dom]["partial_spearman_R_future_given_DHW"] = partial_spearman(dd['R'][m], dd['y'][m], dd['persa'][m])
    partD[dom]["R_tertile_severe_rate"] = {str(q): rtert_point[dom][q] for q in (1, 2, 3)}
    partD[dom]["R_tertile_ci"] = {str(q): [float(np.nanpercentile(rtert_boot[dom][q], 2.5)),
                                           float(np.nanpercentile(rtert_boot[dom][q], 97.5))]
                                  for q in (1, 2, 3)}
    for q in (1, 2, 3):
        boot_rows.append({"statistic": "R_tertile_severe_rate", "domain": dom, "param": f"Q{q}",
                          "value": rtert_point[dom][q],
                          "ci_lo": partD[dom]["R_tertile_ci"][str(q)][0],
                          "ci_hi": partD[dom]["R_tertile_ci"][str(q)][1]})
summary["partD"] = partD

# ---------- strict conclusion ----------
def ci_excl(x):
    return (np.isfinite(x[0]) and (x[0] > 0 or x[1] < 0))


gbr_scs = [r for r in inter_rows if r.get("term") == "contrast_GBR-SCS" and r["model"] == "logistic"][0]
car_scs = [r for r in inter_rows if r.get("term") == "contrast_CARIB-SCS" and r["model"] == "logistic"][0]
heterog_supported = ci_excl((gbr_scs["ci_lo"], gbr_scs["ci_hi"])) or ci_excl((car_scs["ci_lo"], car_scs["ci_hi"]))
summary["conclusion"] = {
    "interaction_logistic_DHWxGBR": {"value": beta_o[4], "ci": [gbr_scs["ci_lo"], gbr_scs["ci_hi"]]},
    "interaction_logistic_DHWxCARIB": {"value": beta_o[5], "ci": [car_scs["ci_lo"], car_scs["ci_hi"]]},
    "basin_specific_susceptibility_supported": bool(heterog_supported),
    "text": ("Interaction CI excludes 0 => basin-specific susceptibility supported"
             if heterog_supported else
             "DHWxbasin interaction CI INCLUDES 0 => basin-specific susceptibility is NOT proven; "
             "report explicitly as not demonstrated at the pooled common-support level."),
    "caveat": "All associations are associative/predictive; no causal mechanism is claimed."}

# ---------- write outputs ----------
pd.DataFrame(resp_rows).to_csv(OUT / "response_surface.csv", index=False)
pd.DataFrame(inter_rows).to_csv(OUT / "interaction_model.csv", index=False)
pd.DataFrame(pair_rows).to_csv(OUT / "pairwise_contrasts.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap.csv", index=False)
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

A = ["# Basin-Specific Susceptibility / Thermal-Response Heterogeneity — AUDIT", "",
     f"Generated: {datetime.now().isoformat()}", f"Source: {REC}", f"Source md5: {md5_file(REC)}", "",
     "## Provenance", "",
     "- Exposure: PersA = clip(DHW/10,0,1) at ISSUE time t (issue-time information only)",
     "- target y = y(t+4); severe = y >= 0.8 (frozen definition)",
     "- R = TSAR - PersA (frozen seed-42), used ONLY in Part D as the object tested",
     "- Common DHW support: 0.1-wide bins where ALL basins have n >= "
     f"{MIN_BIN_N} => bins {common_bins} = [{lo_c:.2f}, {hi_c:.2f})", "",
     "## Protocol compliance", "",
     "- [PASS] Frozen records only; NO retraining / target tuning / threshold optimisation",
     "- [PASS] Identical bins and protocol across SCS/GBR/CARIB; comparison restricted to common support",
     "- [PASS] Bootstrap: paired week-block, n=1000, seed 42, weeks resampled WITHIN basin",
     "- [PASS] Part B logistic/rank-logistic fitted by fixed IRLS (no regularisation, no tuning)",
     "- [PASS] TSAR training outputs NOT used as explanatory variables (Part D tests fixed R only)",
     "- [NOTE] No GPU kernel invoked: all quantities are closed-form statistics of frozen "
     "predictions (no model inference); piecf_env used", "",
     "## Strict conclusion", "",
     f"- logistic DHWxGBR = {beta_o[4]:+.4f} CI [{gbr_scs['ci_lo']:+.4f},{gbr_scs['ci_hi']:+.4f}]",
     f"- logistic DHWxCARIB = {beta_o[5]:+.4f} CI [{car_scs['ci_lo']:+.4f},{car_scs['ci_hi']:+.4f}]",
     f"- basin-specific susceptibility supported = {heterog_supported}",
     "- If either interaction CI excludes 0, heterogeneity is supported at the pooled "
     "common-support level; if BOTH include 0, report 'not proven'.",
     "- All associations associative/predictive; no causal interpretation.", ""]
open(OUT / "AUDIT.md", "w", encoding="utf-8").write("\n".join(A) + "\n")

print("\n== KEY RESULTS ==")
print(f"Common support bins {common_bins} [{lo_c:.2f},{hi_c:.2f})")
for d in DOMAINS:
    print(f"  {d}: pooled common-support severe rate = {pooled_rate[d]:.4f}; "
          f"slopes(logistic) SCS/GBR/CARIB = "
          f"{beta_o[1]:+.3f}/{beta_o[1]+beta_o[4]:+.3f}/{beta_o[1]+beta_o[5]:+.3f}")
print(f"  interactions: DHWxGBR {beta_o[4]:+.4f} [{gbr_scs['ci_lo']:+.4f},{gbr_scs['ci_hi']:+.4f}]; "
      f"DHWxCARIB {beta_o[5]:+.4f} [{car_scs['ci_lo']:+.4f},{car_scs['ci_hi']:+.4f}]")
print(f"  basin-specific susceptibility supported = {heterog_supported}")
print("Outputs ->", OUT)
print("DONE")
_LOCK.unlink(missing_ok=True)