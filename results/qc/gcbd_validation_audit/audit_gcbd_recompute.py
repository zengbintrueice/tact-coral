# -*- coding: utf-8 -*-
"""Independent recomputation of the GCBD validation statistics.

Source of truth: results/step9_gcbd_validation/step9_gcbd_results.csv
(one row per matched GCBD observation; TSCF score only).

Cross-checks the published values in:
  results/step9_gcbd_validation/step9_bootstrap.csv
  results/review2_revision/gcbd_model_wise_enrichment.json
  results/review2_revision/gcbd_common_cohort.json
  results/step10_gcbd_matching_audit/step10_matching_results.csv

Writes only into results/qc/gcbd_validation_audit/.
"""
import io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CSV = ROOT / "results/step9_gcbd_validation/step9_gcbd_results.csv"

df = pd.read_csv(CSV)
p = df.pred.values.astype(float)
sev = (df.pb.values >= 20).astype(int)
anyb = (df.pb.values > 0).astype(int)
bg = (df.pb.values == 0).astype(int)
site = df.Site_ID.astype(str).values
print(f"n={len(df)}  any={anyb.sum()}  severe={sev.sum()}  background={bg.sum()}  "
      f"sites={len(np.unique(site))}  max_per_site={int(pd.Series(site).value_counts().max())}")

def cliffs(x, y):
    n1, n2 = len(x), len(y)
    r = rankdata(np.concatenate([x, y]))
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return 2.0 * u / (n1 * n2) - 1.0

mask_sb = (sev == 1) | (bg == 1)
au_sb = roc_auc_score(sev[mask_sb], p[mask_sb])
au_sr = roc_auc_score(sev, p)
au_any = roc_auc_score(anyb, p)
mw_p = mannwhitneyu(p[sev == 1], p[bg == 1], alternative="greater").pvalue
cd = cliffs(p[sev == 1], p[bg == 1])

q = df.groupby("quintile").severe.mean()
q1, q5 = float(q.loc[1]), float(q.loc[5])
rr = q5 / q1
print(f"AUROC sev-vs-bg={au_sb:.6f}  sev-vs-rest={au_sr:.6f}  any={au_any:.6f}")
print(f"MW p={mw_p:.3e}  Cliff d={cd:.4f}")
print("quintile severe rates:", [round(float(x), 4) for x in q.values],
      " RR=", round(rr, 4))

# ---------- site-cluster bootstrap (review2 convention), seed 42 ----------
def rr_of(ps, s, k=5):
    labels = np.asarray(pd.qcut(ps, k, labels=False, duplicates="drop"))
    top = labels == np.nanmax(labels); bot = labels == np.nanmin(labels)
    pt = s[top].mean() if top.sum() else np.nan
    pb_ = s[bot].mean() if bot.sum() else np.nan
    return (pt / pb_) if (pb_ and pb_ > 0) else np.nan

rng = np.random.default_rng(42)
site_names = np.unique(site)
site_idx = {s: np.where(site == s)[0] for s in site_names}
boot_au, boot_rr = [], []
for _ in range(1000):
    pick = rng.integers(0, len(site_names), len(site_names))
    idx = np.concatenate([site_idx[site_names[k]] for k in pick])
    pp, ss, bb = p[idx], sev[idx], bg[idx]
    if len(np.unique(pp)) < 2 or ss.sum() == 0 or bb.sum() == 0:
        continue
    m = (ss == 1) | (bb == 1)
    boot_au.append(roc_auc_score(ss[m], pp[m]))
    v = rr_of(pp, ss)
    boot_rr.append(v)
boot_au = np.array(boot_au); boot_rr = np.array(boot_rr)
rrv = boot_rr[np.isfinite(boot_rr)]
ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
print(f"site-cluster AUROC CI {ci(boot_au)}  (n_valid={len(boot_au)})")
print(f"site-cluster RR    CI {ci(rrv)}  (n_valid={len(rrv)})  mean={rrv.mean():.3f}")

# ---------- naive observation bootstrap ----------
rng2 = np.random.default_rng(42)
nau, nrr = [], []
for _ in range(1000):
    idx = rng2.integers(0, len(p), len(p))
    pp, ss, bb = p[idx], sev[idx], bg[idx]
    if len(np.unique(pp)) < 2 or ss.sum() == 0 or bb.sum() == 0:
        continue
    m = (ss == 1) | (bb == 1)
    nau.append(roc_auc_score(ss[m], pp[m]))
    v = rr_of(pp, ss)
    nrr.append(v)
nrr = np.array(nrr); nrr = nrr[np.isfinite(nrr)]
print(f"naive AUROC CI {ci(np.array(nau))}  RR mean={nrr.mean():.3f} CI {ci(nrr)}")

# ---------- year composition ----------
# issue_t is a week index; map to year via the SCS causal calendar if available
print("\nsevere by issue_t range:", int(df[sev == 1].issue_t.min()), int(df[sev == 1].issue_t.max()))

summary = {
    "n": int(len(df)), "any": int(anyb.sum()), "severe": int(sev.sum()),
    "background": int(bg.sum()), "sites": int(len(np.unique(site))),
    "max_per_site": int(pd.Series(site).value_counts().max()),
    "auroc_sev_vs_bg": float(au_sb), "auroc_sev_vs_rest": float(au_sr),
    "auroc_any_vs_rest": float(au_any), "mw_p": float(mw_p), "cliffs_delta": float(cd),
    "quintile_rates": [float(x) for x in q.values], "rr_q5_q1": float(rr),
    "site_cluster_auroc_ci": ci(boot_au), "site_cluster_rr_ci": ci(rrv),
    "site_cluster_rr_mean": float(rrv.mean()), "site_cluster_n_valid": int(len(boot_au)),
    "naive_auroc_ci": ci(np.array(nau)), "naive_rr_mean": float(nrr.mean()),
    "naive_rr_ci": ci(nrr),
}
json.dump(summary, open(OUT / "audit_gcbd_recompute.json", "w", encoding="utf-8"), indent=1)
print("\nwrote", OUT / "audit_gcbd_recompute.json")
