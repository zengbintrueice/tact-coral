# finalize_transfer_coordinate.py — merge forward + reverse into the full
# 6-direction matrix and write summary.md answering the three questions.
import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / "results" / "transfer_coordinate"

fwd = pd.read_csv(OUT / "transfer_matrix.csv")        # PersA(all) + SCS->GBR/CARIB TSCF/TSAR
rev = pd.read_csv(OUT / "reverse_matrix.csv")          # GBR/CARIB source TSCF/TSAR (incl self rows)
fwd_ci = pd.read_csv(OUT / "transfer_matrix_ci.csv")   # SCS->* week-block
rev_ci = pd.read_csv(OUT / "reverse_matrix_ci.csv")    # reverse diff bootstrap
coord = pd.read_csv(OUT / "coordinate_metrics.csv")
year = pd.read_csv(OUT / "yearly_coordinate_metrics.csv")

MODELS = ["TSCF", "TSAR", "PersA"]
BASINS = ["SCS", "GBR", "CARIB"]
def directed(js, target):
    return {m: float(js[(js.source==s)&(js.target==target)&(js.model==m)].roc_auc.iloc[0])
            for s in BASINS if s!=target for m in MODELS if m in js.model.values}

# ---- PersA is source-independent: inject PersA rows for GBR/CARIB sources
persa_rows = fwd[fwd.model=="PersA"].copy()
# ensure every (source,target) directed pair covered for TSCF/TSAR/PersA
rows = []
sources_all = {"SCS","GBR","CARIB"}
covers = {}
for _,r in fwd.iterrows(): covers[(r.source,r.target,r.model)] = r
for _,r in rev.iterrows():
    if r.source!=r.target and r.model in ("TSCF","TSAR"):
        covers[(r.source,r.target,r.model)] = r
for s in BASINS:
    for t in BASINS:
        if s==t: continue
        for m in MODELS:
            key=(s,t,m)
            if key not in covers and m=="PersA":
                # derive from any PersA row on target t (source-independent)
                pr = persa_rows[persa_rows.target==t].iloc[0]
                covers[key]=pr.assign(source=s)
cols = ["source","target","model","n","n_severe","severe_prevalence","roc_auc","pr_auc",
        "csi_tau09","pod_tau09","far_tau09","f1_tau09","brier","ece"] if "ece" in fwd.columns else \
        ["source","target","model","n","n_severe","severe_prevalence","roc_auc","pr_auc",
         "csi_tau09","pod_tau09","far_tau09","f1_tau09","brier"]
mat = pd.DataFrame([covers[k] for k in sorted(covers, key=lambda x:(x[1],x[0],x[2]))])
mat = mat[cols].sort_values(["target","source","model"]).reset_index(drop=True)
mat.to_csv(OUT / "transfer_matrix.csv", index=False)

# ---- merged degradation CI ----
ci_rows = []
for _,r in fwd_ci.iterrows(): ci_rows.append(r)
for _,r in rev_ci.iterrows():
    if r.source!=r.target: ci_rows.append(r)
ci = pd.DataFrame(ci_rows).sort_values(["target","source","model"]).reset_index(drop=True)
ci.to_csv(OUT / "transfer_matrix_ci.csv", index=False)

# ================= summary =================
def roc(s,t,m):
    r=mat[(mat.source==s)&(mat.target==t)&(mat.model==m)]
    return float(r.roc_auc.iloc[0]) if len(r) else float("nan")
def dci(s,t,m):
    r=ci[(ci.source==s)&(ci.target==t)&(ci.model==m)]
    return (float(r.d_roc.iloc[0]),float(r.d_roc_lo.iloc[0]),float(r.d_roc_hi.iloc[0])) if len(r) else (np.nan,np.nan,np.nan)

L=[]
L.append("# Transfer Matrix + Transferable Thermal Coordinate Stability — summary")
L.append("**Experiment:** all-to-all zero-shot transfer, SCS / GBR / CARIB. Models: TSCF (contextual), "
         "TSAR (anchored gated residual), PersA_DHW = clip(DHW/10,0,1). lead=4 wk, history=24 wk, "
         "τ=0.90 pinned, severe label y(t+4) ≥ 0.8. Source-only training; target never trained/tuned/thresholded; "
         "normalization = the fixed SCS-style channel normalizer baked into every model (`piecf_model.py:193`), "
         "applied identically to all input basins (no target scaler fitting).")
L.append("")
L.append("## ROC-AUC transfer matrix (rows=source, columns=target)")
hdr = "| source\\target | " + " | ".join(BASINS) + " |"
L.append(hdr); L.append("|"+"---|"*(len(BASINS)+1))
for s in BASINS:
    cells=[]
    for t in BASINS:
        if s==t: cells.append("**self**")
        else:
            cells.append(f"TSCF {roc(s,t,'TSCF'):.3f} / TSAR {roc(s,t,'TSAR'):.3f} / PersA {roc(s,t,'PersA'):.3f}")
    L.append(f"| {s} | " + " | ".join(cells) + " |")
L.append("")
L.append("PersA is source-independent, so its value depends only on the target basin "
         "(SCS 0.982, GBR 0.959, CARIB 0.979).")
L.append("")

L.append("## Coordinate stability: PersA_DHW vs future thermal risk y(t+4)")
L.append("| basin | n | n_severe | Spearman ρ | ROC-AUC | PR-AUC |")
L.append("|---|---|---|---|---|---|")
for _,r in coord.iterrows():
    L.append(f"| {r.basin} | {int(r.n)} | {int(r.n_severe)} | {r.spearman:.3f} | {r.roc_auc:.3f} | {r.pr_auc:.3f} |")
L.append("")
L.append("Per-year Spearman ρ: SCS 0.73–0.82, GBR 0.69–0.83, CARIB 0.70–0.83 "
         "(GBR 2019 has zero severe events → ROC/PR undefined). The anchor's rank association "
         "with future thermal risk is stable both across basins and across years within the study period.")
L.append("")

L.append("## Transfer degradation Δ(target − source), ΔROC (95% CI)")
L.append("| source→target | TSCF | TSAR | PersA |")
L.append("|---|---|---|---|")
for s in BASINS:
    for t in BASINS:
        if s==t or not (t in BASINS): continue
        cells=[]
        for m in ["TSCF","TSAR","PersA"]:
            d,lo,hi=dci(s,t,m)
            cells.append(f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]")
        L.append(f"| {s}→{t} | " + " | ".join(cells) + " |")
L.append("")

L.append("## Answer to the three questions")
L.append("### (1) Which transfer directions succeed / fail")
L.append("- **TSAR (anchored residual) and PersA: succeed in every direction** (ROC ≥ 0.96 and ≥ 0.96, "
         "respectively); the anchor-level transfer never collapses.")
L.append("- **TSCF (full contextual representation): succeeds on SCS→GBR (0.962), CARIB→SCS (0.981), "
         "CARIB→GBR (0.906), but fails on SCS→CARIB (0.610), GBR→CARIB (0.635) and GBR→SCS (0.668).** "
         "The contextual model collapses specifically when the target is CARIB (from any source) and for "
         "GBR→SCS; it does not collapse when CARIB is the source.")
L.append("")
L.append("### (2) Is TSCF/TSAR cross-basin degradation greater than PersA?")
L.append("- **TSCF: yes, and by a large margin on the failing directions.** E.g. SCS→CARIB ΔROC −0.376 "
         "[−0.40, −0.35]; GBR→CARIB −0.263 [−0.28, −0.25]; GBR→SCS −0.230 [−0.26, −0.20]. PersA's "
         "corresponding ΔROC is ≈ 0 (source-independent; basin-to-basin ≈ −0.01 to +0.02).")
L.append("- **TSAR: no** — its degradation is ≈ 0 on every direction (e.g. GBR→CARIB +0.019 "
         "[+0.015, +0.023]; SCS→CARIB +0.001 [−0.006, +0.009]), i.e. as stable as PersA. So the "
         "instability is specific to the unanchored contextual TSCF, not to the neural model per se.")
L.append("")
L.append("### (3) Is PersA_DHW's future-risk association more stable across basins?")
L.append("- Yes, within the three basins studied and the study period: Spearman ρ is 0.775 / 0.783 / 0.802 "
         "for SCS / GBR / CARIB (Δ ≈ 0.03), ROC-AUC 0.982 / 0.959 / 0.979, and per-year ρ stays 0.70–0.83 "
         "in every basin. The DHW anchor is a consistently stable rank coordinate across the six transfer "
         "directions, whereas the full contextual representation is not.")
L.append("")
L.append("## Conclusion (scoped)")
L.append("Within the three basins (SCS, GBR, CARIB) and the study period, PersA_DHW — and the gated, bounded "
         "TSAR built around it — exhibit a **more stable transferable thermal-risk coordinate and rank "
         "relationship than the full contextual TSCF representation**. We do not claim the coordinate is "
         "basin-invariant, universal, or globally transferable; the observed stability is demonstrated for "
         "these three basins and this study period only. Note: reverse-source (GBR/CARIB) degradation CIs use "
         "a record-level difference bootstrap; SCS-source CIs re-use the frozen week-block record set. "
         "Source-metric windows are SCS 2016–2019 and GBR/CARIB 2017–2020.")
L.append("")
L.append("## Provenance")
L.append("- Data (md5): see `provenance_tier1.json`; reverse checkpoints: `results/transfer_coordinate/weights/{GBR,CARIB}/{tscf,tsar}_model.pt` "
         "(md5 in each summary.json).")
L.append("- Record sets: frozen SCS→GBR/CARIB (`rec_structural_ablation.npz`); reverse rows generated from the "
         "shared test populations (n: SCS 49,322 / GBR 19,065 / CARIB 83,025).")
OUT.joinpath("summary.md").write_text("\n".join(L), encoding="utf-8")
print("wrote transfer_matrix.csv rows:", len(mat), " ci rows:", len(ci))
print("wrote summary.md")
print(mat[["source","target","model","roc_auc"]].to_string(index=False))