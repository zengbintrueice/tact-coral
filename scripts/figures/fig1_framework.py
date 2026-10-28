# -*- coding: utf-8 -*-
"""Figure 1 — Cross-basin thermal-risk transferability framework.

Panels
  a  study-domain map (domain boxes over a minimal light-grey coastline)
  b  temporal train/val/test protocol (explicit years; from step4 protocol JSON)
  c  frozen TSCF architecture
  d  conceptual decomposition: standardized thermal-stress anchor
     + gated contextual residual (bounded two-sided correction)

Only protocol constants are read from results/step4_causal_zero_shot/
causal_zero_shot_protocol.json. Coastlines are geographic reference from the
project's ETOPO1 bathymetry (D:\\piecf_dns); if that source is unavailable the
map falls back to domain boxes only (with a printed warning). No experimental
result is hard-coded.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import xarray as xr

import nc_style as S

PROT = S.require(S.RESULTS / "step4_causal_zero_shot" / "causal_zero_shot_protocol.json",
                 fig="fig1")
SEV_THR = PROT.get("severe_threshold", "y >= 0.8")


def draw_coastline(ax):
    """Minimal light-grey coastline from the downloaded ETOPO1 global grid.
    Pure geographic reference (land = z>0), down-sampled ~0.3°; never an
    experimental result. Falls back silently if the source is absent."""
    src = os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"\bathymetry\ETOPO1_Ice_g_gmt4.grd"
    if not os.path.isfile(src):
        print("[warn] fig1a: ETOPO1 coastline source missing — drawing domain "
              "boxes only")
        return
    try:
        ds = xr.open_dataset(src)
        z = ds["z"].values
        y = np.asarray(ds["y"].values)   # lat ascending, -90..90
        x = np.asarray(ds["x"].values)   # lon ascending, -180..180
        ds.close()
    except Exception as e:
        print(f"[warn] fig1a: ETOPO1 read failed ({type(e).__name__}); "
              "drawing domain boxes only")
        return
    step = max(1, len(y) // 540)          # ~0.3° resolution
    land = (z[::step, ::step] > 0).astype(float)
    land[land == 0] = np.nan              # transparent ocean
    ax.imshow(land, origin="lower", cmap="Greys", vmin=0, vmax=1,
              extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
              aspect="auto", alpha=0.30, zorder=0,
              interpolation="nearest")
    ax.set_facecolor("white")


fig = plt.figure(figsize=(S.WIDTH_2COL, 5.4))  # 180 mm double column
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                      left=0.06, right=0.97, top=0.94, bottom=0.06)

# ---------------------------------------------------------------- (a) map ---
ax = fig.add_subplot(gs[0, 0])
ax.set_title("Study domains", loc="left", fontsize=8)
ax.set_xlim(-120, 175)          # wide enough to show SCS 100.5–129.5°E fully
ax.set_ylim(-40, 40)
ax.set_xticks([-90, 0, 90, 150])
ax.set_yticks([-30, 0, 30])
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
for lat in (23.4, -23.4):
    ax.axhline(lat, color="#E3E3E3", lw=0.5, ls="--", zorder=0)
draw_coastline(ax)
# domain boxes (region bounds per the study-domain framing)
boxes = [
    ("SCS", 100.5, 129.5, 3, 24, "source domain\ntrain 2003–2010"),
    ("GBR", 142, 154, -24, -10, "zero-shot\ntest 2017–2020"),
    ("CARIB", -91, -59, 4, 26, "zero-shot boundary\ntest 2017–2020"),
]
for name, lo_x, hi_x, lo_y, hi_y, lab in boxes:
    ax.add_patch(mpatches.Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                                    facecolor=S.DOMAIN_COLOR[name], alpha=0.28,
                                    edgecolor=S.DOMAIN_COLOR[name], lw=1.2,
                                    zorder=2))
    ax.text((lo_x + hi_x) / 2, hi_y + 2.0, name, ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=S.DOMAIN_COLOR[name])
    ax.text((lo_x + hi_x) / 2, (lo_y + hi_y) / 2, lab, ha="center", va="center",
            fontsize=5.8, color="#222222")
S.panel_label(ax, "a")

# ----------------------------------------------------------- (b) timeline ---
ax = fig.add_subplot(gs[0, 1])
ax.set_title("Temporal protocol (year-based, no overlap)", loc="left", fontsize=8)
# explicit labels per requirement: SCS train/validation/test; GBR & CARIB
# zero-shot test 2017–2020
rows = [
    ("SCS", [(2003, 2010, "train\n2003–2010"),
             (2011, 2015, "validation\n2011–2015"),
             (2016, 2019, "test\n2016–2019")], S.COLORS["SCS"]),
    ("GBR", [(2003, 2013, "unused"), (2014, 2016, "val (diag.)"),
             (2017, 2020, "zero-shot test\n2017–2020")], S.COLORS["GBR"]),
    ("CARIB", [(2003, 2013, "unused"), (2014, 2016, "val (diag.)"),
               (2017, 2020, "zero-shot test\n2017–2020")], S.COLORS["CARIB"]),
]
for yi, (name, spans, col) in enumerate(rows):
    for k, (y0, y1, lab) in enumerate(spans):
        ax.barh(yi, y1 - y0 + 1, left=y0, height=0.52,
                color=col, alpha=0.25 + 0.25 * k, edgecolor=col, lw=0.7)
        signed = max(1, y1 - y0)
        if signed > 3:
            ax.text((y0 + y1 + 1) / 2, yi, lab, ha="center", va="center",
                    fontsize=5.8)
        else:
            ax.text(y1 + 0.3, yi, f"val (diag.)\n{y0}–{y1}", ha="left",
                    va="center", fontsize=5.4, color="#444444")
ax.set_yticks(range(3))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(2002, 2021.6)
ax.set_xticks([2003, 2008, 2013, 2018])
ax.set_xlabel("Year")
ax.invert_yaxis()
ax.annotate("", xy=(2019.9, 2.78), xytext=(2019.9 - 4 / 52 * 4, 2.78),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
ax.text(2015.8, 2.60, "prediction lead = 4 weeks", fontsize=5.8, va="top")
ax.set_ylim(2.95, -0.7)
S.panel_label(ax, "b")

# ------------------------------------------------------- (c) architecture ---
ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.set_title("Frozen TSCF (source-trained)", loc="left", fontsize=8)
blocks = [
    (0.02, 0.20, "Causal inputs\n$\\leq t$\n(13 weekly\nchannels,\n24-wk\nwindow)"),
    (0.245, 0.17, "Channel\nnormalizer\n(SCS-train\nstatistics, frozen)"),
    (0.44, 0.17, "ConvLSTM\nencoder"),
    (0.635, 0.17, "Temporal\nattention"),
    (0.83, 0.17, "Regression\nhead\n$\\hat{y}_{main}(t{+}4)$"),
]
for x0, w, txt in blocks:
    ax.add_patch(FancyBboxPatch((x0, 0.42), w, 0.34,
                                boxstyle="round,pad=0.012", fc="#EDF2F8",
                                ec="#5B7DA8", lw=0.8,
                                transform=ax.transAxes))
    ax.text(x0 + w / 2, 0.59, txt, ha="center", va="center",
            fontsize=5.6, transform=ax.transAxes)
for x0, w, _ in blocks[:-1]:
    ax.add_patch(FancyArrowPatch((x0 + w + 0.002, 0.59), (x0 + w + 0.022, 0.59),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color="#5B7DA8", transform=ax.transAxes))
ax.text(0.5, 0.16, "frozen during all zero-shot evaluation  (no target-domain "
        "fitting, no normalization updates)",
        ha="center", fontsize=6.0, style="italic", transform=ax.transAxes)
ax.text(0.5, 0.86, f"severe thermal-risk threshold {SEV_THR}  |  lead 4 wk  |  "
        "history 24 wk", ha="center", fontsize=5.8, transform=ax.transAxes)
S.panel_label(ax, "c")

# ------------------------------------------------- (d) decomposition --------
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
ax.set_title("Thermal-stress anchoring + gated residual", loc="left", fontsize=8)
# unified symbols: PersA, ŷ_main, R(h_t), g(t), s
ax.text(0.5, 0.93,
        r"$\hat{y}_{final}(t{+}4)=\mathrm{PersA}(t)+g(t)\cdot\tanh(R(h_t))\cdot s$",
        ha="center", va="center", fontsize=8.4, transform=ax.transAxes)
ax.text(0.5, 0.76,
        r"$\mathrm{PersA}(t)=\mathrm{clip}(\mathrm{DHW}(t)/10,\,0,\,1)$"
        r"$\quad\quad g(t)=\sigma((\hat{y}_{main}-\tau)/T)$",
        ha="center", va="center", fontsize=7.0, transform=ax.transAxes)
# anchor box (widened & shortened on the left so both boxes fit with margins)
ax.add_patch(FancyBboxPatch((0.02, 0.26), 0.44, 0.36, boxstyle="round,pad=0.015",
                            fc="#E7EBEE", ec=S.COLORS["anchor"], lw=1.0,
                            transform=ax.transAxes))
ax.text(0.24, 0.44,
        "standardized thermal-stress anchor\n$\\mathrm{PersA}(t)=\\mathrm{clip}(\\mathrm{DHW}(t)/10,0,1)$\n"
        "(fixed DHW reference)",
        ha="center", va="center", fontsize=5.6, transform=ax.transAxes)
# residual/gate box (wider and taller so the full text fits inside the box)
ax.add_patch(FancyBboxPatch((0.50, 0.26), 0.48, 0.36, boxstyle="round,pad=0.015",
                            fc="#F3EAF7", ec=S.COLORS["gated"], lw=1.0,
                            transform=ax.transAxes))
ax.text(0.74, 0.44,
        "gated contextual correction\nresidual head: $R(h_t)$ from\n"
        "attention features\n"
        "bounded as $\\tanh(R(h_t))\\cdot s$\n"
        "with $g(t)\\in[0,1]$ limiting\n"
        "the correction magnitude",
        ha="center", va="center", fontsize=5.6, transform=ax.transAxes)
ax.add_patch(FancyArrowPatch((0.465, 0.44), (0.495, 0.44), arrowstyle="-|>",
                             mutation_scale=8, lw=1.0, color="k",
                             transform=ax.transAxes))
ax.text(0.5, 0.045, "contextual residual = bounded two-sided correction to the\n"
        "anchor; question: does the anchor give a more domain-stable\n"
        "coordinate for cross-basin generalization?",
        ha="center", va="bottom", fontsize=6.0, style="italic", transform=ax.transAxes)
S.panel_label(ax, "d")

out = S.save_fig(fig, "fig1_framework")

# ---------------------------------------------------------- source data -----
import pandas as pd
sd_rows = [
    dict(panel="a", domain="SCS", lon_lo=100.5, lon_hi=129.5, lat_lo=3, lat_hi=24,
         role="source domain train 2003-2010"),
    dict(panel="a", domain="GBR", lon_lo=142, lon_hi=154, lat_lo=-24, lat_hi=-10,
         role="zero-shot test 2017-2020"),
    dict(panel="a", domain="CARIB", lon_lo=-91, lon_hi=-59, lat_lo=4, lat_hi=26,
         role="zero-shot boundary test 2017-2020"),
]
for dom, spans, _ in rows:
    for (y0, y1, lab) in spans:
        sd_rows.append(dict(panel="b", domain=dom, span=lab, y0=y0, y1=y1,
                            lon_lo=np.nan, lon_hi=np.nan, lat_lo=np.nan,
                            lat_hi=np.nan, role=""))
pd.DataFrame(sd_rows).to_csv(S.SRC_DIR / "fig1_source_data.csv", index=False)
print("[source] fig1_source_data.csv")

# manifest rows
man = S.OUT_DIR / "figure_manifest.csv"
new = pd.DataFrame([
    dict(figure="fig1", panel="a", source_file="domain boxes (study framing) + ETOPO1 coastline (PIECF_DNS_DIR/bathymetry/ETOPO1 grid)",
         source_columns="lon/lat bounds; z>0 land", statistical_method="schematic",
         n=3, unit="degrees", output_file="fig1_framework.pdf"),
    dict(figure="fig1", panel="b", source_file="results/step4_causal_zero_shot/causal_zero_shot_protocol.json",
         source_columns="splits", statistical_method="protocol constant", n=3, unit="years",
         output_file="fig1_framework.pdf"),
    dict(figure="fig1", panel="c", source_file="results/step4_causal_zero_shot/causal_zero_shot_protocol.json",
         source_columns="checkpoint.md5, normalization", statistical_method="schematic", n="n/a", unit="n/a",
         output_file="fig1_framework.pdf"),
    dict(figure="fig1", panel="d", source_file="conceptual (equations per TSAR definition)",
         source_columns="n/a", statistical_method="schematic", n="n/a", unit="n/a",
         output_file="fig1_framework.pdf"),
])
if man.exists():
    old = pd.read_csv(man)
    new = pd.concat([old[old.figure != "fig1"], new], ignore_index=True)
new.to_csv(man, index=False)
print("[manifest] fig1 rows written")