# plot_counterfactual_response_surface.py — main-text figure (3x2).
# ALL data read from results/counterfactual_response_surface/surface.csv.
#
#   Top row:    TSAR counterfactual response surface (x = current DHW,
#               y = ΔDHW_4w) — sequential colormap, common scale
#   Bottom row: residual surface R = TSAR − PersA — diverging colormap,
#               common symmetric scale
#   Both: contour lines, grey mask outside real-data support, support-density
#   contours, shared colorbars, concise titles.
#
# Run (piecf_env):
#   & 'D:\...\piecf_env\Scripts\python.exe' plot_counterfactual_response_surface.py

import io, sys, json, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "figures"))
import nc_style as S  # unified NC typography (rcParams) / palette

RES = HERE.parents[1] / "results" / "counterfactual_response_surface"

BASINS = ["SCS", "GBR", "CARIB"]
cells = pd.read_csv(RES / "surface.csv")
SJSON = json.load(open(RES / "summary.json", encoding="utf-8"))  # noqa: keep nc_style module unshadowed

# pivot to 2D grids (d × delta); NaN where unsupported
def grid(basin, col):
    sub = cells[cells["basin"] == basin]
    return sub.pivot_table(index="delta", columns="d", values=col)

grids_tsar = {b: grid(b, "tsar_mean") for b in BASINS}
grids_r = {b: grid(b, "r_mean") for b in BASINS}
grids_sup = {b: grid(b, "support") for b in BASINS}

tsar_lo = np.nanpercentile(cells["tsar_mean"], 2)
tsar_hi = np.nanpercentile(cells["tsar_mean"], 98)
r_abs = np.nanmax(np.abs(cells["r_mean"]))

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), sharex=True, sharey=True)
fig.patch.set_facecolor("white")

def draw_panel(ax, G, Sup, cmap, vmin, vmax, norm=None, contour_color="k"):
    x = G.columns.values
    y = G.index.values
    Z = G.values
    if norm is None:
        norm = plt.Normalize(vmin, vmax)
    masked = np.ma.masked_invalid(Z)
    pc = ax.pcolormesh(x, y, masked, cmap=cmap, norm=norm, shading="auto",
                       rasterized=True)
    # grey mask outside support
    ax.pcolor(x, y, np.where(np.isnan(Z), 1.0, np.nan), cmap=matplotlib.colors
              .ListedColormap(["#d9d9d9"]), shading="auto")
    # support-density contours (0 where unsupported)
    SupF = np.where(np.isnan(Sup.values), 0.0, Sup.values)
    if SupF.max() > 0:
        cs = ax.contour(x, y, SupF, levels=3, colors=contour_color,
                        linewidths=0.6, alpha=0.5)
    # surface contours
    if np.isfinite(Z).sum() > 3:
        ax.contour(x, y, Z, levels=5, colors="white", linewidths=0.7, alpha=0.8)
    return pc

for k, b in enumerate(BASINS):
    ax = axes[0, k]
    pc = draw_panel(ax, grids_tsar[b], grids_sup[b], "viridis", tsar_lo, tsar_hi)
    if k == 0:
        ax.set_ylabel("ΔDHW over last 4 weeks (°C-weeks)\n← falling   |   rising →",
                      fontsize=9.5)
    ax.set_title(f"{b}", fontsize=11, fontweight="bold")
    S.panel_label(ax, "abc"[k])
    if k == 2:
        cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label("TSAR (counterfactual)", fontsize=8.5)

for k, b in enumerate(BASINS):
    ax = axes[1, k]
    pc = draw_panel(ax, grids_r[b], grids_sup[b], "RdBu_r", -r_abs, r_abs,
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-r_abs, vmax=r_abs))
    ax.set_xlabel("Current DHW (°C-weeks)", fontsize=9.5)
    if k == 0:
        ax.set_ylabel("ΔDHW over last 4 weeks (°C-weeks)\n← falling   |   rising →",
                      fontsize=9.5)
    S.panel_label(ax, "def"[k])
    if k == 2:
        cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label("R = TSAR − PersA", fontsize=8.5)

axes[0, 0].text(0.01, 0.99, "top: TSAR model response", transform=axes[0, 0].transAxes,
                fontsize=8.5, va="top", style="italic", alpha=0.7)
axes[1, 0].text(0.01, 0.99, "bottom: residual R = TSAR − PersA",
                transform=axes[1, 0].transAxes, fontsize=8.5, va="top",
                style="italic", alpha=0.7)
axes[1, 2].text(0.99, -0.32,
                "Counterfactual model response of the frozen TSAR, not a causal "
                "effect.\nGrey = outside real-data support; thin contours = "
                "support density.",
                transform=axes[1, 2].transAxes, fontsize=8, ha="right", va="top",
                alpha=0.8)

plt.tight_layout()
out = RES / "figure_counterfactual_response_surface.png"
plt.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
plt.savefig(RES / "figure_counterfactual_response_surface.pdf",
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out}")
