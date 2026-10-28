# -*- coding: utf-8 -*-
"""Shared Nature-Communications style for the cross-basin transferability figures.

Domain colors are used CONSISTENTLY across every figure:
  SCS (source domain)        -> blue
  GBR (successful transfer)  -> green
  CARIB (transfer boundary)  -> red
  PersA / thermal-stress anchor -> neutral slate
  gated residual (TSAR)      -> purple accent

Every figure script reads numerical values from result files under
LTTS_CRW/results/ — never hard-coded results. If a file/column is missing the
script raises with an explicit message (see require()).
"""
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# UTF-8 console output (Windows GBK console cannot print some symbols)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LTTS_ROOT = Path(__file__).resolve().parents[2]
RESULTS = LTTS_ROOT / "results"
OUT_DIR = RESULTS / "figures"
SRC_DIR = OUT_DIR / "source_data"
for _d in (OUT_DIR, SRC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette ---
# Domain colors follow the Okabe-Ito colour-blind-safe set: the GBR/CARIB
# contrast is bluish-green vs vermillion (distinguishable under all common
# CVD types), never pure green vs pure red.
COLORS = {
    "SCS": "#0072B2",      # blue            — source domain (Okabe-Ito blue)
    "GBR": "#009E73",      # bluish green    — successful transfer
    "CARIB": "#D55E00",    # vermillion      — transfer boundary
    "anchor": "#6E7B8B",   # slate           — PersA / thermal-stress anchor
    "gated": "#7D3C98",    # purple          — gated residual (TSAR)
    "ungated": "#C3A6D0",  # light purple    — un-gated residual
    "tscf": "#2C3E50",     # dark slate      — frozen TSCF
    "neutral": "#9A9A9A",
}
DOMAIN_COLOR = {"SCS": COLORS["SCS"], "GBR": COLORS["GBR"], "CARIB": COLORS["CARIB"]}
DOMAIN_LABEL = {
    "SCS": "SCS (source)",
    "GBR": "GBR (zero-shot)",
    "CARIB": "CARIB (zero-shot)",
}

# -------------------------------------------------------------- typography --
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

PANEL_KW = dict(fontsize=9, fontweight="bold", ha="left", va="bottom")

# ---------------------------------------------------- journal page widths ---
# Nature-family final print widths (inches); figures must be authored AT the
# target width so that 5-7 pt text prints true-size (no downscaling).
WIDTH_1COL = 89 / 25.4     # single column  (89 mm  = 3.50 in)
WIDTH_15COL = 140 / 25.4   # 1.5 column     (140 mm = 5.51 in)
WIDTH_2COL = 183 / 25.4    # double column  (183 mm = 7.20 in)


def panel_label(ax, letter, dx=-0.12, dy=1.04):
    """Small bold panel label (a, b, c, ...) at the top-left of an axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, **PANEL_KW)


def require(path, cols=None, fig=""):
    """Load a CSV/JSON result file; raise an explicit error if missing."""
    import json
    import pandas as pd
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"[{fig}] MISSING source file: {p}\n"
            f"  -> figure cannot be generated without it; do NOT fabricate values.")
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        if cols:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"[{fig}] MISSING columns {missing} in {p}")
        return df
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, figname, source_files=None):
    """Save PDF + SVG + high-res PNG; return the list of written paths."""
    outs = []
    for ext in ("pdf", "svg", "png"):
        p = OUT_DIR / f"{figname}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    print(f"[saved] {figname}: " + ", ".join(o.name for o in outs))
    return outs


def diverging_cmap():
    """Standard scientific diverging map centered at zero (blue-white-red)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "nc_div", ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"])


def reliability_curve(pred, target, n_bins=10):
    """Bin-based reliability curve + Brier + ECE from stored predictions.

    Unified convention (2026-10-08 audit): predictions are clipped to
    [0, 1] before binning and before the Brier/ECE computation, so every
    model is evaluated on the same effective sample set (no silent
    exclusion of out-of-range predictions)."""
    import numpy as np
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    target = np.asarray(target, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    sev = (target >= 0.8).astype(float)
    centers, freqs, sizes = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        centers.append(float(pred[m].mean()))
        freqs.append(float(sev[m].mean()))
        sizes.append(int(m.sum()))
    brier = float(np.mean((pred - target) ** 2))
    ece = float(sum((s / len(pred)) * abs(f - c)
                    for s, f, c in zip(sizes, freqs, centers)))
    return np.array(centers), np.array(freqs), np.array(sizes), brier, ece


def ece_official(pred, target, n_bins=10):
    """ECE using the EXACT binning convention of the official evaluation
    (compute_structural_ablation_analysis.py): bins are [lo,hi) over
    linspace(0,1,n_bins+1). Returns the value that MUST match the
    structural_ablation.json 'ece' field (boundary pred==1.0 excluded)."""
    import numpy as np
    sev = (target >= 0.8).astype(float)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        msk = (pred >= lo) & (pred < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / len(pred)) * abs(sev[msk].mean() - pred[msk].mean())
    return float(ece)


def roc_pr_curve(pred, target):
    """ROC + PR curves from stored predictions (sklearn)."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    sev = (target >= 0.8).astype(int)
    fpr, tpr, _ = roc_curve(sev, pred)
    rec, prec, _ = precision_recall_curve(sev, pred)
    return fpr, tpr, rec, prec
