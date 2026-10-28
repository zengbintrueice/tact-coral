#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble ALL supplementary figures into one montage image and collect all
supplementary source data into one Markdown file.

  - Montage : results/figures/supplementary_montage.png (+ .pdf)
  - Source  : results/figures/SOURCE_DATA_SUPPLEMENTARY.md
              (each supplementary figure -> its source-data file(s), embedded)

Figures included (ordered): edfig_s1..s11, S8_seed_sensitivity.
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image

# NOTE: do NOT re-wrap sys.stdout here — nc_style already wraps it on import
# (a second wrapper is closed by the first, raising "I/O operation on closed file").
import nc_style as S

FIG_ORDER = [
    "edfig_s1_leakage_audit", "edfig_s2_temporal_purge",
    "edfig_s3_feature_ablation", "edfig_s4_temporal_scale",
    "edfig_s5_context_attribution", "edfig_s6_threshold_sensitivity",
    "edfig_s7_bootstrap_effects", "edfig_s8_seed_sensitivity",
    "edfig_s9_spatial_maps", "edfig_s10_carib_diagnostics",
    "edfig_s11_hyperparameters",
    "S8_seed_sensitivity",
]
SRC_MAP = {  # fig base name -> list of source_data filenames
    "edfig_s1_leakage_audit": ["edfig_s1_source_data.csv"],
    "edfig_s2_temporal_purge": ["edfig_s2_source_data.csv"],
    "edfig_s3_feature_ablation": ["edfig_s3_source_data.csv"],
    "edfig_s4_temporal_scale": ["edfig_s4_source_data.csv"],
    "edfig_s5_context_attribution": ["edfig_s5_source_data.csv"],
    "edfig_s6_threshold_sensitivity": ["edfig_s6_source_data.csv"],
    "edfig_s7_bootstrap_effects": ["edfig_s7_source_data.csv"],
    "edfig_s8_seed_sensitivity": ["edfig_s8_source_data.csv"],
    "edfig_s9_spatial_maps": ["edfig_s9_source_data.npz"],
    "edfig_s10_carib_diagnostics": ["edfig_s10_source_data.json"],
    "edfig_s11_hyperparameters": ["edfig_s11_source_data.json"],
    "S8_seed_sensitivity": ["S8_seed_sensitivity.csv"],
}
# figures whose source-data CSV is a verbatim copy of a LARGE upstream step
# CSV — embed the header + first N rows + note (keeps the md readable).
TRUNCATE_ROWS = {"edfig_s2_source_data.csv": 20, "edfig_s3_source_data.csv": 20,
                 "edfig_s4_source_data.csv": 30, "edfig_s6_source_data.csv": 30}


# ----------------------------------------------------------------- montage ---
def assemble_montage(ncols=3, cell_width=1400):
    keep = []
    for base in FIG_ORDER:
        p = S.OUT_DIR / f"{base}.png"
        if not p.exists():
            print(f"[warn] missing figure png: {p}")
            continue
        keep.append(base)
    imgs = [Image.open(S.OUT_DIR / f"{b}.png").convert("RGB") for b in keep]
    # scale each to a common height (columns fixed, rows auto)
    heights = [im.height / im.width * cell_width for im in imgs]
    nrows = (len(imgs) + ncols - 1) // ncols
    cell_h = int(max(heights) + 120)     # per-cell height incl. a label strip
    W = ncols * (cell_width + 40)
    H = nrows * cell_h
    canvas = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for k, im in enumerate(imgs):
        r, c = divmod(k, ncols)
        h = int(im.height / im.width * cell_width)
        im2 = im.resize((cell_width, h), Image.LANCZOS)
        x = c * (cell_width + 40) + 20
        y = r * cell_h + 20
        canvas.paste(im2, (x, y))
        draw.text((x, y + h + 8), keep[k], fill="black")
    canvas.save(S.OUT_DIR / "supplementary_montage.png")
    print(f"[montage] {len(keep)} figures -> "
          f"{S.OUT_DIR / 'supplementary_montage.png'} "
          f"({canvas.width}x{canvas.height})")
    return keep


def csv_to_md(path, trunc=None):
    df = pd.read_csv(path)
    used = df.head(trunc) if trunc else df
    rows = ["| " + " | ".join(map(str, used.columns)) + " |",
            "|" + "|".join(["---"] * len(used.columns)) + "|"]
    for _, r in used.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r.values) + " |")
    if trunc and len(df) > trunc:
        rows.append(f"*… {len(df) - trunc} more rows omitted (verbatim upstream "
                    "copy)…*")
    return "\n".join(rows)


def build_md(keep):
    L = ["# Supplementary Figures — Source Data", "",
         f"Collected: {pd.Timestamp.now().isoformat()}", "",
         "One section per supplementary figure; each lists its source-data file(s) "
         "under `results/figures/source_data/` and embeds the values (CSV as markdown "
         "tables; JSON as fenced blocks; the spatial-map NPZ is listed with its "
         "contents since it is binary).", ""]
    for base in keep:
        L.append(f"## {base}")
        L.append("")
        for sfn in SRC_MAP.get(base, []):
            sp = S.SRC_DIR / sfn
            if not sp.exists():
                L.append(f"- `{sfn}` — MISSING")
                L.append("")
                continue
            if sfn.endswith(".csv"):
                trunc = TRUNCATE_ROWS.get(sfn)
                L.append(f"Source: `{sfn}` ({'all rows' if not trunc else f'{trunc} shown'}):")
                L.append("")
                L.append(csv_to_md(sp, trunc))
            elif sfn.endswith(".json"):
                L.append(f"Source: `{sfn}`:")
                L.append("")
                L.append("```json")
                L.append(json.dumps(json.load(open(sp, encoding="utf-8")),
                                    indent=2, ensure_ascii=False))
                L.append("```")
            elif sfn.endswith(".npz"):
                import numpy as np
                z = np.load(sp)
                L.append(f"Source: `{sfn}` (binary NPZ) — keys/shapes:")
                L.append("")
                for k in z.files:
                    L.append(f"- `{k}`: {z[k].shape} {z[k].dtype}")
            L.append("")
    out = "\n".join(L) + "\n"
    (S.OUT_DIR / "SOURCE_DATA_SUPPLEMENTARY.md").write_text(out, encoding="utf-8")
    print(f"[md] SOURCE_DATA_SUPPLEMENTARY.md ({len(L)} sections)")


if __name__ == "__main__":
    keep = assemble_montage()
    build_md(keep)
    print("MONTAGE + SOURCE-MD DONE")