#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""edfig_s9 — Spatial prediction maps for the frozen TSCF (reconstructed).

Re-runs the FROZEN SCS-trained TSCF sliding-window PFV inference on the three
domains and draws the FULL-GRID forecast risk field for a representative
test-window issue week per domain (the week with the most severe pixels at
t+4), with the observed severe target overlaid and the domain ROC/PR/CSI
annotated. This reconstructs forecast maps that the evaluation pipeline did
not persist (record-level arrays only); NO new experiment — same model, same
data, same protocol (lead 4 wk, history 24 wk, ocean mask, tau=0.90).

Outputs -> results/figures/edfig_s9_spatial_maps.{pdf,svg,png} +
results/figures/source_data/edfig_s9_source_data.npz
"""
import io, os, sys, json, warnings, importlib
from pathlib import Path

# NOTE: do NOT wrap sys.stdout here and do NOT import nc_style — ablate
# re-wraps stdout on import; importing both would double-wrap and crash at
# shutdown. Only ablate's single wrap is active.
warnings.filterwarnings("ignore")
import numpy as np
import torch

LTTS = Path(__file__).resolve().parents[2]  # was: author-machine path)
sys.path.insert(0, str(LTTS))
sys.path.insert(0, str(LTTS / "experiments" / "04_structural_ablation"))
OUT_DIR = LTTS / "results" / "figures"
SRC_DIR = OUT_DIR / "source_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, figname):
    outs = []
    for ext in ("pdf", "svg", "png"):
        p = OUT_DIR / f"{figname}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=600 if ext == "png" else fig.get_dpi())
        outs.append(p)
    print(f"[saved] {figname}: " + ", ".join(o.name for o in outs))
    return outs

from paths import DATA_DIR
from src.data.aligned_dataset import SCSPIECFDataset
from sklearn.metrics import roc_auc_score, average_precision_score

SEEDM = np.random.default_rng(42)
SEV = 0.8
DOMAINS = [
    ("SCS", "scs_causal_v2.nc", (2016, 2019), dict(train_years=(2003, 2010),
        val_years=(2011, 2015), test_years=(2016, 2019))),
    ("GBR", "gbr_causal_v2.nc", (2017, 2020), dict(train_years=(2003, 2010),
        val_years=(2014, 2016), test_years=(2017, 2020))),
    ("CARIB", "caribbean_causal_v2.nc", (2017, 2020), dict(train_years=(2003, 2010),
        val_years=(2014, 2016), test_years=(2017, 2020))),
]


def restore(device):
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    m = mod.FinalSingleHeadModel(dynamic_channels=13)
    m.load_state_dict(torch.load(LTTS / "results" / "model_final_Btail3" / "model.pt",
                                 map_location="cpu", weights_only=False))
    m.eval()
    return m.to(device)


def land_grid():
    import xarray as xr
    src = os.environ.get("PIECF_DNS_DIR", r"D:\piecf_dns") + r"\bathymetry\ETOPO1_Ice_g_gmt4.grd"
    if not os.path.isfile(src):
        print("[warn] edfig_s9: ETOPO1 grid not found (set PIECF_DNS_DIR); coastlines skipped")
        return np.zeros((180, 360)), (-180, 180, -90, 90)
    ds = xr.open_dataset(src)
    z = ds["z"].values; y = np.asarray(ds["y"].values); x = np.asarray(ds["x"].values)
    ds.close()
    step = max(1, len(y) // 540)
    land = (z[::step, ::step] > 0).astype(float)
    return land, (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))


def read_grid(file):
    """lat/lon coordinates of a causal v2 netCDF (for map extent/scatter)."""
    import xarray as xr
    ds = xr.open_dataset(file)
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    ds.close()
    return lat, lon


def main():
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    model = restore(device)
    land, lextent = land_grid()

    sab = json.load(open(LTTS / "results" / "physics_anchored" / "structural_ablation" /
                         "structural_ablation.json", encoding="utf-8"))

    maps, coords = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.6))
    for ax, (name, file, yrs, split) in zip(axes, DOMAINS):
        ds = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                             lead_weeks=4, dataset_file=file, year_based_split=True,
                             **split)
        T, H, W = ds.history_weeks, ds.n_lat, ds.n_lon
        n, C = ds.n_dates, ds.C_dyn
        y = ds.y_regression_full
        fm = np.full((n, H, W), np.nan, np.float32)
        X = ds.X_dynamic_full
        lm = torch.BoolTensor(ds.land_mask).to(device)
        with torch.no_grad():
            for b in range(0, n - T + 1, 16):
                be = min(b + 16, n - T + 1)
                xd = _bw(X, b, be, T, H, W, C)
                xd = torch.FloatTensor(xd).to(device)
                xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
                out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
                for k in range(be - b):
                    fm[b + k + T - 1] = out[0][k, -1].cpu().numpy()
        # representative issue week: max severe pixels in the test window
        best_t, best_n = None, -1
        for ti in range(T - 1, n):
            if not (yrs[0] <= int(ds.date_strings[ti][:4]) <= yrs[1]):
                continue
            tt = ti + 4
            if tt >= n:
                continue
            sev = np.sum((y[tt] >= SEV) & ~ds.land_mask)
            if sev > best_n:
                best_n = int(sev); best_t = ti
        # PFV metrics on the ocean-masked test record (for annotation)
        p, tg = [], []
        for ti in range(T - 1, n):
            if not (yrs[0] <= int(ds.date_strings[ti][:4]) <= yrs[1]):
                continue
            tt = ti + 4
            if tt >= n:
                continue
            v = ~np.isnan(y[tt]) & ~ds.land_mask & ~np.isnan(fm[ti])
            if v.sum() == 0:
                continue
            p.append(fm[ti][v]); tg.append(y[tt][v])
        p = np.concatenate(p); tg = np.concatenate(tg)
        sev = tg >= SEV
        roc = float(roc_auc_score(sev, p)); pr = float(average_precision_score(sev, p))
        csi = float(sab["domains"][name]["models"]["D_TSCF"]["csi_tau09"])

        maps[name] = (fm[best_t], y[best_t + 4], ds.date_strings[best_t],
                      best_n, roc, pr, csi)
        lat, lon = read_grid(DATA_DIR / file)
        coords[name] = (lat, lon)

        # ---- draw map ----
        ax.imshow(land, origin="lower", cmap="Greys", vmin=0, vmax=1,
                  extent=lextent, aspect="auto", alpha=0.35, zorder=0)
        risk = fm[best_t]
        risk[ds.land_mask] = np.nan
        im = ax.imshow(risk, origin="lower", cmap="magma", vmin=0, vmax=1,
                       extent=(float(lon[0]) - 0.5, float(lon[-1]) + 0.5,
                               float(lat[0]) - 0.5, float(lat[-1]) + 0.5),
                       aspect="auto", zorder=1)
        # overlay observed severe target (t+4) as white/black-ring markers
        sev_map = (y[best_t + 4] >= SEV) & ~ds.land_mask
        sy, sx = np.where(sev_map)
        if len(sy):
            ax.scatter(lon[sx], lat[sy], s=4.0, facecolors="white",
                       edgecolors="#111111", linewidths=0.4, marker="o",
                       zorder=3, alpha=0.9)
        ax.set_title(f"{name}  ({ds.date_strings[best_t][:10]} issue)\n"
                     f"ROC {roc:.3f} · PR {pr:.3f} · CSI@0.9 {csi:.3f}",
                     loc="left", fontsize=7.2)
        ax.set_xlim(lon[0] - 0.5, lon[-1] + 0.5)
        ax.set_ylim(lat[0] - 0.5, lat[-1] + 0.5)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.tick_params(labelsize=5.5)
        print(f"  {name}: issue_fm_date={ds.date_strings[best_t]} severe_at_t4={best_n} "
              f"ROC={roc:.4f} PR={pr:.4f} CSI={csi:.4f}", flush=True)
        del ds, X, fm
        torch.cuda.empty_cache()
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("frozen TSCF predicted severe thermal risk", fontsize=6.6)
    cbar.ax.tick_params(labelsize=6)
    ax0 = axes[0]
    ax0.scatter([], [], s=4.0, facecolors="white", edgecolors="#111111",
                linewidths=0.4, marker="o", label="observed severe (t+4)")
    ax0.legend(loc="upper left", fontsize=5.6, frameon=False)
    fig.suptitle("Spatial prediction maps (frozen TSCF; week = max severe pixels in test window)",
                 fontsize=8.5, x=0.03, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    save_fig(fig, "edfig_s9_spatial_maps")
    np.savez_compressed(SRC_DIR / "edfig_s9_source_data.npz",
                        **{f"{k}_map": v[0] for k, v in maps.items()},
                        **{f"{k}_sev_t4": v[1] for k, v in maps.items()},
                        **{f"{k}_date": np.array(v[2]) for k, v in maps.items()},
                        **{f"{k}_nsev": np.array(v[3]) for k, v in maps.items()},
                        **{f"SCS_lat": coords["SCS"][0], f"SCS_lon": coords["SCS"][1]},
                        **{f"GBR_lat": coords["GBR"][0], f"GBR_lon": coords["GBR"][1]},
                        **{f"CARIB_lat": coords["CARIB"][0], f"CARIB_lon": coords["CARIB"][1]})
    print("[source] edfig_s9_source_data.npz", flush=True)
    print("PHASE DONE (of edfig_s9)", flush=True)


def _bw(X, b, be, T, H, W, C):
    """windows [b,be) -> [be-b,T,H,W,C] sliding windows ending at t-relative."""
    xd = np.zeros((be - b, T, H, W, C), np.float32)
    for k in range(be - b):
        te = b + k + T - 1
        xd[k] = X[te - T + 1:te + 1]
    return xd


if __name__ == "__main__":
    main()