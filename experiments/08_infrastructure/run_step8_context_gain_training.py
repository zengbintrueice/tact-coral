# run_step8_context_gain_training.py — Step 8: why/when does contextual
# fusion add value? (training phase)
#
#   Nested ablation matrix on top of the two principal recent-memory
#   controls identified by Step 7 (C1 = trailing 1-week sum of HotSpot>=1C,
#   C4 = trailing 4-week sum). All C_W definitions/statistics are loaded
#   FROZEN from step7_feature_manifest.json (audit A3: identical to Step 7).
#
#   Spec name mapping (spec §5 uses "C1/C4" for the env-context models;
#   renamed here to avoid clashing with the C_W channel names):
#     A1 (spec A1): C1 only                        1 channel
#     A2 (spec A2): C4 only                        1 channel
#     B1 (spec B1): C1 + instantaneous thermal     6 channels
#     B2 (spec B2): C4 + instantaneous thermal     6 channels
#     ENV1 (spec §5 "C1"): C1 + env/calendar       7 channels
#     ENV2 (spec §5 "C4"): C4 + env/calendar       7 channels
#     D1 (spec D1 "FULL-C1"): BASE(11) + C1       12 channels
#     D2 (spec D2 "FULL-C4"): BASE(11) + C4       12 channels
#   BASE(11) = instantaneous thermal (5) + env/calendar (6) = the original
#   13 dataset channels minus hotspot_cumsum_4w/12w. D is therefore FULL
#   with the two original cumulative channels REPLACED by C_W (spec §6
#   "exact replacement"); documented in step8_feature_manifest.json.
#
#   Feature groups (Step-6 manifest, NOT redefined):
#     instantaneous thermal (group A): sst, ssta, hotspot, sst_anomaly,
#                                      d_hotspot_dt
#     env/calendar (group D): wind_speed, ssr, chlorophyll, par,
#                             seasonal_sin, seasonal_cos
#
#   Training protocol: identical to run_step6/7 (D1 protocol verbatim):
#     weighted BCE pos_weight=1000, Adam lr=1e-3 wd=1e-5, cosine T_max=60,
#     batch 32, 60 epochs, early stop patience 10, grad clip 1.0, seed 42.
#     Train from scratch — no initialization from any checkpoint.
#
#   Normalization: per-channel stats from SCS TRAIN 2003-2010 ONLY
#     (audit A6/A8): dataset channels from frozen SCS_CHANNEL_MEAN/STD;
#     C_W from step7_feature_manifest.json (computed on train windows only).
#
#   Weights -> results/step8_contextual_gain/weights/<MODEL>/{model.pt,training_log.json}
#
#   Run:
#     cd LTTS_CRW/experiments/08_infrastructure
#     & 'D:\...\piecf_env\Scripts\python.exe' run_step8_context_gain_training.py

import io, sys, json, time, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))

from paths import DATA_DIR
from src.data.aligned_dataset import create_scs_dataloaders
from src.configs.config import SCS_CHANNEL_MEAN, SCS_CHANNEL_STD
import ablate_a_dh_stopgrad as trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH = 32
LR = 1e-3
SEED = 42
SEV = 0.8
POS_WEIGHT = 1000.0
PATIENCE = 10

DATASET_FILE = "scs_causal_v2.nc"
TRAIN_YEARS = (2003, 2010)
OUT_DIR = LTTS_ROOT / "results" / "step8_contextual_gain"
WEIGHTS_DIR = OUT_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
S7_MANIFEST = LTTS_ROOT / "results" / "step7_temporal_scale" / "step7_feature_manifest.json"

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
HS_IDX = DATASET_ORDER.index("hotspot")
BASE_NAMES = [c for c in DATASET_ORDER if c not in ("hotspot_cumsum_4w", "hotspot_cumsum_12w")]
BASE_IDX = [DATASET_ORDER.index(c) for c in BASE_NAMES]

# Step-6 feature manifest groups (verbatim; no new features — audit A4/A5)
INST_NAMES = ["sst", "ssta", "hotspot", "sst_anomaly", "d_hotspot_dt"]
INST_IDX = [DATASET_ORDER.index(c) for c in INST_NAMES]
ENV_NAMES = ["wind_speed", "ssr", "chlorophyll", "par", "seasonal_sin", "seasonal_cos"]
ENV_IDX = [DATASET_ORDER.index(c) for c in ENV_NAMES]

C_NAME = {1: "hotspot_cumsum1c_1w", 4: "hotspot_cumsum1c_4w"}

# nested model matrix: name -> (base channel names, C window or None)
MODELS = {
    "A1": ([], 1),
    "A2": ([], 4),
    "B1": (INST_NAMES, 1),
    "B2": (INST_NAMES, 4),
    "ENV1": (ENV_NAMES, 1),
    "ENV2": (ENV_NAMES, 4),
    "D1": (BASE_NAMES, 1),
    "D2": (BASE_NAMES, 4),
}


def compute_cumsum_hs1c(hotspot, W):
    """Trailing W-week sum of HotSpot >= 1 degC — verbatim identical to
    run_step7_temporal_scale_training.compute_cumsum_hs1c (audit A2/A3),
    including the documented partial-window caveat for the first W-1 record
    dates (train period only)."""
    hs = np.ascontiguousarray(hotspot, dtype=np.float64)
    D = hs.shape[0]
    hs_ge1 = np.where(np.isfinite(hs), np.where(hs >= 1.0, hs, 0.0), 0.0)
    cum = np.cumsum(hs_ge1, axis=0)
    cs = np.empty_like(cum)
    cs[:W] = cum[:W]
    if D > W:
        cs[W:] = cum[W:] - cum[:-W]
    cs[np.isnan(hs)] = np.nan
    return cs.astype(np.float32)


class SubsetDataset(torch.utils.data.Dataset):
    """Wrap a SCSPIECFDataset: keep the selected base channels (possibly
    none) and append the C_W trailing maps as extra per-timestep channels."""

    def __init__(self, base, base_idx, c_window, c_map):
        self.base = base
        self.base_idx = list(base_idx)
        T = base.history_weeks
        self.c_window = c_window
        if c_window is not None:
            starts = np.asarray(base.window_indices, dtype=np.int64)
            self.c_win = c_map[starts[:, None] + np.arange(T)[None, :]]  # [N,T,H,W]
        else:
            self.c_win = None

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, i):
        item = dict(self.base[i])
        xd = item["x_dynamic"][..., self.base_idx] if self.base_idx \
            else item["x_dynamic"][..., :0]
        nm = item.get("nan_mask_dyn")
        if self.c_win is not None:
            ex = torch.FloatTensor(self.c_win[i][..., None])         # [T,H,W,1]
            item["x_dynamic"] = torch.cat([xd, ex], dim=-1)
            valid_ex = torch.BoolTensor(
                np.isfinite(self.c_win[i][..., None]).transpose(3, 0, 1, 2))    # [1,T,H,W]
            if nm is not None:
                nm = nm[self.base_idx] if self.base_idx else nm[:0]
                item["nan_mask_dyn"] = torch.cat([nm, valid_ex], dim=0)
            else:
                item["nan_mask_dyn"] = valid_ex
        else:
            item["x_dynamic"] = xd
            if nm is not None:
                item["nan_mask_dyn"] = nm[self.base_idx] if self.base_idx else nm[:0]
        return item


def weighted_bce(p, y_sev, valid, pos_weight):
    v = valid.float()
    y_clean = torch.nan_to_num(y_sev.float(), nan=0.0)
    eps = 1e-7
    loss = -(pos_weight * y_clean * torch.log(p.clamp(eps, 1 - eps))
             + (1 - y_clean) * torch.log((1 - p).clamp(eps, 1 - eps)))
    return (loss * v).sum() / (v.sum() + 1e-8)


# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print(f"Step 8: contextual-gain nested training (D1 protocol, causal v2) — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)
print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_FILE} in {DATA_DIR}")

with open(S7_MANIFEST, encoding="utf-8") as f:
    s7 = json.load(f)
c_stats = {}
for W in (1, 4):
    key = f"C{W}"
    entry = next(v for v in s7["channel_definition"]["C_channels"].values()
                 if v["feature"] == C_NAME[W])
    assert entry["window_weeks"] == W and entry["feature"] == C_NAME[W]
    c_stats[C_NAME[W]] = entry
print(f"Step-7 manifest loaded; C1/C4 stats frozen from {S7_MANIFEST.name}")

torch.manual_seed(SEED)
np.random.seed(SEED)
loaders = create_scs_dataloaders(
    data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
    random_split=False, random_seed=SEED, lead_weeks=4, year_based_split=True,
    train_years=TRAIN_YEARS, val_years=(2011, 2015), test_years=(2016, 2019),
    dataset_file=DATASET_FILE)
tr_ds, va_ds, te_ds = loaders[0].dataset, loaders[1].dataset, loaders[2].dataset
assert list(tr_ds.dyn_feature_names) == DATASET_ORDER

# ---- C_W maps (full record) + trailing-window spot verification --------------
hotspot_full = tr_ds.X_dynamic_full[..., HS_IDX].astype(np.float32)
c_maps = {W: compute_cumsum_hs1c(hotspot_full, W) for W in (1, 4)}

rng = np.random.default_rng(7)
ver = []
for W in (1, 4):
    for _ in range(5):
        t = int(rng.integers(W, hotspot_full.shape[0]))
        i = int(rng.integers(0, hotspot_full.shape[1]))
        j = int(rng.integers(0, hotspot_full.shape[2]))
        hs_win = hotspot_full[t - W + 1: t + 1, i, j]
        if np.isfinite(hs_win[-1]):
            manual = float(np.nansum(np.where(np.isfinite(hs_win) & (hs_win >= 1.0), hs_win, 0.0)))
        else:
            manual = np.nan
        got = c_maps[W][t, i, j]
        ok = (np.isnan(manual) and np.isnan(got)) or abs(float(got) - manual) < 1e-3
        ver.append(bool(ok))
        assert (t - W + 1) >= 0 and t < hotspot_full.shape[0]
trailing_ok = all(ver)
print(f"Trailing-window verification ({len(ver)} spot checks): "
      f"{'PASS' if trailing_ok else 'FAIL'}")
assert trailing_ok

base_stats = {c: (SCS_CHANNEL_MEAN[DATASET_ORDER.index(c)],
                  SCS_CHANNEL_STD[DATASET_ORDER.index(c)]) for c in DATASET_ORDER}


def channel_spec(base_names, W):
    names = list(base_names) + ([C_NAME[W]] if W is not None else [])
    mean = [base_stats[n][0] if n in base_stats else c_stats[n]["train_mean"]
            for n in names]
    std = [base_stats[n][1] if n in base_stats else c_stats[n]["train_std"]
           for n in names]
    return names, mean, std


def train_one(name, base_names, W):
    out_dir = WEIGHTS_DIR / name
    ckpt = out_dir / "model.pt"
    log_path = out_dir / "training_log.json"
    if ckpt.exists() and log_path.exists():
        print(f"[{name}] checkpoint already exists — skipping (resume mode)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ch_names, mean_v, std_v = channel_spec(base_names, W)
    n_ch = len(ch_names)

    print(f"\n{'=' * 64}\n[{name}] {n_ch} channels: {ch_names}\n{'=' * 64}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    base_idx = [DATASET_ORDER.index(c) for c in base_names]
    tr_s = DataLoader(SubsetDataset(tr_ds, base_idx, W, c_maps[W] if W else None),
                      batch_size=BATCH, shuffle=True, num_workers=0)
    va_s = DataLoader(SubsetDataset(va_ds, base_idx, W, c_maps[W] if W else None),
                      batch_size=BATCH, shuffle=False, num_workers=0)

    ytr = tr_ds.y_lead
    labtr = tr_ds.label_mask
    n_pos = int(np.sum((ytr >= SEV) & labtr))
    n_neg = int(np.sum((ytr < SEV) & labtr))
    print(f"  train severe: {n_pos}/{n_pos+n_neg} ({n_pos/(n_pos+n_neg)*100:.3f}%)  "
          f"pos_weight={POS_WEIGHT:.1f}")

    model = trainer.FinalSingleHeadModel(dynamic_channels=n_ch)
    model.channel_norm.mean = torch.tensor(mean_v, dtype=torch.float32).view(1, 1, 1, 1, -1)
    model.channel_norm.std = torch.tensor(std_v, dtype=torch.float32).view(1, 1, 1, 1, -1)
    model = model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    pw = torch.tensor(POS_WEIGHT, device=DEVICE)

    best_val = float("inf")
    best_state = None
    patience = 0
    stopped_at = EPOCHS
    hist = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss, n_steps = 0.0, 0
        for batch in tr_s:
            xd = batch["x_dynamic"].to(DEVICE)
            xs = batch["x_static"].to(DEVICE)
            lm = batch["land_mask"].to(DEVICE)
            nm = batch.get("nan_mask_dyn")
            if nm is not None:
                nm = nm.to(DEVICE)
            lab = batch.get("label_mask")
            if lab is not None:
                lab = lab.to(DEVICE)
            y = batch["y_lead"].to(DEVICE)
            valid = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
            y_sev = (y >= SEV).float()
            opt.zero_grad()
            out = model(xd, xs, lm, nan_mask=nm)
            p = out[0][:, -1]
            loss = weighted_bce(p, y_sev, valid, pw)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            tr_loss += loss.item()
            n_steps += 1
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in va_s:
                xd = batch["x_dynamic"].to(DEVICE)
                xs = batch["x_static"].to(DEVICE)
                lm = batch["land_mask"].to(DEVICE)
                nm = batch.get("nan_mask_dyn")
                if nm is not None:
                    nm = nm.to(DEVICE)
                lab = batch.get("label_mask")
                if lab is not None:
                    lab = lab.to(DEVICE)
                y = batch["y_lead"].to(DEVICE)
                valid = torch.isfinite(y).float() * (lab.float() if lab is not None else (~lm).float())
                y_sev = (y >= SEV).float()
                out = model(xd, xs, lm, nan_mask=nm)
                vl += weighted_bce(out[0][:, -1], y_sev, valid, pw).item()
                nv += 1
        sched.step()
        v = vl / max(nv, 1)
        hist.append({"epoch": ep, "train_bce": tr_loss / max(n_steps, 1), "val_bce": v})
        if v < best_val:
            best_val = v
            patience = 0
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else:
            patience += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep {ep:3d} train-bce={tr_loss/max(n_steps,1):.4f} val-bce={v:.4f}")
        if patience >= PATIENCE:
            stopped_at = ep
            print(f"    early stop at epoch {ep} (patience {PATIENCE})")
            break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), ckpt)
    log = {
        "model": name, "channels": ch_names,
        "base_channels": list(base_names), "c_window_weeks": W,
        "normalization_mean": mean_v, "normalization_std": std_v,
        "normalization_source": ("dataset channels: SCS_CHANNEL_MEAN/STD (SCS train "
                                 "2003-2010, frozen); C_W: step7_feature_manifest.json "
                                 "train-window stats (frozen)"),
        "dataset_file": DATASET_FILE,
        "protocol": {"epochs_max": EPOCHS, "batch": BATCH, "lr": LR,
                     "weight_decay": 1e-5, "scheduler": "cosine T_max=60 -> 1e-5",
                     "early_stop_patience": PATIENCE, "pos_weight": POS_WEIGHT,
                     "grad_clip": 1.0, "seed": SEED,
                     "loss": "weighted BCE on y_sev=(y_lead>=0.8), valid pixels only",
                     "selection": "best val weighted-BCE epoch",
                     "init": "from scratch (no Step-4/6/7 checkpoint init)"},
        "stopped_at_epoch": stopped_at, "best_val_bce": float(best_val),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": round(time.time() - t0, 1), "history": hist,
        "timestamp": datetime.now().isoformat(),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  saved {ckpt}  (best val-bce {best_val:.4f}, {log['n_params']} params, "
          f"{time.time()-t0:.0f}s)")


for name, (base_names, W) in MODELS.items():
    train_one(name, base_names, W)

# ---- feature manifest (spec §15) ---------------------------------------------
feat_rows = []
for idx, feat in enumerate(DATASET_ORDER):
    if feat in ("hotspot_cumsum_4w", "hotspot_cumsum_12w"):
        group, thermal, kind, window = "replaced_by_C_W", "thermal", "cumulative", \
            ("4" if feat.endswith("4w") else "12")
    elif feat in INST_NAMES:
        group, thermal, kind, window = "A_instantaneous_thermal", "thermal", \
            "instantaneous", "0"
    else:
        group, thermal, kind, window = "D_other_environmental_calendar", "nonthermal", \
            ("calendar" if feat.startswith("seasonal") else "instantaneous"), \
            ("annual-cycle" if feat.startswith("seasonal") else "0")
    feat_rows.append({
        "feature_name": feat, "channel_index_dataset": idx,
        "feature_group": group, "thermal": thermal, "temporal_kind": kind,
        "window_weeks": window,
        "norm_mean": base_stats[feat][0], "norm_std": base_stats[feat][1],
    })
for W in (1, 4):
    e = c_stats[C_NAME[W]]
    feat_rows.append({
        "feature_name": C_NAME[W], "channel_index_dataset": None,
        "feature_group": "C_recent_cumulative_thermal", "thermal": "thermal",
        "temporal_kind": "cumulative", "window_weeks": str(W),
        "norm_mean": e["train_mean"], "norm_std": e["train_std"],
    })

manifest = {
    "experiment": "step8_contextual_gain",
    "dataset": {"file": DATASET_FILE, "dir": str(DATA_DIR)},
    "spec_name_mapping": {
        "A1": "spec A1 (C1 only)", "A2": "spec A2 (C4 only)",
        "B1": "spec B1 (C1+instantaneous thermal)",
        "B2": "spec B2 (C4+instantaneous thermal)",
        "ENV1": "spec section-5 'C1' (C1+environmental context)",
        "ENV2": "spec section-5 'C4' (C4+environmental context)",
        "D1": "spec D1 FULL-C1", "D2": "spec D2 FULL-C4",
    },
    "feature_groups_source": "Step-6 manifest (groups A/D verbatim; no new features)",
    "features": feat_rows,
    "models": {name: {"base_channels": bn, "c_window_weeks": W,
                      "n_channels": len(bn) + (1 if W else 0)}
               for name, (bn, W) in MODELS.items()},
    "c_channels_source": str(S7_MANIFEST),
    "note": ("cumulative HotSpot channels are physically meaningful historical "
             "thermal-stress variables (NOAA CRW DHW = 12-week accumulation of "
             "HotSpot >= 1 degC); attribution experiment, NOT a leakage claim"),
    "timestamp": datetime.now().isoformat(),
}
with open(OUT_DIR / "step8_feature_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nManifest: {OUT_DIR / 'step8_feature_manifest.json'}")
print("Done.")
