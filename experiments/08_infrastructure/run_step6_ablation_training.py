# run_step6_ablation_training.py — Step 6: retrain feature-ablation models
# under the exact D1 training protocol (train_severe_classifier.py).
#
#   Models (all trained on scs_causal_v2.nc, SCS splits 2003-2010 / 2011-2015
#   / 2016-2019, lead 4, history 24, seed 42):
#     FULL          all 13 dynamic channels (reference)
#     NO_CUMSUM12   remove hotspot_cumsum_12w                 (12 ch)
#     NO_CUMULATIVE remove hotspot_cumsum_4w + _12w           (11 ch)
#     THERMAL_ONLY  thermal channels only                     (7 ch)
#
#   Protocol (identical to experiments/01_training/train_severe_classifier.py,
#   the D1 formal-model entry point):
#     FinalSingleHeadModel, weighted BCE pos_weight=1000 (fixed, not recomputed),
#     Adam lr=1e-3 wd=1e-5, cosine annealing T_max=60 -> 1e-5, batch 32,
#     60 epochs, early stop patience 10 on val weighted-BCE, grad clip 1.0,
#     seed 42. Normalization = SCS_CHANNEL_MEAN/STD from src/configs/config.py
#     (SCS train 2003-2010 only, frozen), subset per model to the retained
#     channels in dataset order.
#
#   Weights -> results/step6_feature_ablation/weights/<MODEL>/model.pt
#   Logs    -> results/step6_feature_ablation/weights/<MODEL>/training_log.json
#
#   Run:
#     cd LTTS_CRW/experiments/08_infrastructure
#     & 'D:\...\piecf_env\Scripts\python.exe' run_step6_ablation_training.py

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
OUT_DIR = LTTS_ROOT / "results" / "step6_feature_ablation"
WEIGHTS_DIR = OUT_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Channel order == dataset order == SCS_CHANNEL_MEAN/STD order (asserted below)
EXPECTED_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                  "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                  "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]

GROUP_A = ["sst", "ssta", "hotspot", "sst_anomaly", "d_hotspot_dt"]
GROUP_B = ["hotspot_cumsum_4w"]
GROUP_C = ["hotspot_cumsum_12w"]
GROUP_D = ["wind_speed", "ssr", "chlorophyll", "par", "seasonal_sin", "seasonal_cos"]

MODELS = {
    "FULL": EXPECTED_ORDER,
    "NO_CUMSUM12": [c for c in EXPECTED_ORDER if c != "hotspot_cumsum_12w"],
    "NO_CUMULATIVE": [c for c in EXPECTED_ORDER
                      if c not in ("hotspot_cumsum_4w", "hotspot_cumsum_12w")],
    "THERMAL_ONLY": GROUP_A + GROUP_B + GROUP_C,
}


class ChannelSubset(torch.utils.data.Dataset):
    """Wrap a SCSPIECFDataset and keep only the selected dynamic channels.
    x_dynamic: [T,H,W,C] -> [..., idx]; nan_mask: [C,T,H,W] -> [idx,...]."""

    def __init__(self, base, idx):
        self.base = base
        self.idx = list(idx)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, i):
        item = dict(self.base[i])
        item["x_dynamic"] = item["x_dynamic"][..., self.idx]
        for key in ("nan_mask_dyn", "nan_mask"):
            if key in item:
                item[key] = item[key][self.idx]
        return item


def weighted_bce(p, y_sev, valid, pos_weight):
    v = valid.float()
    y_clean = torch.nan_to_num(y_sev.float(), nan=0.0)
    eps = 1e-7
    loss = -(pos_weight * y_clean * torch.log(p.clamp(eps, 1 - eps))
             + (1 - y_clean) * torch.log((1 - p).clamp(eps, 1 - eps)))
    return (loss * v).sum() / (v.sum() + 1e-8)


def make_subset_loader(loader, idx, shuffle):
    ds = ChannelSubset(loader.dataset, idx)
    return DataLoader(ds, batch_size=loader.batch_size,
                      shuffle=shuffle, num_workers=0)


def train_one(name, channels):
    out_dir = WEIGHTS_DIR / name
    ckpt = out_dir / "model.pt"
    log_path = out_dir / "training_log.json"
    if ckpt.exists() and log_path.exists():
        print(f"[{name}] checkpoint already exists — skipping (resume mode)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = [EXPECTED_ORDER.index(c) for c in channels]
    assert channels == [EXPECTED_ORDER[i] for i in idx]
    mean_sub = [SCS_CHANNEL_MEAN[i] for i in idx]
    std_sub = [SCS_CHANNEL_STD[i] for i in idx]

    print(f"\n{'=' * 64}\n[{name}] {len(channels)} channels: {channels}\n{'=' * 64}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    loaders = create_scs_dataloaders(
        data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
        random_split=False, random_seed=SEED, lead_weeks=4, year_based_split=True,
        train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019),
        dataset_file=DATASET_FILE)
    tr, va, te = loaders
    tr_s = make_subset_loader(tr, idx, shuffle=True)
    va_s = make_subset_loader(va, idx, shuffle=False)

    ytr = tr.dataset.y_lead
    labtr = tr.dataset.label_mask
    n_pos = int(np.sum((ytr >= SEV) & labtr))
    n_neg = int(np.sum((ytr < SEV) & labtr))
    print(f"  train severe: {n_pos}/{n_pos+n_neg} ({n_pos/(n_pos+n_neg)*100:.3f}%)  "
          f"pos_weight={POS_WEIGHT:.1f}")

    model = trainer.FinalSingleHeadModel(dynamic_channels=len(channels))
    # Subset normalization: same SCS-train stats, restricted to retained
    # channels in dataset order (re-assign buffers -> persistent -> baked
    # into ckpt). ChannelNormalizer's constructor always builds 13-wide
    # buffers, so the tensors are replaced, not copied into.
    model.channel_norm.mean = torch.tensor(
        mean_sub, dtype=torch.float32).view(1, 1, 1, 1, -1)
    model.channel_norm.std = torch.tensor(
        std_sub, dtype=torch.float32).view(1, 1, 1, 1, -1)
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
    n_params = sum(p.numel() for p in model.parameters())
    log = {
        "model": name,
        "channels": channels,
        "channel_indices": idx,
        "normalization_mean": mean_sub,
        "normalization_std": std_sub,
        "normalization_source": ("src/configs/config.py SCS_CHANNEL_MEAN/STD "
                                 "(SCS train 2003-2010 only, frozen); subset to "
                                 "retained channels in dataset order"),
        "dataset_file": DATASET_FILE,
        "protocol": {"epochs_max": EPOCHS, "batch": BATCH, "lr": LR,
                     "weight_decay": 1e-5, "scheduler": "cosine T_max=60 -> 1e-5",
                     "early_stop_patience": PATIENCE, "pos_weight": POS_WEIGHT,
                     "grad_clip": 1.0, "seed": SEED,
                     "loss": "weighted BCE on y_sev=(y_lead>=0.8), valid pixels only",
                     "selection": "best val weighted-BCE epoch"},
        "stopped_at_epoch": stopped_at,
        "best_val_bce": float(best_val),
        "n_params": int(n_params),
        "train_seconds": round(time.time() - t0, 1),
        "history": hist,
        "timestamp": datetime.now().isoformat(),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  saved {ckpt}  (best val-bce {best_val:.4f}, {n_params} params, "
          f"{time.time()-t0:.0f}s)")


print("=" * 64)
print(f"Step 6: feature-ablation retraining (D1 protocol, causal v2) — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)
print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_FILE} in {DATA_DIR}")

# Hard assertion: hardcoded normalization order == dataset channel order
_dummy = create_scs_dataloaders(
    data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
    random_split=False, random_seed=SEED, lead_weeks=4, year_based_split=True,
    train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019),
    dataset_file=DATASET_FILE)
_ds = _dummy[0].dataset
assert list(_ds.dyn_feature_names) == EXPECTED_ORDER, (
    f"channel order mismatch: {_ds.dyn_feature_names}")
print(f"Channel order verified: {EXPECTED_ORDER}")
del _dummy, _ds

for name, channels in MODELS.items():
    train_one(name, channels)

# manifest
manifest = {
    "experiment": "step6_feature_ablation",
    "dataset": {"files": ["scs_causal_v2.nc", "gbr_causal_v2.nc"],
                "dir": str(DATA_DIR)},
    "channel_order": EXPECTED_ORDER,
    "groups": {
        "A_instantaneous_thermal": GROUP_A,
        "B_short_cumulative": GROUP_B,
        "C_long_cumulative": GROUP_C,
        "D_other_environmental_calendar": GROUP_D,
        "E_static_not_consumed": ["bathymetry", "dist_to_shore",
                                  "cyclone_frequency", "salinity", "ocean_mask"],
    },
    "group_rationale": {
        "A": "instantaneous thermal state/tendency (d_hotspot_dt = 1-week trailing HotSpot difference)",
        "B": "short trailing cumulative HotSpot (4 weeks)",
        "C": "long trailing cumulative HotSpot (12 weeks; CRW DHW is a 12-week accumulation of HotSpot>=1C)",
        "D": "nonthermal environmental (wind_speed, ssr, chlorophyll, par) and calendar-phase (seasonal_sin/cos) channels",
        "E": "static channels; FinalSingleHeadModel.forward_features consumes x_dynamic only (piecf_model.py:302-357) — zero contribution, not claimed",
    },
    "models": {name: {"channels": ch, "n_channels": len(ch)}
               for name, ch in MODELS.items()},
    "note": "cumulative HotSpot channels are scientifically legitimate "
            "thermal-stress predictors (NOAA CRW DHW definition); this "
            "experiment is attribution, NOT a leakage claim",
    "timestamp": datetime.now().isoformat(),
}
with open(OUT_DIR / "step6_feature_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nManifest: {OUT_DIR / 'step6_feature_manifest.json'}")
print("Done.")
