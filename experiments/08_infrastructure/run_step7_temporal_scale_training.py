# run_step7_temporal_scale_training.py — Step 7: temporal-scale thermal-stress
# decomposition (training phase).
#
#   NEW cumulative channels (CRW-style, per spec §2):
#     C_W = trailing W-week sum of HotSpot values >= 1 degC, ending at issue
#     time t (window (t-W+1 .. t]), no future observations, no centered
#     windows, no interpolation. Semantics replicated verbatim from the
#     existing derived-channel implementation (aligned_dataset
#     _compute_derived_channels): NaN weeks contribute 0 to the sum; the
#     result is NaN where hotspot[t] is NaN. Unit convention: degC-weeks of
#     the weekly 7-day-mean HotSpot channel (same as hotspot_cumsum_4w/12w).
#     NOTE: these are NEW channels — the existing dataset channels
#     hotspot_cumsum_4w/12w accumulate ALL HotSpot (>=0) and are REPLACED,
#     not retained.
#
#   Models (architecture/protocol identical to Step-6 FULL; only the
#   cumulative thermal-stress representation changes):
#     Tier 1 (single scale):  M_C1, M_C2, M_C4, M_C8, M_C12
#         = 11 non-cumulative channels + [C_W]
#     Tier 2 (multi-scale):   M_C4_C12, M_C2_C4_C12, M_C1_C4_C12
#         = 11 non-cumulative channels + the named C_W set
#
#   Normalization: per-channel stats from SCS TRAIN 2003-2010 ONLY.
#     - the 11 non-cumulative channels: SCS_CHANNEL_MEAN/STD (frozen,
#       SCS-train-only, src/configs/config.py) at their dataset indices
#     - each C_W: mean/std/min/max/n_valid computed from the train-window
#       tensors (train windows x 24 timesteps x H x W, finite values only)
#       and stored in step7_feature_manifest.json
#
#   Training protocol (identical to run_step6_ablation_training.py, itself a
#   verbatim copy of train_severe_classifier.py / D1):
#     weighted BCE pos_weight=1000, Adam lr=1e-3 wd=1e-5, cosine T_max=60,
#     batch 32, 60 epochs, early stop patience 10, grad clip 1.0, seed 42.
#     Train from scratch — no initialization from Step-4/6 checkpoints.
#
#   Weights -> results/step7_temporal_scale/weights/<MODEL>/{model.pt,training_log.json}
#
#   Run:
#     cd LTTS_CRW/experiments/08_infrastructure
#     & 'D:\...\piecf_env\Scripts\python.exe' run_step7_temporal_scale_training.py

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
OUT_DIR = LTTS_ROOT / "results" / "step7_temporal_scale"
WEIGHTS_DIR = OUT_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = ["sst", "ssta", "hotspot", "chlorophyll", "par", "seasonal_sin",
                 "wind_speed", "ssr", "sst_anomaly", "d_hotspot_dt",
                 "hotspot_cumsum_4w", "hotspot_cumsum_12w", "seasonal_cos"]
HS_IDX = DATASET_ORDER.index("hotspot")
# 11 non-cumulative channels (everything except hotspot_cumsum_4w/12w)
BASE_NAMES = [c for c in DATASET_ORDER if c not in ("hotspot_cumsum_4w", "hotspot_cumsum_12w")]
BASE_IDX = [DATASET_ORDER.index(c) for c in BASE_NAMES]

WINDOWS = [1, 2, 4, 8, 12]
C_NAME = {W: f"hotspot_cumsum1c_{W}w" for W in WINDOWS}

MODELS = {
    "M_C1": ["C1"], "M_C2": ["C2"], "M_C4": ["C4"], "M_C8": ["C8"], "M_C12": ["C12"],
    "M_C4_C12": ["C4", "C12"],
    "M_C2_C4_C12": ["C2", "C4", "C12"],
    "M_C1_C4_C12": ["C1", "C4", "C12"],
}
Ckey = {"C1": 1, "C2": 2, "C4": 4, "C8": 8, "C12": 12}


def compute_cumsum_hs1c(hotspot, W):
    """Trailing W-week sum of HotSpot >= 1 degC. Verbatim NaN semantics of
    aligned_dataset._compute_derived_channels: NaN weeks contribute 0;
    result NaN where hotspot[t] is NaN. hotspot: [D,H,W].

    Known partial-window caveat (audit minor #4): for the first W-1 dates of
    the record (t < W-1) this returns the partial sum over the available
    history only. This affects at most the first ~12 weeks of 2003 (train
    period, SCS record start); val/test and GBR zero-shot windows start years
    later, so the semantic is identical between training and evaluation — both
    use THIS function. Do not change it without retraining: the saved Step-7
    checkpoints were trained with partial-sum semantics."""
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


class ScaleDataset(torch.utils.data.Dataset):
    """Wrap a SCSPIECFDataset: keep the selected base channels and append the
    C_W trailing maps as extra per-timestep channels."""

    def __init__(self, base, extra_names, c_maps):
        self.base = base
        self.extra_names = list(extra_names)
        T = base.history_weeks
        self.extras = np.stack([c_maps[n] for n in self.extra_names], axis=-1) \
            if self.extra_names else None  # [n_dates,H,W,E]
        if self.extras is not None:
            starts = np.asarray(base.window_indices, dtype=np.int64)
            self.extra_win = self.extras[starts[:, None] + np.arange(T)[None, :]]
        else:
            self.extra_win = None

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, i):
        item = dict(self.base[i])
        xd = item["x_dynamic"][..., BASE_IDX]
        if self.extra_win is not None:
            ex = torch.FloatTensor(self.extra_win[i])            # [T,H,W,E]
            item["x_dynamic"] = torch.cat([xd, ex], dim=-1)
            nm = item.get("nan_mask_dyn")
            valid_ex = torch.BoolTensor(
                np.isfinite(self.extra_win[i]).transpose(3, 0, 1, 2))  # [E,T,H,W]
            if nm is not None:
                nm = nm[BASE_IDX]  # base mask covers all 13 dataset channels
                item["nan_mask_dyn"] = torch.cat([nm, valid_ex], dim=0)
            else:
                item["nan_mask_dyn"] = valid_ex
        else:
            item["x_dynamic"] = xd
        return item


def weighted_bce(p, y_sev, valid, pos_weight):
    v = valid.float()
    y_clean = torch.nan_to_num(y_sev.float(), nan=0.0)
    eps = 1e-7
    loss = -(pos_weight * y_clean * torch.log(p.clamp(eps, 1 - eps))
             + (1 - y_clean) * torch.log((1 - p).clamp(eps, 1 - eps)))
    return (loss * v).sum() / (v.sum() + 1e-8)


def make_loader(base_ds, c_maps, extra_names, shuffle):
    ds = ScaleDataset(base_ds, extra_names, c_maps)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print(f"Step 7: temporal-scale training (D1 protocol, causal v2) — "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 64)
print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_FILE} in {DATA_DIR}")

torch.manual_seed(SEED)
np.random.seed(SEED)
loaders = create_scs_dataloaders(
    data_dir=str(DATA_DIR), batch_size=BATCH, num_workers=0,
    random_split=False, random_seed=SEED, lead_weeks=4, year_based_split=True,
    train_years=TRAIN_YEARS, val_years=(2011, 2015), test_years=(2016, 2019),
    dataset_file=DATASET_FILE)
tr_ds, va_ds, te_ds = loaders[0].dataset, loaders[1].dataset, loaders[2].dataset
assert list(tr_ds.dyn_feature_names) == DATASET_ORDER

# ---- C_W maps (full record) + trailing-window verification -------------------
hotspot_full = tr_ds.X_dynamic_full[..., HS_IDX].astype(np.float32)
c_maps = {C_NAME[W]: compute_cumsum_hs1c(hotspot_full, W) for W in WINDOWS}

rng = np.random.default_rng(7)
ver = []
for W in WINDOWS:
    name = C_NAME[W]
    for _ in range(3):
        t = int(rng.integers(W, hotspot_full.shape[0]))
        i = int(rng.integers(0, hotspot_full.shape[1]))
        j = int(rng.integers(0, hotspot_full.shape[2]))
        hs_win = hotspot_full[t - W + 1: t + 1, i, j]
        if np.isfinite(hs_win[-1]):
            manual = float(np.nansum(np.where(np.isfinite(hs_win) & (hs_win >= 1.0), hs_win, 0.0)))
        else:
            manual = np.nan  # convention: NaN where hotspot[t] is NaN
        got = c_maps[name][t, i, j]
        ok = (np.isnan(manual) and np.isnan(got)) or abs(float(got) - manual) < 1e-3
        ver.append(bool(ok))
        # A1/A2/A3: window ends exactly at t and contains no future values
        assert (t - W + 1) >= 0 and t < hotspot_full.shape[0]
trailing_ok = all(ver)
print(f"Trailing-window verification ({len(ver)} spot checks): "
      f"{'PASS' if trailing_ok else 'FAIL'}")
assert trailing_ok

# ---- per-channel train statistics (SCS train 2003-2010 ONLY) -----------------
feat_manifest = {}
for W in WINDOWS:
    name = C_NAME[W]
    tmp = ScaleDataset(tr_ds, [name], c_maps)
    vals = tmp.extra_win[..., 0]
    finite = vals[np.isfinite(vals)]
    feat_manifest[name] = {
        "feature": name, "window_weeks": W,
        "definition": ("trailing W-week sum of weekly 7-day-mean HotSpot with "
                       "values >= 1 degC only; window (t-W+1..t]; NaN weeks "
                       "contribute 0; NaN where hotspot[t] is NaN"),
        "train_mean": float(finite.mean()), "train_std": float(finite.std()),
        "train_min": float(finite.min()), "train_max": float(finite.max()),
        "n_valid_train_obs": int(finite.size),
        "stats_source": "SCS train windows only (2003-2010); no val/test/GBR",
    }
    print(f"  {name}: mean={finite.mean():.4f} std={finite.std():.4f} "
          f"min={finite.min():.3f} max={finite.max():.3f} n={finite.size}")
base_stats = {c: (SCS_CHANNEL_MEAN[DATASET_ORDER.index(c)],
                  SCS_CHANNEL_STD[DATASET_ORDER.index(c)]) for c in BASE_NAMES}


def norm_vectors(extra_names):
    names = BASE_NAMES + [C_NAME[int(k[1:])] for k in extra_names]
    mean = [base_stats[n][0] if n in base_stats else feat_manifest[n]["train_mean"]
            for n in names]
    std = [base_stats[n][1] if n in base_stats else feat_manifest[n]["train_std"]
           for n in names]
    return names, mean, std


# ---- training loop ------------------------------------------------------------
def train_one(name, extra_keys):
    out_dir = WEIGHTS_DIR / name
    ckpt = out_dir / "model.pt"
    log_path = out_dir / "training_log.json"
    if ckpt.exists() and log_path.exists():
        print(f"[{name}] checkpoint already exists — skipping (resume mode)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_names = [C_NAME[Ckey[k]] for k in extra_keys]
    ch_names, mean_v, std_v = norm_vectors(extra_keys)
    n_ch = len(ch_names)

    print(f"\n{'=' * 64}\n[{name}] {n_ch} channels: cumulative={extra_keys}\n{'=' * 64}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr_s = make_loader(tr_ds, c_maps, extra_names, shuffle=True)
    va_s = make_loader(va_ds, c_maps, extra_names, shuffle=False)

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
        "model": name, "channels": ch_names, "cumulative_keys": extra_keys,
        "normalization_mean": mean_v, "normalization_std": std_v,
        "normalization_source": ("non-cumulative: SCS_CHANNEL_MEAN/STD (SCS train "
                                 "2003-2010, frozen); C_W: train-window stats in "
                                 "step7_feature_manifest.json"),
        "dataset_file": DATASET_FILE,
        "protocol": {"epochs_max": EPOCHS, "batch": BATCH, "lr": LR,
                     "weight_decay": 1e-5, "scheduler": "cosine T_max=60 -> 1e-5",
                     "early_stop_patience": PATIENCE, "pos_weight": POS_WEIGHT,
                     "grad_clip": 1.0, "seed": SEED,
                     "loss": "weighted BCE on y_sev=(y_lead>=0.8), valid pixels only",
                     "selection": "best val weighted-BCE epoch",
                     "init": "from scratch (no Step-4/6 checkpoint init)"},
        "stopped_at_epoch": stopped_at, "best_val_bce": float(best_val),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": round(time.time() - t0, 1), "history": hist,
        "timestamp": datetime.now().isoformat(),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  saved {ckpt}  (best val-bce {best_val:.4f}, {log['n_params']} params, "
          f"{time.time()-t0:.0f}s)")


for name, extra_keys in MODELS.items():
    train_one(name, extra_keys)

manifest = {
    "experiment": "step7_temporal_scale",
    "dataset": {"file": DATASET_FILE, "dir": str(DATA_DIR)},
    "channel_definition": {
        "C_channels": {f"C{W}": feat_manifest[C_NAME[W]] for W in WINDOWS},
        "base_channels": BASE_NAMES,
        "replaced": ("hotspot_cumsum_4w / hotspot_cumsum_12w (which accumulate ALL "
                     "HotSpot >= 0) are REPLACED by the C_W channels (HotSpot >= 1 "
                     "degC only, CRW DHW convention); d_hotspot_dt retained"),
    },
    "trailing_verification": {"n_spot_checks": len(ver), "passed": bool(trailing_ok)},
    "models": {name: {"cumulative": keys, "n_channels": len(BASE_NAMES) + len(keys)}
               for name, keys in MODELS.items()},
    "note": ("cumulative HotSpot channels are physically meaningful historical "
             "thermal-stress variables (NOAA CRW DHW = 12-week accumulation of "
             "HotSpot >= 1 degC); this is a temporal-scale decomposition, NOT a "
             "leakage/shortcut claim"),
    "timestamp": datetime.now().isoformat(),
}
with open(OUT_DIR / "step7_feature_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nManifest: {OUT_DIR / 'step7_feature_manifest.json'}")
print("Done.")
