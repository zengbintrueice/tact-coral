# compute_review2_spatial_holdout.py
# Reviewer 2 - Major 1 follow-up: WITHIN-BASIN SPATIAL HOLDOUT within SCS.
#
# Goal: test whether the paper's "thermal coordinate transfers,
# contextual representation is location-dependent" hierarchy holds SPATIALLY
# within a single basin (the alternative the reviewer offered to adding a
# new basin). SCS (source) is the only basin where retraining is permitted
# (GBR/CARIB must stay frozen zero-shot SCS-trained-only).
#
# Design:
#   - Spatial partition by latitude: SOUTH band (rows 0-10, lat <= 13.5 N,
#     abundant severe events) = training region; NORTH band (rows 11-20,
#     lat >= 14.5 N) = within-basin HELD-OUT region (spatially unseen target).
#   - A fresh TSCF (FinalSingleHeadModel, seed 42, weighted-BCE pos_weight 1000,
#     same hyperparameters as the frozen classifier) is trained on SCS 2003-2010
#     with the NORTH cells' labels MASKED OUT of the loss, so the model never
#     sees held-out-region severe signals during training or validation.
#   - Both regions are then evaluated (PFV, test 2016-2019) for PersA
#     (DHW anchor, no training), the spatial-holdout-trained TSCF, the frozen
#     full-SCS TSCF (reference), and TSAR (frozen, reference).
#   - Interpretation: does the DHW anchor retain strong ranking on the
#     spatially-held-out NORTH region (thermal coordinate spatially
#     transferable), and does the contextual model trained without the NORTH
#     region generalise there (contextual spatial generalisation)?
#
# One GPU training run (the training is the dominant cost). Outputs ->
# results/review2_revision/spatial_holdout/.

import io, sys, json, time, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

LTTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LTTS_ROOT))
sys.path.insert(0, str(LTTS_ROOT / "experiments" / "04_structural_ablation"))

from paths import DATA_DIR, RESULTS_DIR
from src.data.aligned_dataset import create_scs_dataloaders, SCSPIECFDataset
import ablate_a_dh_stopgrad as trainer
from physics_anchored_model import PhysicsAnchoredResidualModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = RESULTS_DIR / "review2_revision" / "spatial_holdout"
OUT.mkdir(parents=True, exist_ok=True)
CKPT_FROZEN = LTTS_ROOT / "results" / "model_final_Btail3" / "model.pt"
CKPT_TSAR = RESULTS_DIR / "physics_anchored" / "model_seed_42.pt"

# Spatial partition: rows 0-10 = SOUTH (train/visible); rows 11-20 = NORTH (held out)
H_ROWS = 21
NORTH_FROM_ROW = 11          # rows >= this are the held-out NORTH band
HOLD_OUT_ROW = np.zeros(H_ROWS)   # training mask: 1 = visible (south), 0 = held out (north)
HOLD_OUT_ROW[:NORTH_FROM_ROW] = 1.0
TRAIN_MASK = HOLD_OUT_ROW[:, None].astype(np.float32)   # [H,1] broadcast over W

EPOCHS, BATCH, LR, SEED, SEV = 60, 32, 1e-3, 42, 0.8
POS_WEIGHT = 1000.0
SPLIT = dict(year_based_split=True, train_years=(2003, 2010),
             val_years=(2011, 2015), test_years=(2016, 2019))


def weighted_bce(p, y_sev, valid, pos_weight):
    v = valid.float()
    y_clean = torch.nan_to_num(y_sev.float(), nan=0.0)
    eps = 1e-7
    loss = -(pos_weight * y_clean * torch.log(p.clamp(eps, 1 - eps))
             + (1 - y_clean) * torch.log((1 - p).clamp(eps, 1 - eps)))
    return (loss * v).sum() / (v.sum() + 1e-8)


def predict_sliding(model, ds, device):
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    fc = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    bs = 16
    with torch.no_grad():
        for b in range(0, n - T + 1, bs):
            be = min(b + bs, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            out = model(xd, xs, lm, scale=0.3, tau=0.5, temp=0.1)
            fc[b + T - 1: be + T - 1] = out[0][:, -1].cpu().numpy()
    return fc


def infer_tsar(model, ds, device):
    T, H, W, C = ds.history_weeks, ds.n_lat, ds.n_lon, ds.C_dyn
    n = ds.n_dates
    gt = np.full((n, H, W), np.nan, np.float32)
    rr = np.full((n, H, W), np.nan, np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X = ds.X_dynamic_full
    Y = ds.y_regression_full
    bs = 16
    with torch.no_grad():
        for b in range(0, n - T + 1, bs):
            be = min(b + bs, n - T + 1)
            xd = np.zeros((be - b, T, H, W, C), np.float32)
            for k in range(be - b):
                te = b + k + T - 1
                xd[k] = X[te - T + 1: te + 1]
            xd = torch.FloatTensor(xd).to(device)
            xs = torch.FloatTensor(np.tile(ds.X_static[0], (be - b, 1, 1, 1))).to(device)
            anch = torch.FloatTensor(Y[b + T - 1: be + T - 1].copy()).to(device)
            out = model(xd, xs, lm, anch, scale=1.0, tau=0.5, temp=0.1)
            gt[b + T - 1: be + T - 1] = out[2][:, -1].cpu().numpy()
            rr[b + T - 1: be + T - 1] = out[3][:, -1].cpu().numpy()
    return gt, rr


def collect_records(ds, maps, region_vec):
    """PFV records over test years for cells in region_vec (>0.5) & ocean &
    finite target at t+4. Returns per-method pred arrays + target."""
    T, n = ds.history_weeks, ds.n_dates
    y = ds.y_regression_full
    reg = region_vec.astype(bool)
    rec = {m: [] for m in maps}
    rec["target"] = []
    for t in range(T - 1, n):
        if not (2016 <= int(ds.date_strings[t][:4]) <= 2019):
            continue
        tt = t + 4
        if tt >= n:
            continue
        v = reg & ~ds.land_mask & np.isfinite(y[tt])
        for m in maps:
            v &= np.isfinite(maps[m][t])
        if not v.any():
            continue
        for m in maps:
            rec[m].append(maps[m][t][v])
        rec["target"].append(y[tt][v])
    return {k: (np.concatenate(v) if v else np.array([], np.float32)) for k, v in rec.items()}


def roc_pr(pred, target):
    sev = (target >= SEV).astype(int)
    if len(np.unique(sev)) < 2 or sev.sum() < 2:
        return float("nan"), float("nan"), int(sev.sum())
    return (float(roc_auc_score(sev, pred)),
            float(average_precision_score(sev, pred)), int(sev.sum()))


print("=" * 66)
print(f"Within-SCS spatial holdout (south train / north held-out) - "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"Device: {DEVICE}  Hold-out: rows {NORTH_FROM_ROW}..{H_ROWS - 1} (NORTH band)")
print("=" * 66)

# ---- data + spatial mask ----
loaders = create_scs_dataloaders(data_dir=str(DATA_DIR), batch_size=BATCH,
                                 num_workers=0, random_split=False, random_seed=SEED,
                                 lead_weeks=4, **SPLIT)
tr, va, te = loaders
ds_test = SCSPIECFDataset(data_dir=str(DATA_DIR), mode="test", random_split=False,
                          lead_weeks=4, **SPLIT)
H, W = ds_test.n_lat, ds_test.n_lon
mask_b = (TRAIN_MASK.shape[0] == H) and (TRAIN_MASK[:H].shape == (H,))
assert H == H_ROWS, f"unexpected H={H}"
tmask = torch.tensor(TRAIN_MASK[:H], device=DEVICE)   # [H,1]

print(f"Train mask: rows 0..{NORTH_FROM_ROW - 1} visible (south), "
      f"rows {NORTH_FROM_ROW}..{H - 1} held out (north). Grid {H}x{W}")

# ---- train fresh TSCF with north cells masked ----
torch.manual_seed(SEED); np.random.seed(SEED)
model = trainer.FinalSingleHeadModel(dynamic_channels=13).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
best_val = float("inf"); best_state = None; patience = 0; t0 = time.time()


def batch_valid(batch):
    lm = batch["land_mask"].to(DEVICE)
    lab = batch.get("label_mask")
    lab = lab.to(DEVICE) if lab is not None else (~lm).float()
    y = batch["y_lead"].to(DEVICE)
    valid = torch.isfinite(y).float() * lab.float()
    valid = valid * tmask.unsqueeze(0)      # [B,H,W] * [1,H,1] -> masks north cells
    return valid, (y >= SEV).float(), y


for ep in range(1, EPOCHS + 1):
    model.train()
    for batch in tr:
        xd = batch["x_dynamic"].to(DEVICE); xs = batch["x_static"].to(DEVICE)
        nm = batch.get("nan_mask_dyn"); nm = nm.to(DEVICE) if nm is not None else None
        lm = batch["land_mask"].to(DEVICE)
        valid, y_sev, _ = batch_valid(batch)
        opt.zero_grad()
        out = model(xd, xs, lm, nan_mask=nm)
        loss = weighted_bce(out[0][:, -1], y_sev, valid, torch.tensor(POS_WEIGHT, device=DEVICE))
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    model.eval()
    vl, nv = 0.0, 0
    with torch.no_grad():
        for batch in va:
            xd = batch["x_dynamic"].to(DEVICE); xs = batch["x_static"].to(DEVICE)
            nm = batch.get("nan_mask_dyn"); nm = nm.to(DEVICE) if nm is not None else None
            lm = batch["land_mask"].to(DEVICE)
            valid, y_sev, _ = batch_valid(batch)
            out = model(xd, xs, lm, nan_mask=nm)
            vl += weighted_bce(out[0][:, -1], y_sev, valid,
                               torch.tensor(POS_WEIGHT, device=DEVICE)).item(); nv += 1
    sched.step()
    v = vl / max(nv, 1)
    if v < best_val:
        best_val = v; patience = 0; best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
    else:
        patience += 1
    if ep % 15 == 0 or ep == EPOCHS:
        print(f"  ep {ep:3d} val-bce(south)={v:.4f}  t={time.time() - t0:.0f}s", flush=True)
    if patience >= 10:
        break
model.load_state_dict(best_state); model.eval()
torch.save(model.state_dict(), OUT / "tscf_spatial_holdout.pt")
print(f"Trained spatial-holdout TSCF ({time.time() - t0 / 60:.0f} min) -> {OUT / 'tscf_spatial_holdout.pt'}", flush=True)

# ---- inference maps ----
print("Inference (PersA / spatial-TSCF / frozen-TSCF / TSAR)...", flush=True)
persa = ds_test.y_regression_full.astype(np.float32)
map_hold = predict_sliding(model, ds_test, DEVICE); del model
frozen = trainer.FinalSingleHeadModel(dynamic_channels=13)
frozen.load_state_dict(torch.load(CKPT_FROZEN, map_location="cpu", weights_only=False))
frozen.eval().to(DEVICE); map_frozen = predict_sliding(frozen, ds_test, DEVICE); del frozen
tsar_m = PhysicsAnchoredResidualModel(dynamic_channels=13)
tsar_m.load_state_dict(torch.load(CKPT_TSAR, map_location="cpu", weights_only=False))
tsar_m.eval().to(DEVICE); gt, rr = infer_tsar(tsar_m, ds_test, DEVICE); del tsar_m
map_tsar = (persa + gt * rr * 0.4).astype(np.float32)
del gt, rr
torch.cuda.empty_cache()

maps = {"PersA": persa, "spatial_TSCF": map_hold,
        "frozen_TSCF": map_frozen, "TSAR": map_tsar}

south_vec = np.zeros((H, W), np.float32); south_vec[:NORTH_FROM_ROW, :] = 1.0
north_vec = np.zeros((H, W), np.float32); north_vec[NORTH_FROM_ROW:, :] = 1.0

rows = []
for region, rvec in (("south (train-visible)", south_vec), ("north (held-out)", north_vec)):
    rec = collect_records(ds_test, maps, rvec)
    for m in maps:
        roc, pr, nsev = roc_pr(rec[m], rec["target"])
        print(f"  [{region}] {m:12s} ROC={roc:.4f} PR={pr:.4f} n={len(rec['target'])} n_sev={nsev}", flush=True)
        rows.append({"region": region, "method": m, "roc_auc": roc, "pr_auc": pr,
                     "n": int(len(rec["target"])), "n_severe": int(nsev)})

import pandas as pd
res = pd.DataFrame(rows)
res.to_csv(OUT / "spatial_holdout_metrics.csv", index=False)
out = {"experiment": "within_SCS_spatial_holdout", "device": str(DEVICE),
       "split": {"train_region": "south rows 0-10 (lat<=13.5N)",
                 "held_out_region": "north rows 11-20 (lat>=14.5N)",
                 "train_mask": "north cells masked from loss/val during training"},
       "training": {"seed": SEED, "pos_weight": POS_WEIGHT, "epochs": EPOCHS,
                    "patience": 10}, "metrics": rows,
       "note": ("PFV test 2016-2019, severe y>=0.8; spatial-TSCF trained on "
                "south only (north unseen); frozen-TSCF trained on full SCS "
                "(reference); TSAR frozen (reference)."),
       "generated": datetime.now().isoformat()}
with open(OUT / "spatial_holdout_metrics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print("DONE ->", OUT)