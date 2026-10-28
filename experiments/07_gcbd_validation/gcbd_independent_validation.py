# ==========================================================================
# Source: paper/a7_dual_output/gcbd_independent_validation.py (verbatim except path fix)
# Experiment: Independent GCBD validation (leakage-free, self-contained)
# Paper: Results 5, Fig. 6, Supplementary Table S7
# Purpose: distribution shift, AUROC, Precision@K, decile stratification on 1,898 matched obs
# Output: results/experiment8_v2_stats.json, results/experiment8_v2_figure.png
# ==========================================================================

# -*- coding: utf-8 -*-
"""gcbd_independent_validation.py — GCBD ecological validation (v2) SELF-CONTAINED

Replaces the original RR enrichment analysis (v1) with four NC-appropriate analyses:
  Exp 8.1 — Distribution shift (violin + Mann-Whitney U + Cliff's δ)
  Exp 8.2 — AUROC (any bleaching / severe ≥20%)
  Exp 8.3 — Precision@K (Top 5/10/20%) vs random baseline
  Exp 8.4 — Decile calibration + isotonic/Loess trend (no RR)

SELF-CONTAINED: does NOT import experiment8_gcbd_validation (v1). All matching
logic (load_scs_coords / restore_a5 / predict_sliding_temporal / match_gcbd_to_model)
is inlined here. Leakage-free: test-window filter (2016-2019) + lead-time alignment
(obs at D matched to forecast issued at D - LEAD_WEEKS, which targets D).

Outputs:
  - results/experiment8_v2_stats.json     ← all numerical results
  - results/experiment8_v2_figure.png      ← 4-panel figure
"""
import os, sys, io, json, warnings, importlib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import PIECF_ROOT, DATA_DIR, RESULTS_DIR, A6_WEIGHT
from src.data.aligned_dataset import SCSPIECFDataset

# ── Config ──────────────────────────────────────────────
SCS_DATASET_FILE = "scs_coral_bleaching_dataset_v9_1deg_raw.nc"
A5_MODEL = A6_WEIGHT
A5_CFG = {"gamma": 0.5, "scale": 0.3, "tau": 0.5, "temp": 0.1, "lambda_sev": 20}
LEAD_WEEKS = 4
SPATIAL_BUFFER_DEG = 0.5
TEMPORAL_WINDOW_WEEKS = 4
TEST_YEAR_MIN = 2016
TEST_YEAR_MAX = 2019
SEVERE_THRESHOLD = 0.8
_gcbd_env = os.environ.get("GCBD_CSV", "")
GCBD_PATH = Path(_gcbd_env) if _gcbd_env else (Path(__file__).resolve().parents[2] / "data" / "china_bco_dmo_bleaching.csv")


# ═══════════════════════════════════════════════════════════════
# Inlined matching logic (independent of experiment8 v1)
# ═══════════════════════════════════════════════════════════════
def load_scs_coords(data_dir):
    """Load lat/lon/time from SCS NetCDF (full time axis)."""
    ds = xr.open_dataset(os.path.join(data_dir, SCS_DATASET_FILE))
    lats = ds['lat'].values.astype(float)
    lons = ds['lon'].values.astype(float)
    times = pd.DatetimeIndex(ds['time'].values)
    ds.close()
    return lats, lons, times


def restore_a5():
    """Load a6 (sg_lam20) model."""
    mod = importlib.import_module("ablate_a_dh_stopgrad")
    model = mod.FinalSingleHeadModel(dynamic_channels=13)
    sd = torch.load(A5_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_sliding_temporal(model, ds, device):
    """Sliding-window inference -> forecast_map [n_dates, H, W].
    forecast_map[t] = prediction for week t + LEAD_WEEKS, from input window [t-T+1, t]."""
    T = ds.history_weeks
    H, W = ds.n_lat, ds.n_lon
    n_dates = ds.n_dates
    C_dyn = ds.C_dyn
    forecast_map = np.full((n_dates, H, W), np.nan, dtype=np.float32)
    lm = torch.BoolTensor(ds.land_mask).to(device)
    X_full = ds.X_dynamic_full
    bs = 8
    n_valid = n_dates - T + 1
    t_start_valid = T - 1
    with torch.no_grad():
        for start_batch in range(0, n_valid, bs):
            batch_end = min(start_batch + bs, n_valid)
            bs_actual = batch_end - start_batch
            xd_batch = np.zeros((bs_actual, T, H, W, C_dyn), dtype=np.float32)
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                start_idx = t_end - T + 1
                xd_batch[k] = X_full[start_idx:t_end + 1]
            xd = torch.FloatTensor(xd_batch).to(device)
            xs_t = torch.FloatTensor(np.tile(ds.X_static[0], (bs_actual, 1, 1, 1))).to(device)
            out = model(xd, xs_t, lm, scale=A5_CFG["scale"],
                        tau=A5_CFG["tau"], temp=A5_CFG["temp"])
            preds = out[0][:, -1, :, :].cpu().numpy()
            for k in range(bs_actual):
                t_end = t_start_valid + start_batch + k
                forecast_map[t_end] = preds[k]
    return forecast_map


def match_gcbd_to_model(row, forecast_map, lats, lons, times, valid_time_mask):
    """Match GCBD obs to forecast with CORRECT lead-time alignment + test filter.

    For an observation at date D, use the forecast ISSUED at D - LEAD_WEEKS
    (which targets D). Using forecast_map[D] would compare obs(D) to pred(D+LEAD)
    — thermal autocorrelation. Only test-window forecast indices are eligible.
    Returns {'risk_at': pred} or None.
    """
    try:
        lat = float(row['Latitude_Degrees'])
        lon = float(row['Longitude_Degrees'])
        year = int(row['Date_Year'])
        month = int(row['Date_Month'])
        day = int(row.get('Date_Day', 15))
        if not (1 <= month <= 12):
            return None
        day = min(max(day, 1), 28)
        date = pd.Timestamp(year, month, day)
    except Exception:
        return None
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))
    if abs(lats[lat_idx] - lat) > SPATIAL_BUFFER_DEG or abs(lons[lon_idx] - lon) > SPATIAL_BUFFER_DEG:
        return None
    target_issue_date = date - pd.Timedelta(weeks=LEAD_WEEKS)
    time_diffs = np.abs(times - target_issue_date)
    week_window = pd.Timedelta(weeks=TEMPORAL_WINDOW_WEEKS)
    in_window = time_diffs <= week_window
    valid_and_in_window = in_window & valid_time_mask
    if not valid_and_in_window.any():
        return None
    time_diffs_float = time_diffs.values.astype(np.float64)
    masked_diffs = np.where(valid_and_in_window, time_diffs_float, np.inf)
    nearest_issue_idx = int(np.argmin(masked_diffs))
    risk_at = forecast_map[nearest_issue_idx, lat_idx, lon_idx]
    if not np.isfinite(risk_at):
        return None
    return {'risk_at': float(risk_at)}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("GCBD Validation v2 — leakage-free ecological analyses (SELF-CONTAINED)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device} | Weight: {A5_MODEL}")

print("\n[1] Loading SCS test set + a6 sliding-window inference...")
ds_scs = SCSPIECFDataset(
    data_dir=str(DATA_DIR), mode="test", random_split=False,
    lead_weeks=LEAD_WEEKS, year_based_split=True,
    train_years=(2003, 2010), val_years=(2011, 2015), test_years=(2016, 2019),
)
H, W = ds_scs.n_lat, ds_scs.n_lon
lats, lons, times = load_scs_coords(DATA_DIR)
n_dates = ds_scs.n_dates
print(f"  SCS grid: {H}x{W}, n_dates={n_dates}")

model = restore_a5().to(device)
forecast_map = predict_sliding_temporal(model, ds_scs, device)
print(f"  Forecast map: {forecast_map.shape}")

# test-window mask (ONLY 2016-2019 forecast indices — no train/val leakage)
times_year = times.year
valid_time_mask = (times_year >= TEST_YEAR_MIN) & (times_year <= TEST_YEAR_MAX)
print(f"  test window: {int(valid_time_mask.sum())}/{len(times)} forecast indices")

print("\n[2] Loading GCBD + matching (lead-time aligned, test-only)...")
df = pd.read_csv(GCBD_PATH, low_memory=False)
df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')
print(f"  GCBD total records: {len(df)}")

def match_to_pred(row):
    res = match_gcbd_to_model(row, forecast_map, lats, lons, times, valid_time_mask)
    if res is None:
        return None
    pb = row['Percent_Bleaching']
    if pd.isna(pb):
        return None
    return {'pred': res['risk_at'], 'pb': float(pb)}

df['match'] = df.apply(match_to_pred, axis=1)
valid = df[df['match'].notna()].copy()
valid['pred'] = valid['match'].apply(lambda m: m['pred'])
valid['pb'] = valid['match'].apply(lambda m: m['pb'])
valid = valid[(valid['pb'] >= 0) & (valid['pb'] <= 100)].copy()
valid['any_bleach'] = (valid['pb'] > 0).astype(int)
valid['severe_20'] = (valid['pb'] >= 20).astype(int)
valid['severe_50'] = (valid['pb'] >= 50).astype(int)
print(f"  Matched: {len(valid)} GCBD records with valid test-window predictions")

pred = valid['pred'].values
pb = valid['pb'].values
any_bleach = valid['any_bleach'].values
severe_20 = valid['severe_20'].values
severe_50 = valid['severe_50'].values


# ═══════════════════════════════════════════════════════════════
# Exp 8.1 — Distribution shift (violin plot + Mann-Whitney + Cliff's δ)
# ═══════════════════════════════════════════════════════════════
print("\n[Exp 8.1] Distribution shift: bleached vs background...")

bleached_pred = pred[any_bleach == 1]
background_pred = pred[any_bleach == 0]

u_stat, p_mw = mannwhitneyu(bleached_pred, background_pred, alternative='greater')

# Cliff's delta
def cliffs_delta(x, y):
    """Cliff's delta: dominance measure."""
    n = 0
    d = 0
    for xi in x:
        for yj in y:
            n += 1
            if xi > yj:
                d += 1
            elif xi < yj:
                d -= 1
    return d / n

# Vectorized version for speed
def cliffs_delta_fast(x, y):
    nx, ny = len(x), len(y)
    diff = np.subtract.outer(x, y)
    delta = (np.sum(diff > 0) - np.sum(diff < 0)) / (nx * ny)
    return float(delta)

delta_any = cliffs_delta_fast(bleached_pred, background_pred)
delta_sev20 = cliffs_delta_fast(pred[severe_20 == 1], pred[severe_20 == 0])

print(f"  Bleached n={len(bleached_pred)}, Background n={len(background_pred)}")
print(f"  Mann-Whitney U p-value (any bleach): {p_mw:.4e}")
print(f"  Cliff's δ (any bleach): {delta_any:.4f}")
print(f"  Cliff's δ (severe ≥20%): {delta_sev20:.4f}")


# ═══════════════════════════════════════════════════════════════
# Exp 8.2 — AUROC (binary: any bleaching)
# ═══════════════════════════════════════════════════════════════
print("\n[Exp 8.2] AUROC for any-bleach discrimination...")
auroc_any = roc_auc_score(any_bleach, pred)
auroc_sev20 = roc_auc_score(severe_20, pred) if severe_20.sum() > 5 else float('nan')
fpr, tpr, _ = roc_curve(any_bleach, pred)
print(f"  AUROC (any bleach > 0%): {auroc_any:.4f}")
print(f"  AUROC (severe ≥ 20%):    {auroc_sev20:.4f}")


# ═══════════════════════════════════════════════════════════════
# Exp 8.3 — Precision@K
# ═══════════════════════════════════════════════════════════════
print("\n[Exp 8.3] Precision@K (operational value)...")

n_total = len(pred)
sorted_idx = np.argsort(-pred)  # descending
random_rate = any_bleach.mean()

precision_at_k = {}
for k_pct in [5, 10, 20]:
    k = int(n_total * k_pct / 100)
    top_k_idx = sorted_idx[:k]
    precision = any_bleach[top_k_idx].mean()
    lift = precision / random_rate if random_rate > 0 else float('nan')
    precision_at_k[k_pct] = {
        'k_n': k,
        'precision': float(precision),
        'lift_vs_random': float(lift),
    }
    print(f"  Top {k_pct}%: precision={precision:.4f} (lift={lift:.2f}x vs random={random_rate:.4f})")


# ═══════════════════════════════════════════════════════════════
# Exp 8.4 — Decile calibration + Loess trend
# ═══════════════════════════════════════════════════════════════
print("\n[Exp 8.4] Decile calibration + Loess trend...")

decile_edges = np.quantile(pred, np.linspace(0, 1, 11))
decile_bins = np.digitize(pred, decile_edges[1:-1])  # 0..9

decile_stats = []
for d in range(10):
    mask = decile_bins == d
    if mask.sum() < 5:
        continue
    decile_stats.append({
        'decile': d + 1,
        'n': int(mask.sum()),
        'pred_mean': float(pred[mask].mean()),
        'pred_range': [float(np.quantile(pred[mask], 0.05)), float(np.quantile(pred[mask], 0.95))],
        'obs_rate': float(any_bleach[mask].mean()),
        'obs_severe_rate': float(severe_20[mask].mean()) if severe_20[mask].sum() > 0 else 0.0,
    })

for ds in decile_stats:
    print(f"  Decile {ds['decile']}: pred_mean={ds['pred_mean']:.3f}, "
          f"obs_rate={ds['obs_rate']:.4f}, n={ds['n']}")


# ═══════════════════════════════════════════════════════════════
# Generate 4-panel figure
# ═══════════════════════════════════════════════════════════════
print("\n[Figure] Generating 4-panel figure...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('white')

# ── Panel 8.1: Violin plot ───────────────────────────────────
ax = axes[0, 0]
data_v = [background_pred, bleached_pred]
pos = [1, 2]
parts = ax.violinplot(data_v, positions=pos, widths=0.7, showmeans=False,
                       showmedians=True, showextrema=False)
for pc, color in zip(parts['bodies'], ['#5DADE2', '#E74C3C']):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)

ax.scatter([1]*len(background_pred) + [2]*len(bleached_pred),
           list(background_pred) + list(bleached_pred),
           alpha=0.15, s=8, c='gray', zorder=2)

ax.set_xticks(pos)
ax.set_xticklabels([f'Background\n(n={len(background_pred):,})',
                    f'Any bleaching\n(n={len(bleached_pred):,})'], fontsize=9)
ax.set_ylabel('Predicted risk score', fontsize=10)
ax.set_title(f'(8.1) Distribution shift\n'
             f'Mann-Whitney p={p_mw:.2e}, Cliff\'s δ={delta_any:.3f}',
             fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# ── Panel 8.2: AUROC ─────────────────────────────────────────
ax = axes[0, 1]
ax.plot(fpr, tpr, color='#2E86C1', linewidth=2.2, label=f'AUROC = {auroc_any:.3f}')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.5)')
ax.fill_between(fpr, tpr, alpha=0.15, color='#2E86C1')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title(f'(8.2) ROC: predicted risk → any bleaching\n'
             f'(severe≥20% AUROC = {auroc_sev20:.3f})',
             fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# ── Panel 8.3: Precision@K ───────────────────────────────────
ax = axes[1, 0]
ks = [5, 10, 20]
precisions = [precision_at_k[k]['precision'] for k in ks]
lifts = [precision_at_k[k]['lift_vs_random'] for k in ks]
x_pos = np.arange(len(ks))
bars = ax.bar(x_pos, precisions, color=['#C0392B', '#E67E22', '#F1C40F'],
              edgecolor='black', linewidth=1.2, alpha=0.85)
ax.axhline(y=random_rate, color='gray', linestyle='--', linewidth=1.2,
           label=f'Random baseline = {random_rate:.3f}')
for i, (bar, p, l) in enumerate(zip(bars, precisions, lifts)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{p:.3f}\n({l:.1f}×)', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Top {k}%\n(n={precision_at_k[k]["k_n"]})' for k in ks], fontsize=9)
ax.set_ylabel('Precision (observed bleaching rate)', fontsize=10)
ax.set_title('(8.3) Precision@K — operational value',
             fontsize=10, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(precisions) * 1.4)

# ── Panel 8.4: Decile calibration + Loess ────────────────────
ax = axes[1, 1]
decile_pred = [d['pred_mean'] for d in decile_stats]
decile_obs = [d['obs_rate'] for d in decile_stats]
decile_sev = [d['obs_severe_rate'] for d in decile_stats]

ax.scatter(decile_pred, decile_obs, c='#2C3E50', s=80, alpha=0.85,
           edgecolors='white', linewidth=1.5,
           label='Any bleaching (per decile)', zorder=5)
ax.scatter(decile_pred, decile_sev, c='#C0392B', s=80, alpha=0.85,
           edgecolors='white', linewidth=1.5, marker='D',
           label='Severe ≥20% (per decile)', zorder=5)

# Isotonic (monotonic) trend as a non-parametric smoother
iso_any = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
iso_any.fit(decile_pred, decile_obs)
xx = np.linspace(min(decile_pred), max(decile_pred), 100)
ax.plot(xx, iso_any.predict(xx), color='#2C3E50', linewidth=2.0,
        linestyle='-', alpha=0.7, label='Isotonic trend (any)')

iso_sev = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
iso_sev.fit(decile_pred, decile_sev)
ax.plot(xx, iso_sev.predict(xx), color='#C0392B', linewidth=2.0,
        linestyle='--', alpha=0.7, label='Isotonic trend (severe)')

# Identity reference (perfect calibration if pred == obs_rate)
ax.plot([0, 1], [0, 1], ':', color='gray', alpha=0.5, label='Identity')

ax.set_xlabel('Mean predicted risk (per decile)', fontsize=10)
ax.set_ylabel('Observed bleaching rate', fontsize=10)
ax.set_title('(8.4) Decile calibration\n(trend non-parametric, not RR)',
             fontsize=10, fontweight='bold')
ax.legend(loc='upper left', fontsize=7.5)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, max(max(decile_obs), max(decile_sev)) * 1.4 + 0.02)

plt.tight_layout()
fig_path = RESULTS_DIR / "gcbd/gcbd_independent_validation.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved figure: {fig_path}")


# ═══════════════════════════════════════════════════════════════
# Save stats JSON
# ═══════════════════════════════════════════════════════════════
stats = {
    "experiment": "gcbd_validation_v2",
    "description": "Leakage-free GCBD validation: distribution shift + AUROC + Precision@K + decile calibration",
    "data_scope": "SCS test 2016-2019 + GCBD observations",
    "self_contained": True,
    "no_v1_dependency": True,
    "n_matched": int(len(valid)),
    "n_any_bleach": int(any_bleach.sum()),
    "n_severe_20": int(severe_20.sum()),
    "n_severe_50": int(severe_50.sum()),
    "random_baseline_bleach_rate": float(random_rate),
    "exp_8_1_distribution_shift": {
        "mannwhitney_p_any": float(p_mw),
        "mannwhitney_u_stat": float(u_stat),
        "cliffs_delta_any": float(delta_any),
        "cliffs_delta_severe_20": float(delta_sev20),
        "mean_pred_background": float(background_pred.mean()),
        "mean_pred_bleached": float(bleached_pred.mean()),
    },
    "exp_8_2_auroc": {
        "auroc_any_bleach": float(auroc_any),
        "auroc_severe_20": float(auroc_sev20),
    },
    "exp_8_3_precision_at_k": precision_at_k,
    "exp_8_4_decile_calibration": decile_stats,
    "interpretation": {
        "exp_8_1": "Predicted risk is significantly higher at bleached sites vs background (Mann-Whitney). Cliff's δ indicates effect size.",
        "exp_8_2": f"AUROC={auroc_any:.3f} measures binary discrimination (any bleaching > 0).",
        "exp_8_3": "Precision@K shows operational value: enrichment above random baseline when selecting top-K% pixels.",
        "exp_8_4": "Decile trend shows monotonicity (or lack thereof) between predicted risk and observed bleaching; isotonic smoothing is non-parametric.",
    },
}
stats_path = RESULTS_DIR / "gcbd/gcbd_independent_validation.json"
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
print(f"  Saved stats: {stats_path}")

# ── Save raw distribution data for violin plot ──────────────
dist_path = RESULTS_DIR / "gcbd/gcbd_violin_distribution.npz"
np.savez_compressed(dist_path,
                    background_pred=background_pred,
                    bleached_pred=bleached_pred)
print(f"  Saved violin distribution: {dist_path}")
print(f"    background_pred: n={len(background_pred)}, "
      f"mean={float(background_pred.mean()):.4f}, "
      f"range=[{float(background_pred.min()):.4f}, {float(background_pred.max()):.4f}]")
print(f"    bleached_pred:   n={len(bleached_pred)}, "
      f"mean={float(bleached_pred.mean()):.4f}, "
      f"range=[{float(bleached_pred.min()):.4f}, {float(bleached_pred.max()):.4f}]")

print("\n" + "=" * 60)
print("DONE — All 4 leakage-free GCBD analyses complete (self-contained)")
print("=" * 60)
