# -*- coding: utf-8 -*-
"""
Jensen-Aware DHW Target Construction
=====================================

Implements the Jensen-aware Degree Heating Week (DHW) target as described in
the PIECF-a6 paper. This module makes the target construction **explicit and
reusable**, extracting the logic that was previously embedded in the monolithic
``build_gbr_1deg_from_raw.py`` / ``build_v9_1deg_from_raw.py`` scripts.

Jensen's Inequality & Why Order Matters
---------------------------------------
DHW is a **non-linear** function (clip + rolling sum). Computing it at native
resolution BEFORE spatial aggregation preserves thermal-stress peaks that would
be smoothed out by prior block-averaging:

  Correct (Jensen-aware):  DHW_1deg = mean( compute_DHW_5km(Hotspot_5km) )
  Wrong (naive):           DHW_1deg = compute_DHW_1deg( mean(Hotspot_5km) )

The former is ~10–20 % higher at peak stress events because sub-degree hotspots
are not diluted by cooler surrounding pixels.

Pipeline
--------
1. ``compute_dhw_5km``        — clip hotspot ≥ 0, 84-day rolling sum at native 5 km.
2. ``build_aggregation_matrix`` — construct the 5 km → 1° block-averaging matrix.
3. ``aggregate_to_1deg``       — apply the matrix to get mean DHW per 1° cell.
4. ``dhw_to_target``          — convert degree-weeks → [0, 1] regression target.
5. ``build_jensen_target``     — end-to-end convenience wrapper (chains 1→4).

Reference: CRW (Coral Reef Watch) DHW definition —
           DHW = Σ max(Hotspot, 0) over trailing 12 weeks (84 days).
"""
import numpy as np
from scipy.sparse import csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Constants (CRW standard)
# ─────────────────────────────────────────────────────────────────────────────
DHW_WINDOW_DAYS = 84            # 12 weeks × 7 days
DHW_CLIP_LOWER = 1.0            # CRW standard: only HotSpot ≥ 1 °C contributes to DHW
TARGET_SCALE = 10.0             # y_regression = clip(dhw_weekly / 10, 0, 1)
WARNING_SCALE = 8.0             # y_warning    = clip(dhw_weekly /  8, 0, 1)
CLASSIFICATION_THRESHOLD = 4.0  # DHW ≥ 4 °C-weeks → bleaching warning


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Native-5km rolling DHW
# ─────────────────────────────────────────────────────────────────────────────
def compute_dhw_5km(hotspot_5km, window_days=DHW_WINDOW_DAYS,
                    require_full_window=True):
    """Compute DHW at native 5 km resolution (CRW standard).

    DHW(t) = Σ_{d=t-W+1}^{t} H_d · 1(H_d ≥ 1 °C)

    NOAA Coral Reef Watch defines DHW as the accumulation, over the past 12
    weeks (84 days), of daily HotSpot values that reach or exceed 1 °C. Days
    with HotSpot < 1 °C (or missing) contribute 0; they do not subtract.

    Parameters
    ----------
    hotspot_5km : np.ndarray, shape (n_days, n_lat, n_lon)
        Daily CRW Hotspot field at native 5 km resolution.
        May contain NaN (land / no-data).
    window_days : int, default 84
        Rolling-sum window in days (CRW standard = 84 = 12 weeks).
    require_full_window : bool, default True
        If True, only pixels with all ``window_days`` valid observations
        receive a DHW value; partial windows are set to NaN. Treating a
        missing day as 0 artificially lowers DHW and biases the target low.

    Returns
    -------
    dhw_5km : np.ndarray, shape (n_days, n_lat, n_lon)
        DHW at native 5 km resolution (degree-Celsius · weeks).
        Pixels without a full valid window are NaN.
    """
    n_days = hotspot_5km.shape[0]
    dhw_5km = np.full_like(hotspot_5km, np.nan, dtype=np.float32)

    # CRW clip: only HotSpot ≥ 1 °C contributes (value kept, not excess)
    hs_clipped = np.where(np.isfinite(hotspot_5km) & (hotspot_5km >= DHW_CLIP_LOWER),
                          hotspot_5km, 0.0)
    hs_nan = np.isfinite(hotspot_5km)

    # Rolling sum via cumulative-sum trick (O(n) vs O(n·W)).
    # Use float64 for the cumulative sum to avoid float32 accumulation error
    # (the original rolling-buffer sums only W values; naive cumsum over
    # all n_days loses precision when the tail is subtracted).
    cumsum = np.cumsum(hs_clipped.astype(np.float64), axis=0)
    cumcnt = np.cumsum(hs_nan.astype(np.int32), axis=0)

    # First window_days-1 entries: partial window (will be NaN if full window required)
    dhw_5km[:window_days] = cumsum[:window_days].astype(np.float32)

    # Full window: subtract the tail
    if n_days > window_days:
        dhw_5km[window_days:] = (cumsum[window_days:] -
                                  cumsum[:-window_days]).astype(np.float32)
        valid_cnt = np.empty_like(cumcnt)
        valid_cnt[:window_days] = cumcnt[:window_days]
        valid_cnt[window_days:] = cumcnt[window_days:] - cumcnt[:-window_days]
    else:
        valid_cnt = cumcnt

    # CRW-standard validity: require the FULL window of valid observations.
    # Partial windows treat missing days as 0 and would bias DHW low.
    if require_full_window:
        dhw_5km[valid_cnt < window_days] = np.nan
    else:
        dhw_5km[valid_cnt == 0] = np.nan
    return dhw_5km


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build the 5 km → 1° aggregation matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_aggregation_matrix(crw_lat, crw_lon, lat_edges, lon_edges):
    """Build a sparse block-averaging matrix from 5 km pixels to 1° cells.

    For each 1° cell, finds all 5 km pixels whose centers fall inside the cell
    and constructs a CSR matrix such that::

        dhw_1deg[cell_k] = mean( dhw_5km[pixels_in_cell] )
                          = (M · dhw_5km_flat)[k] / cell_size[k]

    Parameters
    ----------
    crw_lat : np.ndarray, 1-D
        5 km pixel latitudes. May be ascending (e.g., GBR: -23.5..-9.5) or
        descending (e.g., SCS/CRW: 24.025..3.025). The function auto-detects
        the order and applies the appropriate flip to match the data array
        storage convention (row 0 = southernmost).
    crw_lon : np.ndarray, 1-D
        5 km pixel longitudes (ascending).
    lat_edges : np.ndarray, 1-D
        Latitude edges of 1° cells (len = n_lat + 1).
    lon_edges : np.ndarray, 1-D
        Longitude edges of 1° cells (len = n_lon + 1).

    Returns
    -------
    M_csr : scipy.sparse.csr_matrix, shape (n_cells, n_5km_pixels)
        Unnormalized row-selector matrix. Divide output by ``cell_size`` for
        the mean. ``n_cells = n_lat * n_lon``.
    cell_size : np.ndarray, shape (n_cells,)
        Number of 5 km pixels inside each 1° cell.
    """
    n_lat_px = len(crw_lat)
    n_lon_px = len(crw_lon)
    n_lat = len(lat_edges) - 1
    n_lon = len(lon_edges) - 1
    n_5km = n_lat_px * n_lon_px

    # Pixel-in-cell membership: 1-D masks per axis (coordinate space)
    lat_in = (crw_lat[:, None] >= lat_edges[:-1]) & (crw_lat[:, None] < lat_edges[1:])  # (n_lat_px, n_lat)
    lon_in = (crw_lon[:, None] >= lon_edges[:-1]) & (crw_lon[:, None] < lon_edges[1:])  # (n_lon_px, n_lon)

    # Handle descending-latitude convention (e.g., CRW SCS: lat 24.025→3.025).
    # The coordinate array is descending but the data array is stored ascending
    # (row 0 = southernmost). The flip converts coordinate-space indices to
    # array-space indices, matching the original build_regrid_mask behaviour.
    lat_descending = crw_lat[0] > crw_lat[-1]
    if lat_descending:
        lat_in = lat_in[::-1]

    rows, cols = [], []
    for i in range(n_lat):
        for j in range(n_lon):
            cell_k = i * n_lon + j
            # Outer product: pixel (a, b) is in cell (i, j) iff lat_in[a, i] AND lon_in[b, j]
            lat_idx = np.where(lat_in[:, i])[0]
            lon_idx = np.where(lon_in[:, j])[0]
            # Cartesian product of matching lat and lon pixels
            ii, jj = np.meshgrid(lat_idx, lon_idx, indexing='ij')
            pix = np.ravel_multi_index([ii.ravel(), jj.ravel()], (n_lat_px, n_lon_px))
            rows.extend([cell_k] * len(pix))
            cols.extend(pix.tolist())

    M_csr = csr_matrix((np.ones(len(rows), dtype=np.float32),
                        (rows, cols)),
                       shape=(n_lat * n_lon, n_5km))
    cell_size = np.asarray(M_csr.sum(axis=1)).ravel().astype(np.float32)
    return M_csr, cell_size


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Apply aggregation — 5 km → 1°
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_to_1deg(dhw_5km_flat, M_csr, cell_size):
    """Block-average a single-day 5 km DHW field to 1° cells.

    Parameters
    ----------
    dhw_5km_flat : np.ndarray, shape (n_5km_pixels,)
        Flattened 5 km DHW field for one day. NaN = no data.
    M_csr : scipy.sparse.csr_matrix, shape (n_cells, n_5km_pixels)
        Aggregation matrix from ``build_aggregation_matrix``.
    cell_size : np.ndarray, shape (n_cells,)
        Pixels-per-cell counts.

    Returns
    -------
    dhw_1deg : np.ndarray, shape (n_cells,)
        Mean DHW within each 1° cell. Cells with no valid 5 km data are NaN.
    """
    valid = np.isfinite(dhw_5km_flat)
    tmp = dhw_5km_flat.copy()
    tmp[~valid] = 0.0

    cell_sum   = np.asarray(M_csr.dot(tmp)).ravel()
    cell_valid = np.asarray(M_csr.dot(valid.astype(np.float32))).ravel()

    dhw_1deg = np.full(M_csr.shape[0], np.nan, dtype=np.float32)
    ok = cell_valid > 0
    dhw_1deg[ok] = (cell_sum[ok] / cell_valid[ok]).astype(np.float32)
    return dhw_1deg


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: DHW → regression / warning / classification targets
# ─────────────────────────────────────────────────────────────────────────────
def dhw_to_target(dhw_1deg_weekly, scale=TARGET_SCALE):
    """Convert weekly DHW (°C · weeks) to a regression target in [0, 1].

    y = clip( dhw_weekly / scale, 0, 1 )

    The default ``scale = 10`` maps the typical DHW range [0, 10+] onto
    [0, 1], where y ≈ 1 corresponds to severe bleaching risk
    (DHW ≥ 10 °C-weeks).

    Parameters
    ----------
    dhw_1deg_weekly : np.ndarray
        Weekly DHW at 1° resolution (degree-weeks).
    scale : float, default 10.0
        Normalisation factor (degree-weeks per unit target).

    Returns
    -------
    y : np.ndarray, same shape as input
        Regression target in [0, 1].
    """
    return np.clip(dhw_1deg_weekly / float(scale), 0.0, 1.0)


def dhw_to_warning(dhw_1deg_weekly, scale=WARNING_SCALE):
    """Convert DHW to a warning-level target (more sensitive than regression)."""
    return np.clip(dhw_1deg_weekly / float(scale), 0.0, 1.0)


def dhw_to_classification(dhw_1deg_weekly, threshold=CLASSIFICATION_THRESHOLD):
    """Convert DHW to a binary bleaching-warning label.

    Returns 1.0 where DHW ≥ threshold (default 4 °C-weeks, the CRW Level-1
    bleaching warning), 0.0 otherwise, NaN where DHW is NaN.
    """
    return np.where(np.isnan(dhw_1deg_weekly), np.nan,
                    (dhw_1deg_weekly >= threshold).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────
def build_jensen_target(hotspot_5km_daily, M_csr, cell_size,
                        window_days=DHW_WINDOW_DAYS,
                        target_scale=TARGET_SCALE):
    """End-to-end Jensen-aware target from daily 5 km Hotspot.

    Chains the three steps::

        dhw_5km      = compute_dhw_5km(hotspot_5km_daily, window_days)
        dhw_1deg[d]  = aggregate_to_1deg(dhw_5km[d].ravel(), M_csr, cell_size)
        dhw_weekly   = dhw_1deg / 7          (degree-days → degree-weeks)
        y_regression = dhw_to_target(dhw_weekly, target_scale)

    Parameters
    ----------
    hotspot_5km_daily : np.ndarray, shape (n_days, n_lat, n_lon)
        Daily 5 km Hotspot field.
    M_csr : scipy.sparse.csr_matrix
        5 km → 1° aggregation matrix.
    cell_size : np.ndarray
        Pixels-per-cell counts.
    window_days : int
        Rolling-sum window (CRW = 84).
    target_scale : float
        DHW → target normalisation.

    Returns
    -------
    y_regression : np.ndarray, shape (n_days, n_cells)
        Regression target in [0, 1].
    """
    dhw_5km = compute_dhw_5km(hotspot_5km_daily, window_days)
    n_days, n_cells = dhw_5km.shape[0], M_csr.shape[0]

    dhw_1deg = np.full((n_days, n_cells), np.nan, dtype=np.float32)
    for d in range(n_days):
        dhw_1deg[d] = aggregate_to_1deg(dhw_5km[d].ravel(), M_csr, cell_size)

    # degree-days → degree-weeks
    dhw_weekly = dhw_1deg / 7.0
    return dhw_to_target(dhw_weekly, target_scale)
