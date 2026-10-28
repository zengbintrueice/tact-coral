"""
PIECF Aligned Dataset
Loads the multi-source aligned grid data (CRW + SOCAT + BCO-DMO + GLODAP)
from aligned_grid_data.nc for training the PIECF model.

This replaces the old PIECFDataset which used salinity_2012_2025.npy.
The aligned dataset uses a unified (date, lat, lon) grid with 6 dynamic
channels and 4 static channels.
"""

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
import os
import sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_DIR, DATA_CONFIG


class AlignedPIECFDataset(Dataset):
    """Dataset that reads from aligned_grid_data.nc.

    Dynamic features (6 channels, [T, H, W]):
        0: SST  - Sea Surface Temperature (from SOCAT where available, else CRW)
        1: SSTA - Sea Surface Temperature Anomaly (from CRW)
        2: DHW  - Degree Heating Weeks (from CRW)
        3: SST_trend - SST anomaly trend within window
        4: Salinity (from SOCAT where available)
        5: fCO2 - Interfacial CO2 partial pressure (from SOCAT where available)

    Static features (4 channels, [H, W]):
        0: Depth (from BCO-DMO)
        1: Distance_to_Shore (from BCO-DMO)
        2: Cyclone_Frequency (from BCO-DMO)
        3: Turbidity (from BCO-DMO)

    Targets:
        y_regression:     [T, H, W] bleaching percentage (from BCO-DMO, mostly zeros)
        y_classification: [H, W] event probability (soft label from BCO-DMO/DHW)
        y_warning:        [T, H, W] warning level
    """

    def __init__(self, data_dir=DATA_DIR, mode="train", transform=None,
                 history_weeks=None, min_lat=None, max_lat=None,
                 min_lon=None, max_lon=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.history_weeks = history_weeks or DATA_CONFIG["history_weeks"]
        self.grid_lat = DATA_CONFIG["grid_lat"]
        self.grid_lon = DATA_CONFIG["grid_lon"]
        self._load_data()

    def _load_data(self):
        print(f"Loading aligned dataset ({self.mode})...")
        ds = xr.open_dataset(os.path.join(self.data_dir, "aligned_grid_data.nc"))

        # Spatial subsetting handled by constructor args (not used by default)

        # Support both "time" (new) and "date" (legacy) dimension names
        time_dim = "time" if "time" in ds.dims else "date"
        self.n_dates = ds.sizes[time_dim]
        self.n_lat = ds.sizes["lat"]
        self.n_lon = ds.sizes["lon"]
        T = self.history_weeks

        # Dynamic features: [date, lat, lon, C_dyn=6]
        # Channel order matches config: SST, SSTA, SST_anomaly_trend, Wind_Speed, Salinity, fCO2
        # From aligned_grid_data.nc we have: SST, SSTA, DHW, BAA, fco2, salinity
        # We map: SST→0, SSTA→1, DHW→2 (as proxy for SST_anomaly_trend), BAA→3, fCO2→5, salinity→4
        sst = ds["SST"].values.astype(np.float32)           # [D, H, W]
        ssta = ds["SSTA"].values.astype(np.float32)         # [D, H, W]
        dhw = ds["DHW"].values.astype(np.float32)           # [D, H, W]
        baa = ds["BAA"].values.astype(np.float32)           # [D, H, W]
        fco2 = ds["fco2_ave_weighted_year"].values.astype(np.float32)   # [D, H, W]
        salinity = ds["salinity_ave_weighted_year"].values.astype(np.float32)  # [D, H, W]

        # Static features: [lat, lon, C_stat=4]
        # NetCDF 中静态场是 (time, lat, lon)，取第一个时间步即可
        depth = ds["Depth_m"].values[0].astype(np.float32) if "Depth_m" in ds else ds["Distance_to_Shore"].values[0].astype(np.float32)
        dist_shore = ds["Distance_to_Shore"].values[0].astype(np.float32)
        cyclone = ds["Cyclone_Frequency"].values[0].astype(np.float32)
        turbidity = ds["Turbidity"].values[0].astype(np.float32)

        # Handle NaN in dynamic data: fill with 0 (ocean points with no data = default)
        sst = np.nan_to_num(sst, nan=0.0)
        ssta = np.nan_to_num(ssta, nan=0.0)
        dhw = np.nan_to_num(dhw, nan=0.0)
        baa = np.nan_to_num(baa, nan=0.0)
        fco2 = np.nan_to_num(fco2, nan=0.0)
        salinity = np.nan_to_num(salinity, nan=0.0)

        # Handle NaN in static data: fill with column mean
        depth = self._fill_nan(depth)
        dist_shore = self._fill_nan(dist_shore)
        cyclone = self._fill_nan(cyclone)
        turbidity = self._fill_nan(turbidity)

        # Build sliding windows
        n_windows = self.n_dates - T + 1
        if n_windows <= 0:
            raise ValueError(f"Not enough dates ({self.n_dates}) for window size {T}")

        # Train/val/test split on time axis (70/15/15), 带 gap 隔离带防止时序泄漏
        gap = self.history_weeks  # = T
        raw_train_end = int(n_windows * 0.70)
        raw_val_end = int(n_windows * 0.85)

        train_end = raw_train_end
        val_start = raw_train_end + gap
        val_end = raw_val_end
        test_start = min(raw_val_end + gap, n_windows - 1)

        if self.mode == "train":
            window_indices = range(0, train_end)
        elif self.mode == "val":
            window_indices = range(val_start, val_end)
        else:  # test
            window_indices = range(test_start, n_windows)

        self.window_indices = list(window_indices)
        print(f"  Window indices: {len(self.window_indices)} windows for {self.mode}")

        # Pre-compute dynamic features as numpy arrays
        # Shape: [N_windows, T, H, W, C_dyn]
        C_dyn = 6
        self.X_dynamic = np.zeros((len(self.window_indices), T, self.n_lat, self.n_lon, C_dyn), dtype=np.float32)

        for i, start_idx in enumerate(self.window_indices):
            end_idx = start_idx + T
            self.X_dynamic[i, :, :, :, 0] = sst[start_idx:end_idx]       # SST
            self.X_dynamic[i, :, :, :, 1] = ssta[start_idx:end_idx]      # SSTA
            self.X_dynamic[i, :, :, :, 2] = dhw[start_idx:end_idx]       # DHW (thermal stress)
            self.X_dynamic[i, :, :, :, 3] = baa[start_idx:end_idx]       # BAA (bleaching alert)
            self.X_dynamic[i, :, :, :, 4] = salinity[start_idx:end_idx]  # Salinity
            self.X_dynamic[i, :, :, :, 5] = fco2[start_idx:end_idx]      # fCO2

        # Static: [N, H, W, C_stat]
        C_stat = 4
        self.X_static = np.zeros((len(self.window_indices), self.n_lat, self.n_lon, C_stat), dtype=np.float32)
        for i in range(len(self.window_indices)):
            self.X_static[i, :, :, 0] = depth
            self.X_static[i, :, :, 1] = dist_shore
            self.X_static[i, :, :, 2] = cyclone
            self.X_static[i, :, :, 3] = turbidity

        # Land mask
        self.land_mask = ds["land_mask"].values.astype(bool)  # [H, W]

        # Targets
        self._generate_targets(sst, ssta, dhw, baa, T)

        # DHW raw for physical consistency — 取窗口内 DHW 最大值（与 compute_dhw 输出对齐）
        self.dhw_raw = np.zeros((len(self.window_indices), self.n_lat, self.n_lon), dtype=np.float32)
        for i, start_idx in enumerate(self.window_indices):
            end_idx = start_idx + T
            window_dhw = dhw[start_idx:end_idx]  # [T, H, W]
            # 窗口内 DHW 最大值（物理约束已移除，此字段保留仅为兼容）
            self.dhw_raw[i, :, :] = window_dhw.max(axis=0)

        ds.close()
        print(f"  Loaded {len(self)} samples: X_dynamic={self.X_dynamic.shape}, X_static={self.X_static.shape}")

    def _generate_targets(self, sst, ssta, dhw, baa, T):
        """Generate targets from CRW and BCO-DMO data.

        For each window, the target is based on the state at the end of the window
        (the most recent week). BCO-DMO Percent_Bleaching provides sparse labels.
        """
        n_windows = len(self.window_indices)
        H, W = self.n_lat, self.n_lon

        self.y_regression = np.zeros((n_windows, T, H, W), dtype=np.float32)
        self.y_classification = np.zeros((n_windows, H, W), dtype=np.float32)
        self.y_warning = np.zeros((n_windows, T, H, W), dtype=np.float32)

        for i, start_idx in enumerate(self.window_indices):
            end_idx = start_idx + T
            window_dhw = dhw[start_idx:end_idx]  # [T, H, W]
            window_baa = baa[start_idx:end_idx]  # [T, H, W]
            window_ssta = ssta[start_idx:end_idx]  # [T, H, W]

            # Regression target: bleaching percentage based on thermal stress
            # Week t: bleaching = clip(dhw_pixel * t * 0.5, 0, 100) / 100
            # 保留空间异质性：每个像素独立计算
            for t in range(T):
                dhw_t = window_dhw[t]  # [H, W] — 保留空间结构
                bleaching_pct = np.clip(dhw_t * (t + 1) * 0.5, 0, 100) / 100.0
                self.y_regression[i, t, :, :] = bleaching_pct

            # Classification target: event probability from DHW/BAA
            max_dhw = window_dhw.max(axis=0)  # [H, W]
            max_baa = window_baa.max(axis=0)  # [H, W]
            dhw_score = 1.0 / (1.0 + np.exp(-(max_dhw - 4.0) * 2.0))
            baa_score = 1.0 / (1.0 + np.exp(-(max_baa - 2.0) * 2.0))
            self.y_classification[i, :, :] = np.maximum(dhw_score, baa_score)

            # Warning target: accumulated thermal stress
            # 保留空间异质性：每个像素独立计算
            for t in range(T):
                dhw_t = window_dhw[t]  # [H, W] — 保留空间结构
                warning_level = np.clip(dhw_t * (t + 1) * 0.3, 0, 8) / 8.0
                self.y_warning[i, t, :, :] = warning_level

    def _fill_nan(self, arr):
        """Fill NaN with column mean."""
        col_mean = np.nanmean(arr, axis=0, keepdims=True)
        col_mean = np.where(np.isnan(col_mean), 0, col_mean)
        return np.where(np.isnan(arr), col_mean, arr)

    def get_class_weights(self):
        """Compute class weights for imbalanced classification."""
        pos = np.sum(self.y_classification > 0.5)
        neg = np.sum(self.y_classification <= 0.5)
        total = pos + neg
        if total == 0 or pos == 0:
            return torch.FloatTensor([1.0, 1.0])
        weight_neg = total / (2.0 * neg) if neg > 0 else 1.0
        weight_pos = total / (2.0 * pos) if pos > 0 else 1.0
        return torch.FloatTensor([weight_neg, weight_pos])

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        x_dynamic = torch.FloatTensor(self.X_dynamic[idx])
        x_static = torch.FloatTensor(self.X_static[idx])
        y_reg = torch.FloatTensor(self.y_regression[idx])
        y_cls = torch.FloatTensor(self.y_classification[idx])
        y_warn = torch.FloatTensor(self.y_warning[idx])
        land_mask = torch.BoolTensor(self.land_mask)
        dhw_raw = torch.FloatTensor(self.dhw_raw[idx])

        # y_alert: binary alert label derived from y_warning (max across time)
        # Shape: [H, W] - 1 if any time step exceeds threshold (0.5/8 = 0.0625 normalized)
        y_alert = torch.FloatTensor((y_warn.max(dim=0)[0] > 0.0625).float())

        return {
            "x_dynamic": x_dynamic,
            "x_static": x_static,
            "y_regression": y_reg,
            "y_classification": y_cls,
            "y_warning": y_warn,
            "y_alert": y_alert,
            "land_mask": land_mask,
            "dhw_raw": dhw_raw,
        }


def create_aligned_dataloaders(data_dir=DATA_DIR, batch_size=16, num_workers=0,
                                use_weighted_sampling=True):
    train_dataset = AlignedPIECFDataset(data_dir, mode="train")
    val_dataset = AlignedPIECFDataset(data_dir, mode="val")
    test_dataset = AlignedPIECFDataset(data_dir, mode="test")

    train_sampler = None
    shuffle = True
    if use_weighted_sampling:
        # Weight by whether window contains high DHW event
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=_compute_sample_weights(train_dataset),
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def _compute_sample_weights(dataset):
    """Compute per-sample weights for oversampling windows with bleaching events."""
    has_event = np.any(dataset.y_classification > 0.5, axis=(1, 2)).astype(np.float32)
    n_pos = np.sum(has_event > 0)
    n_neg = np.sum(has_event == 0)
    n_total = len(has_event)

    pos_weight = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
    neg_weight = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0

    weights = np.where(has_event > 0, pos_weight, neg_weight)
    return torch.FloatTensor(weights)


# =========================================================================
# SCS (South China Sea) Coral Bleaching Dataset
# Reads from scs_coral_bleaching_dataset.nc with 8 dynamic + 2 static channels
# and REAL bleaching labels from BCO-DMO.
# =========================================================================

class SCSPIECFDataset(Dataset):
    """Dataset that reads from scs_coral_bleaching_dataset_v9_final2.nc (time-aligned, 23 bad cells removed).

    v9 = v8 features + DHW-based targets from CRW 5km daily data:
      y_regression  = clip(DHW/10, 0, 1)       — continuous [0, 1]
      y_warning    = clip(DHW/8, 0, 1)        — continuous [0, 1], Alert 2 = 1.0
      y_classification = 1 if DHW >= 4 else 0  — 10.8% positive (was 0.11% from BAA)

    Dynamic features (8 channels, [T, H, W]):
        0:    SSTA           - SST anomaly (CRW, raw Celsius, 74% valid)
        1:    Hotspot        - CRW thermal stress (clipped [0,+inf), 74% valid)
        2:    Chlorophyll    - Chlorophyll-a (OC-CCI v6, log1p, 92% valid)
        3:    PAR            - Photosynthetically Active Radiation (MODIS, min-max [0,1])
        4:    Seasonal_Sin   - sin(2π * week_of_year / 52) (100% valid)
        5:    Wind_Speed     - 10 m wind speed (ERA5, raw m/s, 67% valid)
        6:    SSR            - Solar radiation index (ERA5, normalized [0,1], 67% valid)
        7:    SST_Anomaly    - SST monthly anomaly (ERA5, raw Celsius, 67% valid)

    Static features (5 channels, [H, W]):
        0: Bathymetry        - log1p(-depth) from ETOPO1 (range [0, 8.36])
        1: Dist_to_Shore     - log1p(meters) from ETOPO1 (range [0, 10.25])
        2: Cyclone_Frequency - z-score standardized from IBTrACS
        3: Salinity          - WOA23 annual climatology (PSU, 73% valid)
        4: Ocean_Mask        - Binary ocean mask (land_fraction < 0.5 → ocean)

    Targets (from v9 NetCDF, DHW-based):
        y_regression:      [T, H, W] clip(DHW/10, 0, 1) — continuous thermal stress
        y_classification:  [T, H, W] 1 if DHW >= 4 (Alert Level 1+), ~10.8% positive
        y_warning:         [T, H, W] clip(DHW/8, 0, 1) — continuous, 1.0 = Alert Level 2

    NaN Handling:
        - Features: NaN values are replaced with 0 for model input, but tracked via nan_mask
        - Targets: NaN values mean "no label" — loss is only computed where labels exist
    """

    def __init__(self, data_dir=DATA_DIR, mode="train", transform=None,
                 history_weeks=24, dataset_file="scs_coral_bleaching_dataset_v9_1deg_raw.nc",
                 random_split=False, random_seed=42, lead_weeks=4,
                 year_based_split=False,
                 train_years=(2003, 2012),
                 val_years=(2013, 2015),
                 test_years=(2016, 2019),
                 purge_weeks=0):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.history_weeks = history_weeks
        self.dataset_file = dataset_file
        self.random_split = random_split
        self.random_seed = random_seed
        self.lead_weeks = lead_weeks
        self.year_based_split = year_based_split
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.purge_weeks = purge_weeks
        self._load_data()

    def _purge_boundary_windows(self, window_indices, T, dates):
        """Drop windows whose last-input date falls within purge_weeks of an
        internal train/val/test boundary (anti-leakage: prevents target
        cross-boundary contamination). Outermost edges are kept."""
        pw = self.purge_weeks
        if pw <= 0:
            return window_indices
        from datetime import datetime, timedelta
        b1 = datetime(self.val_years[0], 1, 1)   # train|val boundary
        b2 = datetime(self.test_years[0], 1, 1)  # val|test boundary
        d = timedelta(weeks=pw)
        keep = []
        for w in window_indices:
            dt = datetime.strptime(dates[w + T - 1], '%Y-%m-%d')
            if self.mode == 'train':
                if dt.date() > (b1 - d).date():
                    continue
            elif self.mode == 'val':
                if dt.date() < (b1 + d).date() or dt.date() > (b2 - d).date():
                    continue
            elif self.mode == 'test':
                if dt.date() < (b2 + d).date():
                    continue
            keep.append(w)
        return keep

    def _compute_derived_channels(self, dyn_arrays):
        """Compute Plan A derived channels from raw hotspot (dHS/dt, cumsum, seasonal_cos).

        These make bleaching physics explicit to the model:
          - dHS/dt: onset rate of thermal stress (week-over-week difference)
          - cumsum_4w / cumsum_12w: rolling sum (= DHW proxy)
          - seasonal_cos: cos(2π·woy/52), complements existing seasonal_sin

        NOTE: avoids pd.DatetimeIndex — pandas segfaults in subprocess/worker
        context when combined with netCDF4. Uses datetime.strptime instead.
        """
        # Find hotspot channel
        hs_idx = None
        for i, name in enumerate(self.dyn_feature_names):
            if name == 'hotspot':
                hs_idx = i
                break
        if hs_idx is None:
            return [], []

        # Force contiguous copy so netCDF4 buffer can be safely released
        hotspot = np.ascontiguousarray(dyn_arrays[hs_idx])
        D = hotspot.shape[0]
        hs_clean = np.nan_to_num(hotspot, nan=0.0)
        nan_mask = np.isnan(hotspot)

        new_arrays = []
        new_names = []

        # (1) d_hotspot_dt — forward difference, NaN-aligned with original
        d_hs = np.zeros_like(hotspot)
        d_hs[1:] = hs_clean[1:] - hs_clean[:-1]
        d_hs[nan_mask] = np.nan
        new_arrays.append(d_hs.astype(np.float32))
        new_names.append('d_hotspot_dt')

        # (2-3) hotspot_cumsum_4w / cumsum_12w — rolling sum via cumulative trick
        cumsum = np.cumsum(hs_clean, axis=0)
        for window, wname in [(4, 'hotspot_cumsum_4w'), (12, 'hotspot_cumsum_12w')]:
            cs = np.zeros_like(hotspot)
            for i in range(D):
                if i < window:
                    cs[i] = cumsum[i]
                else:
                    cs[i] = cumsum[i] - cumsum[i - window]
            cs[nan_mask] = np.nan
            new_arrays.append(cs.astype(np.float32))
            new_names.append(wname)

        # (4) seasonal_cos from week-of-year (avoids pandas in worker context)
        from datetime import datetime as _dt
        woy = np.array([_dt.strptime(s, '%Y-%m-%d').isocalendar()[1]
                        for s in self.date_strings], dtype=np.float32)
        seasonal_cos = np.cos(2 * np.pi * woy / 52.0).astype(np.float32)
        sc_3d = np.broadcast_to(seasonal_cos[:, None, None], hotspot.shape).copy()
        new_arrays.append(sc_3d.astype(np.float32))
        new_names.append('seasonal_cos')

        return new_arrays, new_names

    def _load_data(self):
        # Import config dynamically to get feature lists
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from configs.config import SCS_DYNAMIC_FEATURES, SCS_STATIC_FEATURES

        print(f"Loading SCS dataset ({self.mode})...")
        ds = xr.open_dataset(os.path.join(self.data_dir, self.dataset_file))

        time_dim = "time"
        self.n_dates = ds.sizes[time_dim]
        self.n_lat = ds.sizes["lat"]
        self.n_lon = ds.sizes["lon"]
        T = self.history_weeks

        # ---- Timestamps (ISO date per time step) ----
        try:
            _time_vals = ds[time_dim].values  # datetime64[ns]
            self.date_strings = [str(t)[:10] for t in _time_vals]  # 'YYYY-MM-DD'
        except Exception:
            self.date_strings = [f'2003-01-05'] * self.n_dates  # safe fallback

        # ---- Dynamic features ----
        # Load by name from SCS_DYNAMIC_FEATURES; skip variables not in dataset file
        # IMPORTANT: force eager copy via np.ascontiguousarray so that netCDF4
        # mmap'd memory can be safely released by ds.close() later. Without this,
        # the underlying buffer is freed and subsequent numpy ops segfault.
        self.dyn_feature_names = []
        dyn_arrays = []
        for name in SCS_DYNAMIC_FEATURES:
            xr_name = name.lower()  # NetCDF variables are lowercase
            if xr_name in ds.data_vars:
                arr = np.ascontiguousarray(ds[xr_name].values.astype(np.float32))
                dyn_arrays.append(arr)
                self.dyn_feature_names.append(xr_name)
        if not dyn_arrays:
            raise ValueError("No dynamic features found in dataset")
        C_dyn = len(dyn_arrays)
        self.C_dyn = C_dyn
        print(f"  Dynamic features ({C_dyn}): {self.dyn_feature_names}")

        # ---- Static features ---- (eager copy before ds.close)
        static_arrays = []
        self.static_feature_names = []
        for name in SCS_STATIC_FEATURES:
            xr_name = name.lower()
            if xr_name in ds.data_vars:
                arr = np.ascontiguousarray(ds[xr_name].values.astype(np.float32))
                static_arrays.append(arr)
                self.static_feature_names.append(xr_name)
        C_stat = len(static_arrays)
        self.C_stat = C_stat
        print(f"  Static features ({C_stat}): {self.static_feature_names}")

        # Land/ocean mask
        if "ocean_mask" in ds.data_vars:
            om = ds["ocean_mask"].values.astype(bool)
            self.land_mask = ~om
        else:
            lm = ds["land_mask"].values
            if lm.ndim == 3:
                lm = lm[0]
            self.land_mask = lm.astype(bool)

        # Targets (eager copy before ds.close)
        y_warning_arr = np.ascontiguousarray(ds["y_warning"].values.astype(np.float32))
        y_regression_arr = np.ascontiguousarray(ds["y_regression"].values.astype(np.float32))
        y_cls_arr = np.ascontiguousarray(ds["y_classification"].values.astype(np.float32)) if "y_classification" in ds.data_vars else None
        ds.close()  # <-- close netCDF handle BEFORE derived-channel numpy work

        # Store full time series for per-lead target computation (XGBoost etc.)
        self.y_regression_full = y_regression_arr  # [n_dates, H, W]

        # ---- Plan A: derived channels (computed on-the-fly from raw hotspot) ----
        # Must run AFTER ds.close() — netCDF4 C lib conflicts with heavy numpy ops
        # on same process. Hotspot already in memory as eager numpy float32 array.
        print("    [load] about to compute derived...", flush=True)
        derived_arrays, derived_names = self._compute_derived_channels(dyn_arrays)
        print(f"    [load] derived done: {len(derived_arrays)} channels", flush=True)
        if derived_arrays:
            dyn_arrays = dyn_arrays + derived_arrays
            self.dyn_feature_names = self.dyn_feature_names + derived_names
            C_dyn = len(dyn_arrays)
            self.C_dyn = C_dyn
            print(f"  + Derived features ({len(derived_arrays)}): {derived_names}")
            print(f"  Total dynamic features ({C_dyn}): {self.dyn_feature_names}")

        # Store full dynamic time series for sliding-window evaluation
        # Shape: [n_dates, H, W, C_dyn] — used by predict_sliding_temporal()
        self.X_dynamic_full = np.stack(dyn_arrays, axis=-1).astype(np.float32)

        # Track NaN mask; keep NaN in tensor (model normalizes with NaN handling)
        self.nan_mask_dyn = np.zeros((C_dyn, self.n_dates, self.n_lat, self.n_lon), dtype=bool)
        for i, arr in enumerate(dyn_arrays):
            self.nan_mask_dyn[i] = np.isnan(arr)
            # Do NOT fill NaN here — model's ChannelNormalizer handles NaN

        # Fill NaN in static features with 0
        static_arrays = [np.nan_to_num(a, nan=0.0) for a in static_arrays]

        # Build sliding windows
        # lead_weeks > 0: target is (weeks after window end) → true future prediction
        # Window start must satisfy: start + T - 1 + lead_weeks < n_dates
        n_windows = self.n_dates - T + 1 - self.lead_weeks
        if n_windows <= 0:
            raise ValueError(f"Not enough dates ({self.n_dates}) for window size {T} + lead {self.lead_weeks}")

        # Train/val/test split 策略 (互斥, 优先级: random > year_based > temporal)
        # 1. random: 随机打乱窗口 (用于分布偏移假设验证)
        # 2. year_based: 严格按窗口最后时间步的年份留出 (无泄漏)
        # 3. temporal: 比例切分 + gap=T 隔离带 (旧方法, 仍有部分信息穿越)

        if self.random_split:
            # 随机打乱窗口索引 (固定种子保证 train/val/test 使用同一排列)
            rng = np.random.RandomState(self.random_seed)
            perm = rng.permutation(n_windows)
            train_end = int(n_windows * 0.70)
            val_end = int(n_windows * 0.85)
            if self.mode == "train":
                window_indices = perm[:train_end]
            elif self.mode == "val":
                window_indices = perm[train_end:val_end]
            else:  # test
                window_indices = perm[val_end:]
            print(f"  [RANDOM SPLIT] seed={self.random_seed}")
        elif self.year_based_split:
            # 严格按窗口最后时间步的年份分配 — 彻底消除跨分割信息穿越
            # 每个窗口 start ∈ [0, n_windows-1], 其最后时间步 = start + T - 1
            import pandas as pd
            dates = pd.DatetimeIndex(self.date_strings)
            win_last_dates = dates[np.array(range(n_windows)) + T - 1]
            win_years = win_last_dates.year

            if self.mode == "train":
                mask = (win_years >= self.train_years[0]) & (win_years <= self.train_years[1])
            elif self.mode == "val":
                mask = (win_years >= self.val_years[0]) & (win_years <= self.val_years[1])
            else:  # test
                mask = (win_years >= self.test_years[0]) & (win_years <= self.test_years[1])
            window_indices = list(np.where(mask)[0])
            print(f"  [YEAR-BASED SPLIT] {self.mode}: years={getattr(self, self.mode+'_years')}, "
                  f"windows={len(window_indices)}")
        else:
            # 旧: 时间顺序分割 (带 gap 隔离带) — 保留但不再作为默认
            # ⚠️ AUDIT WARNING (2026-09-14): legacy branch, NEVER used by any
            # production experiment (Steps 2-7 all pass year_based_split=True).
            # With purge_weeks=0 and stride-1 windows, consecutive windows
            # within a split share 23 weeks of input history, and split
            # isolation relies solely on the T-week gap below. Prefer
            # year_based_split=True for any new experiment.
            print("  ⚠️ [LEGACY SPLIT] year_based_split=False: non-production "
                  "path; windows use stride-1 (23-week intra-split overlap) "
                  "and rely on the gap parameter for split isolation. Use "
                  "year_based_split=True for production runs.")
            gap = self.history_weeks  # = T
            raw_train_end = int(n_windows * 0.70)
            raw_val_end = int(n_windows * 0.85)
            train_end = raw_train_end
            val_start = raw_train_end + gap
            val_end = raw_val_end
            test_start = raw_val_end + gap
            if test_start >= n_windows:
                test_start = max(val_end + 1, n_windows - int(n_windows * 0.10))
            if self.mode == "train":
                window_indices = range(0, train_end)
            elif self.mode == "val":
                window_indices = range(val_start, val_end)
            else:  # test
                window_indices = range(test_start, n_windows)
            print(f"  [TEMPORAL SPLIT] {self.mode}: windows={len(window_indices) if isinstance(window_indices, range) else len(list(window_indices))}")

        self.window_indices = list(window_indices)
        print(f"  Window indices: {len(self.window_indices)} windows for {self.mode}")

        # Pre-compute dynamic features: [N, T, H, W, C_dyn]
        # Anti-leakage purge: drop windows near internal split boundaries (purge_weeks>0)
        self.window_indices = self._purge_boundary_windows(self.window_indices, self.history_weeks, self.date_strings)
        self.X_dynamic = np.zeros((len(self.window_indices), T, self.n_lat, self.n_lon, C_dyn), dtype=np.float32)
        for i, start_idx in enumerate(self.window_indices):
            end_idx = start_idx + T
            for c in range(C_dyn):
                self.X_dynamic[i, :, :, :, c] = dyn_arrays[c][start_idx:end_idx]

        # Static: [N, H, W, C_stat]
        # v8 stores static as [H, W] (no time dim)
        self.X_static = np.zeros((len(self.window_indices), self.n_lat, self.n_lon, C_stat), dtype=np.float32)
        for i in range(len(self.window_indices)):
            for c in range(C_stat):
                arr = static_arrays[c]
                if arr.ndim == 3:
                    arr = arr[0]  # [T, H, W] → [H, W]
                self.X_static[i, :, :, c] = arr

        # Build targets for each window
        nW = len(self.window_indices)
        H, W = self.n_lat, self.n_lon
        self.y_regression = np.full((nW, T, H, W), np.nan, dtype=np.float32)
        self.y_classification = np.full((nW, T, H, W), np.nan, dtype=np.float32)
        self.y_warning_out = np.full((nW, T, H, W), np.nan, dtype=np.float32)

        # Lead-time target: single value at week (start + T - 1 + lead_weeks)
        # This is the TRUE future prediction target (beyond the input window)
        self.y_lead = np.full((nW, H, W), np.nan, dtype=np.float32)

        # v9: y_classification is pre-computed in the NetCDF file (1 if DHW >= 4)
        # y_cls_arr already loaded before ds.close()

        for i, start_idx in enumerate(self.window_indices):
            end_idx = start_idx + T
            self.y_regression[i] = y_regression_arr[start_idx:end_idx]
            self.y_warning_out[i] = y_warning_arr[start_idx:end_idx]

            # Lead target: the value G weeks AFTER the window ends
            lead_idx = start_idx + T - 1 + self.lead_weeks
            if lead_idx < self.n_dates:
                self.y_lead[i] = y_regression_arr[lead_idx]

            if y_cls_arr is not None:
                # v9: use pre-computed DHW-based classification
                self.y_classification[i] = y_cls_arr[start_idx:end_idx]
            else:
                # Fallback for v8: derive from y_warning (BAA-based, very sparse)
                window_warn = y_warning_arr[start_idx:end_idx]
                has_label = ~np.all(np.isnan(window_warn), axis=0)
                max_warn = np.nanmax(window_warn, axis=0)
                cls = np.where(max_warn >= 2.0, 1.0, 0.0).astype(np.float32)
                window_reg = y_regression_arr[start_idx:end_idx]
                has_reg = ~np.all(np.isnan(window_reg), axis=0)
                max_reg = np.nanmax(window_reg, axis=0)
                cls_from_reg = np.where(max_reg > 0.05, 1.0, 0.0).astype(np.float32)
                cls = np.maximum(cls, cls_from_reg)
                cls[~(has_label | has_reg)] = np.nan
                self.y_classification[i] = cls[:, :][np.newaxis, :, :].repeat(T, axis=0)

        # Hotspot (热胁迫) for physical consistency — pick from dyn_arrays by name
        # In v8, Hotspot is clipped [0,+inf) and is the physical constraint proxy
        try:
            hs_idx = self.dyn_feature_names.index('hotspot')
            hs_arr = dyn_arrays[hs_idx]
        except ValueError:
            hs_arr = None

        self.hotspot_raw = np.zeros((nW, H, W), dtype=np.float32)
        if hs_arr is not None:
            for i, start_idx in enumerate(self.window_indices):
                end_idx = start_idx + T
                window_hs = hs_arr[start_idx:end_idx]
                self.hotspot_raw[i] = np.nanmax(window_hs, axis=0)

        # v8 does not include y_baa (removed as duplicate of y_warning in v6)
        self.y_baa = None

        # Label availability mask: [N, H, W].
        # The lead-time target y_lead is the ONLY supervised signal; a window is
        # trainable only where y_lead is finite AND the cell is ocean. This fixes
        # the NaN->0 pseudo-label bug: label_mask was previously based on the
        # input window's y_regression validity, so boundary windows without a
        # future target (y_lead=NaN) were trained/evaluated as target 0.
        self.label_mask = np.isfinite(self.y_lead) & (~self.land_mask)  # [N, H, W]

        n_labels = np.sum(self.label_mask)
        n_total = self.label_mask.size
        print(f"  Loaded {len(self)} samples: X_dynamic={self.X_dynamic.shape}, X_static={self.X_static.shape}")
        print(f"  Label coverage: {n_labels}/{n_total} ({n_labels/n_total*100:.2f}%)")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        x_dynamic = torch.FloatTensor(self.X_dynamic[idx])  # may contain NaN
        x_static = torch.FloatTensor(self.X_static[idx])
        y_reg = torch.FloatTensor(self.y_regression[idx])
        y_cls = torch.FloatTensor(self.y_classification[idx])
        y_warn = torch.FloatTensor(self.y_warning_out[idx])
        y_lead = torch.FloatTensor(self.y_lead[idx])
        land_mask = torch.BoolTensor(self.land_mask)
        hotspot_raw = torch.FloatTensor(self.hotspot_raw[idx])
        label_mask = torch.BoolTensor(self.label_mask[idx])

        # NaN mask for dynamic features: [C_dyn, T, H, W] — True where valid
        nan_mask = torch.BoolTensor(~self.nan_mask_dyn[:, self.window_indices[idx]:self.window_indices[idx]+self.history_weeks, :, :])

        # y_alert: binary alert from y_warning
        y_alert = torch.FloatTensor((y_warn.max(dim=0)[0] > 0.5).float())

        # Timestamp for the LAST timestep of this window (matches prediction target)
        # Used by evaluator for seasonal cycle / climatology computation
        _win_start = self.window_indices[idx]
        _target_step = _win_start + self.history_weeks - 1
        if _target_step < len(self.date_strings):
            timestamp = self.date_strings[_target_step]
        else:
            timestamp = self.date_strings[-1] if self.date_strings else None

        result = {
            "x_dynamic": x_dynamic,
            "x_static": x_static,
            "y_regression": y_reg,
            "y_classification": y_cls,
            "y_warning": y_warn,
            "y_lead": y_lead,
            "y_alert": y_alert,
            "land_mask": land_mask,
            "hotspot_raw": hotspot_raw,
            "label_mask": label_mask,
            "nan_mask_dyn": nan_mask,
            "timestamp": timestamp,
        }

        # BAA 多级分类目标 (若存在)
        if self.y_baa is not None:
            result["y_baa"] = torch.FloatTensor(self.y_baa[idx])

        return result


def create_scs_dataloaders(data_dir=DATA_DIR, batch_size=16, num_workers=0,
                            random_split=False, random_seed=42, lead_weeks=4,
                            year_based_split=False,
                            train_years=(2003, 2012),
                            val_years=(2013, 2015),
                            test_years=(2016, 2019),
                            dataset_file="scs_coral_bleaching_dataset_v9_1deg_raw.nc"):
    """Create dataloaders for the SCS coral bleaching dataset.

    Args:
        random_split: 随机打乱窗口 (用于分布偏移假设验证)
        year_based_split: 严格按窗口末尾年份留出, 完全消除跨分割信息穿越
        train_years, val_years, test_years: 年分割的年份区间 (含端点)
        lead_weeks: 目标 = 窗口末 + lead_weeks 周后的状态 (0=编码, >0=预测)
        dataset_file: NetCDF filename (use GBR file for GBR retraining)
    """
    split_kwargs = dict(year_based_split=year_based_split,
                       train_years=train_years, val_years=val_years, test_years=test_years)
    train_dataset = SCSPIECFDataset(data_dir, mode="train",
                                     random_split=random_split, random_seed=random_seed,
                                     lead_weeks=lead_weeks, dataset_file=dataset_file,
                                     **split_kwargs)
    val_dataset = SCSPIECFDataset(data_dir, mode="val",
                                   random_split=random_split, random_seed=random_seed,
                                   lead_weeks=lead_weeks, dataset_file=dataset_file,
                                   **split_kwargs)
    test_dataset = SCSPIECFDataset(data_dir, mode="test",
                                    random_split=random_split, random_seed=random_seed,
                                    lead_weeks=lead_weeks, dataset_file=dataset_file,
                                    **split_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
