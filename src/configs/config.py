"""
PIECF Configuration
物理约束可解释珊瑚白化预警算法 - 配置文件
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Raw dataset directory (external, not shipped). Resolution order:
#   env PIECF_DATA_DIR > author-machine default D:\piecf_crw_data > data/
# NOTE: config.DATA_DIR is only the *default*; every experiment passes the
# canonical DATA_DIR from paths.py explicitly.
DATA_DIR = os.environ.get('PIECF_DATA_DIR', '') or (
    r'D:\piecf_crw_data' if os.path.isdir(r'D:\piecf_crw_data') else
    os.path.join(PROJECT_ROOT, 'data'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Data parameters (方案一: 2°网格降低空间分辨率，提速4倍)
DATA_CONFIG = {
    'grid_lat': 20,              # 20 lat points (3.5, 5.5, ..., 41.5)
    'grid_lon': 15,              # 15 lon points (100.5, 102.5, ..., 128.5)
    'history_weeks': 24,         # T = 24 weeks (方案二: 缩短历史窗口)
    'dynamic_channels': 6,       # C_dyn = 6
    'static_channels': 4,        # C_stat = 4
    'grid_resolution': 2.0,      # 网格分辨率 (度)，用于BCO-DMO坐标匹配
    'gap_weeks': 24,             # 时序分割隔离带: train/val/test 之间至少间隔 T 周, 防止窗口重叠泄漏
}

# SCS (South China Sea) dataset config — v9 (DHW-based targets from CRW 5km, 10.8% positive)
# Grid encrypted to 1° (21×23) to increase ocean cells from 190 → ~317 (+67%)
# Dataset: scs_coral_bleaching_dataset_v9_1deg_raw.nc (built from raw sources, 9 dynamic channels)
SCS_CONFIG = {
    'grid_lat': 21,              # 21 lat points (3.5, 4.5, ..., 23.5)
    'grid_lon': 23,              # 23 lon points (100.5, 101.5, ..., 122.5)
    'history_weeks': 24,         # T = 24 weeks
    'dynamic_channels': 9,       # C_dyn = 9 (sst(CRW raw), ssta, hotspot, chlorophyll, par, seasonal_sin, wind_speed, ssr, sst_anomaly)
    'static_channels': 5,        # C_stat = 5 (bathymetry, dist_to_shore, cyclone_frequency, salinity, ocean_mask)
    'grid_resolution': 1.0,      # 网格分辨率 (度)，从 2° 加密到 1°
    'gap_weeks': 24,             # 时序分割隔离带: train/val/test 之间至少间隔 T 周, 防止窗口重叠泄漏
}

# Dynamic feature channels
DYNAMIC_FEATURES = [
    'SST',                       # Sea Surface Temperature
    'SSTA',                      # Sea Surface Temperature Anomaly
    'SST_anomaly_trend',         # SST anomaly trend
    'Wind_Speed',                # Wind speed
    'Salinity',                  # Salinity
    'fCO2_interp',               # Interfacial CO2 partial pressure (interpolated)
]

# Static feature channels
STATIC_FEATURES = [
    'Depth',                     # Water depth
    'Distance_to_Shore',         # Distance to shore
    'Cyclone_Frequency',         # Cyclone frequency
    'Turbidity',                 # Water turbidity
]

# SCS dataset feature names (v6: dropped fCO2, y_baa, y_bleaching, overall_quality)
SCS_DYNAMIC_FEATURES = [
    'SST',                       # Sea Surface Temperature (CRW, raw °C, 74% valid) [NEW in v9_1deg_raw]
    'SSTA',                      # Sea Surface Temperature Anomaly (CRW, 74% valid)
    'Hotspot',                   # CRW Hotspot thermal stress indicator (°C, 74% valid)
    'Chlorophyll',               # Chlorophyll-a (OC-CCI v6.0, 3.8% valid in SCS — very sparse)
    'PAR',                       # Photosynthetically Active Radiation proxy (MODIS, 81% valid)
    'Seasonal_Sin',              # sin(2π·week_of_year/52) encoding (100% valid)
    'Wind_Speed',                # 10 m wind speed (ERA5, 67% valid)
    'SSR',                       # Solar Radiation Index (ERA5, normalized [0,1], 67% valid)
    'SST_Anomaly',               # SST anomaly monthly (ERA5, 67% valid)
]

# Plan A: derived dynamic channels (computed on-the-fly from raw hotspot + timestamps)
# These make bleaching physics explicit so model generalizes across year splits
SCS_DERIVED_FEATURES = [
    'D_Hotspot_Dt',              # Week-over-week change in thermal stress (dHS/dt onset rate)
    'Hotspot_Cumsum_4w',         # 4-week rolling sum (recent accumulation)
    'Hotspot_Cumsum_12w',        # 12-week rolling sum (= DHW proxy, the actual bleaching driver)
    'Seasonal_Cos',              # cos(2π·woy/52) completes circular seasonal encoding
]
# fCO2 已移除 (仅 93/563400 有效 = 0.016%)
# y_baa 已移除 (与 y_warning 完全重复, r=1.0)
# y_bleaching 已移除 (100% NaN 占位符)
# overall_quality 已移除 (恒等于 2, 无信息)

SCS_STATIC_FEATURES = [
    'Bathymetry',                # Water depth (ETOPO1, land cells set to 0)
    'Dist_to_Shore',             # Distance to coastline (ETOPO1)
    'Cyclone_Frequency',         # Tropical cyclone frequency (IBTrACS climatology)
    'Salinity',                  # Sea surface salinity (WOA23 annual climatology, 73% valid)
    'Ocean_Mask',                # Binary ocean mask (land_fraction < 0.5 → ocean)
]

# Per-channel normalization statistics (RECOMPUTED 2026-07-31 from TRAIN period only)
# Order matches SCS_DYNAMIC_FEATURES + SCS_DERIVED_FEATURES:
#   sst, ssta, hotspot, chlorophyll, par, seasonal_sin, wind_speed, ssr, sst_anomaly,
#   d_hotspot_dt, hotspot_cumsum_4w, hotspot_cumsum_12w, seasonal_cos
# Updated 2026-07-31: recomputed from train-only (2003-2010) weeks to eliminate
#   normalization-statistics leakage (previous values used full 2003-2019 dataset).
#   See paper/a7_dual_output/compute_norm_stats_train_only.py for reproducibility.
# Updated 2026-07-08: sst_anomaly now in °C (climatology-subtracted), was Kelvin before
# Updated 2026-07-15: +4 Plan A derived channels (dHS/dt, cumsum_4w, cumsum_12w, seasonal_cos)
SCS_CHANNEL_MEAN = [27.97262, 0.26027, 0.077869, 0.438681, 0.618502, 0.000578, 3.769291, 16686431.0, -0.10284,
                    1.8e-05, 0.311349, 0.930024, 0.002363]
SCS_CHANNEL_STD = [2.159708, 0.570275, 0.219589, 0.517891, 0.200923, 0.706308, 2.583638, 3551082.25, 0.484529,
                   0.146682, 0.758721, 1.78879, 0.707901]

# Model architecture (ConvLSTM spatial-temporal encoder)
MODEL_CONFIG = {
    'convlstm_hidden_dims': [16, 16],  # 2层 ConvLSTM: 16→16 channels
    'attention_heads': 2,              # Temporal self-attention heads
    'attention_dim': 32,               # Attention output dimension
    'regression_hidden': 32,           # Regression head hidden dim
    'warning_hidden': 16,              # Warning head hidden dim
    'dropout': 0.2,
}

# =========================================================================
# 训练参数 (路线 A: 单任务回归 + 物理约束)
# =========================================================================
TRAINING_MODE = 'baseline'  # 'baseline' (纯回归) | 'hard_physics' (物理由构造产生)

TRAINING_CONFIG = {
    'random_seed': 42,
    'batch_size': 32,
    'num_epochs': 80,
    'learning_rate': 1e-3,
    'min_learning_rate': 1e-5,
    'early_stopping_patience': 20,
    'grad_norm_interval': 100,
}

# =========================================================================
# 损失权重（降低默认值，Phase 1 动态覆盖）
# =========================================================================
# 路线 A 损失权重 (单任务回归 + Hotspot 物理约束)
# =========================================================================
# Hard physics mode: no auxiliary loss weights (physics is in model by construction)
LOSS_CONFIG = {
    'lambda_hotspot': 0.0,               # (unused in hard_physics mode)
    'lambda_phys': 0.0,                  # (unused in hard_physics mode)
}

# Evaluation thresholds
EVALUATION_CONFIG = {
    'r2_target': 0.45,
    'rmse_target': 0.05,                 # 5%
    'mae_target': 0.03,                  # 3%
    'auc_roc_target': 0.85,
    'f1_target': 0.75,
    'brier_target': 0.15,
    'lead_time_target': 4,               # weeks
    'hit_rate_target': 0.80,
    'physical_consistency_threshold': 0.95,
}

# Data files
DATA_FILES = {
    'china_bco_dmo_bleaching': 'china_bco_dmo_bleaching.csv',
    'china_socatv2025': 'china_socatv2025.nc',
    'crw_china_daily': 'crw_china_daily.csv',
    'crw_hong_kong_daily': 'crw_hong_kong_daily.csv',
    'crw_singapore_daily': 'crw_singapore_daily.csv',
    'crw_southern_ryukyu_islands_daily': 'crw_southern_ryukyu_islands_daily.csv',
    'glodap_china': 'glodap_china.csv',
    'glodap_geomar': 'glodap_geomar.mat',
    'glodapv2_2023_obs': 'glodapv2_2023_obs.csv.gz',
}
