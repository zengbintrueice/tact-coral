# Figure 4 — data QC report (formal J2/J4 verification)

Generated: 2026-10-04T08:42:57.911702

## 1. Panel d data source correction

The previous version of panel d read `y_assoc.csv` — the output of the
**marginal** domain-shift analysis (per-pair association drift on
week-pooled populations), NOT the formal J2 definition. The formal J2
output is `joint_assoc_shift.csv` (Spearman(channel at issue t, y(t+4))
on the **shared PFV record set** — physics_anchored_records.npz,
ocean-masked — for SCS test / GBR test / CARIB test). Panel d now reads
the formal J2 file; its values match the J2 table in the figure
specification exactly (e.g. sst .708/.605/.393).

## 2. Independent recomputation vs formal outputs

Recomputed: Spearman per channel on the shared record set (gathered
from physics_anchored_records.npz t/i/j + causal v2 arrays, definition
identical to compute_domain_shift_joint_analysis.py), compared against
`joint_assoc_shift.csv` (J2) and `joint_pred_corr.csv` (J4).
Checks: 81  |  mismatches: 0  |  max |diff|: 9.19e-17

| Check | Population | Channel | CSV (formal) | Recomputed | \|diff\| | Status |
|---|---|---|---|---|---|---|
| J2 | SCS_test | sst | +0.708027 | +0.708027 | 0.00e+00 | PASS |
| J4 | SCS_test | sst | +0.582531 | +0.582531 | 0.00e+00 | PASS |
| J2 | SCS_test | ssta | +0.244635 | +0.244635 | 5.55e-17 | PASS |
| J4 | SCS_test | ssta | +0.166265 | +0.166265 | 2.78e-17 | PASS |
| J2 | SCS_test | hotspot | +0.740781 | +0.740781 | 0.00e+00 | PASS |
| J4 | SCS_test | hotspot | +0.437828 | +0.437828 | 5.55e-17 | PASS |
| J2 | SCS_test | chlorophyll | +0.104959 | +0.104959 | 8.33e-17 | PASS |
| J4 | SCS_test | chlorophyll | +0.158314 | +0.158314 | 0.00e+00 | PASS |
| J2 | SCS_test | par | +0.592886 | +0.592886 | 0.00e+00 | PASS |
| J4 | SCS_test | par | +0.365324 | +0.365324 | 5.55e-17 | PASS |
| J2 | SCS_test | seasonal_sin | -0.088176 | -0.088176 | 8.33e-17 | PASS |
| J4 | SCS_test | seasonal_sin | -0.584450 | -0.584450 | 0.00e+00 | PASS |
| J2 | SCS_test | wind_speed | -0.417290 | -0.417290 | 0.00e+00 | PASS |
| J4 | SCS_test | wind_speed | -0.379125 | -0.379125 | 5.55e-17 | PASS |
| J2 | SCS_test | ssr | +0.476123 | +0.476123 | 5.55e-17 | PASS |
| J4 | SCS_test | ssr | +0.350604 | +0.350604 | 5.55e-17 | PASS |
| J2 | SCS_test | sst_anomaly | +0.129670 | +0.129670 | 5.55e-17 | PASS |
| J4 | SCS_test | sst_anomaly | -0.019975 | -0.019975 | 2.08e-17 | PASS |
| J2 | SCS_test | d_hotspot_dt | +0.020905 | +0.020905 | 5.90e-17 | PASS |
| J4 | SCS_test | d_hotspot_dt | -0.134064 | -0.134064 | 5.55e-17 | PASS |
| J2 | SCS_test | hotspot_cumsum_4w | +0.814206 | +0.814206 | 0.00e+00 | PASS |
| J4 | SCS_test | hotspot_cumsum_4w | +0.581526 | +0.581526 | 0.00e+00 | PASS |
| J2 | SCS_test | hotspot_cumsum_12w | +0.750742 | +0.750742 | 0.00e+00 | PASS |
| J4 | SCS_test | hotspot_cumsum_12w | +0.756534 | +0.756534 | 0.00e+00 | PASS |
| J2 | SCS_test | seasonal_cos | -0.742796 | -0.742796 | 0.00e+00 | PASS |
| J4 | SCS_test | seasonal_cos | -0.380476 | -0.380476 | 5.55e-17 | PASS |
| J4 | SCS_test | (PersA) | +0.706920 | +0.706920 | 0.00e+00 | PASS |
| J2 | GBR_test | sst | +0.604722 | +0.604722 | 0.00e+00 | PASS |
| J4 | GBR_test | sst | +0.256900 | +0.256900 | 0.00e+00 | PASS |
| J2 | GBR_test | ssta | +0.362754 | +0.362754 | 0.00e+00 | PASS |
| J4 | GBR_test | ssta | +0.387707 | +0.387707 | 0.00e+00 | PASS |
| J2 | GBR_test | hotspot | +0.577177 | +0.577177 | 0.00e+00 | PASS |
| J4 | GBR_test | hotspot | +0.313976 | +0.313976 | 0.00e+00 | PASS |
| J2 | GBR_test | chlorophyll | +0.029322 | +0.029322 | 2.78e-17 | PASS |
| J4 | GBR_test | chlorophyll | +0.215109 | +0.215109 | 8.33e-17 | PASS |
| J2 | GBR_test | par | +0.672015 | +0.672015 | 0.00e+00 | PASS |
| J4 | GBR_test | par | +0.604433 | +0.604433 | 0.00e+00 | PASS |
| J2 | GBR_test | seasonal_sin | +0.654547 | +0.654547 | 0.00e+00 | PASS |
| J4 | GBR_test | seasonal_sin | +0.620843 | +0.620843 | 0.00e+00 | PASS |
| J2 | GBR_test | wind_speed | -0.462980 | -0.462980 | 0.00e+00 | PASS |
| J4 | GBR_test | wind_speed | -0.334536 | -0.334536 | 5.55e-17 | PASS |
| J2 | GBR_test | ssr | +0.535132 | +0.535132 | 0.00e+00 | PASS |
| J4 | GBR_test | ssr | -0.193270 | -0.193270 | 0.00e+00 | PASS |
| J2 | GBR_test | sst_anomaly | +0.255745 | +0.255745 | 0.00e+00 | PASS |
| J4 | GBR_test | sst_anomaly | +0.331901 | +0.331901 | 5.55e-17 | PASS |
| J2 | GBR_test | d_hotspot_dt | -0.018611 | -0.018611 | 3.47e-18 | PASS |
| J4 | GBR_test | d_hotspot_dt | -0.139713 | -0.139713 | 5.55e-17 | PASS |
| J2 | GBR_test | hotspot_cumsum_4w | +0.666320 | +0.666320 | 0.00e+00 | PASS |
| J4 | GBR_test | hotspot_cumsum_4w | +0.411026 | +0.411026 | 5.55e-17 | PASS |
| J2 | GBR_test | hotspot_cumsum_12w | +0.654601 | +0.654601 | 0.00e+00 | PASS |
| J4 | GBR_test | hotspot_cumsum_12w | +0.624859 | +0.624859 | 0.00e+00 | PASS |
| J2 | GBR_test | seasonal_cos | +0.409765 | +0.409765 | 5.55e-17 | PASS |
| J4 | GBR_test | seasonal_cos | -0.392990 | -0.392990 | 5.55e-17 | PASS |
| J4 | GBR_test | (PersA) | +0.518038 | +0.518038 | 0.00e+00 | PASS |
| J2 | CARIB_test | sst | +0.393136 | +0.393136 | 0.00e+00 | PASS |
| J4 | CARIB_test | sst | +0.099040 | +0.099040 | 1.39e-17 | PASS |
| J2 | CARIB_test | ssta | +0.284472 | +0.284472 | 0.00e+00 | PASS |
| J4 | CARIB_test | ssta | +0.141131 | +0.141131 | 5.55e-17 | PASS |
| J2 | CARIB_test | hotspot | +0.440338 | +0.440338 | 0.00e+00 | PASS |
| J4 | CARIB_test | hotspot | +0.109409 | +0.109409 | 4.16e-17 | PASS |
| J2 | CARIB_test | chlorophyll | -0.014205 | -0.014205 | 9.19e-17 | PASS |
| J4 | CARIB_test | chlorophyll | +0.362879 | +0.362879 | 5.55e-17 | PASS |
| J2 | CARIB_test | par | +0.019418 | +0.019418 | 5.90e-17 | PASS |
| J4 | CARIB_test | par | -0.453162 | -0.453162 | 0.00e+00 | PASS |
| J2 | CARIB_test | seasonal_sin | -0.499163 | -0.499163 | 0.00e+00 | PASS |
| J4 | CARIB_test | seasonal_sin | -0.109718 | -0.109718 | 2.78e-17 | PASS |
| J2 | CARIB_test | wind_speed | -0.190904 | -0.190904 | 0.00e+00 | PASS |
| J4 | CARIB_test | wind_speed | -0.581088 | -0.581088 | 0.00e+00 | PASS |
| J2 | CARIB_test | ssr | +0.002772 | +0.002772 | 1.08e-17 | PASS |
| J4 | CARIB_test | ssr | -0.518293 | -0.518293 | 0.00e+00 | PASS |
| J2 | CARIB_test | sst_anomaly | +0.186936 | +0.186936 | 2.78e-17 | PASS |
| J4 | CARIB_test | sst_anomaly | +0.308557 | +0.308557 | 0.00e+00 | PASS |
| J2 | CARIB_test | d_hotspot_dt | -0.044833 | -0.044833 | 5.55e-17 | PASS |
| J4 | CARIB_test | d_hotspot_dt | -0.137621 | -0.137621 | 5.55e-17 | PASS |
| J2 | CARIB_test | hotspot_cumsum_4w | +0.467450 | +0.467450 | 0.00e+00 | PASS |
| J4 | CARIB_test | hotspot_cumsum_4w | +0.170060 | +0.170060 | 2.78e-17 | PASS |
| J2 | CARIB_test | hotspot_cumsum_12w | +0.423059 | +0.423059 | 5.55e-17 | PASS |
| J4 | CARIB_test | hotspot_cumsum_12w | +0.340696 | +0.340696 | 0.00e+00 | PASS |
| J2 | CARIB_test | seasonal_cos | -0.126660 | -0.126660 | 2.78e-17 | PASS |
| J4 | CARIB_test | seasonal_cos | +0.260848 | +0.260848 | 5.55e-17 | PASS |
| J4 | CARIB_test | (PersA) | +0.219272 | +0.219272 | 0.00e+00 | PASS |

Resolution per the data-integrity rules: the FORMAL J2/J4 result files
are authoritative for plotted values; any mismatch would be listed
above with its reason.

## 3. Colour mapping

Panels d/e use a blue–white–red diverging palette centred at zero
(vmin=-1, vmax=1), colour-blind-safe; no rainbow colormap is used
anywhere in the figure.

## 4. Wording / layout changes applied (per specification)

- Panel d annotation: 'thermal-stress associations remain positive
  across basins, whereas several contextual and seasonal associations
  weaken or change sign.'
- Panel e annotation: 'prediction–anchor alignment is substantially
  weaker in CARIB' (replaces 'collapses').
- Summary: 'The observed transfer boundary is better explained by
  relational instability than by marginal feature shift.'
- Reading-the-diagnostics compressed to 3 items.
- Panel b annotation moved to blank space above the bars.
- Panel c legend simplified to 'HotSpot: lag 1 / lag 4; C12: 12-week
  cumulative; SST'.