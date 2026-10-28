# Domain Shift Analysis — SCS train vs SCS/GBR/CARIB test

Generated: 2026-09-18T00:36:38+00:00

Populations (ocean cells only; weeks pooled with ocean pixels):
- **SCS_train**: scs_causal_v2.nc, years 2003-2010, 417 weeks x 271 ocean cells
- **SCS_test**: scs_causal_v2.nc, years 2016-2019, 186 weeks x 271 ocean cells
- **GBR_test**: gbr_causal_v2.nc, years 2017-2020, 209 weeks x 93 ocean cells
- **CARIB_test**: caribbean_causal_v2.nc, years 2017-2020, 209 weeks x 405 ocean cells

Standardization: SCS-train (A) statistics ONLY (no per-domain normalization).
Metrics: SMD = (mean_target − mean_A)/std_A; Wasserstein-1 and energy distance on z-scored values.

## Group-level summary

| source | target | group | SMD median | SMD |max| | W1(z) median | W1 max | energy median | energy max |
|---|---|---|---|---|---|---|---|
| SCS_train | SCS_test | instantaneous_thermal | +0.449 | 0.796 | 0.449 | 0.801 | 0.293 | 0.593 |
| SCS_train | SCS_test | cumulative_thermal | +0.397 | 0.497 | 0.397 | 0.497 | 0.201 | 0.287 |
| SCS_train | SCS_test | context_calendar | -0.000 | 0.121 | 0.124 | 0.157 | 0.098 | 0.132 |
| SCS_train | GBR_test | instantaneous_thermal | +0.400 | 0.591 | 0.429 | 0.591 | 0.305 | 0.450 |
| SCS_train | GBR_test | cumulative_thermal | +0.666 | 0.975 | 0.666 | 0.975 | 0.288 | 0.447 |
| SCS_train | GBR_test | context_calendar | +0.002 | 0.615 | 0.432 | 0.801 | 0.349 | 0.585 |
| SCS_train | CARIB_test | instantaneous_thermal | +0.243 | 0.639 | 0.298 | 0.639 | 0.227 | 0.515 |
| SCS_train | CARIB_test | cumulative_thermal | +0.466 | 0.702 | 0.466 | 0.702 | 0.205 | 0.330 |
| SCS_train | CARIB_test | context_calendar | +0.202 | 0.564 | 0.273 | 0.564 | 0.212 | 0.425 |
| SCS_train | SCS_test | THERMAL_ALL | +0.423 | 0.796 | 0.423 | 0.801 | 0.228 | 0.593 |
| SCS_train | SCS_test | CONTEXT_ALL | -0.000 | 0.121 | 0.124 | 0.157 | 0.098 | 0.132 |
| SCS_train | GBR_test | THERMAL_ALL | +0.540 | 0.975 | 0.540 | 0.975 | 0.296 | 0.450 |
| SCS_train | GBR_test | CONTEXT_ALL | +0.002 | 0.615 | 0.432 | 0.801 | 0.349 | 0.585 |
| SCS_train | CARIB_test | THERMAL_ALL | +0.427 | 0.702 | 0.427 | 0.702 | 0.216 | 0.515 |
| SCS_train | CARIB_test | CONTEXT_ALL | +0.202 | 0.564 | 0.273 | 0.564 | 0.212 | 0.425 |

## Per-feature table

| source | target | feature | group | SMD | W1(z) | energy(z) |
|---|---|---|---|---|---|---|
| SCS_train | SCS_test | sst | instantaneous_thermal | +0.146 | 0.149 | 0.141 |
| SCS_train | SCS_test | ssta | instantaneous_thermal | +0.608 | 0.611 | 0.477 |
| SCS_train | SCS_test | hotspot | instantaneous_thermal | +0.449 | 0.449 | 0.293 |
| SCS_train | SCS_test | sst_anomaly | instantaneous_thermal | +0.796 | 0.801 | 0.593 |
| SCS_train | SCS_test | d_hotspot_dt | instantaneous_thermal | +0.014 | 0.217 | 0.124 |
| SCS_train | SCS_test | C1 | cumulative_thermal | +0.295 | 0.295 | 0.136 |
| SCS_train | SCS_test | C2 | cumulative_thermal | +0.338 | 0.338 | 0.160 |
| SCS_train | SCS_test | C4 | cumulative_thermal | +0.397 | 0.397 | 0.201 |
| SCS_train | SCS_test | C8 | cumulative_thermal | +0.471 | 0.471 | 0.256 |
| SCS_train | SCS_test | C12 | cumulative_thermal | +0.497 | 0.497 | 0.287 |
| SCS_train | SCS_test | chlorophyll | context_calendar | -0.066 | 0.157 | 0.132 |
| SCS_train | SCS_test | par | context_calendar | +0.040 | 0.151 | 0.116 |
| SCS_train | SCS_test | wind_speed | context_calendar | +0.025 | 0.028 | 0.023 |
| SCS_train | SCS_test | ssr | context_calendar | -0.061 | 0.128 | 0.089 |
| SCS_train | SCS_test | seasonal_sin | context_calendar | +0.121 | 0.121 | 0.108 |
| SCS_train | SCS_test | seasonal_cos | context_calendar | -0.025 | 0.025 | 0.023 |
| SCS_train | GBR_test | sst | instantaneous_thermal | -0.434 | 0.471 | 0.430 |
| SCS_train | GBR_test | ssta | instantaneous_thermal | +0.429 | 0.429 | 0.305 |
| SCS_train | GBR_test | hotspot | instantaneous_thermal | +0.400 | 0.400 | 0.224 |
| SCS_train | GBR_test | sst_anomaly | instantaneous_thermal | +0.591 | 0.591 | 0.450 |
| SCS_train | GBR_test | d_hotspot_dt | instantaneous_thermal | +0.025 | 0.134 | 0.065 |
| SCS_train | GBR_test | C1 | cumulative_thermal | +0.506 | 0.506 | 0.221 |
| SCS_train | GBR_test | C2 | cumulative_thermal | +0.573 | 0.573 | 0.245 |
| SCS_train | GBR_test | C4 | cumulative_thermal | +0.666 | 0.666 | 0.288 |
| SCS_train | GBR_test | C8 | cumulative_thermal | +0.831 | 0.831 | 0.368 |
| SCS_train | GBR_test | C12 | cumulative_thermal | +0.975 | 0.975 | 0.447 |
| SCS_train | GBR_test | chlorophyll | context_calendar | -0.335 | 0.335 | 0.311 |
| SCS_train | GBR_test | par | context_calendar | -0.554 | 0.801 | 0.585 |
| SCS_train | GBR_test | wind_speed | context_calendar | +0.615 | 0.641 | 0.536 |
| SCS_train | GBR_test | ssr | context_calendar | +0.528 | 0.528 | 0.388 |
| SCS_train | GBR_test | seasonal_sin | context_calendar | -0.000 | 0.002 | 0.002 |
| SCS_train | GBR_test | seasonal_cos | context_calendar | +0.003 | 0.003 | 0.003 |
| SCS_train | CARIB_test | sst | instantaneous_thermal | +0.052 | 0.298 | 0.227 |
| SCS_train | CARIB_test | ssta | instantaneous_thermal | +0.639 | 0.639 | 0.515 |
| SCS_train | CARIB_test | hotspot | instantaneous_thermal | +0.520 | 0.520 | 0.341 |
| SCS_train | CARIB_test | sst_anomaly | instantaneous_thermal | +0.243 | 0.243 | 0.199 |
| SCS_train | CARIB_test | d_hotspot_dt | instantaneous_thermal | -0.000 | 0.128 | 0.094 |
| SCS_train | CARIB_test | C1 | cumulative_thermal | +0.335 | 0.335 | 0.155 |
| SCS_train | CARIB_test | C2 | cumulative_thermal | +0.389 | 0.389 | 0.173 |
| SCS_train | CARIB_test | C4 | cumulative_thermal | +0.466 | 0.466 | 0.205 |
| SCS_train | CARIB_test | C8 | cumulative_thermal | +0.594 | 0.594 | 0.268 |
| SCS_train | CARIB_test | C12 | cumulative_thermal | +0.702 | 0.702 | 0.330 |
| SCS_train | CARIB_test | chlorophyll | context_calendar | -0.087 | 0.120 | 0.097 |
| SCS_train | CARIB_test | par | context_calendar | +0.428 | 0.428 | 0.327 |
| SCS_train | CARIB_test | wind_speed | context_calendar | +0.401 | 0.425 | 0.379 |
| SCS_train | CARIB_test | ssr | context_calendar | +0.564 | 0.564 | 0.425 |
| SCS_train | CARIB_test | seasonal_sin | context_calendar | -0.001 | 0.003 | 0.004 |
| SCS_train | CARIB_test | seasonal_cos | context_calendar | +0.003 | 0.003 | 0.003 |

## Auto-flagged extremes (per pair)

### SCS_train -> SCS_test
- Largest |SMD|: sst_anomaly (+0.80, instantaneous_thermal); ssta (+0.61, instantaneous_thermal); C12 (+0.50, cumulative_thermal); C8 (+0.47, cumulative_thermal); hotspot (+0.45, instantaneous_thermal)
- Largest W1(z): sst_anomaly (0.80); ssta (0.61); C12 (0.50); C8 (0.47); hotspot (0.45)

### SCS_train -> GBR_test
- Largest |SMD|: C12 (+0.97, cumulative_thermal); C8 (+0.83, cumulative_thermal); C4 (+0.67, cumulative_thermal); wind_speed (+0.62, context_calendar); sst_anomaly (+0.59, instantaneous_thermal)
- Largest W1(z): C12 (0.97); C8 (0.83); par (0.80); C4 (0.67); wind_speed (0.64)

### SCS_train -> CARIB_test
- Largest |SMD|: C12 (+0.70, cumulative_thermal); ssta (+0.64, instantaneous_thermal); C8 (+0.59, cumulative_thermal); ssr (+0.56, context_calendar); hotspot (+0.52, instantaneous_thermal)
- Largest W1(z): C12 (0.70); ssta (0.64); C8 (0.59); ssr (0.56); hotspot (0.52)

## Performance anchor (frozen TSCF, ocean-masked)

| target | n | severe | ROC | PR | CSI@0.9 |
|---|---|---|---|---|---|
| SCS_test | 49322 | 419 | 0.9865 | 0.5463 | 0.3374 |
| GBR_test | 19065 | 676 | 0.9624 | 0.5543 | 0.3648 |
| CARIB_test | 83025 | 1356 | 0.6101 | 0.0321 | 0.0262 |

NOTE: association language only — this analysis does NOT establish that distribution shift causes the transfer performance difference.

## Interpretation (2026-09-17) — association language only

### Headline: marginal shift magnitude does NOT rank-order the transfer outcome

| target | thermal W1(z) median | context W1(z) median | TSCF ROC |
|---|---|---|---|
| SCS_test (in-domain) | 0.423 | 0.124 | 0.9865 (transfer OK) |
| GBR_test | 0.540 | 0.432 | 0.9624 (transfer OK) |
| CARIB_test | **0.427** | **0.273** | 0.6101 (FAILURE) |

CARIB test shows the SMALLEST overall marginal shift among the two zero-shot
domains (thermal W1 median 0.427 vs GBR 0.540; context 0.273 vs 0.432), yet
the frozen TSCF fails exactly there and transfers to GBR. The simple story
"larger marginal shift causes the failure" is NOT supported by this analysis.

### Specific observations

- **Cumulative thermal shift is SMALLER for CARIB than for GBR on every
  C_W channel** (SMD: C1 0.335 vs 0.506, C2 0.389 vs 0.573, C4 0.466 vs 0.666,
  C8 0.594 vs 0.831, C12 0.702 vs 0.975). The transferable thermal-stress
  representation is therefore NOT more shifted on the failing basin — the
  CARIB failure is not associated with a larger cumulative-thermal marginal
  shift.
- **Absolute SST is nearly SCS-matched on CARIB** (SMD +0.052) while GBR is
  cooler (−0.434); CARIB's ERA5 sst_anomaly is the CLOSEST of the three test
  populations (+0.243 vs GBR +0.591 and in-domain SCS_test +0.796). The
  failing basin is not the one with the most shifted absolute-thermal
  marginals.
- **Context/calendar shifts do not separate the outcomes either**: GBR's
  context W1 median (0.432) is LARGER than CARIB's (0.273) — including
  PAR (W1 0.801 vs 0.428) and wind (0.641 vs 0.425) — yet GBR transfers.
  ssr is the single context feature with comparable shifts on both
  (0.528 vs 0.564).
- Positive SMD signs on C4/C8/C12 and hotspot/ssta across ALL test
  populations reflect that every test window is more thermally stressed than
  the 2003-2010 training period (warming + event-rich test years), in-domain
  and zero-shot alike.
- The only features where CARIB exceeds GBR in shift: hotspot (SMD 0.520 vs
  0.400), ssta (0.639 vs 0.429), ssr (0.564 vs 0.528) — small differences in
  the same direction as real 2017-2020 Caribbean thermal events, and of
  similar magnitude to the IN-DOMAIN SCS_test temporal drift (ssta +0.608,
  hotspot +0.449), which the model handles.

### Reading

Per-feature marginal distances (SMD, W1, energy) on the SCS-standardized
input space do NOT correspond to the cross-basin transfer outcome ordering.
This is consistent with the earlier audit findings (CARIB 13-channel z-scores
in-family; test-time re-normalization with CARIB statistics did not recover
skill) and indicates that the CARIB failure is not associated with a larger
marginal input shift. A marginal analysis cannot capture shifts in the JOINT
dependence structure (channel co-variation and temporal dynamics) that the
ConvLSTM+attention backbone may have learned on SCS; the failure pattern is
consistent with such a relational/representational mismatch rather than with
per-feature distributional distance. Establishing that mechanism would
require joint-distribution or representation-level analyses, which are
outside the scope of this marginal audit.

ED Fig 9 base data: feature_shift.csv / wasserstein.csv /
energy_distance.csv / ed_fig9_draft.png (draft only; formal figure untouched).
