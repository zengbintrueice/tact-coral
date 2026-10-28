# Joint-Dependence Structure Shift (follow-up to the marginal audit)

Tests the hypothesis the marginal analysis could not: relational mismatch (channel co-variation, channel→future-target mapping, temporal dynamics).
Ocean-masked; no model training; reference = SCS train 2003-2010.

## J1. Correlation-matrix distance vs SCS-train

| Population | n_px | Frobenius vs ref | mean |dcorr| (off-diag) |
|---|---|---|---|
| SCS_train | 113007 | 0.000 | 0.000 |
| SCS_test | 50406 | 0.867 | 0.050 |
| GBR_test | 19437 | 4.722 | 0.247 |
| CARIB_test | 84645 | 2.300 | 0.135 |

### Largest channel-pair departures (top 5 per population)

| Population | Pair | corr(ref) | corr(dom) | |dcorr| |
|---|---|---|---|---|
| SCS_train | wind_speed × sst_anomaly | -0.110 | -0.110 | 0.000 |
| SCS_train | wind_speed × ssr | -0.053 | -0.053 | 0.000 |
| SCS_train | wind_speed × seasonal_cos | +0.384 | +0.384 | 0.000 |
| SCS_train | wind_speed × hotspot_cumsum_4w | -0.242 | -0.242 | 0.000 |
| SCS_train | wind_speed × hotspot_cumsum_12w | -0.275 | -0.275 | 0.000 |
| SCS_test | hotspot_cumsum_12w × seasonal_cos | -0.351 | -0.581 | 0.231 |
| SCS_test | hotspot_cumsum_4w × seasonal_cos | -0.402 | -0.596 | 0.194 |
| SCS_test | hotspot × seasonal_cos | -0.362 | -0.537 | 0.174 |
| SCS_test | hotspot × ssr | +0.234 | +0.360 | 0.126 |
| SCS_test | par × hotspot_cumsum_12w | +0.247 | +0.373 | 0.126 |
| GBR_test | ssr × seasonal_cos | -0.512 | +0.841 | 1.352 |
| GBR_test | sst × seasonal_cos | -0.586 | +0.573 | 1.160 |
| GBR_test | seasonal_sin × hotspot_cumsum_12w | -0.352 | +0.686 | 1.038 |
| GBR_test | par × seasonal_sin | -0.005 | +0.998 | 1.004 |
| GBR_test | sst × seasonal_sin | -0.264 | +0.560 | 0.825 |
| CARIB_test | par × wind_speed | -0.254 | +0.251 | 0.505 |
| CARIB_test | hotspot_cumsum_12w × seasonal_cos | -0.351 | +0.145 | 0.496 |
| CARIB_test | wind_speed × seasonal_cos | +0.384 | -0.038 | 0.422 |
| CARIB_test | wind_speed × ssr | -0.053 | +0.358 | 0.411 |
| CARIB_test | hotspot × seasonal_sin | -0.046 | -0.447 | 0.401 |

## J2. Channel → y(t+4) association (Spearman, shared PFV record set)

| Channel | SCS_test | GBR_test | CARIB_test |
|---|---|---|---|
| sst | +0.708 | +0.605 | +0.393 |
| ssta | +0.245 | +0.363 | +0.284 |
| hotspot | +0.741 | +0.577 | +0.440 |
| chlorophyll | +0.105 | +0.029 | -0.014 |
| par | +0.593 | +0.672 | +0.019 |
| seasonal_sin | -0.088 | +0.655 | -0.499 |
| wind_speed | -0.417 | -0.463 | -0.191 |
| ssr | +0.476 | +0.535 | +0.003 |
| sst_anomaly | +0.130 | +0.256 | +0.187 |
| d_hotspot_dt | +0.021 | -0.019 | -0.045 |
| hotspot_cumsum_4w | +0.814 | +0.666 | +0.467 |
| hotspot_cumsum_12w | +0.751 | +0.655 | +0.423 |
| seasonal_cos | -0.743 | +0.410 | -0.127 |

## J3. Temporal dynamics (pooled lag-1 / lag-4 autocorrelation)

| Channel | SCS_train l1/l4 | SCS_test l1/l4 | GBR_test l1/l4 | CARIB_test l1/l4 |
|---|---|---|---|---|
| sst | +0.978 / +0.866 | +0.978 / +0.866 | +0.977 / +0.879 | +0.971 / +0.843 |
| ssta | +0.804 / +0.426 | +0.804 / +0.426 | +0.805 / +0.507 | +0.849 / +0.567 |
| hotspot | +0.801 / +0.404 | +0.801 / +0.404 | +0.803 / +0.418 | +0.895 / +0.611 |
| chlorophyll | +0.994 / +0.975 | +0.994 / +0.975 | +0.996 / +0.984 | +0.996 / +0.985 |
| par | +0.930 / +0.720 | +0.930 / +0.720 | +0.993 / +0.886 | +0.946 / +0.784 |
| seasonal_sin | +0.993 / +0.886 | +0.993 / +0.886 | +0.993 / +0.886 | +0.993 / +0.887 |
| wind_speed | +0.932 / +0.726 | +0.932 / +0.726 | +0.921 / +0.685 | +0.963 / +0.853 |
| ssr | +0.948 / +0.793 | +0.948 / +0.793 | +0.949 / +0.795 | +0.957 / +0.828 |
| sst_anomaly | +0.889 / +0.554 | +0.889 / +0.554 | +0.939 / +0.755 | +0.930 / +0.721 |
| d_hotspot_dt | -0.049 / -0.052 | -0.049 / -0.052 | -0.080 / -0.047 | -0.017 / +0.038 |
| hotspot_cumsum_4w | +0.952 / +0.556 | +0.952 / +0.556 | +0.953 / +0.572 | +0.972 / +0.695 |
| hotspot_cumsum_12w | +0.985 / +0.819 | +0.985 / +0.819 | +0.985 / +0.816 | +0.989 / +0.842 |
| seasonal_cos | +0.993 / +0.886 | +0.993 / +0.886 | +0.993 / +0.886 | +0.993 / +0.885 |

## J4. What the frozen TSCF prediction actually tracks (Spearman, shared record set)

| Population | ρ(pred, PersA) | ρ(pred, hotspot) | ρ(pred, cumsum12) | ρ(pred, sst_anomaly) | ρ(pred, seasonal_sin) |
|---|---|---|---|---|---|
| SCS_test | +0.707 | +0.438 | +0.757 | -0.020 | -0.584 |
| GBR_test | +0.518 | +0.314 | +0.625 | +0.332 | +0.621 |
| CARIB_test | +0.219 | +0.109 | +0.341 | +0.309 | -0.110 |

Interpretation: if J1-J3 show CARIB departing the reference joint structure more than GBR (while the marginal audit showed the opposite ordering), the failure is associated with relational/representational mismatch rather than marginal distribution distance. Association language only; no causal claims.

## Findings (2026-09-18)

1. **Static co-variation (J1) does NOT rank-order the outcome either**:
   GBR departs the SCS-train correlation structure MORE than CARIB
   (Frobenius 4.72 vs 2.30) yet transfers. The naive "relational distance
   causes failure" story fails at the static-covariance level too.
2. **Temporal dynamics (J3) are essentially domain-invariant**: lag-1/lag-4
   autocorrelations of the thermal channels are near-identical across all
   populations (hotspot lag1 0.80-0.90; cumsum12 lag1 0.985-0.989). The
   ConvLSTM's exploitable temporal structure is preserved everywhere.
3. **The discriminator is the channel→FUTURE-TARGET mapping (J2)**:
   - Thermal channels stay (moderately) predictive on both targets:
     cumsum4→y Spearman 0.814 (SCS) / 0.666 (GBR) / 0.467 (CARIB).
   - Context/calendar/light associations are BASIN-IDIOSYNCRATIC and flip or
     collapse: seasonal_sin→y −0.088 (SCS) vs +0.655 (GBR) vs −0.499 (CARIB);
     seasonal_cos→y −0.743 / +0.410 / −0.127; par→y +0.593 / +0.672 / +0.019;
     ssr→y +0.476 / +0.535 / +0.003. The southern-hemisphere seasonal flip
     (GBR) and the Caribbean collapse of light/seasonal associations are
     exactly the kind of sign/level changes a frozen SCS-learned blend cannot
     absorb.
4. **J4 confirms the prediction is aligned with the learned SCS joint pattern
   and loses that alignment only on CARIB**: ρ(pred, cumsum12) 0.757 (SCS) /
   0.625 (GBR) / 0.341 (CARIB); ρ(pred, PersA) 0.707 / 0.518 / 0.219; on
   SCS/GBR the prediction also tracks seasonal phase (|ρ| 0.58-0.62), which is
   precisely the learned component whose target-association flips across
   basins.

**Synthesis with the marginal audit**: marginals, static co-variation, and
temporal dynamics all FAIL to rank-order the transfer outcome; the
channel→future-target association table (J2) is the one view in which the
failing basin stands out (context/calendar/light associations flip or
collapse). Combined with Step-6/8 (env context is SCS-local; thermal channels
carry transfer) this explains why the thermal-only physics anchor (PersA_DHW,
which by construction uses ONLY the surviving thermal association) rescues the
Caribbean while the full contextual representation collapses.
