# Weather Forecast Degradation Methodology

## Overview

To evaluate model robustness under realistic operational conditions, observed weather variables were degraded to simulate forecast uncertainty at lead times of 6, 24, 48, and 168 hours. Error growth functions were parameterized from published numerical weather prediction (NWP) verification statistics where available, with conservative assumptions applied for variables lacking specific verification data.

## Error Growth Parameterization

### Temperature

Temperature (2-meter) error growth was derived from ECMWF verification statistics for global numerical weather prediction. Root mean square error (RMSE) values were extracted from multiple verification reports covering both upper-air (850 hPa) and surface (2-meter) temperature forecasts, and linearly interpolated across forecast horizons:

**σ(h) = 0.77 + 0.0181·h** (°C)

This linear approximation is anchored to σ(24h) ≈ 1.2°C and σ(168h) ≈ 3.8°C, matching reported RMSE values from operational verification over Europe and the Northern Hemisphere. The formula provides acceptable accuracy over the 1-7 day forecast range, though actual verification curves show slight nonlinearity beyond day 5.

### Wind Speed

Wind speed (10-meter) error growth was derived from ECMWF surface wind verification statistics:

**σ(h) = 1.4 + 0.008·h** (m/s)

This parameterization is consistent with reported RMSE values of approximately 1.3-1.7 m/s at 24 hours and 2.5-3.0 m/s at 168 hours from operational 10-meter wind forecasts.

### Relative Humidity

Humidity error growth was parameterized from operational NWP verification characteristics:

**σ(h) = 4.5 + 0.068·h** (%-points)

The chosen parameters align with reported ranges of 5-7% at 24 hours and 15-20% at 168 hours observed in operational forecasting systems. This formulation treats the reported values as RMSE directly. If verification statistics report mean absolute error (MAE) instead of RMSE, the implementation overestimates σ by approximately 25% (since correct conversion would be σ = 1.253 × MAE), representing a conservative forecast uncertainty assumption.

### Solar Radiation

Solar irradiance forecast errors were modeled using relative error growth based on Kleissl (2013), which reports 10-20% day-ahead errors and 25-40% multi-day errors for solar forecasting systems. A conservative linear growth schedule was assumed:

**Relative MAE(h) = 15 + 0.15·h** (percent)

The hourly growth rate (0.15%/hour ≈ 3.6%/day) represents a conservative estimate, as specific hourly growth rates are not explicitly reported in published solar forecasting verification studies. This yields relative MAE values of approximately 18.6% at 24 hours and 40.2% at 168 hours, consistent with the upper range of reported multi-day errors.

Solar radiation errors are modeled using additive Gaussian noise rather than multiplicative lognormal noise, as forecast errors scale approximately linearly with irradiance magnitude but do not exhibit the heavy-tailed behavior typical of precipitation.

### Precipitation

Precipitation forecast errors were modeled using a two-component approach: event detection errors (false alarms and missed events) and magnitude errors for correctly detected events.

#### Magnitude Errors

For correctly detected precipitation events, a lognormal multiplicative noise model was applied following Jolliffe & Stephenson (2003) with coefficient of variation:

**CV(h) = 30 + 0.15·h** (percent)

This schedule represents an assumed conservative degradation, as published precipitation verification reports typically focus on categorical skill scores rather than continuous magnitude errors. The 30% base error reflects typical precipitation forecast uncertainty in well-observed regions, while the growth rate (0.15%/hour ≈ 3.6%/day) produces CV values of 33.6% at 24 hours and 55.2% at 168 hours.

#### Event Detection Errors

In addition to magnitude errors, precipitation forecasts exhibit significant event detection errors (false alarms and missed events). Based on typical probability of detection (POD) and false alarm ratio (FAR) characteristics observed in operational mid-latitude forecasting, miss rates and false alarm rates were estimated:

**Event Error Rates:**
- **6h:** 10% miss rate, 10% false alarm rate
- **24h:** 17% miss rate, 17% false alarm rate  
- **48h:** 24% miss rate, 24% false alarm rate
- **168h:** 50% miss rate, 50% false alarm rate

**Implementation:**
- **False alarms** (actual = 0, forecast > 0): Generated with probability = false_alarm_rate. When triggered, a small precipitation amount is sampled from lognormal(mean=0.5mm, σ=0.5), representing typical light false alarm precipitation.
  
- **Missed events** (actual > 0, forecast = 0): Occur with probability = miss_rate. When triggered, forecast returns 0 regardless of actual amount.
  
- **Detected events** (actual > 0, forecast > 0): Magnitude error applied using lognormal multiplicative model as described above.

**Note:** The linear growth rates (0.003/hour ≈ 7.2%/day) and specific percentages represent conservative estimates based on published categorical skill scores rather than directly reported event error rates. The 50% cap at 168h reflects near-random skill for week-ahead precipitation event detection, consistent with operational NWP performance.

### Visibility

In the absence of published visibility forecast verification statistics, lognormal multiplicative noise was applied with assumed relative error:

**CV(h) = 10 + 0.1·h** (percent)

This parameterization is entirely assumption-based and represents a conservative estimate for visibility degradation in the absence of empirical verification data.

## Noise Application

### Two-Phase Degradation Process

Weather degradation is applied in two sequential phases to maintain physical consistency:

**Phase 1: Independent Variable Degradation**
All weather variables are degraded independently according to their respective error models (described below).

**Phase 2: Physical Consistency Correction**
After independent degradation, precipitation types (rain vs. snow) are corrected based on degraded temperature to prevent physically impossible combinations (e.g., snowfall at 15°C or rainfall at -5°C).

### Additive Homoscedastic Gaussian (Temperature, Humidity, Wind)

For temperature, humidity, and wind speed, zero-mean Gaussian noise was applied:

**X' = X + ε**, where **ε ~ N(0, σ(h)²)**

with σ(h) derived from the error growth functions described above.

**Wind speed** uses truncation at zero to prevent negative values:

**X' = max(0, X + ε)**

This introduces minor negative bias for low wind speeds (approximately -0.2 m/s at low speeds) but avoids the large positive bias that results from reflection (folded normal). Testing on the Seoul dataset showed truncation produces more realistic degraded wind speeds, with mean bias of +0.57 m/s at 168h compared to +1.04 m/s with reflection.

**Humidity** was clipped to [0, 100] after noise application to enforce physical bounds.

### Additive Heteroscedastic Gaussian (Solar Radiation)

Solar radiation uses additive Gaussian noise with standard deviation proportional to the observed value:

**X' = X + ε**, where **ε ~ N(0, σ²)** and **σ = 1.253 × (relative_mae/100) × X**

This heteroscedastic formulation produces errors that scale with radiation intensity, consistent with the relative error characteristics observed in solar forecasting verification studies. The factor 1.253 converts relative MAE to standard deviation for Gaussian noise (since σ ≈ 1.253 × MAE for normal distributions).

**Night-time handling:** Zero solar radiation values (nighttime) are not perturbed and remain at 0, correctly representing no solar irradiance.

Degraded values were clipped to [0, P₉₉.₅], where P₉₉.₅ represents the 99.5th percentile of observed solar radiation in the training set (3.18 MJ/m² for the Seoul Bike dataset). This prevents physically implausible values while preserving the natural variability of the original data.

### Multiplicative Lognormal (Precipitation Magnitude, Visibility)

For precipitation magnitude (when events are correctly detected) and visibility, mean-preserving lognormal multiplicative noise was applied:

**X' = X × M**, where **M ~ Lognormal(μ, σ²)**

with parameters chosen to preserve the mean (E[M] = 1) while achieving target coefficient of variation CV(h):

**μ = -0.5 × σ²**
**σ = √(ln(1 + CV(h)²))**

This ensures E[X'] = E[X] while introducing realistic relative errors that scale with magnitude.

**Precipitation night-time handling:** Zero precipitation values (dry conditions) are handled through the event detection error model rather than lognormal noise. False alarms may introduce small precipitation amounts even when actual = 0.

**Visibility floor:** Degraded visibility was clipped to a minimum of 100 meters to prevent physically unrealistic values approaching zero.

## Random Seed Control

All degradation functions accept an optional `seed` parameter for reproducibility:

```python
degraded = degrade_weather(data, horizon_hours=24, seed=42)
```

- Use default (seed=42): Reproducible results across runs
- Specify custom seed (seed=N): Reproducible with different perturbations
- Use random seed (seed=None): Non-reproducible, different results each run

For production pipelines, the default reproducible behavior is recommended.

## Limitations

1. **Timing and displacement errors not modeled**: The degradation model perturbs variable magnitudes and simulates event detection errors, but does not simulate timing errors (temporal phase shifts) or spatial displacement. These factors contribute to precipitation forecast errors beyond 48 hours (Jolliffe & Stephenson, 2003) and temperature/wind errors at longer lead times.

2. **Independent errors across variables**: Errors were treated as independent across weather variables, except for the temperature-precipitation type coupling. Actual NWP forecast errors exhibit substantial cross-variable correlations—for example, temperature and humidity errors are coupled through thermodynamic relationships, and wind errors correlate with temperature gradients. This independence assumption may underestimate error in derived quantities or physically coupled processes.

3. **Linear error growth**: Error growth was modeled as linear in forecast lead time. Actual verification curves show modest nonlinearity, with error growth accelerating slightly beyond 5-7 days as predictability limits are approached, and asymptotic behavior at very long ranges (>10 days) where forecast skill approaches climatology.

4. **Simplified precipitation phase transition**: The 2°C threshold for rain/snow conversion is a simplification. Real precipitation phase transitions occur over a range (typically 0-4°C) with mixed precipitation possible. This threshold represents typical operational practice but does not capture the full complexity of precipitation phase physics.

5. **Assumption-based parameters**: Precipitation event detection rates (10% at 6h, 50% at 168h), coefficient of variation growth (30% + 0.15·h), solar radiation hourly growth rate (0.15%/h), and visibility error schedule (10% + 0.1·h) are based on qualitative guidance from verification literature but lack specific numerical validation. These represent conservative estimates rather than empirically calibrated values.

6. **Geographic and seasonal specificity**: Error growth parameters are derived from global or European verification statistics and mid-latitude precipitation skill characteristics. Parameters may not fully represent forecast error characteristics for all locations or seasons. Seasonal variations in forecast skill (e.g., summer convection vs. winter synoptic patterns) are not explicitly captured.

7. **Single-model representation**: Parameters represent a composite of operational NWP systems and may not accurately reflect errors from other forecast systems or ensemble spread characteristics.

8. **Wind speed truncation bias**: Truncation at zero introduces minor negative bias for low wind speeds (approximately -0.2 m/s), though this is significantly smaller than the positive bias from reflection. At 168h, mean bias is approximately +0.57 m/s (33% of original mean), which is acceptable but non-zero.

Despite these limitations, the degradation methodology provides a realistic and conservative estimate of operational forecast uncertainty appropriate for evaluating machine learning model robustness under forecast input conditions.

## References

European Centre for Medium-Range Weather Forecasts (2024). *Evaluation of ECMWF forecasts, including the 2023-2024 upgrade*. ECMWF Technical Memorandum No. 918. Reading, UK.

Jolliffe, I. T., & Stephenson, D. B. (Eds.). (2003). *Forecast Verification: A Practitioner's Guide in Atmospheric Science*. John Wiley & Sons, Chichester, UK.

Kleissl, J. (Ed.). (2013). *Solar Energy Forecasting and Resource Assessment*. Academic Press, Oxford, UK.

---

## Appendix: Error Growth Tables

### Temperature Error Growth

| Horizon | σ (°C) | Expected MAE (°C) |
|---------|--------|-------------------|
| 6h      | 0.88   | 0.70              |
| 24h     | 1.20   | 0.96              |
| 48h     | 1.64   | 1.31              |
| 72h     | 2.07   | 1.65              |
| 168h    | 3.81   | 3.04              |

*Note: Expected MAE = 0.798 × σ for Gaussian distributions.*

### Wind Speed Error Growth

| Horizon | σ (m/s) | Expected MAE (m/s) | Mean Bias (m/s) |
|---------|---------|-------------------|-----------------|
| 6h      | 1.45    | 1.16              | ~0.1            |
| 24h     | 1.59    | 1.27              | ~0.2            |
| 48h     | 1.78    | 1.42              | ~0.3            |
| 72h     | 1.98    | 1.58              | ~0.4            |
| 168h    | 2.74    | 2.19              | ~0.6            |

*Note: Truncation at zero introduces positive bias, especially at longer horizons. Bias values are approximate from Seoul dataset testing.*

### Humidity Error Growth

| Horizon | σ (%-pts) | Expected MAE (%-pts) |
|---------|-----------|----------------------|
| 6h      | 4.91      | 3.92                 |
| 24h     | 6.13      | 4.89                 |
| 48h     | 7.76      | 6.19                 |
| 72h     | 9.40      | 7.50                 |
| 168h    | 15.92     | 12.70                |

### Solar Radiation Error Growth

| Horizon | Relative MAE (%) | Absolute σ at 2.0 MJ/m² | Absolute MAE at 2.0 MJ/m² |
|---------|------------------|-------------------------|---------------------------|
| 6h      | 15.9             | 0.40 MJ/m²              | 0.32 MJ/m²                |
| 24h     | 18.6             | 0.47 MJ/m²              | 0.37 MJ/m²                |
| 48h     | 22.2             | 0.56 MJ/m²              | 0.44 MJ/m²                |
| 72h     | 25.8             | 0.65 MJ/m²              | 0.52 MJ/m²                |
| 168h    | 40.2             | 1.01 MJ/m²              | 0.80 MJ/m²                |

*Note: Solar radiation uses heteroscedastic Gaussian noise with σ = 1.253 × (relative_mae/100) × value. Values shown are for a typical daytime radiation level of 2.0 MJ/m². Actual σ scales linearly with observed radiation intensity. Night-time values (0 MJ/m²) remain at 0.*

### Precipitation Error Growth

#### Magnitude Errors (for detected events)

| Horizon | CV (%) | Multiplier Range (90% interval) |
|---------|--------|---------------------------------|
| 6h      | 30.9   | 0.52 - 1.62                     |
| 24h     | 33.6   | 0.48 - 1.75                     |
| 48h     | 37.2   | 0.43 - 1.94                     |
| 72h     | 40.8   | 0.38 - 2.17                     |
| 168h    | 55.2   | 0.26 - 3.16                     |

*Note: Multiplier ranges represent 5th to 95th percentiles of the mean-preserving lognormal distribution. These apply only to correctly detected precipitation events.*

#### Event Detection Errors

| Horizon | Miss Rate (%) | False Alarm Rate (%) |
|---------|---------------|----------------------|
| 6h      | 10            | 10                   |
| 24h     | 17            | 17                   |
| 48h     | 24            | 24                   |
| 72h     | 32            | 32                   |
| 168h    | 50            | 50                   |

*Note: Miss rate = probability of missing actual precipitation (forecast = 0 when actual > 0). False alarm rate = probability of forecasting precipitation when none occurs (forecast > 0 when actual = 0). Rates grow linearly at 0.003/hour.*

### Visibility Error Growth

| Horizon | CV (%) | Multiplier Range (90% interval) |
|---------|--------|---------------------------------|
| 6h      | 10.6   | 0.81 - 1.23                     |
| 24h     | 12.4   | 0.77 - 1.29                     |
| 48h     | 14.8   | 0.73 - 1.37                     |
| 72h     | 17.2   | 0.69 - 1.45                     |
| 168h    | 26.8   | 0.59 - 1.69                     |

*Note: Visibility error schedule is assumption-based due to lack of published verification statistics.*
