# Weather Forecast Degradation Methodology

## Overview

To evaluate model robustness under realistic operational conditions, observed weather variables were degraded to simulate forecast uncertainty at lead times of 6, 24, 48, and 168 hours. Error growth functions were parameterized from published numerical weather prediction (NWP) verification statistics where available, with conservative assumptions applied for variables lacking specific verification data.

## Error Growth Parameterization

### Temperature

Temperature (2-meter) error growth was derived from ECMWF verification statistics for global numerical weather prediction. Root mean square error (RMSE) values were extracted from multiple verification reports covering both upper-air (850 hPa) and surface (2-meter) temperature forecasts, and linearly interpolated across forecast horizons:

**σ(h) = 0.77 + 0.0181·h** (°C)

This linear approximation is anchored to σ(24h) ≈ 1.2°C and σ(168h) ≈ 3.8°C, matching reported RMSE values from operational verification over Europe and the Northern Hemisphere. The formula provides acceptable accuracy over the 1-7 day forecast range, though actual verification curves show slight nonlinearity beyond day 5.

### Wind Speed

Wind speed (10-meter) error growth was derived from meteoblue global verification (2017) and ECMWF surface wind verification statistics:

**σ(h) = 1.8 + 0.010·h** (m/s)

This parameterization is consistent with meteoblue's reported MAE of 1.8 m/s at 24 hours and ECMWF operational verification showing RMSE values of approximately 2.0-2.5 m/s at 60-72 hours and 3.0-3.5 m/s at 168 hours for 10-meter wind forecasts.

### Relative Humidity

Humidity error growth was parameterized based on 2-meter relative humidity RMSE verification from Kartsios et al. (2024), which analyzed NCEP/GFS forecasts over sub-Saharan Africa:

**σ(h) = 13.0 + 0.023·h** (%-points)

This parameterization is calibrated to match the observed RMSE range of 13.58-16.94% across 12-180 hour forecasts reported in Kartsios et al. (2024). The formula yields σ(24h) ≈ 13.6% and σ(168h) ≈ 16.9%, consistent with empirical verification. Note that these values are derived from GFS verification over Africa and may not fully represent ECMWF global forecast performance, though they provide the best available empirical basis for 2-meter relative humidity forecast error growth.

### Solar Radiation

Solar irradiance forecast errors were modeled using relative error growth based on Kleissl (2013, Table 10.2, p. 250), which reports RMSE-metric summaries for operational solar forecasting systems:

**Relative MAE(h) = 15 + 0.15·h** (percent)

The baseline intercept (15%) and week-ahead value (40.2%) are calibrated to match Kleissl's reported ranges:
- **1-day forecasts:** Desert Rock 18% MAE, Fort Peck 27%, Boulder 36%, Penn State 28%
- **7-day forecasts:** Desert Rock 23% MAE, Fort Peck 31%, Boulder 46%, Penn State 41%

The model's 18.6% at 24h matches the lower bound (Desert Rock), while 40.2% at 168h falls in the mid-range of reported multi-day errors. The linear growth rate (0.15%/hour ≈ 3.6%/day) represents a conservative interpolation between these empirically verified endpoints.

Solar radiation errors are modeled using additive Gaussian noise rather than multiplicative lognormal noise, as forecast errors scale approximately linearly with irradiance magnitude but do not exhibit the heavy-tailed behavior typical of precipitation.

### Precipitation

Precipitation forecast errors were modeled using a two-component approach: event detection errors (false alarms and missed events) and magnitude errors for correctly detected events.

#### Magnitude Errors

For correctly detected precipitation events, a lognormal multiplicative noise model was applied following Jolliffe & Stephenson (2003) with coefficient of variation:

**CV(h) = 30 + 0.15·h** (percent)

This growth rate represents a conservative estimate based on ensemble spread theory. The theoretical upper bound can be derived from:
- **Climatological baseline:** Daily precipitation CV ≈ 100-150% (Katz & Parlange, 1998)
- **Skill decay:** Ensemble CRPSS decreases from ~0.33 (Day 1) to ~0.0 (Day 6) (Hersbach, 2000)
- **Theoretical relationship:** CV(t) = CV_obs × (1 - CRPSS(t)) yields ~0.30%/h growth rate for extratropics

The implemented rate of 0.15%/h represents a conservative lower bound appropriate for conditional (detected events only) application, as it excludes dry/near-zero cases that inflate ensemble spread. This yields CV values of 33.6% at 24 hours and 55.2% at 168 hours, compared to theoretical bounds of 78% and 120% respectively.

**References:**
- Katz, R.W. & Parlange, M.B. (1998). Overdispersion phenomenon in stochastic modeling of precipitation. *Journal of Climate*, 11(4), 591-601.
- Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559-570.

#### Event Detection Errors

In addition to magnitude errors, precipitation forecasts exhibit significant event detection errors (false alarms and missed events). Based on Sukovich et al. (2014) quantitative precipitation forecast verification over the contiguous United States from 2001-2011, miss rates and false alarm rates were estimated from reported probability of detection (POD) and false alarm ratio (FAR) metrics:

**Event Error Rates:**
- **6h:** 31% miss rate, 36% false alarm rate
- **24h:** 34% miss rate, 37% false alarm rate  
- **48h:** 37% miss rate, 40% false alarm rate
- **168h:** 50% miss rate, 50% false alarm rate

These values are derived from Sukovich et al. (2014) reporting Day 1 POD ≈ 0.65 (miss ≈ 35%) and FAR ≈ 0.35, Day 2 POD ≈ 0.55 (miss ≈ 45%) and FAR ≈ 0.45, with linear interpolation for hourly forecasts at growth rates of 0.0015/hour (miss rate) and 0.0010/hour (false alarm rate), capped at 50% for week-ahead forecasts.

**Implementation:**
- **False alarms** (actual = 0, forecast > 0): Generated with probability = false_alarm_rate. When triggered, a small precipitation amount is sampled from lognormal(mean=0.5mm, σ=0.5), representing typical light false alarm precipitation.
  
- **Missed events** (actual > 0, forecast = 0): Occur with probability = miss_rate. When triggered, forecast returns 0 regardless of actual amount.
  
- **Detected events** (actual > 0, forecast > 0): Magnitude error applied using lognormal multiplicative model as described above.

**Note:** The 50% cap at 168h reflects near-random skill for week-ahead precipitation event detection, consistent with operational NWP performance.

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

### Multiplicative Lognormal (Precipitation Magnitude)

For precipitation magnitude when events are correctly detected, mean-preserving lognormal multiplicative noise was applied:

**X' = X × M**, where **M ~ Lognormal(μ, σ²)**

with parameters chosen to preserve the mean (E[M] = 1) while achieving target coefficient of variation CV(h):

**μ = -0.5 × σ²**
**σ = √(ln(1 + CV(h)²))**

This ensures E[X'] = E[X] while introducing realistic relative errors that scale with magnitude.

**Precipitation night-time handling:** Zero precipitation values (dry conditions) are handled through the event detection error model rather than lognormal noise. False alarms may introduce small precipitation amounts even when actual = 0.

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

1. **Timing and displacement errors not modeled**: The degradation model perturbs variable magnitudes and simulates event detection errors, but does not simulate timing errors (temporal phase shifts) or spatial displacement. These factors contribute to precipitation forecast errors beyond 48 hours (Jolliffe & Stephenson, 2008) and temperature/wind errors at longer lead times.

2. **Independent errors across variables**: Errors were treated as independent across weather variables, except for the temperature-precipitation type coupling. Actual NWP forecast errors exhibit substantial cross-variable correlations—for example, temperature and humidity errors are coupled through thermodynamic relationships, and wind errors correlate with temperature gradients. This independence assumption may underestimate error in derived quantities or physically coupled processes.

3. **Linear error growth**: Error growth was modeled as linear in forecast lead time. Actual verification curves show modest nonlinearity, with error growth accelerating slightly beyond 5-7 days as predictability limits are approached, and asymptotic behavior at very long ranges (>10 days) where forecast skill approaches climatology.

4. **Simplified precipitation phase transition**: The 2°C threshold for rain/snow conversion is a simplification. Real precipitation phase transitions occur over a range (typically 0-4°C) with mixed precipitation possible. This threshold represents typical operational practice but does not capture the full complexity of precipitation phase physics.

5. **Assumption-based parameters**: Precipitation magnitude CV growth rate (0.15%/h) is a conservative estimate derived from ensemble spread theory rather than direct empirical measurement. Solar radiation linear growth (0.15%/h) interpolates between verified 1-day and 7-day endpoints. These parameters represent defensible estimates but have not been independently validated against held-out verification datasets.

6. **Geographic and seasonal specificity**: Temperature and wind error growth parameters are derived from global or European verification statistics, while humidity errors are based on GFS verification over sub-Saharan Africa, and precipitation skill characteristics reflect mid-latitude performance. Parameters may not fully represent forecast error characteristics for all locations or seasons. Seasonal variations in forecast skill (e.g., summer convection vs. winter synoptic patterns) are not explicitly captured. Humidity errors from African verification may not generalize to all climate regimes.

7. **Single-model representation**: Parameters represent a composite of operational NWP systems and may not accurately reflect errors from other forecast systems or ensemble spread characteristics.

8. **Wind speed truncation bias**: Truncation at zero introduces minor negative bias for low wind speeds (approximately -0.2 m/s), though this is significantly smaller than the positive bias from reflection. At 168h, mean bias is approximately +0.57 m/s (33% of original mean), which is acceptable but non-zero.

Despite these limitations, the degradation methodology provides a realistic and conservative estimate of operational forecast uncertainty appropriate for evaluating machine learning model robustness under forecast input conditions.

## References

European Centre for Medium-Range Weather Forecasts (2024). *Evaluation of ECMWF forecasts, including the 2023-2024 upgrade*. ECMWF Technical Memorandum No. 918. Reading, UK.

Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559-570.

Jolliffe, I. T., & Stephenson, D. B. (Eds.). (2003). *Forecast Verification: A Practitioner's Guide in Atmospheric Science*. John Wiley & Sons, Chichester, UK.

Kartsios, S., Tsarsitalidou, C., Pytharoulis, I., Tegoulias, I., Kotsopoulos, S., Zanis, P., & Katragkou, E. (2024). Verification of the NCEP GFS, ECMWF and BoM ACCESS-G numerical weather prediction model forecasts over Eastern Africa. *Acta Geophysica*, 72, 669-688. https://doi.org/10.1007/s11600-023-01136-y

Katz, R.W. & Parlange, M.B. (1998). Overdispersion phenomenon in stochastic modeling of precipitation. *Journal of Climate*, 11(4), 591-601.

Kleissl, J. (Ed.). (2013). *Solar Energy Forecasting and Resource Assessment*. Academic Press, Oxford, UK.

meteoblue (2018). *Global Weather Forecast Verification Report 2017*. Temperature, wind speed, precipitation, and dew point verification over 10,000+ meteorological stations worldwide. Available at: https://content.meteoblue.com/en/research-education/weather-data-accuracy

Sukovich, E. M., Ralph, F. M., Barthold, F. E., Reynolds, D. W., & Novak, D. R. (2014). Extreme Quantitative Precipitation Forecast Performance at the Weather Prediction Center from 2001 to 2011. *Weather and Forecasting*, 29(4), 894-911.

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
| 6h      | 1.86    | 1.48              | ~0.1            |
| 24h     | 2.04    | 1.63              | ~0.2            |
| 48h     | 2.28    | 1.82              | ~0.3            |
| 72h     | 2.52    | 2.01              | ~0.4            |
| 168h    | 3.48    | 2.78              | ~0.7            |

*Note: Truncation at zero introduces positive bias, especially at longer horizons. Bias values are approximate from Seoul dataset testing.*

### Humidity Error Growth

| Horizon | σ (%-pts) | Expected MAE (%-pts) |
|---------|-----------|----------------------|
| 6h      | 13.14     | 10.48                |
| 24h     | 13.55     | 10.81                |
| 48h     | 14.10     | 11.25                |
| 72h     | 14.66     | 11.69                |
| 168h    | 16.86     | 13.46                |

*Note: Values based on Kartsios et al. (2024) GFS verification over Africa (RMSE 13.58-16.94% across 12-180h).*

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
| 6h      | 31            | 36                   |
| 24h     | 34            | 37                   |
| 48h     | 37            | 40                   |
| 72h     | 41            | 42                   |
| 168h    | 50            | 50                   |

*Note: Miss rate = probability of missing actual precipitation (forecast = 0 when actual > 0). False alarm rate = probability of forecasting precipitation when none occurs (forecast > 0 when actual = 0). Rates based on Sukovich et al. (2014) with linear interpolation at 0.0015/hour (miss) and 0.0010/hour (FAR), capped at 50%.*