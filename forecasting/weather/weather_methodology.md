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

These values are an assumption: no published verification of the magnitude error of correctly detected hourly precipitation by lead time was found. The formula yields CV values of 33.6% at 24 hours and 55.2% at 168 hours. The magnitude error only affects hours in which precipitation is observed and correctly forecast.

#### Event Detection Errors

In addition to magnitude errors, precipitation forecasts exhibit significant event detection errors (false alarms and missed events). Based on Sukovich et al. (2014) quantitative precipitation forecast verification over the contiguous United States from 2001-2011, miss rates and false alarm ratios were estimated from reported probability of detection (POD) and false alarm ratio (FAR) metrics:

**Event Error Statistics:**
- **6h:** 27.5% miss rate, FAR 27.5%
- **24h:** 35% miss rate, FAR 35%
- **48h:** 45% miss rate, FAR 45%
- **≥60h (incl. 168h):** 50% miss rate, FAR 50%

Sukovich et al. (2014) report Day 1 POD ≈ 0.65 (miss ≈ 35%) and FAR ≈ 0.35, Day 2 POD ≈ 0.55 (miss ≈ 45%) and FAR ≈ 0.45, and Day 3 miss rate and FAR ≈ 45–55%. Miss rate and FAR both follow the straight line through the Day-1 (24 h) and Day-2 (48 h) values:

**miss rate(h) = FAR(h) = 0.25 + 0.10 · h / 24**, capped at 50%

The cap is reached at 60 h, within the Day-3 range. Below 24 h the line is extrapolated (25.4% at 1 h).

**From false alarm ratio to false-alarm probability per dry hour:**

The miss rate (1 − POD) is conditional on an observed event, P(no forecast | precipitation), and is applied directly to wet hours. The false alarm ratio is conditional on a *forecast* event: FAR = false alarms / (false alarms + hits) = P(no precipitation | precipitation forecast). To simulate forecasts for dry hours, the probability P(precipitation forecast | no precipitation) is needed. With hits H = POD · N_wet and FAR = F / (F + H), the number of false alarms is F = FAR / (1 − FAR) · POD · N_wet, so per dry hour:

**P(false alarm | dry) = FAR / (1 − FAR) · POD · p / (1 − p)**

where *p* is the share of wet hours (rain or snow > 0), computed from the clean training window of each fold (in the same way as the solar cap). With this probability the simulated forecasts reproduce the source statistics: the simulated FAR equals the reported FAR and the simulated POD equals the reported POD. For a wet-hour share of 6%, P(false alarm | dry) is 2.2% at 24 h. A training window without any precipitation gives *p* = 0 and therefore no false alarms in that fold.

**Implementation:**
- **One precipitation variable:** Rainfall and snowfall are treated as one precipitation variable. An hour is wet if rain or snow > 0. Each hour receives one event-detection draw.
- **False alarms** (actual = 0, forecast > 0): Generated with probability P(false alarm | dry). When triggered, a small precipitation amount is sampled from a lognormal distribution with median 0.5 mm (σ_log = 0.5, mean 0.57 mm), representing typical light false alarm precipitation. The amount is written to the rain column; the phase correction (below) converts it to snow when the degraded temperature is below 2 °C.
  
- **Missed events** (actual > 0, forecast = 0): Occur with probability = miss_rate. When triggered, forecast returns 0 regardless of actual amount.
  
- **Detected events** (actual > 0, forecast > 0): Magnitude error applied using lognormal multiplicative model as described above; one multiplier per hour is applied to all non-zero precipitation columns.

**Note:** The 50% cap (from 60 h) reflects near-random skill for event detection beyond Day 2, consistent with the Day-3 values of Sukovich et al. (2014).

### Visibility

Visibility forecast errors were modelled as lognormal multiplicative noise with a constant coefficient of variation:

**CV = 25% (horizon-independent)**

The ECMWF Forecast User Guide (Owens & Hewson, 2018, Section 9.4) explicitly characterises visibility as the lowest-skill surface forecast variable in IFS, noting that the product is experimental and that "expectations regarding the quality of this product should remain low." Critically, Section 9.4.1 states that forecasts with shorter lead times will not necessarily be more skilful than those from longer lead times. This non-monotonic skill relationship makes any horizon-dependent growth rate inconsistent with the primary source; a flat CV is therefore more faithful to the evidence than a formula such as CV(h) = a + b·h.

The CV value of 25% is anchored to Bari & Ouagabi (2020), who report MAE ≈ 1300 m and RMSE ≈ 2000 m at 24h for ML-corrected mesoscale NWP forecasts over mid-latitude stations, implying approximately 25% relative error for a typical mean visibility of 7–8 km. This represents a best-case (post-processed) lower bound; raw NWP parametrization errors frequently exceed 50% (Gultepe et al., 2006).

The lognormal model is mean-preserving (E[multiplier] = 1) with a physical floor at zero.

## Noise Application

### Train / Test Split

Degradation is applied **only to the test (forecast) window**. Training data always uses clean observed weather.

This reflects the operational reality: a model is fitted on historical observations, then deployed with NWP forecast inputs. Degrading training covariates would conflate fitting-time and inference-time uncertainty, obscuring the robustness signal the experiment is designed to measure.

### Per-Row Lead Times

Within a test window of length *h*, each row receives noise calibrated to its own lead time rather than the maximum horizon. Row *i* (0-indexed) represents the forecast for step *i + 1* hours ahead, so it is degraded using σ(i + 1):

- **Row 0** → σ(1 h): the smallest noise of the window  
- **Row h/2** → σ(h/2): mid-range noise  
- **Row h − 1** → σ(h): full-horizon noise

The noise at 1 h is not zero: every error formula has an intercept. At 1 h the errors are σ = 0.79 °C (temperature), 13.0 %-points (humidity), 1.81 m/s (wind), 15% relative MAE (solar radiation), CV 25% (visibility), and a 25% miss rate and FAR (precipitation). The shortest lead time verified in the sources is 12 h (temperature, humidity) or 24 h (wind, solar radiation, precipitation, visibility); below that, the formulas are extrapolated (see Limitations).

This is physically correct because a horizon-*h* NWP forecast covers *h* consecutive future hours, and error grows continuously with lead time. The previous implementation applied the maximum-horizon noise uniformly to every row, which overestimated degradation for all but the final prediction step.

### Two-Phase Degradation Process

Weather degradation is applied in two sequential phases to maintain physical consistency:

**Phase 1: Independent Variable Degradation**
All weather variables are degraded independently according to their respective error models, using per-row lead times as described above. Rainfall and snowfall are degraded together as one precipitation variable (one event-detection draw per hour).

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

3. **Independent errors from hour to hour**: Each hour of a test window receives an independent error draw. Actual NWP forecast errors persist over many hours (e.g. a forecast that is too warm stays too warm for most of a day). The per-hour error magnitude matches the error formulas, but the temporal structure of the errors is not represented.

4. **Extrapolation below the shortest verified lead time**: The shortest lead time verified in the sources is 12 h (temperature, humidity) or 24 h (wind speed, solar radiation, precipitation, visibility). Below that, the error formulas are extrapolated to their intercepts; at 1 h the errors are σ = 0.79 °C (temperature), 13.0 %-points (humidity), 1.81 m/s (wind), 15% relative MAE (solar radiation), CV 25% (visibility), and 25% miss rate and FAR (precipitation). All lead times of the 6 h horizon and the first hours of every longer horizon rely on this extrapolation.

5. **Linear error growth**: Error growth was modeled as linear in forecast lead time. Actual verification curves show modest nonlinearity, with error growth accelerating slightly beyond 5-7 days as predictability limits are approached, and asymptotic behavior at very long ranges (>10 days) where forecast skill approaches climatology.

6. **Simplified precipitation phase transition**: The 2°C threshold for rain/snow conversion is a simplification. Real precipitation phase transitions occur over a range (typically 0-4°C) with mixed precipitation possible. This threshold represents typical operational practice but does not capture the full complexity of precipitation phase physics.

7. **Assumption-based parameters**: Precipitation magnitude CV (30% + 0.15%/h) is an assumption; no published verification of the magnitude error of correctly detected hourly precipitation by lead time was found. Solar radiation linear growth (0.15%/h) interpolates between verified 1-day and 7-day endpoints. These parameters represent defensible estimates but have not been independently validated against held-out verification datasets.

8. **Geographic and seasonal specificity**: Temperature and wind error growth parameters are derived from global or European verification statistics, while humidity errors are based on GFS verification over sub-Saharan Africa, and precipitation skill characteristics reflect mid-latitude performance. Parameters may not fully represent forecast error characteristics for all locations or seasons. Seasonal variations in forecast skill (e.g., summer convection vs. winter synoptic patterns) are not explicitly captured. Humidity errors from African verification may not generalize to all climate regimes.

9. **Single-model representation**: Parameters represent a composite of operational NWP systems and may not accurately reflect errors from other forecast systems or ensemble spread characteristics.

10. **Event detection statistics from extreme events**: The POD and FAR values of Sukovich et al. (2014) were verified for the top 1% of 24-hour precipitation events on a 32-km grid. They are applied here to all hourly precipitation events. Detection skill for ordinary hourly precipitation may differ.

11. **Wind speed truncation bias**: Truncation at zero introduces minor negative bias for low wind speeds (approximately -0.2 m/s), though this is significantly smaller than the positive bias from reflection. At 168h, mean bias is approximately +0.57 m/s (33% of original mean), which is acceptable but non-zero.

Despite these limitations, the degradation methodology provides a realistic and conservative estimate of operational forecast uncertainty appropriate for evaluating machine learning model robustness under forecast input conditions.

## References

Bari, D. & Ouagabi, A. (2020). Machine-learning regression applied to diagnose horizontal visibility from mesoscale NWP model forecasts. Discover Applied Sciences, 2, 389. https://doi.org/10.1007/s42452-020-2327-x

European Centre for Medium-Range Weather Forecasts (2024). *Evaluation of ECMWF forecasts, including the 2023-2024 upgrade*. ECMWF Technical Memorandum No. 918. Reading, UK.

Gultepe, I., Müller, M. D., & Boybeyi, Z. (2006). A new visibility parameterization for warm-fog applications in numerical weather prediction models. Journal of Applied Meteorology and Climatology, 45(11), 1469–1480.

Jolliffe, I. T., & Stephenson, D. B. (Eds.). (2003). *Forecast Verification: A Practitioner's Guide in Atmospheric Science*. John Wiley & Sons, Chichester, UK.

Kartsios, S., Tsarsitalidou, C., Pytharoulis, I., Tegoulias, I., Kotsopoulos, S., Zanis, P., & Katragkou, E. (2024). Verification of the NCEP GFS, ECMWF and BoM ACCESS-G numerical weather prediction model forecasts over Eastern Africa. *Acta Geophysica*, 72, 669-688. https://doi.org/10.1007/s11600-023-01136-y

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

| Horizon | Miss Rate (%) | FAR (%) | P(false alarm \| dry) (%), p = 3% | p = 6% | p = 10% |
|---------|---------------|---------|-----------------------------------|--------|---------|
| 1h      | 25.4          | 25.4    | 0.8                               | 1.6    | 2.8     |
| 6h      | 27.5          | 27.5    | 0.9                               | 1.8    | 3.1     |
| 24h     | 35.0          | 35.0    | 1.1                               | 2.2    | 3.9     |
| 48h     | 45.0          | 45.0    | 1.4                               | 2.9    | 5.0     |
| 72h     | 50.0          | 50.0    | 1.5                               | 3.2    | 5.6     |
| 168h    | 50.0          | 50.0    | 1.5                               | 3.2    | 5.6     |

*Note: Miss rate = 1 − POD = probability of missing actual precipitation (forecast = 0 when actual > 0). FAR = false alarm ratio = share of forecast precipitation events that do not occur (actual = 0 when forecast > 0). P(false alarm | dry) = FAR / (1 − FAR) · POD · p / (1 − p) = probability of forecasting precipitation for a dry hour, where p is the wet-hour share of the training window. Miss rate and FAR = 0.25 + 0.10 · h / 24 (line through the Day-1 and Day-2 values of Sukovich et al., 2014), capped at 50% from 60 h.*

### Visibility Error Growth

| Horizon | CV (%) | Multiplier Range (90% interval) |
|---------|--------|---------------------------------|
| 6h      | 25.0   | 0.64 – 1.50                     |
| 24h     | 25.0   | 0.64 – 1.50                     |
| 48h     | 25.0   | 0.64 – 1.50                     |
| 72h     | 25.0   | 0.64 – 1.50                     |
| 168h    | 25.0   | 0.64 – 1.50                     |

*Note: CV is horizon-independent. ECMWF Section 9.4.1 establishes that visibility forecast skill is non-monotonic with lead time; a growth term is therefore not supported by the primary source. Multiplier ranges represent 5th to 95th percentiles of the mean-preserving lognormal distribution.*