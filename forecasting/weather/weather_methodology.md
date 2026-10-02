# Weather Forecast Degradation Methodology

Two error models are implemented (`config.degradation_model`):

- **Measured NWP error model** (`"nwp_measured"`, default since 1 Oct 2026; `nwp_error_model.py`): per-city errors measured from real ECMWF forecasts. Described in the next section.
- **Literature error model** (`"literature"`; `weather_degradation.py`): one set of published error sizes for all cities. Described from "Overview (literature model)" on; kept to reproduce earlier results.

## Measured NWP Error Model (default)

### Summary

Observed weather covariates in the test windows were degraded with forecast errors measured for each city from 1,828 operational-resolution ECMWF IFS HRES (9 km) forecasts issued at 00 and 12 UTC between 14 March 2024 and 16 September 2026 (Open-Meteo Single Runs API). Errors were measured against the data source of each city's covariates: the Seoul weather station (WMO 47108, the station of the Seoul data) for Seoul (precipitation included since 2 Oct 2026), and ERA5/ERA5-Land reanalysis for London and Washington, whose covariates come from the Open-Meteo archive. For temperature, humidity and wind speed, the actual errors of a whole forecast run are replayed; solar radiation, precipitation and visibility errors are simulated from per-city, per-lead-time error statistics with measured hour-to-hour persistence. Precipitation is total precipitation including melted snow in every city; snow depth on the ground, for which no forecast errors are available, is forecast by persistence (its value at the time the forecast is issued).

Default setting (revised 1 Oct 2026): an operator with an up-to-date, locally corrected weather forecast. The weather forecast starts when the demand forecast is made, so test hour *i* gets the error of an (*i*+1)-hour forecast in every city (`config.nwp_fresh_forecast = True`), and the average error (bias) of the forecasts is removed (`config.nwp_remove_bias = True`). Both can be switched off; with both off the model is the version first delivered on 1 Oct 2026 (6–17 h old forecast, bias kept). Since 2 Oct 2026 the rain errors also depend on the time of year (`config.nwp_seasonal_rain = True`), Seoul rain is measured against the station, and the rain-frequency bias of the raw forecast is removed (`config.nwp_rain_frequency_unbiased = True`).

### Data

- Forecasts: ECMWF IFS HRES 9 km, nearest grid cell, no elevation downscaling, hourly values as served by the Open-Meteo Single Runs API (beyond 90 h ECMWF outputs 3-/6-hourly values, which Open-Meteo interpolates to hourly). Runs until 12 May 2026 are ECMWF re-runs with IFS cycle 49r1, later runs operational cycle 50r1. Variables: 2 m temperature, 2 m relative humidity, 10 m wind speed, global horizontal irradiance, precipitation (rain + snow, mm), visibility.
- References: NOAA ISD hourly station reports (Seoul 47108, London Heathrow 03772, Washington Reagan National 72405-13743; nearest report within ±30 min; until Aug 2025) and ERA5-seamless reanalysis (ERA5-Land temperature and humidity, ERA5 otherwise; Open-Meteo Historical Weather API, nearest grid cell).
- Downloaded with `weather/nwp/fetch_nwp_data.py`; statistics with `weather/nwp/analyze_nwp_errors.py`; calibration files with `weather/nwp/build_nwp_calibration.py` (`weather/nwp/calibration/<city>.npz`).

### Lead times

A demand forecast is issued at the end of the training window (one hour before the first test hour). The issue time is converted to UTC with the city's time zone (`config.timezone`) and selects the replayed run. Lead times count test hours (rows), not clock time, so a daylight-saving change inside a test window does not shift them.

- **Fresh forecast (default, `nwp_fresh_forecast = True`).** The weather forecast starts at the issue time, so test hour *i* (0-indexed) has lead time *i* + 1 h, the same in every city and at every time of day. This represents a regularly updated forecast (weather services update their short-range forecasts several times a day, some hourly). Errors are replayed from runs whose start hour (00 or 12 UTC) is closest to the issue time of day (at most 6 h apart; on a tie the earlier start).
- **Newest available run (`nwp_fresh_forecast = False`).** The newest ECMWF run available at the issue time is used: runs start at 00 and 12 UTC and are assumed available 6 h later. Test hour *i* gets lead time run age + *i* + 1 h (run age = issue time − run start). The run is 6–17 h old depending on the time of day, so the first test hour has a lead time of 7–18 h. With the experiments' fixed issue hours this gives a fixed age per city (Seoul 14 h, London 10–11 h, Washington 15–16 h for horizons ≥ 24 h), i.e. differences between cities that come from their time zones.

### Temperature, relative humidity, wind speed: replayed forecast errors

For each test window one ECMWF run of the same city is drawn at random (fold seed) among the runs that (a) start at the selected hour (00 or 12 UTC, see Lead times), (b) start within ±30 days of the test window's day of year (any year) and (c) have complete errors at the needed lead times. Its errors e(L) = forecast − reference at the test window's lead times are added to the clean values. With `nwp_remove_bias = True` (default) the average error ē(L) of all these candidate runs (same start hour, same season) is subtracted first:

X'ₜ = Xₜ + e(Lₜ) − ē(Lₜ)  (default),  or  X'ₜ = Xₜ + e(Lₜ)  (bias kept);  humidity clipped to [0, 100] %, wind speed truncated at 0.

Removing ē(L) represents a forecast corrected to local measurements: the systematic part of the error (e.g. Seoul's forecast being on average 1.5 °C colder than the city-centre station, London's wind on average 0.7–0.8 m/s too low) is gone; the hour-to-hour errors remain. The degraded covariates thus carry the real error growth, hour-to-hour persistence and cross-variable relations (e.g. a run that is too warm is often too dry) of ECMWF forecasts for that city and season. Gaps of up to 3 h in the reference are interpolated along the lead time.

### Solar radiation

Heteroscedastic relative error with persistence, per lead time L:

X'ₜ = Xₜ · (1 + b(L) + s(L)·zₜ),  zₜ = φ zₜ₋₁ + √(1−φ²) εₜ

b(L) and s(L) are weighted least-squares estimates over daylight hours of the error model error = observed · (b + s·z) against ERA5; φ is the lag-1 correlation of the standardised errors (leads 1–89 h, observed > 50 W/m²). With `nwp_remove_bias = True`, b = 0. Night hours (X = 0) stay 0; values are capped at the 99.5th percentile of the training fold.

### Precipitation

Precipitation is total precipitation including melted snow (mm, column `precipitation_mm` in every city; Seoul = KMA station amounts, November–March 3-hour totals spread evenly over their three hours; London/Washington = Open-Meteo rain + snowfall / 0.7), one decision per hour (wet = > 0):
- Wet hour: missed (forecast 0) with the measured miss rate m(L, season); otherwise every precipitation column is multiplied by exp(μ_h(L) + σ_h(L)·zₜ), the measured distribution of log(forecast/observed) for hits. With `nwp_remove_bias = True`, μ_h = −σ_h²/2 (mean-preserving multiplier, no systematic over- or underestimate); the miss rate and false-alarm amounts are event rates and stay as measured; false alarms follow `nwp_rain_frequency_unbiased` (below).
- Cap: degraded amounts are at most the training fold's maximum of the column, or the hour's measured amount if larger (since 2 Oct 2026; the lognormal amount error has no upper limit: without the cap, Seoul's degraded rain reached 306 mm/h against a data maximum of 35 mm/h, Washington 52 against 13).
- Dry hour: false alarm with probability FAR(L)/(1 − FAR(L)) · (1 − m(L)) · p/(1 − p) (p = wet-hour share of the training fold; reproduces the measured false-alarm ratio FAR; with `nwp_rain_frequency_unbiased`, the default, m(L)·p/(1 − p), see below), amount lognormal with the measured median and spread of ECMWF amounts in false alarms; written to the precipitation column.
- Persistence: the miss and false-alarm decisions use latent Gaussian AR(1) series whose lag-1 correlations ρ are the tetrachoric correlations of consecutive hours in the ECMWF data (an event that is missed tends to be missed for hours); the hit amount errors follow an AR(1) series with the measured lag-1 correlation φ.
- Rates are pooled over ±12 h of lead time; wet = ≥ 0.1 mm/h (the data resolution).
- Time of year (`nwp_seasonal_rain = True`, default): miss rate, false-alarm ratio and the hit amount error (μ_h, σ_h) are estimated for 12 bins centred on the 15th of each month, from runs starting within ±30 days of the centre (any year); the window is widened in 15-day steps until every pooled lead window (leads 1–168 h) has ≥ 200 wet hours, ≥ 200 forecast-wet hours and ≥ 50 hits. A test window uses the values of its forecast start date, interpolated linearly between the two nearest bin centres. Windows used: London and Washington ±30 days in every month; Seoul ±30 to ±120 days (few wet hours at the station in winter). False-alarm amounts and the persistence parameters are year-round.
- Reference: London and Washington ERA5 (the source of their covariates). Seoul: the station (its bike-data rain is this station's rain: 2017–18 6-/12-hour station totals equal the bike-data sums in 99 % of cases). From 2024 the station reports hourly SYNOPs with a 1-hour amount in every wet hour (the hourly amounts add up to the station's running 24-hour totals, median ratio 1.00); a routine report without an amount and without precipitation in the present-weather group is a dry hour, one with precipitation in the present-weather group but no amount (mostly snow or traces, 5 % of hours) is left out. The resulting series (Mar 2024 – Aug 2025, 1,049 runs) has 5.7 % wet hours and 0.89 % hours ≥ 5 mm, as the bike data in 2017–18 (6.0 % and 0.8 %); ERA5 has 10.7 % and 0.3 %.
- Rain frequency (`nwp_rain_frequency_unbiased = True`, default since 2 Oct 2026): false alarms are set equal to misses, so the forecast is wet as often as observed (P(false alarm | dry) = m(L)·p/(1 − p); the applied false-alarm ratio equals the miss rate). This removes the rain-frequency bias of the raw forecast, as the temperature, humidity and wind biases are removed: against the Seoul station the raw forecast is wet about 2.2 times as often as observed (FAR 0.65 at 24 h), against ERA5 about 0.8–0.9 times (London, Washington). `False`: the measured false-alarm ratio.

### Snow depth

Snow depth on the ground (cm, column `snow_depth_cm` in every city; Seoul = KMA 적설 at the station, London/Washington = ERA5-Land) has no measured forecast errors. It is forecast by persistence: every test hour gets the value of the last training hour (the issue time). Snow depth is not precipitation: it does not enter the event draw, the wet-hour share or the rain cap. There is no rain/snow phase correction (removed 2 Oct 2026): with total precipitation and snow depth there is no rain/snow split to correct.

### Visibility

Lognormal error in log space with persistence (AR(1), measured φ). For hours below the covariate's cap: X' = X · exp(μ_b + σ_b·zₜ), with μ_b and σ_b the mean and SD of log(forecast/observed) against the station; with `nwp_remove_bias = True`, μ_b = −σ_b²/2 (mean-preserving). The probability of falling below the cap is an event rate and stays as measured. Seoul (cap 20 km) and Washington (cap 16 km) have capped visibility covariates: an hour at the cap stays at the cap unless Φ(zₜ) is below the measured probability that the forecast falls below the cap when the station reports the cap (0.30 and 0.11); the depth below the cap, −log(X'/C), is then lognormal as measured. Results are cut at the cap (London: at the training fold's maximum).

### Random numbers

Seed per test window = `degradation_seed` + 10000 × horizon + fold + a city term (10,000,000 × CRC32 of the dataset name mod 1000; since 2 Oct 2026), so the three cities draw different random numbers for the same horizon and fold.

### Noise-magnitude sensitivity (not run by default since 2 Oct 2026)

Optional scenarios in `config.degradation_scales` (e.g. `degraded_x050`, `degraded_x150`; removed from the defaults because the errors are now measured per city) multiply every error magnitude by 0.5 and 1.5: the replayed errors (their bias included if it is kept), the relative solar error b + s·z, the log errors of precipitation amounts and of visibility, and the depth below the visibility cap. Event detection (misses, false alarms, visibility falling below the cap) and false-alarm amounts are not scaled. The same run is replayed and the same random numbers are used in all three scenarios.

### Calibration per city

| Quantity | Seoul | London | Washington |
| --- | --- | --- | --- |
| Reference for temperature, humidity, wind | station 47108 | ERA5 / ERA5-Land | ERA5 / ERA5-Land |
| ECMWF runs with complete errors (lead 1–192 h) | 844 | 1817 | 1817 |
| Temperature error, bias / SD at 24 h (°C) | -1.50 / 1.34 | +0.47 / 0.98 | +0.02 / 1.38 |
| Temperature error, bias / SD at 168 h (°C) | -1.53 / 2.45 | +0.19 / 2.55 | +0.08 / 3.27 |
| Humidity error, bias / SD at 24 h (%) | +4.5 / 10.8 | -2.6 / 6.5 | -3.4 / 9.3 |
| Humidity error, bias / SD at 168 h (%) | +4.6 / 16.1 | -1.7 / 11.8 | -3.1 / 15.7 |
| Wind speed error, bias / SD at 24 h (m/s) | -0.23 / 0.97 | -0.73 / 0.74 | +0.14 / 0.84 |
| Wind speed error, bias / SD at 168 h (m/s) | -0.26 / 1.28 | -0.82 / 1.90 | +0.16 / 1.48 |
| Solar: relative bias b / SD s at 24 h | +0.025 / 0.257 | +0.031 / 0.239 | -0.024 / 0.192 |
| Solar: relative bias b / SD s at 168 h | -0.078 / 0.416 | -0.049 / 0.384 | -0.036 / 0.369 |
| Solar: hour-to-hour correlation φ | 0.71 | 0.65 | 0.69 |
| Reference for precipitation | station 47108 | ERA5 | ERA5 |
| Precipitation: miss rate / false-alarm ratio at 24 h (year-round) | 0.23 / 0.65 | 0.39 / 0.30 | 0.42 / 0.29 |
| Precipitation: miss rate / false-alarm ratio at 24 h, January / July | 0.22 / 0.52, 0.20 / 0.68 | 0.31 / 0.20, 0.50 / 0.37 | 0.30 / 0.13, 0.50 / 0.35 |
| Precipitation: miss rate / false-alarm ratio at 168 h (year-round) | 0.59 / 0.83 | 0.71 / 0.67 | 0.67 / 0.65 |
| Precipitation hits: mean / SD of log(forecast/observed) at 24 h | -0.22 / 1.66 | +0.03 / 1.07 | -0.04 / 1.32 |
| False-alarm amount: median (mm) / SD of log | 0.36 / 1.19 | 0.26 / 0.89 | 0.37 / 1.09 |
| Persistence: misses ρ / false alarms ρ / hit amounts φ | 0.89 / 0.92 / 0.45 | 0.78 / 0.84 / 0.42 | 0.82 / 0.88 / 0.45 |
| Visibility cap of the covariate (km) | 20 | none | 16 |
| Visibility at the cap: P(forecast below cap) | 0.30 | – | 0.11 |
| Visibility below the cap: mean / SD of log(forecast/observed) | +0.46 / 0.98 | +0.12 / 0.94 | +0.47 / 1.03 |
| Visibility: hour-to-hour correlation φ | 0.82 | 0.77 | 0.67 |

Precipitation rates here use wet = ≥ 0.1 mm/h and ±12 h pooling; the verification report quotes > 0.1 mm/h per lead hour (slightly higher miss rates and false-alarm ratios). In `nwp_error_statistics.csv` the `gt0.1` statistics are strictly > 0.1 mm/h: for the Seoul station, whose amounts come in 0.1 mm steps, this drops every 0.1 mm hour (wet share per lead about 2 % instead of 5.7 %); for statistics comparable with the model use the `gt0` rows (= ≥ 0.1 mm/h for these data). The table lists the measured values; the default model removes the biases (temperature, humidity and wind bias, solar b, mean log errors of rain amounts and visibility).

Errors applied by the default model (fresh forecast, bias removed), from the calibration files; typical error = mean absolute error after removing each run pool's average error (`weather/nwp/expected_errors.py`):

| Typical error | Seoul 24 h / 168 h | London 24 h / 168 h | Washington 24 h / 168 h |
| --- | --- | --- | --- |
| Temperature (°C) | 0.92 / 1.82 | 0.67 / 1.89 | 0.96 / 2.33 |
| Relative humidity (%-points) | 7.4 / 11.9 | 4.4 / 8.8 | 6.4 / 11.6 |
| Wind speed (m/s) | 0.73 / 0.93 | 0.53 / 1.45 | 0.64 / 1.12 |
| Solar radiation (relative) | 0.21 / 0.33 | 0.19 / 0.31 | 0.15 / 0.29 |

### Validation on the bike data

Default setting (fresh forecast, bias removed, seasonal rain, Seoul rain against the station, rain frequency unbiased; 2 Oct 2026), run on the real data with the experiment's CV folds (seeds 42, 1042, 2042): mean temperature error −0.06 to +0.04 °C; rain miss rate within 0.035 of the seasonal calibration values for the test windows' dates (e.g. Seoul 0.21–0.27 for an expected 0.24); degraded data wet 0.99–1.07 times as often as the clean data in every city; false rain in 2.1–2.5 % of dry hours in Seoul, 6.1–6.7 % in London and Washington. With `nwp_rain_frequency_unbiased = False`: false-alarm ratio within 0.03 of the measured values; false rain in 15 % of dry hours in Seoul (London, Washington 4–5 %).

Fresh forecast, bias removed (1 Oct 2026), run on the real data with the experiment's CV folds (all horizons, all folds, scenario `degraded`, seeds 42, 1042, 2042; `weather/nwp/validate_on_real_data.py`): the mean temperature error is between −0.06 and +0.04 °C in every city (bias removed; with the bias kept, Seoul's is about −1.5 °C); typical errors at leads 6, 24 and 48 h are within 0.17 °C (temperature), 0.8 %-points (humidity) and 0.09 m/s (wind) of the calibration values above, solar relative error at lead ~6 h within 0.015; precipitation miss rates within 0.04 of the expected rate across the three seeds. Cost: about 8 ms per fold.

First version (newest available run, bias kept), same check: the simulated covariate errors reproduce the calibration: temperature error SD within 0.3 °C of the replayed runs' SD at the same lead times, humidity within 2.3 %-points, wind speed within 0.11 m/s; solar relative MAE within 0.05; precipitation miss rate and false-alarm ratio within sampling variation (misses persist, so the rate varies between seeds, e.g. Seoul 0.26–0.32 for an expected 0.31); visibility at the cap falls below it in 27 % (Seoul, measured 30 %) and 11 % (Washington, measured 11 %) of hours, and visibility below the cap is pushed to the cap in 46–47 % (Seoul, measured 47 %) and 54–56 % (Washington, measured 53 %) of hours. Cost: about 8 ms per fold.

### Limitations of the measured model

1. Forecast years (2024–26) differ from the data years (2011–18); forecasts were less accurate then, so the errors are probably too small for those years. The noise-magnitude sensitivity scenarios that bracketed this are not run by default since 2 Oct 2026 (they can be added in `config.degradation_scales`).
1a. Fresh, locally corrected forecast (default) is approximated: no free archive of past local forecasts exists, so the errors are those of the global 9 km model at short lead times with its average bias removed. Local high-resolution forecasts are usually more accurate in the first hours, so these may be slightly pessimistic. The replayed run's start hour can differ from the issue time of day by up to 6 h; with the bias removed, only the time-of-day dependence of the remaining errors is shifted.
2. Replayed errors come from another day (same season and run hour) and do not depend on the weather of the test window.
3. Solar radiation, precipitation and visibility errors are independent of each other and of temperature, humidity and wind.
4. References: London and Washington covariates could not be reproduced exactly from the Open-Meteo archive (median differences 0.3–0.5 °C), so the ERA5 reference is close to, not identical with, their source. Solar radiation uses ERA5 also for Seoul, whose covariates are station data. Station reports end in Aug 2025; Seoul's station gaps leave 844 complete runs for temperature, humidity and wind, and 1,049 runs with station rain (Mar 2024 – Aug 2025). Seoul's seasonal rain values in winter rest on wide windows (up to ±120 days) because the station has few wet winter hours.
4a. Precipitation errors do not depend on the intensity of the rain (one miss rate for drizzle and downpours); the amount error of hits is lognormal with one spread per lead time and time of year.
5. Only with `nwp_fresh_forecast = False`: issue time and availability (6 h) are assumptions; the 06/18 UTC runs (90 h only) are not used. Beyond 90 h the hourly errors are those of Open-Meteo's interpolated hourly series.
6. Daylight saving affects only the issue time (run selection), not the lead times: a repeated autumn hour is read as daylight-saving time (the occurrence kept), a non-existent spring hour is shifted forward, so a window starting at such an hour could be issued 1 h off (none does in the experiments: windows start at 00, 06, 12 or 18 h local time).

---

# Literature Error Model (earlier; `degradation_model = "literature"`)


## Overview (literature model)

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

where *p* is the share of wet hours (precipitation > 0), computed from the clean training window of each fold (in the same way as the solar cap). With this probability the simulated forecasts reproduce the source statistics: the simulated FAR equals the reported FAR and the simulated POD equals the reported POD. For a wet-hour share of 6%, P(false alarm | dry) is 2.2% at 24 h. A training window without any precipitation gives *p* = 0 and therefore no false alarms in that fold.

**Implementation:**
- **One precipitation variable:** total precipitation (one column per city). An hour is wet if precipitation > 0. Each hour receives one event-detection draw. Snow depth gets the persistence forecast, as in the measured model.
- **False alarms** (actual = 0, forecast > 0): Generated with probability P(false alarm | dry). When triggered, a small precipitation amount is sampled from a lognormal distribution with median 0.5 mm (σ_log = 0.5, mean 0.57 mm), representing typical light false alarm precipitation. The amount is written to the precipitation column.
  
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

### Degradation Process

All weather variables are degraded independently according to their respective error models, using per-row lead times as described above. Precipitation is one variable (one event-detection draw per hour); snow depth is forecast by persistence. Until 2 Oct 2026 a second phase re-assigned rain and snow by the degraded temperature (2 °C threshold). It was removed because every city now has total precipitation and snow depth instead of a rain/snow split (Seoul's snow column is snow depth, which the correction had turned into rain).

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

2. **Independent errors across variables**: Errors were treated as independent across weather variables. Actual NWP forecast errors exhibit substantial cross-variable correlations—for example, temperature and humidity errors are coupled through thermodynamic relationships, and wind errors correlate with temperature gradients. This independence assumption may underestimate error in derived quantities or physically coupled processes.

3. **Independent errors from hour to hour**: Each hour of a test window receives an independent error draw. Actual NWP forecast errors persist over many hours (e.g. a forecast that is too warm stays too warm for most of a day). The per-hour error magnitude matches the error formulas, but the temporal structure of the errors is not represented.

4. **Extrapolation below the shortest verified lead time**: The shortest lead time verified in the sources is 12 h (temperature, humidity) or 24 h (wind speed, solar radiation, precipitation, visibility). Below that, the error formulas are extrapolated to their intercepts; at 1 h the errors are σ = 0.79 °C (temperature), 13.0 %-points (humidity), 1.81 m/s (wind), 15% relative MAE (solar radiation), CV 25% (visibility), and 25% miss rate and FAR (precipitation). All lead times of the 6 h horizon and the first hours of every longer horizon rely on this extrapolation.

5. **Linear error growth**: Error growth was modeled as linear in forecast lead time. Actual verification curves show modest nonlinearity, with error growth accelerating slightly beyond 5-7 days as predictability limits are approached, and asymptotic behavior at very long ranges (>10 days) where forecast skill approaches climatology.

6. **Snow depth by persistence**: No forecast errors of snow depth are available; snow depth is held at its value at the issue time, so melting and new snow within the horizon are missed.

7. **Assumption-based parameters**: Precipitation magnitude CV (30% + 0.15%/h) is an assumption; no published verification of the magnitude error of correctly detected hourly precipitation by lead time was found. Solar radiation linear growth (0.15%/h) interpolates between verified 1-day and 7-day endpoints. These parameters represent defensible estimates but have not been independently validated against held-out verification datasets.

8. **Geographic and seasonal specificity**: Temperature and wind error growth parameters are derived from global or European verification statistics, while humidity errors are based on GFS verification over sub-Saharan Africa, and precipitation skill characteristics reflect mid-latitude performance. Parameters may not fully represent forecast error characteristics for all locations or seasons. Seasonal variations in forecast skill (e.g., summer convection vs. winter synoptic patterns) are not explicitly captured. Humidity errors from African verification may not generalize to all climate regimes.

9. **Single-model representation**: Parameters represent a composite of operational NWP systems and may not accurately reflect errors from other forecast systems or ensemble spread characteristics.

10. **Event detection statistics from extreme events**: The POD and FAR values of Sukovich et al. (2014) were verified for the top 1% of 24-hour precipitation events on a 32-km grid. They are applied here to all hourly precipitation events. Detection skill for ordinary hourly precipitation may differ.

11. **Wind speed truncation bias**: Truncation at zero introduces minor negative bias for low wind speeds (approximately -0.2 m/s), though this is significantly smaller than the positive bias from reflection. At 168h, mean bias is approximately +0.57 m/s (33% of original mean), which is acceptable but non-zero.

Despite these limitations, the degradation methodology provides a realistic and conservative estimate of operational forecast uncertainty appropriate for evaluating machine learning model robustness under forecast input conditions.

## References

Measured model:

Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803

Muñoz Sabater, J. (2019). ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store. https://doi.org/10.24381/cds.e2161bac

NOAA National Centers for Environmental Information. Integrated Surface Database (ISD), global hourly. https://www.ncei.noaa.gov/data/global-hourly/access/

Zippenfenig, P. (2023). Open-Meteo.com Weather API (Single Runs API; Historical Weather API). https://doi.org/10.5281/zenodo.7970649

Literature model:

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