# Weather Degradation Model: Summary

*Status 2 Oct 2026 (fresh forecast, bias removed, rain by time of year, Seoul rain against the station, no sensitivity runs). Code: `forecasting/weather/nwp_error_model.py` (settings `degradation_model = "nwp_measured"`, `nwp_fresh_forecast = True`, `nwp_remove_bias = True`, `nwp_seasonal_rain = True`, the defaults). Full technical description: `forecasting/weather/weather_methodology.md`.*

## Purpose

In practice a bike-demand model receives weather **forecasts**, not measured weather. To test how much this costs, the weather inputs of every test period are replaced by realistic forecast weather. The training data stays as measured.

The previous version used error sizes from the literature, the same for every city, and drew a new independent random error for every hour. The new version uses forecast errors measured for each city from real forecasts.

## How the model was built

**1. Real forecasts.** We downloaded 1,828 forecasts per city from the global model of the European Centre for Medium-Range Weather Forecasts (ECMWF IFS, 9 km grid): all forecasts issued at 00 and 12 UTC from 14 March 2024 to 16 September 2026, each hourly up to 10 days ahead, at the grid point nearest to each city's weather station. Source: Open-Meteo (free for research).

**2. What counts as "truth".** For each city the truth is the source of that city's own weather data, because that is what its demand model learned from:

| City | Truth for temperature, humidity, wind | Truth for solar radiation, rain | Truth for visibility |
| --- | --- | --- | --- |
| Seoul | City weather station (KMA 108), the station of the Seoul data | Solar: ERA5; rain: city weather station | City weather station |
| London | ERA5, the source of the London weather data (via Open-Meteo) | ERA5 | Heathrow station |
| Washington | ERA5, the source of the Washington weather data (via Open-Meteo) | ERA5 | Reagan National station |

ERA5 is calculated, not measured: ECMWF re-runs its weather model over past dates, corrected with the measurements available at the time, to estimate the weather for every hour and every square of about 30 km (technical term: reanalysis). It is produced by ECMWF, the same organisation that makes the forecasts.

**Consequence for comparing cities.** Seoul is a comparison between *different* sources: the ECMWF forecast against a station in the city centre. London and Washington are comparisons within the *same* model family: the ECMWF forecast against ECMWF's reconstruction. Different-source comparisons give larger errors. This is intended, because each demand model would face exactly this situation in use. It means, however, that part of Seoul's larger errors comes from the comparison, not from Seoul's weather:
- Seoul against ERA5 instead of the station: the temperature bias would be about +0.3 °C instead of −1.5 °C (24 h ahead). Washington against its airport station instead of ERA5: about −1.5 °C, like Seoul.
- London and Washington against their weather stations instead of ERA5: spread larger by up to 0.4 °C (temperature), 1.6 % (humidity) and 0.9 m/s (wind).
- The model removes the average bias (step 4), which takes out most of this difference; the differences in spread remain.

**3. Measuring the errors.** Error = forecast − truth, per city, variable and hour ahead (lead time). The full statistics are in `nwp_error_statistics.csv` (119,051 values, including Seoul rain against the station), together with the report "Measured weather-forecast errors".

**4. Applying the errors to the bike data.** Every test period is handled the same way. The model represents a bike-sharing operator with an up-to-date weather forecast that is corrected to local measurements:

- *Fresh forecast.* The weather forecast starts when the demand forecast is made, so hour *h* of the demand forecast gets the error of an *h*-hour weather forecast, in every city and at every time of day. (A first version used the newest global forecast available at that moment, which is 6–17 hours old. That made the forecast age depend on each city's time zone, Seoul 14 h, London 10–11 h, Washington 15–16 h, and gave some hours of the day always older forecasts. It is still available as a setting.)
- *Temperature, humidity, wind.* We pick one real forecast of the same city, started at the 00 or 12 UTC run time closest to the time of day of the demand forecast, at the same time of year (±30 days, any year), and add its actual errors hour by hour to the measured values. This carries over the real error sizes, how long errors last, and how the three variables go wrong together.
- *Bias removed.* A forecast corrected to local measurements does not have the average error of the raw forecast. So the average error of all comparable forecasts (same start time, same season) is subtracted, e.g. Seoul's forecast being on average 1.5 °C colder than the city-centre station. What remains are the hour-to-hour errors. The same applies to the average error of solar radiation, rain amounts and visibility; how often rain is missed or falsely forecast stays as measured (an option removes the rain-frequency bias too, see "Seoul in detail").
- *Solar radiation, rain, visibility.* These errors depend on the weather itself (a rain error from a rainy day cannot be added to a dry day), so they are simulated with the measured values: error size per lead time; for rain, how often wet hours are missed, how often dry hours get false rain, and how wrong the amounts are, each for the time of year (monthly values; e.g. Washington's forecast misses 50 % of rainy hours in July and 30 % in January); for visibility, the error size and, where the data are capped, how often the forecast drops below the cap. These errors also persist from hour to hour as measured.
- *Physical limits.* No sun at night; solar radiation at most the 99.5th percentile of the training period; humidity 0–100 %; no negative wind; visibility at most the city's cap (Seoul 20 km, Washington 16 km) or, in London, the training maximum. Rain becomes snow below 2 °C and snow becomes rain above.
- *No sensitivity runs.* Runs with all error sizes ×0.5 and ×1.5 were planned when the error sizes came from the literature. With measured errors they are no longer run by default (they would triple the degraded runs); they can be switched back on in the configuration.

**5. Checks.**
- The model was run on the real bike data with exactly the folds of the experiments (three random seeds). The errors it produces match the calibration: average temperature error between −0.06 and +0.04 °C in every city (bias removed), typical errors within 0.17 °C (temperature), 0.8 %-points (humidity) and 0.09 m/s (wind); rain miss rate within 4 %-points and false-alarm rate within 3 %-points of the values for the time of year. It costs about 8 ms per test period.
- Seoul rain: the Seoul bike-data rain is the station's rain (2017–18: the station's 6- and 12-hour totals equal the bike-data sums in 99 % of cases). The station's hourly rain reports for 2024–25 add up to its own daily totals (no rainy hours missing).
- Seoul: the Seoul bike data are identical to the station used as truth (median difference 0.0 °C).
- London and Washington: their weather data are close to, but not identical with, the ERA5 data used as truth (median difference 0.3–0.5 °C); the original download settings are unknown.
- 38 automated tests (`testing/test_weather_unit.py`).

## Expected forecast errors at different time horizons

Errors the model applies *h* hours ahead with the default setting (fresh forecast, bias removed, rain by time of year; rain rows: year-round, plus 15 January / 15 July), from the calibration files (`weather/nwp/expected_errors.py`); all seasons of 2024–26. "Bias, removed" is the measured average error that the model takes out.

**Seoul** (truth: city station; solar: ERA5)

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (typical error) | 0.89 °C | 0.92 °C | 1.03 °C | 1.82 °C |
| Temperature (bias, removed) | −1.24 °C | −1.50 °C | −1.58 °C | −1.53 °C |
| Humidity (typical error) | 6.2% | 7.4% | 8.4% | 11.9% |
| Humidity (bias, removed) | +3.6% | +4.5% | +4.4% | +4.6% |
| Wind speed (typical error) | 0.69 m/s | 0.73 m/s | 0.76 m/s | 0.93 m/s |
| Wind speed (bias, removed) | −0.12 m/s | −0.23 m/s | −0.23 m/s | −0.26 m/s |
| Solar radiation (rel. err.) | 15.0% | 20.5% | 22.3% | 33.2% |
| Precip. (miss rate) | 16.5% | 22.8% | 30.7% | 58.5% |
| Precip. (miss rate, January / July) | 16% / 13% | 22% / 20% | 37% / 22% | 65% / 49% |
| Precip. (false alarm) | 63.4% | 64.9% | 67.5% | 83.4% |
| Precip. (false alarm, January / July) | 54% / 65% | 52% / 68% | 56% / 70% | 81% / 84% |
| Precip. (rel. var.) | 312.0% | 380.6% | 533.4% | 717.0% |
| Visibility (rel. var.) | 126.5% | 126.5% | 126.5% | 126.5% |
| Visibility (at cap: forecast below cap) | 30.2% | 30.2% | 30.2% | 30.2% |

**London** (truth: ERA5; visibility: Heathrow station)

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (typical error) | 0.57 °C | 0.67 °C | 0.78 °C | 1.89 °C |
| Temperature (bias, removed) | +0.35 °C | +0.47 °C | +0.43 °C | +0.19 °C |
| Humidity (typical error) | 4.0% | 4.4% | 5.1% | 8.8% |
| Humidity (bias, removed) | −1.9% | −2.6% | −2.4% | −1.7% |
| Wind speed (typical error) | 0.47 m/s | 0.53 m/s | 0.62 m/s | 1.45 m/s |
| Wind speed (bias, removed) | −0.75 m/s | −0.73 m/s | −0.77 m/s | −0.82 m/s |
| Solar radiation (rel. err.) | 16.7% | 19.1% | 20.6% | 30.7% |
| Precip. (miss rate) | 36.1% | 39.2% | 43.9% | 71.1% |
| Precip. (miss rate, January / July) | 26% / 46% | 31% / 50% | 36% / 55% | 66% / 80% |
| Precip. (false alarm) | 27.7% | 30.2% | 35.6% | 67.4% |
| Precip. (false alarm, January / July) | 18% / 35% | 20% / 37% | 26% / 41% | 60% / 77% |
| Precip. (rel. var.) | 139.9% | 146.8% | 165.5% | 191.1% |
| Visibility (rel. var.) | 119.0% | 119.0% | 119.0% | 119.0% |

**Washington, DC** (truth: ERA5; visibility: Reagan National station)

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (typical error) | 0.92 °C | 0.96 °C | 1.11 °C | 2.33 °C |
| Temperature (bias, removed) | +0.10 °C | +0.02 °C | −0.02 °C | +0.08 °C |
| Humidity (typical error) | 5.8% | 6.4% | 7.2% | 11.6% |
| Humidity (bias, removed) | −2.1% | −3.4% | −3.1% | −3.1% |
| Wind speed (typical error) | 0.44 m/s | 0.64 m/s | 0.70 m/s | 1.12 m/s |
| Wind speed (bias, removed) | −0.05 m/s | +0.14 m/s | +0.16 m/s | +0.16 m/s |
| Solar radiation (rel. err.) | 14.2% | 15.3% | 18.3% | 29.5% |
| Precip. (miss rate) | 36.6% | 41.8% | 46.6% | 66.5% |
| Precip. (miss rate, January / July) | 24% / 44% | 30% / 50% | 32% / 52% | 57% / 63% |
| Precip. (false alarm) | 29.4% | 29.0% | 36.5% | 65.2% |
| Precip. (false alarm, January / July) | 12% / 37% | 13% / 35% | 16% / 42% | 59% / 62% |
| Precip. (rel. var.) | 202.9% | 217.2% | 254.0% | 303.0% |
| Visibility (rel. var.) | 136.6% | 136.6% | 136.6% | 136.6% |
| Visibility (at cap: forecast below cap) | 11.2% | 11.2% | 11.2% | 11.2% |

**Previous model (literature values, same for all cities):**

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (typical error) | 0.70 °C | 0.96 °C | 1.31 °C | 3.04 °C |
| Humidity (typical error) | 10.5% | 10.8% | 11.3% | 13.5% |
| Wind speed (typical error) | 1.48 m/s | 1.63 m/s | 1.82 m/s | 2.78 m/s |
| Solar radiation (rel. err.) | 15.9% | 18.6% | 22.2% | 40.2% |
| Precip. (miss rate) | 27.5% | 35.0% | 45.0% | 50.0% |
| Precip. (false alarm) | 27.5% | 35.0% | 45.0% | 50.0% |
| Precip. (rel. var.) | 30.9% | 33.6% | 37.2% | 55.2% |
| Visibility (rel. var.) | 25.0% | 25.0% | 25.0% | 25.0% |

**Definitions.**
- *typical error*: average size of the error (mean absolute error) after removing the bias, i.e. what the model applies.
- *bias, removed*: measured average error with sign (forecast − truth; negative = raw forecast too low); the model takes it out.
- *rel. err.*: average size of the error divided by the average measured value, daylight hours only.
- *miss rate*: share of wet hours (≥ 0.1 mm/h) that the forecast had dry.
- *false alarm*: share of forecast wet hours that were dry.
- *rel. var.*: relative spread of the amount error (coefficient of variation); for rain only hours that were wet in both forecast and truth, for visibility only hours below the cap.
- *at cap: forecast below cap*: share of hours at the visibility cap for which the forecast is below the cap.

## How the errors behave over the forecast hours

- **Growth.** The error grows gradually from hour to hour, following how far ahead the weather forecast looks. The first hours of a 7-day forecast have the same error size as a 6-hour forecast. Visibility errors do not grow with lead time.
- **Persistence.** Errors keep their direction for several hours, but they are not fixed for the whole forecast. In the real forecasts the temperature error keeps its sign into the next hour in 87–92 % of cases. A too-warm or too-cold stretch typically lasts 3–5 hours (on average 7–12 hours), and 15–39 % of forecasts stay on one side for their whole first day. Humidity behaves similarly; wind changes sign more often. Solar and visibility errors keep their sign into the next hour in 73–81 % of cases. Missed rain events and false rain usually last several hours.
- **Biases.** The raw forecasts have systematic average errors, the largest Seoul temperature (−1.5 °C), Seoul humidity (+4 %) and London wind (−0.8 m/s against ERA5). The model removes them, as a forecast corrected to local measurements would. With the biases removed, the cities are close: temperature 0.7–1.0 °C at 24 h.
- **Same for every hour of the day.** With a fresh forecast, the error depends only on how far ahead the hour is, not on when the demand forecast is made or in which time zone the city is.

## Seoul in detail

- **Temperature bias.** The raw forecast is colder than the station in every month (−0.6 to −2.5 °C), most at night and in winter (about −2 °C) and least on summer afternoons (about −0.6 °C). This is the typical urban heat effect: the station is in the warm city centre, which a 9 km forecast cannot resolve. A forecast corrected to local measurements would not have this average error, so the model removes it (default setting).
- **Rain, measured against the station (since 2 Oct 2026).** The free station records report Seoul rain hour by hour whenever it rains (from 2024), and the hourly amounts add up to the station's own daily totals, so no rainy hours are missing. Hours with a routine report but no rain amount and no rain in the weather description are dry. This gives 5.7 % rainy hours and 0.89 % hours with ≥ 5 mm (March 2024 – August 2025), close to the bike data in 2017–18 (6.0 % and 0.8 %). ERA5, used before, has 10.7 % and 0.3 %: almost twice as many rainy hours but fewer than half the downpours, so the Seoul rain errors were too small.
- **What changed for Seoul rain.** Against the station, the forecast misses fewer rainy hours (23 % at 24 h, before 31 %), but rains far more often than the station measures: 65 % of the forecast's rainy hours are dry at the station (before 27 %), and the amount error of correctly forecast rain is about twice as large (relative spread 381 % at 24 h, before 195 %; July 567 %), the monsoon downpours. In the degraded Seoul test data about 15 % of dry hours get false rain (London and Washington 4–5 %).
- **Rain-frequency option.** The raw forecast is wet about 2.2 times as often as the Seoul station (London and Washington 0.8–0.9 times against ERA5). A locally corrected forecast might not rain that often. The option `nwp_rain_frequency_unbiased` sets false rain so that the forecast is wet as often as observed; it is off by default.

## Limitations

1. **Years.** The forecasts are from 2024–26, the bike data from 2011–18. Forecasts were less accurate then, so the errors are probably too small for those years. Runs with ×0.5 and ×1.5 error sizes are not part of the default setting. ECMWF forecasts from 2011–18 are not freely available (licensed archive; the free TIGGE archive needs an account and has no visibility or usable solar radiation).
2. **Replayed errors come from another day** of the same season; they do not depend on the weather of the test period.
3. **Solar, rain and visibility errors are independent** of each other and of temperature, humidity and wind (temperature, humidity and wind are linked through the replay).
4. **Rain errors by time of year, but not by intensity.** Miss rate, false-rain rate and amount error are taken for the time of year (monthly values). For Seoul in winter they rest on wide windows (up to ±120 days) because the station has few rainy winter hours. Drizzle and downpours have the same chance of being missed; downpours differ only through the amount error.
5. **Calculated, not measured truth** (see step 2): for London and Washington (all variables except visibility) and for Seoul solar radiation, the truth is ERA5, calculated by a weather model of the same centre as the forecasts. Errors against it are probably smaller than against real measurements. Seoul solar radiation additionally does not match the Seoul bike data, which come from the station.
6. **Station data end in August 2025** (NOAA archive). Seoul has 844 forecasts with complete station data for temperature, humidity and wind and 1,049 with station rain, London and Washington 1,817.
7. **Fresh, corrected forecast is approximated.** No free archive of past local forecasts exists, so the errors are those of the global 9 km model at short lead times with its bias removed. Local high-resolution forecasts are usually more accurate in the first hours, so these may be slightly pessimistic. The replayed forecast's start time (00 or 12 UTC) can differ from the time of day of the demand forecast by up to 6 hours. Beyond 90 hours ECMWF provides 3- and 6-hourly values; the hours in between are interpolated by Open-Meteo.
8. **Daylight saving.** For the two clock-change hours per year the time conversion may be off by one hour.

## Sources

- ECMWF IFS HRES forecasts via the Open-Meteo Single Runs API (Zippenfenig, 2023; CC BY 4.0).
- ERA5 (Hersbach et al., 2020) and ERA5-Land (Muñoz Sabater, 2019) via the Open-Meteo Historical Weather API.
- NOAA Integrated Surface Database, hourly station reports: Seoul 47108 (including hourly rain amounts from 2024), London Heathrow 03772, Washington Reagan National 72405-13743.
