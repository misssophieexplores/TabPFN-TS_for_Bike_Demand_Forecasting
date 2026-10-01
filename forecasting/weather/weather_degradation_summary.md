# Weather Degradation Model: Summary

*Status 1 Oct 2026. Code: `forecasting/weather/nwp_error_model.py` (setting `degradation_model = "nwp_measured"`, the default). Full technical description: `forecasting/weather/weather_methodology.md`.*

## Purpose

In practice a bike-demand model receives weather **forecasts**, not measured weather. To test how much this costs, the weather inputs of every test period are replaced by realistic forecast weather. The training data stays as measured.

The previous version used error sizes from the literature, the same for every city, and drew a new independent random error for every hour. The new version uses forecast errors measured for each city from real forecasts.

## How the model was built

**1. Real forecasts.** We downloaded 1,828 forecasts per city from the global model of the European Centre for Medium-Range Weather Forecasts (ECMWF IFS, 9 km grid): all forecasts issued at 00 and 12 UTC from 14 March 2024 to 16 September 2026, each hourly up to 10 days ahead, at the grid point nearest to each city's weather station. Source: Open-Meteo (free for research).

**2. What counts as "truth".** For each city the truth is the source of that city's own weather data, because that is what its demand model learned from:

| City | Truth for temperature, humidity, wind | Truth for solar radiation, rain | Truth for visibility |
| --- | --- | --- | --- |
| Seoul | City weather station (KMA 108), the station of the Seoul data | ERA5 | City weather station |
| London | ERA5, the source of the London weather data (via Open-Meteo) | ERA5 | Heathrow station |
| Washington | ERA5, the source of the Washington weather data (via Open-Meteo) | ERA5 | Reagan National station |

ERA5 is a reconstruction of past weather (a "reanalysis") that combines observations with a weather model. It is produced by ECMWF, the same organisation that makes the forecasts.

**Consequence for comparing cities.** Seoul is a comparison between *different* sources: the ECMWF forecast against a station in the city centre. London and Washington are comparisons within the *same* model family: the ECMWF forecast against ECMWF's reconstruction. Different-source comparisons give larger errors. This is intended, because each demand model would face exactly this situation in use. It means, however, that part of Seoul's larger errors comes from the comparison, not from Seoul's weather:
- Seoul against ERA5 instead of the station: the temperature offset would be about +0.3 °C instead of −1.5 °C (24 h ahead).
- London and Washington against their weather stations instead of ERA5: spread larger by up to 0.4 °C (temperature), 1.6 % (humidity) and 0.9 m/s (wind).

**3. Measuring the errors.** Error = forecast − truth, per city, variable and hour ahead (lead time). The full statistics are in `nwp_error_statistics.csv` (113,466 values), together with the report "Measured weather-forecast errors".

**4. Applying the errors to the bike data.** Every test period is handled the same way:

- *Which forecast is used.* The demand forecast is made at the end of the training period. At that moment the newest weather forecast is already 6–17 hours old: forecasts start at 00 and 12 UTC and take about 6 hours to become available. So the first hour of every demand forecast uses weather predicted 7–18 hours ahead, and hour *h* uses weather predicted *h* + 6 to *h* + 17 hours ahead.
- *Temperature, humidity, wind.* We pick one real forecast of the same city, issued at the same hour (00 or 12 UTC) and at the same time of year (±30 days, any year), and add its actual errors hour by hour to the measured values. This carries over the real error sizes, the systematic offsets, how long errors last, and how the three variables go wrong together.
- *Solar radiation, rain, visibility.* These errors depend on the weather itself (a rain error from a rainy day cannot be added to a dry day), so they are simulated with the measured values: error size per lead time; for rain, how often wet hours are missed, how often dry hours get false rain, and how wrong the amounts are; for visibility, the error size and, where the data are capped, how often the forecast drops below the cap. These errors also persist from hour to hour as measured.
- *Physical limits.* No sun at night; solar radiation at most the 99.5th percentile of the training period; humidity 0–100 %; no negative wind; visibility at most the city's cap (Seoul 20 km, Washington 16 km) or, in London, the training maximum. Rain becomes snow below 2 °C and snow becomes rain above.
- *Sensitivity.* All error sizes are also run at ×0.5 and ×1.5, with the same forecasts and random numbers. Not scaled: whether rain is missed or falsely forecast, and whether visibility drops below its cap.

**5. Checks.**
- The model was run on the real bike data with exactly the folds of the experiments. The errors it produces match the measured ones: temperature spread within 0.3 °C, humidity within 2.3 %-points, wind within 0.11 m/s, solar relative error within 5 %-points; in Seoul visibility drops below the cap in 27 % of capped hours (measured 30 %), in Washington in 11 % (measured 11 %). It costs about 8 ms per test period.
- Seoul: the Seoul bike data are identical to the station used as truth (median difference 0.0 °C).
- London and Washington: their weather data are close to, but not identical with, the ERA5 data used as truth (median difference 0.3–0.5 °C); the original download settings are unknown.
- 35 automated tests (`testing/test_weather_unit.py`).

## Expected forecast errors at different time horizons

Error *h* hours ahead, for a forecast issued at any time of day with the newest weather forecast available then; averaged over all seasons of 2024–26.

**Seoul** (truth: city station; solar and rain: ERA5)

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (avg. err.) | 1.69 °C | 1.81 °C | 1.89 °C | 2.58 °C |
| Temperature (bias) | -1.43 °C | -1.49 °C | -1.50 °C | -1.51 °C |
| Humidity (avg. err.) | 9.1% | 9.6% | 10.1% | 13.5% |
| Humidity (bias) | +4.3% | +4.1% | +4.1% | +4.2% |
| Wind speed (avg. err.) | 0.77 m/s | 0.80 m/s | 0.82 m/s | 1.06 m/s |
| Wind speed (bias) | -0.21 m/s | -0.21 m/s | -0.24 m/s | -0.26 m/s |
| Solar radiation (rel. err.) | 14.1% | 16.2% | 18.2% | 29.5% |
| Precip. (miss rate) | 28.6% | 35.1% | 41.1% | 63.8% |
| Precip. (false alarm) | 25.2% | 29.4% | 32.7% | 61.2% |
| Precip. (rel. var.) | 175% | 244% | 269% | 378% |
| Visibility (rel. var.) | 134% | 130% | 128% | 116% |
| Visibility (at cap: forecast below cap) | 26.6% | 27.6% | 27.9% | 34.0% |

**London** (truth: ERA5; visibility: Heathrow station)

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (avg. err.) | 0.73 °C | 0.77 °C | 0.88 °C | 2.05 °C |
| Temperature (bias) | +0.41 °C | +0.36 °C | +0.35 °C | +0.03 °C |
| Humidity (avg. err.) | 4.6% | 4.8% | 5.4% | 8.7% |
| Humidity (bias) | -2.2% | -2.0% | -2.1% | -1.6% |
| Wind speed (avg. err.) | 0.85 m/s | 0.90 m/s | 0.99 m/s | 1.65 m/s |
| Wind speed (bias) | -0.76 m/s | -0.78 m/s | -0.79 m/s | -0.92 m/s |
| Solar radiation (rel. err.) | 17.7% | 19.7% | 22.0% | 34.7% |
| Precip. (miss rate) | 37.8% | 40.8% | 47.7% | 73.0% |
| Precip. (false alarm) | 28.8% | 32.2% | 40.5% | 68.5% |
| Precip. (rel. var.) | 142% | 156% | 182% | 200% |
| Visibility (rel. var.) | 112% | 115% | 114% | 121% |

**Washington, DC** (truth: ERA5; visibility: Reagan National station)

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (avg. err.) | 1.02 °C | 1.16 °C | 1.34 °C | 2.70 °C |
| Temperature (bias) | +0.09 °C | +0.04 °C | +0.02 °C | +0.04 °C |
| Humidity (avg. err.) | 6.8% | 7.4% | 8.2% | 13.3% |
| Humidity (bias) | -2.8% | -2.6% | -2.5% | -2.6% |
| Wind speed (avg. err.) | 0.57 m/s | 0.65 m/s | 0.72 m/s | 1.22 m/s |
| Wind speed (bias) | +0.01 m/s | +0.03 m/s | +0.03 m/s | -0.04 m/s |
| Solar radiation (rel. err.) | 13.2% | 15.2% | 17.5% | 28.3% |
| Precip. (miss rate) | 41.2% | 43.0% | 49.7% | 68.8% |
| Precip. (false alarm) | 26.6% | 33.0% | 39.8% | 66.3% |
| Precip. (rel. var.) | 207% | 235% | 257% | 283% |
| Visibility (rel. var.) | 139% | 132% | 139% | 136% |
| Visibility (at cap: forecast below cap) | 10.2% | 11.2% | 11.4% | 11.6% |

**Previous model (literature values, same for all cities):**

| Variable | 6 hours | 24 hours | 48 hours | 7 days |
| --- | ---: | ---: | ---: | ---: |
| Temperature (avg. err.) | 0.70 °C | 0.96 °C | 1.31 °C | 3.04 °C |
| Humidity (avg. err.) | 10.5% | 10.8% | 11.3% | 13.5% |
| Wind speed (avg. err.) | 1.48 m/s | 1.63 m/s | 1.82 m/s | 2.78 m/s |
| Solar radiation (rel. err.) | 15.9% | 18.6% | 22.2% | 40.2% |
| Precip. (miss rate) | 31.0% | 34.0% | 37.0% | 50.0% |
| Precip. (false alarm) | 36.0% | 37.0% | 40.0% | 50.0% |
| Precip. (rel. var.) | 30.9% | 33.6% | 37.2% | 55.2% |
| Visibility (rel. var.) | 25.0% | 25.0% | 25.0% | 25.0% |

**Definitions.**
- *avg. err.*: average size of the error (mean absolute error), offset included.
- *bias*: average error with sign (forecast − truth); negative = forecast too low.
- *rel. err.*: average size of the error divided by the average measured value, daylight hours only.
- *miss rate*: share of wet hours (≥ 0.1 mm/h) that the forecast had dry.
- *false alarm*: share of forecast wet hours that were dry.
- *rel. var.*: relative spread of the amount error (coefficient of variation); for rain only hours that were wet in both forecast and truth, for visibility only hours below the cap.
- *at cap: forecast below cap*: share of hours at the visibility cap for which the forecast is below the cap.

## How the errors behave over the forecast hours

- **Growth.** The error grows gradually from hour to hour, following how far ahead the weather forecast looks. The first hours of a 7-day forecast have the same error size as a 6-hour forecast. Visibility errors do not grow with lead time.
- **Persistence.** Errors keep their direction for several hours, but they are not fixed for the whole forecast. In the real forecasts the temperature error keeps its sign into the next hour in 87–92 % of cases. A too-warm or too-cold stretch typically lasts 3–5 hours (on average 7–12 hours), and 15–39 % of forecasts stay on one side for their whole first day. Humidity behaves similarly; wind changes sign more often. Solar and visibility errors keep their sign into the next hour in 73–81 % of cases. Missed rain events and false rain usually last several hours.
- **Offsets.** Systematic offsets are part of the errors. The largest are Seoul temperature (−1.5 °C), Seoul humidity (+4 %) and London wind (−0.8 m/s against ERA5).

## Seoul in detail

- **Temperature offset.** The forecast is colder than the station in every month (−0.6 to −2.5 °C), most at night and in winter (about −2 °C) and least on summer afternoons (about −0.6 °C). This is the typical urban heat effect: the station is in the warm city centre, which a 9 km forecast cannot resolve. The model reproduces this pattern because it replays forecasts from the same season and the same time of day.
- **Rain.** Free station records for 2024–26 report Seoul rain only when it rains, so ERA5 is the truth for Seoul rain. Compared with the Seoul station rain in the bike data (2017–18), ERA5 has almost twice as many rainy hours (10.7 % vs 6.0 %); 55 % of its rainy hours are dry at the station; it catches 79 % of the station's rainy hours; and it has fewer than half the heavy-rain hours (≥ 5 mm/h: 0.3 % vs 0.8 %). The rain errors used for Seoul are therefore probably smaller than the real errors against the station data, especially for monsoon downpours. Remedy: hourly station rain from the Korean weather service (data.kma.go.kr, free with registration), station 108, March 2024 onwards.

## Limitations

1. **Years.** The forecasts are from 2024–26, the bike data from 2011–18. Forecasts were less accurate then, so the errors are probably too small for those years. The ×0.5 and ×1.5 scenarios bracket the error size. ECMWF forecasts from 2011–18 are not freely available (licensed archive; the free TIGGE archive needs an account and has no visibility or usable solar radiation).
2. **Replayed errors come from another day** of the same season; they do not depend on the weather of the test period.
3. **Solar, rain and visibility errors are independent** of each other and of temperature, humidity and wind (temperature, humidity and wind are linked through the replay).
4. **Rain errors do not change with the season.** The data show strong seasonal differences: in Seoul 26 % of hours are wet in June–September against 6 % in winter, and 1.3 % of summer hours have heavy rain (≥ 5 mm/h), which almost never occurs in London; in Washington the forecast misses 51 % of wet hours in summer against 26 % in winter. The model uses year-round averages per city. (Temperature, humidity and wind are season-matched.)
5. **Different truths per city** (see step 2): Seoul is measured against a station, London and Washington against ERA5; solar and rain against ERA5 everywhere (for Seoul see "Seoul in detail").
6. **Station data end in August 2025** (NOAA archive). Seoul has 844 forecasts with complete station data, London and Washington 1,817.
7. **Timing assumptions.** Weather forecasts are assumed available 6 hours after their start time; the shorter 06 and 18 UTC forecasts are not used. Beyond 90 hours ECMWF provides 3- and 6-hourly values; the hours in between are interpolated by Open-Meteo.
8. **Daylight saving.** For the two clock-change hours per year the time conversion may be off by one hour.

## Sources

- ECMWF IFS HRES forecasts via the Open-Meteo Single Runs API (Zippenfenig, 2023; CC BY 4.0).
- ERA5 (Hersbach et al., 2020) and ERA5-Land (Muñoz Sabater, 2019) via the Open-Meteo Historical Weather API.
- NOAA Integrated Surface Database, hourly station reports: Seoul 47108, London Heathrow 03772, Washington Reagan National 72405-13743.
