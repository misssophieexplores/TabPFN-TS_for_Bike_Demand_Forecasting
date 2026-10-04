# Weather Degradation Model: Summary

*Status 4 Oct 2026 (fresh forecast, bias removed, rain by time of year and intensity, Seoul precipitation against the station with winter 3-hour totals split and compared at 3-hour resolution, rain-frequency bias removed, all covariates and rain amounts unbiased after the physical limits, no sensitivity runs). Code: `forecasting/weather/nwp_error_model.py` (settings `degradation_model = "nwp_measured"`, `nwp_fresh_forecast = True`, `nwp_remove_bias = True`, `nwp_seasonal_rain = True`, `nwp_rain_intensity_dependent = True`, `nwp_rain_frequency_unbiased = True`, `nwp_rain_amount_unbiased = True`, `nwp_mean_preserving_caps = True`, the defaults). Full technical description: `forecasting/weather/weather_methodology.md`.*

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

Visibility: London's and Washington's visibility data come from Visual Crossing, not directly from the airport stations. Washington's match the Reagan National reports (82 % of hours within 0.1 km); London's do not match Heathrow (median difference 4.5 km), so London's visibility errors come from a different series than its data (limitation 9).

ERA5 is calculated, not measured: ECMWF re-runs its weather model over past dates, corrected with the measurements available at the time, to estimate the weather for every hour and every square of about 30 km (technical term: reanalysis). It is produced by ECMWF, the same organisation that makes the forecasts.

**Consequence for comparing cities.** Seoul is a comparison between *different* sources: the ECMWF forecast against a station in the city centre. London and Washington are comparisons within the *same* model family: the ECMWF forecast against ECMWF's reconstruction. Different-source comparisons give larger errors. This is intended, because each demand model would face exactly this situation in use. It means, however, that part of Seoul's larger errors comes from the comparison, not from Seoul's weather:
- Seoul against ERA5 instead of the station: the temperature bias would be about +0.3 °C instead of −1.5 °C (24 h ahead). Washington against its airport station instead of ERA5: about −1.5 °C, like Seoul.
- London and Washington against their weather stations instead of ERA5: spread larger by up to 0.4 °C (temperature), 1.6 % (humidity) and 0.9 m/s (wind).
- The model removes the average bias (step 4), which takes out most of this difference; the differences in spread remain.

**3. Measuring the errors.** Error = forecast − truth, per city, variable and hour ahead (lead time). The full statistics are in `nwp_error_statistics.csv` (119,051 values, including Seoul rain against the station, November–March split like the calibration; for rain use the `gt0` rows: the `gt0.1` rows count only amounts above 0.1 mm/h and drop the Seoul station's 0.1 mm hours and split winter values up to 0.1 mm/h), together with the report "Measured weather-forecast errors".

**4. Applying the errors to the bike data.** Every test period is handled the same way. The model represents a bike-sharing operator with an up-to-date weather forecast that is corrected to local measurements:

- *Fresh forecast.* The weather forecast starts when the demand forecast is made, so hour *h* of the demand forecast gets the error of an *h*-hour weather forecast, in every city and at every time of day. (A first version used the newest global forecast available at that moment, which is 6–17 hours old. That made the forecast age depend on each city's time zone, Seoul 14 h, London 10–11 h, Washington 15–16 h, and gave some hours of the day always older forecasts. It is still available as a setting.)
- *Temperature, humidity, wind.* We pick one real forecast of the same city, started at the 00 or 12 UTC run time closest to the time of day of the demand forecast, at the same time of year (±30 days, any year), and add its actual errors hour by hour to the measured values. This carries over the real error sizes, how long errors last, and how the three variables go wrong together.
- *Bias removed.* A forecast corrected to local measurements does not have the average error of the raw forecast. So the average error of all comparable forecasts (same start time, same season) is subtracted, e.g. Seoul's forecast being on average 1.5 °C colder than the city-centre station. What remains are the hour-to-hour errors. The same applies to solar radiation, rain amounts and visibility. Since 4 Oct 2026 every weather input also keeps the observed value on average after the physical limits (see Physical limits) cut off values: the errors of each hour are shifted just enough to make up for the cut (before, the limits removed only the errors beyond them, see limitation 10). It also applies to rain: false rain is set so that, on average, it occurs in as many hours as rain is missed, so the forecast is wet about as often as observed (this removes the raw forecast's rain-frequency bias), and its amounts are set so that, on average, false rain adds as much rain as missed rain takes away. How often rain is missed stays as measured.
- *Solar radiation, precipitation, visibility.* These errors depend on the weather itself, so they are simulated with the measured values. For precipitation, the miss probability depends on lead time, time of year and two observed-intensity classes: light <1 mm/h and stronger ≥1 mm/h. False-rain frequency and precipitation amount errors otherwise keep the existing model. False rain is balanced against the combined misses when `nwp_rain_frequency_unbiased` is on. These errors also persist from hour to hour as measured.
- *Snow on the ground.* Every city has total precipitation (melted snow included) and snow depth on the ground. No forecast errors of snow depth are available, so the forecast keeps the snow depth measured when the forecast is made (persistence).
- *Physical limits.* No sun at night; solar radiation at most the 99.5th percentile of the training period; humidity 0–100 %; no negative wind; visibility at most the city's cap (Seoul 20 km, Washington 16 km) or, in London, the training maximum (or the hour's measured value if larger); rain at most the training maximum (or twice the hour's measured amount if larger, so heavy rain above the training maximum still has a forecast error; false rain at most the training maximum). Rain and snow are one precipitation amount (melted snow included), so there is no rain/snow switch; snow lying on the ground is taken as it is when the forecast is made (no forecast errors of snow depth are available).
- *No sensitivity runs.* Runs with all error sizes ×0.5 and ×1.5 were planned when the error sizes came from the literature. With measured errors they are no longer run by default (they would triple the degraded runs); they can be switched back on in the configuration.

**5. Checks.**
- The model was run on the real bike data (data folder of 3 Oct 2026) with exactly the folds of the experiments (three random seeds). Light precipitation is missed more often than ≥1 mm/h precipitation in every city and seed (light / stronger: Seoul 29–32 % / 14–20 %, London 48–49 % / 17–22 %, Washington 48–49 % / 28 %). Degraded precipitation is wet 0.96–1.07 times as often as the clean data. Mean temperature error −0.08 to +0.05 °C. It costs about 8 ms per test period.
- Seoul precipitation: the station and bike data use the same KMA quantity. April–October is hourly; November–March is reported as 3-hour totals (checked on the station reports of 2024–25: all 82 positive winter amounts sit at 00, 03, …, 21 h and add up to the station's 24-hour totals). These totals are spread evenly over their three hours in the bike data and in the NWP calibration; in the calibration the forecast is spread over the same three hours, so both are compared at 3-hour resolution.
- Seoul: the Seoul bike data are identical to the station used as truth (median difference 0.0 °C).
- London and Washington: their weather data had been stored on standard time all year, one hour off the bike counts during daylight saving; they were re-aligned to the local clock on 2 Oct 2026. Since then they differ from ERA5 at the airport grid cell by 0.27 °C (London) and 0.30 °C (Washington) on average (the data were downloaded for 51.5074, −0.1278 and 38.9073, −77.0369, not for the airports; see `data/provenance/`).
- Each city draws its own random numbers (before, fold k used the same random numbers in all three cities).
- Every weather input keeps the observed value on average after the physical limits, and false rain balances the missed rain amounts (limitation 10). Turning both off (`nwp_mean_preserving_caps = False`, `nwp_rain_amount_unbiased = False`) gives exactly the earlier degraded data. Turning them on changes nothing else: the same hours of missed and false rain, the same replayed forecast, the same temperature and snow depth, and hours at a limit (visibility at the cap, humidity 100 %, calm wind) are not changed.
- The check script prints, next to every number, how much it varies by chance with one random seed (95 % range).
- 54 automated tests (`testing/test_weather_unit.py`).

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
| Precip. miss, light <1 mm/h | 22.2% | 28.1% | 38.2% | 61.2% |
| Precip. miss, stronger ≥1 mm/h | 7.9% | 15.5% | 20.2% | 52.7% |
| Precip. miss at 24 h, January light / stronger |  | 32.6% / 15.5% |  |  |
| Precip. miss at 24 h, July light / stronger |  | 28.9% / 17.4% |  |  |
| Precip. (rel. var.) | 286.1% | 342.0% | 531.9% | 633.8% |
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
| Precip. miss, light <1 mm/h | 40.1% | 43.3% | 47.8% | 71.9% |
| Precip. miss, stronger ≥1 mm/h | 9.5% | 11.1% | 17.1% | 65.0% |
| Precip. miss at 24 h, January light / stronger |  | 35.0% / 5.8% |  |  |
| Precip. miss at 24 h, July light / stronger |  | 53.3% / 17.6% |  |  |
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
| Precip. miss, light <1 mm/h | 42.7% | 48.1% | 52.1% | 69.1% |
| Precip. miss, stronger ≥1 mm/h | 17.0% | 22.4% | 29.4% | 58.5% |
| Precip. miss at 24 h, January light / stronger |  | 38.7% / 6.4% |  |  |
| Precip. miss at 24 h, July light / stronger |  | 54.7% / 37.2% |  |  |
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
- *miss rate*: share of wet hours that the forecast had dry. Wet: ≥ 0.1 mm/h; Seoul November–March: 3-hour total ≥ 0.1 mm (station and forecast spread over the same three hours).
- *false alarm*: share of forecast wet hours that were dry, as applied by the model (on average equal to the miss rate, because false rain occurs on average in as many hours as rain is missed).
- *rel. var.*: relative spread of the amount error (coefficient of variation); for rain only hours that were wet in both forecast and truth, for visibility only hours below the cap.
- *at cap: forecast below cap*: share of hours at the visibility cap for which the forecast is below the cap.

## How the errors behave over the forecast hours

- **Growth.** The error grows gradually from hour to hour, following how far ahead the weather forecast looks. The first hours of a 7-day forecast have the same error size as a 6-hour forecast. Visibility errors do not grow with lead time.
- **Persistence.** Errors keep their direction for several hours, but they are not fixed for the whole forecast. In the real forecasts the temperature error keeps its sign into the next hour in 87–92 % of cases. A too-warm or too-cold stretch typically lasts 3–5 hours (on average 7–12 hours), and 15–39 % of forecasts stay on one side for their whole first day. Humidity behaves similarly; wind changes sign more often. Solar and visibility errors keep their sign into the next hour in 73–81 % of cases. Missed rain events and false rain usually last several hours.
- **Biases.** The raw forecasts have systematic average errors, the largest Seoul temperature (−1.5 °C), Seoul humidity (+4 %) and London wind (−0.8 m/s against ERA5). The model removes them, as a forecast corrected to local measurements would. With the biases removed, the cities are close: temperature 0.7–1.0 °C at 24 h.
- **Same for every hour of the day.** With a fresh forecast, the error depends only on how far ahead the hour is, not on when the demand forecast is made or in which time zone the city is.

## Seoul in detail

- **Temperature bias.** The raw forecast is colder than the station in every month (−0.6 to −2.5 °C), most at night and in winter (about −2 °C) and least on summer afternoons (about −0.6 °C). This is the typical urban heat effect: the station is in the warm city centre, which a 9 km forecast cannot resolve. A forecast corrected to local measurements would not have this average error, so the model removes it (default setting).
- **Precipitation measured against the station.** Seoul uses KMA station 47108. April–October amounts are hourly; November–March 3-hour totals are divided evenly over their three covered hours, matching the v7 bike data. The forecast is divided over the same three hours, because the station gives no timing within them; an hour is wet if its 3-hour total is at least 0.1 mm, for station and forecast alike. Tested on April–October data treated the same way: this gives a miss rate of 18 % at 24 h where truly hourly data give 22 %; comparing the divided station data with the hourly forecast would give 30 %.
- **Intensity-dependent misses.** At 24 h the year-round miss rates are 28.1 % for light precipitation (<1 mm/h) and 15.5 % for ≥1 mm/h. The same two-class pattern is used by month and lead time; the hit-amount error is unchanged.
- **Rain frequency.** The raw forecast is wet about 2.1 times as often as the Seoul station at 24 h (London and Washington about 0.8–0.9 times against ERA5). Like the temperature bias, this frequency bias is removed by default (`nwp_rain_frequency_unbiased`): false rain balances the combined missed precipitation events.

## Limitations

1. **Years.** The forecasts are from 2024–26, the bike data from 2011–18. Forecasts were less accurate then, so the errors are probably too small for those years. Runs with ×0.5 and ×1.5 error sizes are not part of the default setting. ECMWF forecasts from 2011–18 are not freely available (licensed archive; the free TIGGE archive needs an account and has no visibility or usable solar radiation).
2. **Replayed errors come from another day** of the same season; they do not depend on the weather of the test period.
3. **Solar, rain and visibility errors are independent** of each other and of temperature, humidity and wind (temperature, humidity and wind are linked through the replay).
4. **Only two precipitation-intensity classes.** Miss probability distinguishes light <1 mm/h from stronger ≥1 mm/h, but is not a continuous function of intensity. A third class (≥ 5 mm/h) is too rare: 113 (Seoul), 1 (London) and 46 (Washington) such hours in 2024–26. The hit-amount error is still one distribution per lead time and season. Sparse seasonal classes require wider windows: up to ±165 days for Seoul's stronger class. Seoul's winter rain is checked only at 3-hour resolution (see Seoul in detail): timing errors within the three hours are not counted.
5. **Calculated, not measured truth** (see step 2): for London and Washington (all variables except visibility) and for Seoul solar radiation, the truth is ERA5, calculated by a weather model of the same centre as the forecasts. Errors against it are probably smaller than against real measurements. Seoul solar radiation additionally does not match the Seoul bike data, which come from the station.
6. **Station data end in August 2025** (NOAA archive). Seoul has 844 forecasts with complete station data for temperature, humidity and wind and 1,049 with station rain, London and Washington 1,817.
7. **Fresh, corrected forecast is approximated.** No free archive of past local forecasts exists, so the errors are those of the global 9 km model at short lead times with its bias removed. Local high-resolution forecasts are usually more accurate in the first hours, so these may be slightly pessimistic. The replayed forecast's start time (00 or 12 UTC) can differ from the time of day of the demand forecast by up to 6 hours. Beyond 90 hours ECMWF provides 3- and 6-hourly values; the hours in between are interpolated by Open-Meteo.
8. **Daylight saving.** Lead times count test hours, so clock changes do not affect them. Only the choice of the replayed forecast's start time uses the clock; it could be off by one hour if a test window started at a clock-change hour (none does).
9. **Visibility.** London's visibility data (Visual Crossing) do not match the Heathrow reports the errors were measured against (see step 2). The visibility error is one distribution for all visibility levels, although real forecasts are too high in poor visibility and too low in very good visibility (London: on average about 5 times too high below 5 km, about one third too low above 40 km).
10. **Averages after the physical limits; chance with one seed.** Values beyond the physical limits are cut: visibility above the cap (Seoul 20 km, Washington 16 km; London: highest value of the training period, or the hour's measured value if larger), rain above the training period's maximum (or above twice the hour's measured amount, if that is larger: heavy rain above the training maximum keeps a forecast error), humidity above 100 %, wind below 0, solar radiation above the training period's limit. Until 3 Oct 2026 this removed only the errors beyond the limit, so these inputs were too low (or, for wind, too high) on average; and false rain had much smaller amounts than missed rain in Seoul (0.7 against 2.4 mm per hour). Since 4 Oct 2026 (settings `nwp_mean_preserving_caps` and `nwp_rain_amount_unbiased`) the errors of each hour are shifted just enough that the value after the cut keeps the observed value on average, and false rain amounts are sized so that false rain adds as much rain as missed rain takes away, on average. With the random seed of the experiments (42), Seoul / London / Washington: visibility below the cap 0.99 / 1.00 / 0.99 times the observed value (before 0.71 / 0.80 / 0.66), correctly forecast rain amounts 0.99 / 0.98 / 1.07 times (before 0.65 / 0.77 / 0.78), total precipitation 0.95 / 1.02 / 1.07 times (before 0.57 / 0.84 / 0.85), humidity −0.01 / −0.18 / −0.52 %-points (before −0.17 / −0.32 / −0.83). Over 21 seeds the averages are within about 1 % of the observed values (humidity within 0.04 %-points), except London's total precipitation (1.03: its false rain occurs about 10 % more often than missed rain). Still open: one seed's numbers vary by chance, because a few heavy rain events and, at 7 days, only 35 test periods carry much weight: about ±4–6 % for rain amounts and ±0.2–0.3 %-points for humidity (at seed 42, Washington's rain amounts 1.07 are the highest and its humidity −0.52 the lowest of the 21 seeds). All models see the same degraded data, so comparisons between models are not affected. Also still open: hours exactly at a limit cannot keep their value on average without losing their error (Seoul wind +0.007 m/s from calm hours; solar radiation −0.2 % from hours at the limit; visibility at the cap falls below it as often as measured), and hours close to a limit now sit at the limit more often. Temperature is not affected. (An earlier version of this note put the humidity deficit of seed 42 entirely down to the cut at 100 %; the cut accounted for −0.13 / −0.16 / −0.31 %-points, the rest was chance.)

## Sources

- ECMWF IFS HRES forecasts via the Open-Meteo Single Runs API (Zippenfenig, 2023; CC BY 4.0).
- ERA5 (Hersbach et al., 2020) and ERA5-Land (Muñoz Sabater, 2019) via the Open-Meteo Historical Weather API.
- NOAA Integrated Surface Database, hourly station reports: Seoul 47108 (including hourly rain amounts from 2024), London Heathrow 03772, Washington Reagan National 72405-13743.
