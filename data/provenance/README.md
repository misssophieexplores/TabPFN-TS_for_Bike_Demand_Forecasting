# Data provenance

How `SeoulBikeData.csv`, `LondonBikeData.csv` and `WashingtonBikeData.csv` were produced. Records only, not part of the pipeline.

The scripts in `seoul/`, `london/` and `washington/` were run in an earlier repository. Their paths refer to that layout: bike files in `data/`, weather downloads in `data/weather_data/`. The input files are in `raw/`.

## Seoul

1. `seoul/saving_seoul_data.py`: UCI dataset 560 (via `ucimlrepo`) → `SeoulBikeData.csv`. Rentals on non-functioning days are replaced by the value 24 h earlier, else 168 h earlier, else the mean of functioning hours at that hour of day.
2. `../build_weather_v7.py`: adds `precipitation_mm`, `snow_depth_cm`.

## London

1. `london/load_weather_london.py`: Open-Meteo Historical Weather API, 51.5074, −0.1278, 2015-01-04 to 2017-01-03 → `raw/london_weather_2015-2017.csv`.
2. `london/saving_london_data.py`: `raw/london_og.csv` (Kaggle bike data) joined to the weather on timestamp, keeping all 17,544 weather hours → `london_merged_weather_bikes.csv`.
3. `london/saving_london_data_missing_values.py`: the 130 hours without bike data get the count from 24 h earlier, else 168 h earlier, else the mean at that hour of day; `Functioning Day` = No for these hours → `LondonBikeData.csv`.
4. `london/fill_missing_london_weather.py` → `LondonBikeData.csv`:
   - Visual Crossing fill for missing weather: no weather was missing, nothing was filled.
   - `t1`, `hum`, `wind_speed` overwritten with the Open-Meteo values.
   - `is_weekend` from the date; `is_holiday` and `season` for the added hours from the same date; 2 Sep 2016 set to `is_holiday` 0, `is_weekend` 0, `season` 2.
5. `../add_visibility.py` (run with `TEST_MODE = False`): adds `visibility_km`.
6. `../build_weather_v7.py`: daylight-saving re-alignment, `precipitation_mm`, `snow_depth_cm`.

## Washington

1. `washington/load_weather_washington_openmeteo.py`: Open-Meteo Historical Weather API, 38.9073, −77.0369, 2011-01-01 to 2012-12-31 → `raw/washington_dc_weather_2011-2012.csv`. This is the first version of `data/load_weather_washington.py` in the earlier repository (commit `cde7a49`).
2. `washington/load_weather_washington.py`: `raw/washington_hour.csv` (bike data, see `../README_BIKES.md`) + weather → `WashingtonBikeData.csv`:
   - Complete hourly range 2011-01-01 00:00 to 2012-12-31 23:00; `season`, `holiday`, `workingday` for added hours from the same date.
   - Visual Crossing fill for missing weather: no weather was missing, nothing was filled.
   - Hours without bike data: count from 24 h earlier, else 168 h earlier, else the mean at that hour of day; `Functioning Day` = No.
   - Column selection, added on 3 Oct 2026 (it had been done in a notebook).
3. `../add_visibility.py` (run with `TEST_MODE = False`): adds `visibility_km`.
4. `../build_weather_v7.py`: daylight-saving re-alignment, `precipitation_mm`, `snow_depth_cm`.

## Verification (3 Oct 2026)

- London steps 2–4 and Washington step 2, re-run from `raw/`, reproduce both files as they were before visibility and the v7 columns: all values identical (Washington: up to floating-point rounding in `solar_radiation_mjm2`).
- The columns not changed later are identical in the current files.
- The Open-Meteo weather columns of the current files equal `raw/london_weather_2015-2017.csv` and `raw/washington_dc_weather_2011-2012.csv` in every hour, after the daylight-saving re-alignment. None of these values comes from Visual Crossing.
- Seoul step 1 was not re-run.

## Seasonality

`seasonality/`: autocorrelation, periodogram and average daily, weekly and monthly profiles per city (earlier repository; output kept in the notebooks). Basis for the daily and weekly periodicity stated in the paper. Autocorrelation at 24 h / 168 h: Seoul 0.68 / 0.66, London 0.80 / 0.89, Washington 0.82 / 0.87.
