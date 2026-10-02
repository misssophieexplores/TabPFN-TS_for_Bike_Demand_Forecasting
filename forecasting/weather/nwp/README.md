# Measured NWP forecast errors: data, scripts, results

This folder measures real weather-forecast errors per city and turns them into the calibration files of the degradation model (`weather/nwp_error_model.py`). How the model uses them: `weather/weather_methodology.md` (technical) and `weather/weather_degradation_summary.md` (plain language).

Status 2 Oct 2026.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1. Download forecasts and references (needs internet; run on the Mac) | `fetch_nwp_data.py` | `nwp_data/` raw files, `nwp_extracts.zip` |
| 2. Error statistics per city, variable, lead time | `analyze_nwp_errors.py --data <nwp_extracts> --out <dir>` | `nwp_error_statistics.csv` (119,051 rows), `diagnostics.json` |
| 3. Calibration files for the model | `build_nwp_calibration.py --data <nwp_extracts>` | `calibration/<city>.npz`, `calibration/<city>_summary.json` |
| 4. Check on the bike data (experiment folds) | `validate_on_real_data.py` (run from the folder that contains `data/`; `SEED=42`) | `sim_errors_<seed>.csv` |
| 5. Errors applied by the default model at 6/24/48/168 h | `expected_errors.py` | `expected_errors.json` |

Raw data: `~/GitHub/nwp_errors/nwp_data` on the Mac (not in the repository).

## Data

- **Forecasts:** ECMWF IFS HRES 9 km, Open-Meteo Single Runs API, 00/12 UTC runs, 14 Mar 2024 – 16 Sep 2026, 1,828 runs per city, nearest grid cell, hourly up to 192 h (beyond 90 h interpolated by Open-Meteo from 3-/6-hourly output).
- **Same-years check:** NOAA GEFSv12 reforecast control run, 00 UTC, 3-hourly; 1,196 of 1,827 days (Seoul 183, London 464, Washington 549).
- **References:** NOAA ISD stations Seoul 47108, London Heathrow 03772, Washington Reagan National 72405-13743 (until Aug 2025); Open-Meteo `era5_seamless` (ERA5 / ERA5-Land) for the whole period.
- **Reference used per city** (the source of each city's bike-data weather):

| City | Temperature, humidity, wind | Rain | Solar radiation | Visibility |
| --- | --- | --- | --- | --- |
| Seoul | station 47108 | station 47108 (hourly amounts from 2024) | ERA5 | station 47108 |
| London | ERA5 | ERA5 | ERA5 | Heathrow |
| Washington | ERA5 | ERA5 | ERA5 | Reagan National |

## Fitted lines (raw errors against the reference above, bias included, leads 1–168 h)

Statistic = a + b·h, h = lead time in hours. "Literature" = the earlier degradation model.

| Statistic | Literature | Seoul | London | Washington |
| --- | --- | --- | --- | --- |
| Temperature σ (°C) | 0.77 + 0.0181·h | 1.19 + 0.0087·h | 0.63 + 0.0103·h | 1.06 + 0.0128·h |
| Humidity σ (%) | 13.0 + 0.023·h | 9.4 + 0.042·h | 5.2 + 0.037·h | 7.6 + 0.050·h |
| Wind σ (m/s) | 1.8 + 0.010·h | 0.90 + 0.0024·h | 0.51 + 0.0080·h | 0.64 + 0.0051·h |
| Solar relative MAE (%) | 15 + 0.15·h | 13.5 + 0.10·h | 15.6 + 0.11·h | 12.5 + 0.09·h |
| Visibility CV | 25 % | ~103 % flat | ~110 % flat | 44 % flat (capped at 16 km) |
| Rain miss rate (≥ 0.1 mm/h) | 0.25 + 0.0042·h (max 0.5) | 0.23 + 0.0011·h | 0.41 + 0.0003·h | 0.42 + 0.0005·h |
| Rain false-alarm ratio (≥ 0.1 mm/h) | = miss rate | 0.61 + 0.0016·h | 0.23 + 0.0031·h | 0.24 + 0.0028·h |
| Rain amount CV, hits | 30 % + 0.15 %·h | 342 % + 7.1 %·h (poor fit, r² 0.09) | 115 % + 1.3 %·h | 158 % + 2.1 %·h |

Rain rows: CSV statistics `*_gt0` (amount > 0, which is ≥ 0.1 mm/h for these data in 0.1 mm steps). The CSV's `*_gt0.1` statistics are strictly > 0.1 mm/h and drop all 0.1 mm hours (Seoul station: wet share per lead about 2 % instead of 5.7 %); do not use them for comparison with the model.

The model does not apply these raw values directly: it uses a fresh forecast (lead = hours ahead), removes the biases, matches rain errors to the time of year, sets the forecast to be wet as often as observed, and caps rain at the training maximum. Errors as applied: `expected_errors.py` and `weather_degradation_summary.md`.

## Gaps

- No ECMWF forecasts for the data years 2011–18 (licensed archive; TIGGE needs an account and has no visibility or usable solar radiation).
- No station solar radiation; solar is measured against ERA5 for all cities.
- Seoul station: hourly rain amounts only from 2024; station reports end Aug 2025 (844 complete runs for temperature, humidity and wind; 1,049 with station rain).
- Heathrow before 2024 has only 6-/12-h rain totals; no visibility in GEFS or ERA5.
