# Shared Bike Demand Forecasting - Architecture

## Project Structure
```
forecasting/
├── main.py                  # Multi-city orchestrator — runs all datasets sequentially
├── config.py                # Shared base configuration dataclass (wandb_project, results_version, horizons, etc.)
├── config_seoul.py          # Seoul-specific overrides (get_config())
├── config_london.py         # London-specific overrides (get_config())
├── config_washington.py     # Washington-specific overrides (get_config())
├── features.py              # Calendar time feature engineering (used by XGBoost)
├── provenance.py            # git commit + dirty flag (code ID without .git) + library versions for tuning JSONs and results; output-folder check and params-JSON writer of the tuning scripts
├── run_timesfm_server.py    # Persistent TimesFM server, run in .timesfm_venv (managed by timesfm_model.py)
├── run_timesfm.py           # One-off TimesFM run from the command line (uses the server's load_model()/run_inference(); same results as the pipeline)
├── models/
│   ├── base.py              # BaseForecaster abstract class
│   ├── statistical.py       # Seasonal Naive, ARIMA, SARIMAX, trend_from_intercept()
│   ├── ml_models.py         # XGBoost and XGBoost_NoWeather with lag features
│   ├── tabpfn_pipeline_model.py  # TabPFN pipeline models
│   ├── prophet_models.py    # Prophet, NeuralProphet, NeuralProphet_NoWeather
│   ├── timesfm_model.py     # TimesFMForecaster, TimesFMForecaster_NoWeather
│   └── tuning/              # Hyperparameter tuning scripts
│       ├── arima_search.py          # Order search shared by tune_arima.py / tune_sarimax.py
│       ├── tune_arima.py            # ARIMA order tuning (auto_arima candidates, MAE selection)
│       ├── tune_sarimax.py          # SARIMAX order tuning (auto_arima candidates, MAE selection)
│       ├── tune_xgboost.py          # XGBoost Optuna TPE search
│       ├── tune_prophet.py          # Prophet random search
│       └── tune_neuralprophet.py    # NeuralProphet random search
├── weather/
│   ├── nwp_error_model.py          # Measured NWP error model per city (default degradation model)
│   ├── weather_degradation.py      # Literature error model; training-fold parameters; snow-depth persistence
│   ├── weather_processor.py        # Scenario orchestration
│   ├── weather_methodology.md      # Technical description of both error models
│   ├── weather_degradation_summary.md # Plain-language summary of the measured model
│   ├── EVIDENCE_MAPPING.md         # Literature model: sources of each error size
│   └── nwp/                        # Measured forecast errors: download, analysis, calibration
│       ├── README.md               # Pipeline, data sources, measured error lines per city, gaps
│       ├── fetch_nwp_data.py       # Downloads ECMWF forecasts, reanalysis and station data (normal internet access needed)
│       ├── analyze_nwp_errors.py   # Error statistics per city, variable and lead time (CSV for the paper)
│       ├── build_nwp_calibration.py # Builds the calibration files used by nwp_error_model.py
│       ├── expected_errors.py      # Errors applied by the default setting at 6/24/48/168 h (for the paper)
│       ├── validate_on_real_data.py # Runs the model on the real data with the experiment folds (check)
│       ├── check_seoul_winter_split.py # Evidence for the Seoul winter rain treatment (station format, comparison method)
│       └── calibration/            # seoul.npz, london.npz, washington.npz (+ readable *_summary.json)
├── evaluation/
│   ├── cv.py                # TimeSeriesCV with dynamic fold calculation
│   └── metrics.py           # MAE, RMSE, MASE, sMAPE; win rate and skill score vs Seasonal Naive
├── testing/
│   ├── test_max_degradation.py
│   ├── test_weather_single_model.py
│   ├── test_weather_unit.py
│   ├── test_pipeline_unit.py   # CV folds, metrics, comparative metrics, XGBoost tuning = experiment (with and without weather)
│   ├── test_provenance.py      # Code ID and git provenance
│   └── preflight.py            # Pre-flight check before the paper runs: params files (set, readable, keys, city, scenario, tuning period, covariates incl. order, code IDs), build_models, CV, one fold per model (incl. TabPFN/TimesFM weights), W&B; writes nothing; reports every problem and continues
├── run_experiments.py       # ForecastingExperiment class with W&B logging and checkpointing; load_and_prepare_data(); comparative metrics
└── run_weather_baseline.py  # Per-city experiment runner (called by main.py, or directly with --city)

data/
├── SeoulBikeData.csv
├── LondonBikeData.csv
├── WashingtonBikeData.csv
├── README_BIKES.md          # Sources and licences of the bike data
├── README_WEATHER.md        # Sources and licences of the weather data; v7 weather columns
├── build_weather_v7.py      # Provenance record of the v7 weather columns (not part of the pipeline)
├── add_visibility.py        # Visibility download (Visual Crossing) for London and Washington (not part of the pipeline)
├── open_meteo/              # Open-Meteo downloads used for snow depth (London, Washington)
├── provenance/              # How the three data files were produced: scripts, raw inputs, seasonality notebooks (not part of the pipeline)
└── example_data/            # Not used by the pipeline

results/                                           # Output directory 
├── figures/                 
├── tables/
├── tuning/                                        # Tuning results (JSON)
├── results_master_{version}.csv                   # Aggregated results (all datasets)
├── detailed_results_master_{version}.csv          # Fold-level results (all datasets)
├── forecasts_{dataset_name}_{version}.csv         # Hourly forecasts, one row per forecast hour (one file per dataset)
├── checkpoint_{dataset_name}_{version}.json       # Per-run recovery checkpoints (file name = checkpoint_{experiment_name}.json)
└── errors_{version}.log                           # Fold-, city- and run-level error tracebacks
```

All scripts resolve `data/` and `results/` relative to the current working directory (`Path("data") / config.data_filename`), so run them from the repository root, e.g. `python forecasting/main.py`.

## Data Flow

1. **Load**: `load_and_prepare_data()` reads CSV, parses dates, sorts by time (stable sort), drops duplicate timestamps (e.g. DST clock-back hours, keeps the first occurrence = the daylight-saving hour), applies `config.column_scale_factors`, normalizes holiday and season columns and appends them to `weather_covariates` if not already listed
2. **Scenario Setup**: `WeatherProcessor` selects variables based on scenario
3. **Split**: `TimeSeriesCV.split(df, horizon, partial_last_fold=True)` creates rolling window train/test folds covering the full evaluation period
4. **Weather Preparation**: Per-fold weather preparation via `WeatherProcessor.prepare_weather_data(split=...)`. Training data always uses clean observed weather. In the degraded scenarios ('degraded'; optional noise-magnitude sensitivity scenarios if added to `degradation_scales`) the training fold also provides the degradation parameters (solar cap, wet-hour share, rain maxima), and test data receives per-row lead-time noise, scaled by the scenario's factor: row *i* is degraded using lead time *(i + 1)* hours, so error grows from the 1-hour error at the first step (not zero: the measured model replays the ECMWF error at lead time 1 h; in the literature model every error formula has an intercept, e.g. temperature σ = 0.79 °C, humidity 13.0 %-points) up to full-horizon noise at the last step.
5. **Model inputs** (`run_experiments.prepare_fold_inputs()`, also used by `testing/test_weather_single_model.py`): models with `use_time_features=True` get calendar features appended (`prepare_xgboost_features`); models with `needs_datetime=True` get a real `DatetimeIndex` on `X_train`/`X_test` (an empty DataFrame with that index if the model has no covariates)
6. **Fit**: `model.reset()`, then `model.fit(y_train, X_train)` on each fold (NeuralProphet only stores the data here — see Models). Wall-clock time of `fit()` is recorded as `fit_time_s`
7. **Predict**: `model.predict(n_steps, X_test)` generates forecasts, with `n_steps = len(test_df)`: the horizon, or fewer hours for a partial last fold. Wall-clock time of `predict()` is recorded as `predict_time_s`; `runtime_s = fit_time_s + predict_time_s` (see Runtime Measurement)
8. **Evaluate**: `MetricsCalculator.calculate_all(y_test, y_pred, y_train, test_mask, train_mask)` computes metrics on observed hours only; imputed hours (`functioning_day_col == 'No'`) are excluded from scoring (masks from `MetricsCalculator.observed_mask()`). Imputed hours stay in the model inputs (training data)
9. **Log**: W&B logs aggregated metrics and a per-fold table
10. **Save**: after each successfully completed model-horizon-scenario run, the hourly forecasts are appended to `forecasts_{dataset_name}_{version}.csv`, the fold-level rows to `detailed_results_master_{version}.csv` and the aggregated row to `results_master_{version}.csv` (header must match, otherwise `RuntimeError`); only then is the checkpoint updated. If any fold fails, that whole model-horizon-scenario run is aborted and none of its partial rows are saved or checkpointed
11. **Compare** (once, after all selected cities completed successfully): `compute_and_log_comparative_metrics()` computes win rate and skill score vs `Seasonal_Naive`, pooled across cities, one comparison per (model, scenario)

**Weather Data Flow:**
- **all_weather**: Use all weather columns from `config.weather_covariates` as-is (in `config.weather_scenarios`, but not run by `run_weather_baseline.py`)
- **clean_only**: 7 degradable columns + holiday + season (no degradation)
- **degraded**: 7 degradable columns + holiday + season + apply degradation to degradable columns only with seed(fold_idx, horizon). Degradation applied to **test split only** (training always uses clean observed weather). Each test row receives noise scaled to its own lead time (row *i* → lead time *i + 1* hours).
- Optional noise-magnitude sensitivity scenarios (e.g. **degraded_x050**, **degraded_x150**): not run since 2 Oct 2026; if added to `config.degradation_scales` (and `weather_scenarios`), as degraded with the error magnitudes multiplied by the factor; same seeds as degraded, so only the magnitude differs

## Core Components

### Configuration (`config.py` + city configs)
- `config.py`: Shared base dataclass. Contains the fields that are identical across all datasets: `wandb_project`, `results_version`, `horizons`, `n_folds`, `n_train_samples`, `seasonal_period`, `degradation_seed`, `degradation_model`, `nwp_fresh_forecast`, `nwp_remove_bias`, `nwp_seasonal_rain`, `nwp_rain_intensity_dependent`, `nwp_rain_frequency_unbiased`, `nwp_rain_amount_unbiased`, `nwp_mean_preserving_caps`, `nwp_run_hours_utc`, `nwp_availability_delay_h`, `nwp_season_window_days`, `degradation_scales`, `weather_scenarios`, `output_dir`, `verbose`, `experiment_name`, `tune_folds`, `tune_horizon`. Dataset-specific fields default to `None` (except `holiday_mapping` and `column_scale_factors`, see below).
- `config_seoul.py`, `config_london.py`, `config_washington.py`: Each exposes a `get_config()` function that instantiates `ForecastConfig` and overrides all dataset-specific fields. To update `wandb_project` or `results_version`, change `config.py` only — all cities pick it up automatically.
- Dataset-specific fields (set per city): `data_filename`, `dataset_name`, `date_col`, `target_col`, `functioning_day_col`, `holiday_col`, `holiday_mapping`, `season_col`, `season_mapping`, `weather_covariates`, `weather_degradation_mapping`, `timezone`, `nwp_calibration_file`, `column_scale_factors`, `arima_params_file`, `sarimax_params_file`, `xgb_params_file`, `xgb_noweather_params_file`, `prophet_params_file`, `neuralprophet_params_file`, `neuralprophet_noweather_params_file`
- Horizons: [6, 24, 48, 168] hours
- Training size (`n_train_samples`): 720 observations (30 days). The previous setting (4096 observations, 20 folds) is kept commented out in `config.py`.
- Number of folds (`n_folds`): 35, counted in folds of the longest horizon: evaluation period = `n_folds * max(horizons)` = 35 × 168 = 5,880 h for every horizon. Folds per horizon: 980 (6 h), 245 (24 h), 123 (48 h: 122 full + 1 partial fold of 24 h), 35 (168 h). The evaluation period does not need to be a multiple of each horizon (see Partial Last Fold)
- Seasonal period: 24
- Results version: `v7`
- W&B project: `bike-forecasting`
- Weather scenarios: ['all_weather', 'clean_only', 'degraded'] (sensitivity scenarios removed 2 Oct 2026)
- Degradation seed: 42 (reproducible error simulation)
- `degradation_model`: error model of the degraded scenarios. `"nwp_measured"` (default): per-city errors measured from real ECMWF forecasts (`weather/nwp_error_model.py`). `"literature"`: the earlier model with published error sizes, the same for every city (`weather_degradation.degrade_weather_dataset`); kept to reproduce earlier results. Any other value raises `ValueError` in `WeatherProcessor`
- `nwp_fresh_forecast` (`True`), `nwp_remove_bias` (`True`): measured model only. Default = an operator with an up-to-date, locally corrected weather forecast: test hour i gets the error of an (i+1)-hour forecast in every city, and the average error (bias) of the forecasts is removed. `nwp_fresh_forecast = False`: the newest run available at the issue time is used (6–17 h old); `nwp_remove_bias = False`: the measured bias is kept (both `False` = the first version of 1 Oct 2026)
- `nwp_run_hours_utc` (`[0, 12]`), `nwp_availability_delay_h` (6), `nwp_season_window_days` (30): measured model only. ECMWF runs start at these UTC hours (fresh forecast: the start hour closest to the issue time of day is replayed; otherwise a run can be used `nwp_availability_delay_h` hours after its start); the replayed run starts within ± `nwp_season_window_days` of the test window's day of year
- `nwp_seasonal_rain` (`True`): measured model only; rain miss rate, false-alarm ratio and amount error of the time of year (12 monthly bins). `False`: year-round values
- `nwp_rain_intensity_dependent` (`True`, since 2 Oct 2026): measured model only; precipitation miss probability uses two observed-intensity classes, light <1 mm/h and stronger ≥1 mm/h. `False`: one miss-rate curve for all intensities (same code path as before; Seoul's calibration values differ from the 2 Oct files, see "Seoul winter precipitation in the calibration"). FAR and hit-amount error remain the existing overall calibration.
- `nwp_rain_frequency_unbiased` (`True`, since 2 Oct 2026): measured model only; false alarms balance misses, so the degraded data are wet as often as the clean data (removes the rain-frequency bias like the temperature/humidity/wind bias: with the corrected Seoul winter truth, the raw forecast is wet about 2.1× as often as the station at 24 h). `False`: measured false-alarm ratio
- `nwp_rain_amount_unbiased` (`True`, since 4 Oct 2026): measured model only, in effect with `nwp_rain_frequency_unbiased = True`; false alarms also add as much precipitation as misses remove, in expectation: a false alarm's amount keeps the measured log SD, its mean (after the cap) is the expected amount of a missed hour at that lead time from the training fold (light/strong share, miss rates and mean amounts). `False`: measured false-alarm amounts (Seoul about 0.7 mm against 2.4 mm per missed hour, total precipitation ×0.84; Known Limitation 24). Changes only false-alarm amounts
- `nwp_mean_preserving_caps` (`True`, since 4 Oct 2026): measured model only, in effect with `nwp_remove_bias = True`; every degraded covariate keeps the clean value on average after its caps and bounds: visibility below the cap and rain-hit amounts (the mean of the log error is set per hour), humidity (0–100) and wind (≥ 0; a shift per hour over the candidate runs) and solar radiation (0 to the training cap; a relative shift per hour). Hours at a bound (visibility at the cap, humidity 0 or 100 %, calm wind, solar at or above the cap) are not changed. `False`: the errors are mean-preserving before the cut, which shifts the averages (the version before 4 Oct 2026; output bit-identical to it). See Known Limitation 24
- `degradation_label()`: error model and settings as written to the results, e.g. `nwp_measured(fresh,no_bias,seasonal_rain,intensity_miss,rain_freq_unbiased,rain_amount_unbiased,cap_mean)` (`rain_amount_unbiased` only with `nwp_rain_frequency_unbiased`, `cap_mean` only with `nwp_remove_bias`, i.e. where they act)
- `timezone` (per city): IANA time zone of the date column (Seoul `Asia/Seoul`, London `Europe/London`, Washington `America/New_York`); the measured model converts the first test hour to UTC; the issue time (first hour − 1 h) selects the replayed run (start hour, day of year). Lead times count rows and do not use the clock time. A repeated autumn hour is read as daylight-saving time (the occurrence `load_and_prepare_data()` keeps), a non-existent spring hour is shifted forward (London and Washington have both in their data)
- `nwp_calibration_file` (per city): calibration file of the measured model, relative to `forecasting/` (`weather/nwp/calibration/<city>.npz`)
- `degradation_scales`: `{'degraded': 1.0}` (sensitivity factors removed 2 Oct 2026; add e.g. `'degraded_x150': 1.5` to run one) — every degraded scenario and the factor applied to the calibrated error magnitudes (see Noise-magnitude sensitivity). `config.is_degraded(scenario)` (True for its keys) is the single test for a degraded scenario, used by `WeatherProcessor` and by the skip logic in `run_experiments.py`
- `experiment_name`: Defaults to `{dataset_name}_{results_version}` (set in `__post_init__`; re-set in `run_weather_baseline.main()` if it starts with `None`)
- `tune_horizon`: Horizon used by all tuning scripts (24)
- `tune_folds`: Number of (last) CV folds used by the tuning scripts (90; `None` = all folds)
- `holiday_col`: Optional column name for public holidays (normalized to 0/1, appended to `weather_covariates` at load time)
- `holiday_mapping`: Dict mapping raw holiday string values → 0/1. Default `{'Yes': 1, 'No': 0}`; Seoul overrides with `{'Holiday': 1, 'No Holiday': 0}`. Only used when `holiday_col` is not numeric (object, or `str` in pandas ≥ 3); numeric holiday columns are coerced to int (NaN → 0).
- `column_scale_factors`: Dict `{column: factor}`; each listed column is multiplied by its factor at load time. Keys must match column names exactly (case-sensitive); non-matching keys are silently ignored. Default `{}`; Seoul sets `{"Visibility": 0.01}` (raw unit 10 m → km, the unit used for London and Washington).
- `weather_degradation_mapping` variable types: `temperature`, `humidity`, `wind_speed`, `solar_radiation`, `visibility`, `precipitation` (total precipitation incl. melted snow, mm; exactly one column per city: `precipitation_mm` in all three) and `snow_depth` (snow on the ground, cm: `snow_depth_cm` in all three). `rain_col` / `snow_col` were removed on 2 Oct 2026 together with the rain/snow phase correction.
- `season_col`: Optional column name for season (normalized to 0–3 int via `season_mapping`, appended to `weather_covariates` at load time)
- `season_mapping`: Explicit per-dataset dict mapping raw season values to 0–3 integers (handles strings, 0-based, and 1-based encodings). The codes (0 spring, 1 summer, 2 autumn, 3 winter) are the same in every city, but the source datasets define the seasons differently and the data are used as published (Known Limitation 22): Seoul and London switch on the 1st of March, June, September and December (meteorological seasons); Washington switches on 21 March, 21 June, 23 September and 21 December (astronomical seasons)
- `verbose`: If `True`, prints detailed progress (CV info, data loading, W&B URLs). Default `False` in `config.py` (cluster/server runs where stdout is captured in SLURM logs).
- Params files: each city config points to the tuned-parameter JSON files in `results/tuning/` (all tuned with `n_train_samples=720`). ARIMA and SARIMAX params files must contain `with_intercept`, i.e. they must come from the current tuning scripts; older files are rejected by `run_weather_baseline.py`. `neuralprophet_noweather_params_file` and `xgb_noweather_params_file` must be set (each tuned with `--scenario no_weather`); if one is not set (`None` or `""`), the runners raise `ValueError`. `build_models()` also raises if `xgb_noweather_params_file` was tuned with another scenario. All params files are re-tuned on 3 Oct 2026 (all models, all cities) after the covariate changes of 2 Oct 2026 (total precipitation, snow depth, London/Washington daylight-saving alignment, Seoul winter precipitation spread over 3 hours) and the SARIMAX change of 3 Oct 2026 (see ARIMA / SARIMAX specifics). `tune_xgboost.py` and `tune_neuralprophet.py` raise if a covariate column is missing from the data (before, it was silently left out).

### Weather Degradation (`weather/`)
**WeatherProcessor**: Orchestrates weather data preparation for scenarios
- `prepare_weather_data(split='train'|'test')`: Main entry point, applies scenario logic. Training split always returns clean weather; test split applies degradation with per-row lead times in the degraded scenarios (`config.is_degraded`), with `noise_scale = config.degradation_scales[scenario]`. In a degraded scenario, the train split also computes `degradation_params` (solar cap = 99.5th percentile of the clean training fold; wet-hour share = share of training hours with precipitation > 0; `precip_light_fraction` = among wet hours, share with precipitation <1 mm/h; `precip_max` = largest value of each precipitation column in the training fold; `visibility_max` = largest visibility of the training fold; `persist` = snow depth of the last training hour), which the test split then uses; the train split must be prepared first (otherwise `RuntimeError`). The test split passes the rows' timestamps (`config.date_col`) to the degradation.
- `get_weather_columns()`: Returns appropriate columns per scenario
- `degrade_dataframe(df, horizon, fold_idx, noise_scale=1.0, timestamps=None)`: On-the-fly degradation with proper seeding (the seed does not depend on `noise_scale`), with the model in `config.degradation_model` (no rain/snow phase correction since 2 Oct 2026; raises `ValueError` if `weather_degradation_mapping` has more than one precipitation column). The measured model needs `timestamps` (`ValueError` otherwise) and stores what it used in `last_degradation_info` (settings, forecast start, replayed start hour, first/last lead time, replayed run, number of candidate runs)

**Weather Scenarios:**
1. **all_weather**: All weather variables from `config.weather_covariates`, no degradation (original baseline)
2. **clean_only**: 7 degradable variables + holiday + season (excludes Dew point), no degradation
3. **degraded**: 7 degradable variables + holiday + season with realistic NWP forecast errors on degradable columns only (calibrated error model, scale 1.0)
4. Optional (not run by default since 2 Oct 2026): noise-magnitude sensitivity scenarios, e.g. degraded_x050 / degraded_x150

**Measured NWP error model** (default, `degradation_model = "nwp_measured"`, `weather/nwp_error_model.py`; full description in `weather/weather_methodology.md`):
- Data: 1,828 ECMWF IFS HRES 9 km runs per city (00/12 UTC, Mar 2024 – Sep 2026, Open-Meteo Single Runs API), compared with the source each city's covariates come from: Seoul = station 47108 (= KMA 108, the station of the Seoul data; Mar 2024 – Aug 2025, 844 complete runs); London and Washington = ERA5 / ERA5-Land (their covariates come from the Open-Meteo archive). Precipitation: Seoul = the station (Mar 2024 – Aug 2025, 1,049 runs; April–October hourly amounts, November–March 3-hour totals split over their three covered hours, the forecast split over the same hours), London and Washington = ERA5. Solar radiation uses ERA5 for all cities, visibility the station reports (Seoul 47108, Heathrow 03772, Reagan National 72405-13743; see Known Limitation 23)
- Lead times (`nwp_fresh_forecast = True`, default): the weather forecast starts when the demand forecast is issued (first test hour − 1 h), so row i gets lead time i + 1 in every city; errors are replayed from the 00/12 UTC runs closest to the issue time of day. `False`: the newest run available at the issue time is used (6–17 h old), row i gets lead time run age + i + 1. Lead times count rows (time steps), not clock time, so a clock change inside a test window does not shift them; the UTC issue time only selects the replayed run. (Fixed 2 Oct 2026, before the v7 runs: they were computed from UTC clock time, so in London and Washington the hours after the autumn change got lead time + 1 h in one fold per horizon.)
- Temperature, humidity, wind speed: the actual errors of one real ECMWF run of the city are added (the selected start hour, start within ± 30 days of the test window's day of year, any year; drawn at random with the fold seed). With `nwp_remove_bias = True` (default) the average error of all these candidate runs is subtracted first (bias removed, e.g. Seoul −1.5 °C). Error growth, hour-to-hour persistence and the links between the three variables are those of the real forecasts. Humidity clipped to [0, 100], wind truncated at 0. With the bias removed and `nwp_mean_preserving_caps = True` (default), humidity and wind get a shift per hour so that the value after the clip keeps the clean value on average over the candidate runs (the replayed run is one of them, drawn with equal probability); hours at 0 / 100 % humidity and calm hours are not shifted. Temperature is unchanged
- Solar radiation: X·(1 + b(lead) + s(lead)·z), b and s measured per lead time (b = 0 with the bias removed), z standard normal with the measured hour-to-hour correlation (AR(1)); 0 at night; capped at the solar cap of the training fold. With the bias removed and `nwp_mean_preserving_caps = True` (default), b is set per hour so that the value after the cut at 0 and at the cap keeps the clean value on average (closed form of the clipped normal mean; hours at or above the cap: b = 0)
- Precipitation (total precipitation incl. melted snow, mm; `precipitation_mm` in every city): one decision per hour. With `nwp_rain_intensity_dependent=True` the measured miss rate depends on lead time, season and observed intensity: light <1 mm/h vs stronger ≥1 mm/h. Calibration wet rule, the same for reference and forecast: ≥0.1 mm/h; Seoul Nov–Mar (station and forecast split over the same KMA 3-hour windows) 3-hour total ≥0.1 mm. FAR and hit-amount error remain overall seasonal values. Intensity-specific seasonal windows widen independently until every ±12-lead-hour pool through 168 h has ≥200 observed wet cases (Seoul light ±45–120 days, stronger ±45–165; London ±30 / ±60–105; Washington ±30 / ±30–75). With `nwp_rain_frequency_unbiased=True`, false-alarm probability uses the training fold's light/strong wet mix so false alarms balance combined misses in expectation. Misses and false alarms persist from hour to hour; hits keep the measured lognormal amount error; degraded amounts are capped at the training maximum (never below the observed amount). With the bias removed and `nwp_mean_preserving_caps = True` (default), the mean of the log error is set per hour so that the amount after this cap keeps the observed amount on average (an hour at or above the training maximum keeps its amount); `False`: mean-preserving before the cap, so the cap lowers hit amounts (×0.65–0.78, Known Limitation 24). False alarms: amount from the measured false-alarm distribution; with `nwp_rain_amount_unbiased = True` (default) rescaled so that false alarms add as much as misses remove, in expectation (see the config list).
- Visibility: lognormal error in log space with measured mean, SD and persistence. Seoul (cap 20 km) and Washington (cap 16 km): an hour at the cap stays there unless the forecast falls below it (measured probability 0.30 and 0.11); results are cut at the cap. London (no cap): cut at the training fold's maximum. With the bias removed and `nwp_mean_preserving_caps = True` (default), the mean of the log error of an hour below the cap is set so that the value after the cut keeps the clean value on average (London: cut at the larger of the training maximum and the hour's value); hours at the cap are unchanged. `False`: multiplier mean-preserving before the cut, which removes only the upward errors, so degraded visibility below the cap is ×0.66–0.80 of the clean value (Known Limitation 24)
- Snow depth (`snow_depth_cm` in every city; since 2 Oct 2026): persistence, the value of the last training hour (the issue time) in every test hour; no snow-depth forecast errors were measured. Not precipitation: no event draw, not in the wet-hour share or the rain cap
- Calibration files: `weather/nwp/calibration/<city>.npz`, built by `weather/nwp/build_nwp_calibration.py` from the downloaded data (`weather/nwp/fetch_nwp_data.py`); `<city>_summary.json` lists the main numbers
- Mean change degraded vs clean, default setting (all horizons; Seoul / London / Washington; average over 21 seeds, seed 42 in brackets): humidity −0.01 / −0.02 / −0.04 %-points (−0.01 / −0.18 / −0.52), wind +0.007 / +0.001 / +0.003 m/s, solar radiation in daylight ×0.998 / ×0.998 / ×0.998, visibility below the cap ×1.00 / ×1.00 / ×1.00 (×0.99 / ×1.00 / ×0.99), rain-hit amounts ×1.00 / ×0.99 / ×1.00 (×1.01 / ×0.99 / ×1.04), total precipitation ×1.01 / ×1.03 / ×1.00 (×0.96 / ×1.03 / ×1.05). Seed 42's deviations are within its chance variation, which `weather/nwp/validate_on_real_data.py` prints as (±) next to each number (Known Limitation 24)
- Checked on the real data (data folder of 3 Oct 2026) with the experiment's CV folds (all horizons, every fold, three seeds; `weather/nwp/validate_on_real_data.py`): light precipitation is missed more often than ≥1 mm/h precipitation in every city and seed (light / stronger: Seoul 0.29–0.32 / 0.14–0.20, London 0.48–0.49 / 0.17–0.22, Washington 0.48–0.49 / 0.28); degraded/clean wet-hour ratio 0.96–1.07 with the frequency-unbiased setting; mean temperature error −0.08 to +0.05 °C. The validation output records `precip_ob_mm` and `precip_class` for direct class checks. Temperature/humidity/wind checks remain unchanged; about 8 ms per fold.
- Typical errors applied by the default setting at 24 h: temperature 0.92 / 0.67 / 0.96 °C, humidity 7.4 / 4.4 / 6.4 %-points, wind 0.73 / 0.53 / 0.64 m/s (Seoul / London / Washington; `weather/nwp/expected_errors.py`)

**Literature error model** (`degradation_model = "literature"`, `weather_degradation.py`; same for all cities, row i has lead time i + 1). Degradation variables (7, all mapped in `weather_degradation_mapping` for Seoul, London and Washington):
- Temperature → Additive Gaussian error
- Humidity → Additive Gaussian error, clipped to [0, 100]
- Wind speed → Additive Gaussian, truncated at 0
- Solar Radiation → Heteroscedastic Gaussian error (σ ∝ value, so unit-independent), capped at the solar cap from the training fold
- Precipitation (one total-precipitation column per city) → one event-detection draw per hour (miss rate = FAR = 0.25 + 0.10·h/24, the line through the cited Day-1 and Day-2 values, capped at 50% from 60 h; miss rate = 1 − POD; false alarm for a dry hour with probability FAR / (1 − FAR) · POD · p / (1 − p), p = wet-hour share of the training fold; false-alarm amount written to the precipitation column) + one multiplicative lognormal magnitude error per detected hour
- Visibility → Multiplicative lognormal, constant CV = 25% (horizon-independent, unit-independent)
- Snow depth → persistence (as in the measured model)

**No rain/snow phase correction** (removed 2 Oct 2026): every city has one total-precipitation column (rain + melted snow, mm) and one snow-depth column (cm), so there is no rain/snow split to correct (see Key Design Decisions, "Precipitation and Snow: One Definition for All Cities").

**Non-degradable Variables (excluded from degradation, always passed through as-is):**
- Dew point temperature (excluded from `weather_degradation_mapping`; only included in `all_weather`, via `weather_covariates`)
- Holiday (`config.holiday_col`) — passed through in clean_only and degraded
- Season (`config.season_col`) — passed through in clean_only and degraded

**Noise-magnitude sensitivity** (`noise_scale`, from `config.degradation_scales`):
- Scaled (multiplied by `noise_scale`, before clipping/capping). Measured model: the replayed temperature, humidity and wind errors (bias included); the relative solar error (b and s·z); the log error of precipitation amounts of hits; the log error of visibility and the depth below the cap. Literature model: Gaussian σ of temperature, humidity and wind speed; relative MAE of solar radiation; magnitude CV of detected precipitation; CV of visibility
- Not scaled: precipitation event detection (miss rate, false-alarm probability per dry hour, false-alarm amount), whether visibility falls below the cap (measured model), the replayed run, the wet-hour share, the solar cap and the physical bounds; snow depth (persistence) has no error to scale
- `noise_scale = 1.0` ('degraded') is the calibrated error model, unchanged
- The seed does not depend on the scenario, and neither the detection outcomes nor the number of random draws per hour depend on `noise_scale`, so all degraded scenarios of a (horizon, fold) use the same random numbers (common random numbers): every additive error (before clipping) at 0.5× / 1.5× is 0.5 / 1.5 times the error at 1×, and precipitation is hit, missed and falsely forecast in the same hours (measured model: the same ECMWF run is replayed). The scenarios differ in error magnitude only
- `noise_scale < 0` raises `ValueError`

**Reproducibility:**
- Seed management: `horizon_seed = base_seed + 10000 * horizon + fold_idx` (`degrade_dataframe()` raises `ValueError` if `fold_idx` is outside [0, 10000)); measured model: plus `city_seed_term(config.dataset_name)` = 10,000,000 × (CRC32 of the name mod 1000), so each city draws its own random numbers (since 2 Oct 2026; literature model unchanged)
- Same seed → identical degradation
- Different folds → different realistic errors
- Every (horizon, fold) pair gets its own seed (at most 980 folds, for h=6, so `fold_idx < 10000`)
- All degraded scenarios use the same seed for a (horizon, fold) (see Noise-magnitude sensitivity)

### Models (`models/`)
**BaseForecaster**: Abstract interface requiring:
- `fit(y, X)`: Train model
- `predict(horizon, X)`: Generate forecasts
- `reset()`: Clear state between folds
- `name`, `use_covariates`, `use_time_features`: Properties

**Implemented models:**
- SeasonalNaiveForecaster: Repeats last seasonal period
- ARIMAForecaster: Tuned order via auto_arima (e.g., (2,1,2)); intercept/trend as selected during tuning (see ARIMA / SARIMAX specifics)
- SARIMAXForecaster: Tuned orders via auto_arima (e.g., (4,0,0)×(1,0,1,24)); intercept/trend as selected during tuning (see ARIMA / SARIMAX specifics)
- XGBoostForecaster: Uses lagged features (`n_lags` from tuning, options 12, 24, 48, 168) + weather covariates (including holiday and season via `weather_covariates`) + calendar time features (hour, dayofweek, month, is_weekend). `use_time_features=True` — pipeline appends calendar features automatically at fold time. **Note: XGBoost must be re-tuned whenever `weather_covariates` changes (e.g. after adding holiday/season).**
- XGBoostForecaster_NoWeather (model name `XGBoost_NoWeather`, key `xgboost_noweather`): same model, without weather, holiday and season covariates: lagged demand + calendar time features (hour, dayofweek, month, is_weekend) only. `use_covariates=False`, `use_time_features=True`: the pipeline passes the calendar features only, and the degraded scenarios are skipped. Counterpart of XGBoost in the with/without-weather comparison (as TabPFN / TabPFN_NoWeather and NeuralProphet / NeuralProphet_NoWeather). Tuned with `tune_xgboost.py --scenario no_weather`
- TabPFNPipelineForecaster (model name `TabPFN`): `TabPFNTSPipeline` (tabpfn-time-series), zero-shot, TabPFN v2.5 pinned via `TABPFN_MODEL_CONFIG` (`tabpfn-v2.5-regressor-v2.5_default.ckpt`); all other settings at the pipeline defaults. Features: the pipeline's defaults (running index, calendar, auto-seasonal) + every covariate column (present in both context and future frame). Point forecast = median. Without the explicit pin the checkpoint would depend on the installed tabpfn-time-series version (1.0.10: v2; 1.1.0/1.2.0: v3; 1.3.0: v3.5)
- TabPFNPipelineForecaster_NoWeather (model name `TabPFN_NoWeather`): same pipeline and model, univariate (context and future frames contain only timestamps and target)
- TimesFMForecaster / TimesFMForecaster_NoWeather (`timesfm_model.py`, server `run_timesfm_server.py`): TimesFM 2.5 (200M, `google/timesfm-2.5-200m-pytorch`), zero-shot, no timestamps. Compiled with the authors' recommended forecast config (`normalize_inputs`, `use_continuous_quantile_head`, `force_flip_invariance`, `infer_is_positive`, `fix_quantile_crossing` all `True`), `max_context=1024` (context = the 720 training hours), `max_horizon=168`, `return_backcast=True` (required for covariates). Point forecast = median. With covariates: `forecast_with_covariates` in its default mode `"xreg + timesfm"` — an in-context linear regression on the covariates (standardized with context statistics, ridge 0, pseudo-inverse; constant covariates are harmless), TimesFM forecasts the residual; covariate values for the horizon come from `X_test` (clean or degraded). NoWeather: `forecast()`; with `return_backcast=True` this returns the backcast followed by the forecast, so the forecast is the last `horizon` values. The client checks the context has no NaN (the server finds the horizon rows by NaN target), checks each server reply and deletes its temp files; if the server fails to start or dies, the error includes the server's own error output (its stderr goes to a temporary file, 4 Oct 2026; before it was discarded and the error read `TimesFM server error: ''`)
- ProphetForecaster: Univariate Prophet (`use_covariates=False`). Daily and weekly seasonality on, yearly off (the 30-day training window is far shorter than a year); defaults `seasonality_mode="multiplicative"`, `changepoint_prior_scale=0.05`, `seasonality_prior_scale=10.0`, `holidays_prior_scale=10.0`. Forecast timestamps come from `X_future.index` when available, otherwise an hourly range starting one hour after the last training timestamp.
- NeuralProphetForecaster: NeuralProphet with weather covariates (`use_covariates=True`). Every covariate column is added as a **future regressor** (`add_future_regressor`): the forecast for each step uses that step's covariate values from `X_future`; columns that NeuralProphet silently drops (e.g. all-constant) are removed from the covariate list after fitting. Default `n_lags=24`, `seasonality_mode="multiplicative"`, daily and weekly seasonality on, yearly off, `epochs=None` (NeuralProphet picks based on data size).
- NeuralProphetForecaster_NoWeather: Same as above, univariate (no covariates).

**Model list used for paper results**, built by `run_experiments.build_models(config, keys=None)` — the single place where models are constructed, used by `run_weather_baseline.py`, `run_experiments.main()` and `testing/test_weather_single_model.py`; `keys` selects models from `MODEL_KEYS` and only their params files are read: Seasonal Naive (`seasonal_period`), ARIMA and SARIMAX (orders and `with_intercept` from their params files, translated to a statsmodels `trend`), XGBoost (`n_lags` + `xgb_params`), XGBoost_NoWeather (same keys from its own `xgb_noweather_params_file`), Prophet (`prophet_params`), NeuralProphet (`n_lags` + `neuralprophet_params` from `neuralprophet_params_file`), NeuralProphet_NoWeather (same keys from its own `neuralprophet_noweather_params_file`), TabPFN and TabPFN_NoWeather, TimesFM and TimesFM_NoWeather (default arguments).

**ARIMA / SARIMAX specifics (`statistical.py`):**
- Tuning uses pmdarima, the experiments use statsmodels (`ARIMA`, `SARIMAX`). The two handle the constant differently: pmdarima's `with_intercept` is chosen during the search, statsmodels `SARIMAX` adds no constant unless `trend` is set, and statsmodels `ARIMA` adds one by default only when d = 0.
- Both forecasters therefore take a `trend` argument, and `trend_from_intercept(with_intercept, order, seasonal_order, sarimax)` maps the tuned intercept to it. pmdarima fits every model as statsmodels `SARIMAX(trend="c")` when `with_intercept=True`, whatever d and D are. statsmodels `SARIMAX` puts the trend into the (seasonally) differenced equation; statsmodels `ARIMA` puts trend terms into the levels equation as regressors. Hence:
  - no intercept → `"n"` (passed explicitly, so ARIMA does not add its default constant)
  - SARIMAX (`sarimax=True`), intercept → `"c"` for any d, D (a constant for d + D = 0, a drift for d + D = 1: exactly pmdarima's model)
  - ARIMA, intercept, d + D = 0 → `"c"` (constant)
  - ARIMA, intercept, d + D = 1 → `"t"` (linear trend in levels = drift after differencing)
  - ARIMA, intercept, d + D ≥ 2 → `ValueError`
- ARIMAForecaster: `ARIMA(y, order, trend)` on the raw array, default `fit()`; no covariates.
- SARIMAXForecaster: `SARIMAX(y, exog, order, seasonal_order, trend, enforce_stationarity=False, enforce_invertibility=False)`, fitted with `method='lbfgs'`, `maxiter=1000`; a warning is raised if the optimizer does not converge. Each covariate is divided by its standard deviation in the fold's training window, and the forecast covariates by the same values: the same model with rescaled coefficients, but a better-conditioned optimisation (3 Oct 2026; before, with `maxiter=200` and unscaled covariates, the selected orders did not converge in 85–90 of the 90 tune folds). `y` and `X` get a synthetic hourly `DatetimeIndex` starting 2020-01-01 to silence statsmodels index warnings; the forecast index continues directly after the training index. The real timestamps are not used.
- SARIMAXForecaster drops covariates that are constant in the fold's training window (e.g. season within 30 days, holiday when there is none) for both fit and forecast: their effect cannot be estimated and a constant column duplicates the intercept. Columns that vary in the training window are kept. Same rule as in `tune_sarimax.py` and as NeuralProphet's handling of constant regressors.

**Prophet / NeuralProphet specifics (`prophet_models.py`):**
- All three set `needs_datetime = True`, so `run_experiments.py` attaches a real `DatetimeIndex` (from `config.date_col`) to `X_train`/`X_test`. `fit()`/`predict()` raise `ValueError` without it. This keeps seasonality aligned to real calendar positions.
- **NeuralProphet trains in `predict()`, not `fit()`.** The model uses direct multi-step forecasting with `n_forecasts = horizon`, and the horizon is only known in `predict(horizon, X_future)`. `fit()` therefore only stores the training frame; `predict()` creates, fits and runs the model. Each `predict()` call retrains.
- Forecast extraction: the last `h` rows of the forecast frame are the future steps; step *i* (1-based) is read from column `yhat{i}` in row `-(h - i + 1)`.
- Covariates from `X_future` are written only into the `h` future rows. `predict()` raises `RuntimeError` if the future frame does not contain exactly `h` rows after the last training timestamp.
- Reproducibility: `neuralprophet.set_random_seed(42)` is called immediately before every NeuralProphet model is created.
- Output silencing: NeuralProphet / PyTorch Lightning / cmdstanpy logging is suppressed, and the progress bar is disabled via `fit(..., progress="none")` (with fallbacks for other NeuralProphet versions). Do **not** pass `trainer_config={"enable_progress_bar": False}` — it conflicts with NeuralProphet's own progress-bar callback.
- PyTorch 2.4+ compatibility: `torch.load` is temporarily patched to `weights_only=False` during NeuralProphet fitting (NeuralProphet's checkpoint loader otherwise fails).

**TabPFN Configuration:**
- `TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL, tabpfn_model_config={"model_path": "tabpfn-v2.5-regressor-v2.5_default.ckpt"})` — TabPFN v2.5; all other arguments at the package defaults (runs on the local machine, GPU if available)

### Hyperparameter Tuning (`models/tuning/`)

**Tuning scripts:**
- `tune_arima.py`: Non-seasonal ARIMA (p,d,q)
- `tune_sarimax.py`: Seasonal ARIMA with exogenous variables (p,d,q)×(P,D,Q,s)
- `tune_xgboost.py`: XGBoost with lag features (n_lags + XGBoost hyperparameters); `--scenario no_weather` for XGBoost_NoWeather
- `tune_prophet.py`: Prophet (changepoint and seasonality prior scales + seasonality mode)
- `tune_neuralprophet.py`: NeuralProphet (learning_rate + n_lags)

**Common to all tuning scripts (ARIMA, SARIMAX, XGBoost, Prophet, NeuralProphet):**
- Tuning data is selected with `df[date_col] <= TimeSeriesCV.get_cutoff_date(df)`. `<=` is correct: the cutoff timestamp is the last training hour of evaluation fold 0, not a test hour (test windows are `(test_start, test_end]`, so the held-out test period is `(cutoff, data_end]`). The held-out test period is never touched.
- The tuning period (first and last timestamp of the tuning data, from `TimeSeriesCV.get_tuning_period()`) is saved in every tuning JSON as `tuning_period`; `last_timestamp` equals the cutoff date.
- CV splits are built with `config.tune_horizon`; the tune folds are `TimeSeriesCV.tune_fold_indices(splits)`: the **last** `config.tune_folds` folds (`None` = all folds), without fully imputed test windows — one function for all tuning scripts
- `TimeSeriesCV.spread_fold_indices(tune_folds, n)`: n folds spread evenly over the tune folds, first and last included. Used for the NeuralProphet search (compute) and for the ARIMA/SARIMAX candidate orders; with the default n = 6 these are the same 6 folds
- Run for all cities by default, or one city with `--city {seoul,london,washington}`; results are saved to `--output-dir` (default `results/tuning`)
- The script prints the line to paste into the city config (e.g. `config.prophet_params_file = '...'`)
- Every params JSON contains `provenance`: `{"git_commit": str, "git_dirty": bool, "library_versions": {...}}` (see Code Provenance)
- Output folder checked first: each script calls `provenance.check_output_dir(--output-dir)` before any tuning work and stops with `PermissionError` if no file can be written there (4 Oct 2026; on 3 Oct a read-only folder was found only when the results were saved, after hours of tuning, and the params files were lost)
- The complete params JSON (provenance included) is printed to the log between `----- BEGIN PARAMS JSON <file name> -----` and `----- END PARAMS JSON <file name> -----` before the file is written (`provenance.save_params_json`), so the file can be restored exactly from the job log
- Scoring excludes imputed hours, as in the experiments: `calculate_all(..., test_mask, train_mask)`; folds whose test window is fully imputed are removed from the search/validation fold lists
- The tune folds are all the data there is before the cutoff (Seoul: exactly 90 folds). Re-evaluation numbers in the tuning JSONs (`mae_mean` etc.) are computed on the search folds, so they are in-sample and are not out-of-sample results; the out-of-sample test is the evaluation period. No runner reads them.

**Common to the Prophet and NeuralProphet tuning scripts:**
- Random search optimizing mean MAE across folds. Prophet then re-evaluates the best parameters fold by fold for the reported `mae_mean`/`mae_std`/`rmse_mean`/`rmse_std`; NeuralProphet has no re-evaluation (too slow)
- Parameter sampling uses `np.random.default_rng(seed)` (`--seed`, default 42)
- A failed fold aborts that trial (traceback printed) and the search continues

**ARIMA/SARIMAX approach** (`arima_search.search_orders()`, shared by `tune_arima.py` and `tune_sarimax.py`):
- 1. Candidates: pmdarima `auto_arima` (stepwise, AIC; `d`/`D` from its unit-root tests; `with_intercept` at pmdarima's default `'auto'`) on the training window of each of the `--search-folds` candidate folds (default 6, spread evenly over the tune folds). Every distinct `(order, seasonal_order, with_intercept)` is a candidate
- 2. Selection: every candidate is fitted with the experiment model (`ARIMAForecaster` / `SARIMAXForecaster`, statsmodels, `trend` from `trend_from_intercept()`) on **all** tune folds (the same 90 folds as XGBoost and Prophet), with the experiment inputs (`run_experiments.prepare_fold_inputs()`), and scored on the next `tune_horizon` hours (MAE, imputed hours excluded). The candidate with the lowest mean MAE is selected — the same criterion and folds as XGBoost and Prophet. At most 6 candidates × 90 folds = 540 fits per city. A candidate without a statsmodels trend equivalent, or failing on a tune fold, is not selectable (reason saved)
- ARIMA: non-seasonal, `max_p=7`, `max_q=3`, `max_order=8`, no covariates
- SARIMAX: seasonal with `m = --seasonal-period` (default 24), `max_p=5`, `max_q=3`, `max_P=2`, `max_Q=2`, `max_order=8`; covariates = the experiment's columns for `--scenario` (`WeatherProcessor.get_weather_columns`; `clean_only` default, or `all_weather`); covariates constant in the training window are dropped for the `auto_arima` search (same rule as `SARIMAXForecaster`, which applies it itself when scoring)
- Non-converged `SARIMAXForecaster` fits are counted per candidate (`non_converged_folds`)
- No re-evaluation after the search (the selection scores already come from the experiment model)
- `tune_arima.py` has no `--scenario` argument: ARIMA uses no covariates, so one params file serves all scenarios (same as Prophet)

**ARIMA/SARIMAX Parameters:**
- **p/P**: Autoregressive order (past values)
- **d/D**: Differencing order (trend removal)
- **q/Q**: Moving average order (error correction)
- **s**: Seasonal period (24 for hourly data)
- **with_intercept**: Whether the model includes a constant (drift if differenced)

**XGBoost approach** (`tune_xgboost.py`):
- Optuna TPE search (multivariate, seeded with `--seed`, default 42), default 100 trials (`--trials`), optimizing mean MAE across folds
- Jointly tunes `n_lags` (options `--n-lags-options`, default [12, 24, 48, 168]) and the XGBoost hyperparameters below
- `n_estimators` is not searched: each fold is fitted with early stopping (50 rounds, cap 3,000 trees) on the last 15 % (min 24 rows) of the fold's lagged training rows; the final `n_estimators` is the mean best iteration of the best trial across folds
- Features: lags + covariates chosen by `--scenario` (`clean_only` default: keys of `weather_degradation_mapping` + holiday + season; `all_weather`: all of `config.weather_covariates`; `no_weather`: no covariates, produces the params file for XGBoost_NoWeather (`config.xgb_noweather_params_file`)) + calendar features (hour, dayofweek, month, is_weekend); training rows capped at `--max-train-size` (default 8,000, no effect with 720)
- Forecasts are recursive (predictions fed back as lags), as in `XGBoostForecaster`
- A failed fold prunes the trial
- Features are built with `features.prepare_xgboost_features()`, the same builder `run_experiments.py` uses
- Validation: the best parameters are refitted with the fixed `n_estimators` (no early stopping) on the same `tune_folds` folds as the search

**XGBoost Parameters:**
- **n_lags**: Number of lagged target values as features
- **n_estimators**: Number of boosting trees (from early stopping, see above)
- **learning_rate**: Step size shrinkage, log-uniform in [0.005, 0.2]
- **max_depth**: Maximum tree depth, integer in [2, 12]
- **min_child_weight**: Minimum sum of instance weight in child, uniform in [1, 80]
- **subsample**: Fraction of samples for tree training, uniform in [0.6, 1.0]
- **colsample_bytree**: Fraction of features for tree training, uniform in [0.6, 1.0]
- **gamma**: Minimum loss reduction for split, uniform in [0, 20]
- **reg_lambda**: L2 regularization, log-uniform in [1e-3, 50]
- **reg_alpha**: L1 regularization, log-uniform in [1e-10, 5]

**Prophet approach** (`tune_prophet.py`):
- Random search, default 50 trials (`--trials`)
- Daily and weekly seasonality on, yearly off (same as `ProphetForecaster`)
- Tunes `changepoint_prior_scale`, `seasonality_prior_scale` and `seasonality_mode`; `holidays_prior_scale` is not tuned (the model has no holidays, so it has no effect) and stays at the forecaster default
- Validation re-uses the same `tune_folds` folds as the search

**Prophet Parameters:**
- **changepoint_prior_scale**: log-uniform in [0.001, 0.5] — trend flexibility
- **seasonality_prior_scale**: log-uniform in [0.01, 10] — seasonality strength
- **seasonality_mode**: `multiplicative` or `additive`

**NeuralProphet approach** (`tune_neuralprophet.py`):
- Random search, default 20 trials (`--trials`; NeuralProphet is slow)
- Jointly tunes `learning_rate` and `n_lags`; `n_lags` options default to [12, 24, 48, 168] (`--n-lags-options`)
- Model: `n_forecasts = config.tune_horizon` (direct multi-step, same as the forecaster), multiplicative seasonality, daily and weekly seasonality on, yearly off, `epochs=None`
- Covariates chosen by `--scenario`, added as future regressors (same as the forecaster):
  - `clean_only` (default): keys of `weather_degradation_mapping` + holiday + season
  - `all_weather`: all of `config.weather_covariates`
  - `no_weather`: no covariates; produces the params file for NeuralProphet_NoWeather (`config.neuralprophet_noweather_params_file`)
- `neuralprophet.set_random_seed(42)` before every model is created
- Search on `--search-folds` folds (default 6), spread evenly over the `tune_folds` folds (first and last included), so the search covers the whole tuning period; the chosen fold indices are saved as `search_fold_indices`
- No re-evaluation after the search: NeuralProphet is too slow, and nothing uses those numbers. Cost of one run = trials × search folds model fits (default 20 × 6 = 120)

**NeuralProphet Parameters:**
- **learning_rate**: log-uniform in [1e-4, 0.1]
- **n_lags**: Number of past target values fed to the autoregressive part

**Note on horizons:** NeuralProphet is tuned at a single horizon (`config.tune_horizon`), but at evaluation time `n_forecasts` equals each experiment horizon. The tuned `learning_rate`/`n_lags` are therefore reused for horizons they were not tuned on.

**Output format (ARIMA/SARIMAX)** — `arima_best_params_{city}_{n_train_samples}_{timestamp}.json` / `sarimax_best_params_{city}_{scenario}_{n_train_samples}_{timestamp}.json`:
```json
{
  "city": str,
  "scenario": "clean_only" | "all_weather",
  "n_train_samples": int,
  "tuning_period": {"first_timestamp": str, "last_timestamp": str},
  "order": [p, d, q],
  "seasonal_order": [P, D, Q, s],
  "with_intercept": bool,
  "trend": "n" | "c" | "t",
  "covariates_used": [str],
  "m": int,
  "tuning": {
    "search_type": str,
    "tune_folds": int,
    "scoring_folds": int,
    "candidate_folds": int,
    "candidate_fold_indices": [int],
    "metric_optimized": "MAE",
    "best_tune_mae_mean": float,
    "best_tune_rmse_mean": float,
    "auto_arima_kwargs": {...},
    "candidates": [{"order", "seasonal_order", "with_intercept", "found_on_folds", "aic_on_found_folds",
                    "status", "trend", "fold_mae", "mae_mean", "rmse_mean", "non_converged_folds"}]
  },
  "provenance": {"git_commit": str, "git_dirty": bool, "library_versions": {...}}
}
```
`scenario`, `seasonal_order`, `covariates_used` and `m` are SARIMAX only. `order`, `seasonal_order` and `with_intercept` are read by `run_experiments.build_models()`; `trend` is saved for reference (the runner derives it again with `trend_from_intercept()`). ARIMA params files from before the `--scenario` argument was removed contain a `scenario` key and have it in the file name; they remain valid (the key is not read).

**Output format (XGBoost)** — `xgboost_best_params_{city}_{scenario}_{n_train_samples}_{timestamp}.json`:
```json
{
  "city": str,
  "scenario": "clean_only" | "all_weather" | "no_weather",
  "n_train_samples": int,
  "tuning_period": {"first_timestamp": str, "last_timestamp": str},
  "n_lags": int,
  "xgb_params": {
    "objective": "reg:squarederror",
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
    "n_estimators": int,
    "learning_rate": float,
    "max_depth": int,
    "min_child_weight": float,
    "subsample": float,
    "colsample_bytree": float,
    "gamma": float,
    "reg_lambda": float,
    "reg_alpha": float
  },
  "tuning": {
    "search_type": "optuna_tpe_multivariate_with_early_stopping",
    "trials": int,
    "tune_folds": int,
    "seed": int,
    "metric_optimized": "MAE",
    "best_tune_mae_mean": float,
    "n_lags_options": [int],
    "early_stopping_rounds": int,
    "max_estimators_cap": int
  },
  "mae_mean": float,
  "mae_std": float,
  "rmse_mean": float,
  "rmse_std": float,
  "validation_folds": int,
  "max_train_size": int,
  "covariates_used": [str],
  "time_features_used": ["hour", "dayofweek", "month", "is_weekend"]
}
```

**Output format (Prophet)** — `prophet_best_params_{city}_{n_train_samples}_{timestamp}.json`:
```json
{
  "city": str,
  "n_train_samples": int,
  "tuning_period": {"first_timestamp": str, "last_timestamp": str},
  "prophet_params": {
    "changepoint_prior_scale": float,
    "seasonality_prior_scale": float,
    "seasonality_mode": str
  },
  "tuning": {
    "search_type": "random_search",
    "trials": int,
    "tune_folds": int,
    "seed": int,
    "metric_optimized": "MAE",
    "best_tune_mae_mean": float,
    "best_tune_rmse_mean": float
  },
  "mae_mean": float,
  "mae_std": float,
  "rmse_mean": float,
  "rmse_std": float,
  "validation_folds": int
}
```

**Output format (NeuralProphet)** — `neuralprophet_best_params_{city}_{scenario}_{n_train_samples}_{timestamp}.json`:
```json
{
  "city": str,
  "scenario": "clean_only" | "all_weather" | "no_weather",
  "n_train_samples": int,
  "tuning_period": {"first_timestamp": str, "last_timestamp": str},
  "n_lags": int,
  "neuralprophet_params": { "learning_rate": float },
  "tuning": {
    "search_type": "random_search",
    "trials": int,
    "search_folds": int,
    "search_fold_indices": [int],
    "tune_folds": int,
    "seed": int,
    "np_seed": 42,
    "metric_optimized": "MAE",
    "best_tune_mae_mean": float,
    "best_tune_rmse_mean": float,
    "n_lags_options": [int]
  },
  "covariates_used": [str]
}
```

**Tuning frequency:** Once per dataset. Parameters describe data structure, not forecast length.

### Cross-Validation (`evaluation/cv.py`)
**TimeSeriesCV**:
- Rolling window split — fixed-size training window advances by `horizon` hours with each fold
- First fold date = `get_cutoff_date()`; folds of `horizon` hours follow until the end of the data
- `partial_last_fold` (default `False`): if the evaluation period is not a multiple of `horizon`, the last fold keeps its regular origin and its test window is truncated to the remaining hours instead of being dropped. `run_experiments.py` uses `True`; the tuning scripts use the default (with `tune_horizon=24` and 5,880 h there is no partial fold anyway)
- `get_eval_hours()`: `n_folds * max(horizons)`; `expected_n_folds(horizon)`: `ceil(eval_hours / horizon)`
- Tracks imputed data per kept fold (train_imputed, test_imputed); entry `fold` = index in the returned splits, so `get_split_info()`/`get_imputation_summary()` line up with the splits (previously skipped folds were counted too, which shifted the entries in tuning)
- Ensures minimum training size and exact test size (`horizon` hours, or the remaining hours for a partial last fold); folds that fail this check are skipped (needed for tuning, where the earliest folds start before the data). A fold whose windows lie fully inside the data but fails the check (missing hourly timestamps) is also skipped, and a `WARNING` line is always printed for it
- `get_cutoff_date(df)`: `data_end - evaluation period`. The cutoff is the last training hour of fold 0; the held-out test period is `(cutoff, data_end]`. Tuning scripts use data up to and including this timestamp (`<= cutoff`)
- `get_tuning_period(tune_df)`: `{"first_timestamp", "last_timestamp"}` (ISO strings) of the tuning data, saved in the tuning JSONs

### Metrics (`evaluation/metrics.py`)
**MetricsCalculator**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MASE: Mean Absolute Scaled Error (seasonal naive baseline)
- sMAPE: Symmetric Mean Absolute Percentage Error, range [0, 200]: a step with y = 0 and ŷ ≠ 0 contributes 200, a step with y = ŷ = 0 contributes 0
- `calculate_all` raises `ValueError` if `y_pred` has a different length than `y_true` or contains NaN/inf (NaN metrics are reserved for fully imputed test windows, which are skipped in the means; a broken forecast must not be skipped silently)
- Imputed values are excluded from scoring: `calculate_all(y_true, y_pred, y_train, test_mask=None, train_mask=None)` scores only test steps with `test_mask=True`; MASE scaling uses only seasonal-naive pairs `(t, t-24)` where both training values are observed (`train_mask`). If no test step is observed, all four metrics are NaN. Without masks, behaviour is unchanged
- `observed_mask(df, functioning_day_col)`: boolean array, `True` = observed, `False` = imputed (`== 'No'`); all `True` if the column is not set/present
- `compute_and_save_comparative_metrics(...)`: comparative metrics (win rate, skill score, with CIs) computed from the aggregated results file — called via `run_experiments.compute_and_log_comparative_metrics()`. Raises `ValueError` if the file has more than one row per (dataset, horizon, weather_scenario, model), or if the baseline's error differs between scenarios for the same (dataset, horizon)

### Multi-City Orchestrator (`main.py`)
Runs all datasets sequentially without manual intervention.
- Imports each city config via `get_config()` and passes it to `run_weather_baseline.main()`
- Bypasses the interactive confirmation prompt (`no_confirm=True`)
- Catches per-city failures, writes full traceback to `errors_{version}.log` (path taken from the first selected city's config), and continues to the next city
- After all cities: computes comparative metrics only if all selected cities succeeded, pooled across cities (`compute_and_log_comparative_metrics(first_config, log_wandb=False)`), and prints `model`, `weather_scenario`, `n_tasks`, `win_rate`, `skill_score`
- Prints timestamped `STARTING`, `[OK]`, and `[FAILED]` lines to stdout; prints a pass/fail summary and total wall time at the end
- Exits with code 1 if any city failed
- Accepts `--cities` flag to run a subset (e.g. `python forecasting/main.py --cities seoul london`); default order: seoul, washington, london

### Experiment Runner (`run_experiments.py`)
**ForecastingExperiment**:
- Manages experiment lifecycle
- W&B initialization (`entity` from the `WANDB_ENTITY` environment variable, loaded from `.env`) and logging
- Checkpoint save/load for recovery — checkpoint keys include `dataset_name` so Seoul and Washington runs never conflict
- Code provenance: `git_commit`/`git_dirty` are written to every fold row, every aggregated row, the checkpoint and the W&B config; `library_versions` (JSON string) to every aggregated row and the W&B config. A checkpoint without `code_version` or a `results_master_{version}.csv` without `git_commit` (written by older code) raises `RuntimeError`; resuming from another commit prints a warning (see Code Provenance)
- Runs all model-horizon-scenario combinations
- Saves aggregated and detailed results and the hourly forecasts; on resume, reloads previously completed results from `results_master_{version}.csv`
- `save_results()` merges with the existing aggregated CSV and drops duplicates on (`dataset`, `model`, `horizon`, `weather_scenario`, `run_name`), keeping the latest; the detailed CSV is not touched (already written after every experiment)
- Automatic skip logic: models with `use_covariates=False` skip every degraded scenario (`config.is_degraded`)
- Coverage check: `run_single_experiment()` raises `RuntimeError` unless the splits number `expected_n_folds(horizon)` and their test windows add up to exactly `get_eval_hours()` hours (catches missing hourly timestamps), so no evaluation hours are dropped for any horizon
- Fold-level errors are logged to `errors_{version}.log` with full traceback and to W&B (`error`); `[ERROR]` is always printed regardless of `verbose`. This includes errors in the fold's input preparation (`prepare_fold_inputs()`: weather degradation, calendar features), which runs inside the same `try` (4 Oct 2026; before, its traceback was lost and only the message was printed). A fold error aborts that entire model-horizon-scenario run immediately. `run_all_experiments()` catches the exception, records the failed experiment, leaves it unsaved and uncheckpointed, and continues with the remaining combinations. After all combinations have been attempted, it raises `RuntimeError` if any experiment failed, so `main.py` marks that city failed while preserving all successful completed work
- Runtime: `fit()` and `predict()` are timed per fold with `time.perf_counter()` (`fit_time_s`, `predict_time_s`, `runtime_s`) and aggregated per model, horizon and scenario (see Results Schema)

- `run_all_experiments(models, df, scenarios=None)`: `scenarios=None` uses `config.weather_scenarios`; failed model-horizon-scenario combinations are collected in `failed_experiments` and reported only after all remaining combinations have been attempted
- The module sets `warnings.filterwarnings('ignore')` globally. Per fold, the warnings raised in `fit()`/`predict()` are recorded and those containing "converge" are counted (`convergence_warnings`); none is printed

**`main(config=None)`**: run directly with `python forecasting/run_experiments.py --city {seoul,washington,london}` (`--city` is required when no config is passed). Builds the models with `build_models(config)` (same as `run_weather_baseline.py`; ARIMA/SARIMAX `trend` from `with_intercept` via `trend_from_intercept()`, ARIMA with `seasonal_order=(0, 0, 0, 0)`) but runs all of `config.weather_scenarios` (including `all_weather`) under the experiment name `baseline_models_{version}`.

**`load_and_prepare_data(config)`**: see Data Flow step 1. Returns `(df, dataset_name)`, where `dataset_name` is the CSV file stem. Note: it appends holiday/season to `config.weather_covariates` in place (only if not already listed).

**`compute_and_log_comparative_metrics(config, log_wandb=True)`**: tasks are (`dataset`, `horizon`, `weather_scenario`), pooled across all datasets in `results_master_{version}.csv`; baseline model `Seasonal_Naive`. One comparison per (model, `weather_scenario`), over that scenario's tasks only: optional sensitivity scenarios get their own rows and do not enter the clean_only/degraded comparisons. The baseline error is the clean_only error for every scenario (the baseline ignores covariates). Run once after all cities finish.

### Experiment Runner (`run_weather_baseline.py`)
**`main(config=None, no_confirm=False)`**: Main runner used for paper results
- Requires a city config (from `get_config()` in `config_<city>.py`), passed in by `main.py` or by the `--city` path; raises `ValueError` if `config` is `None`
- Run directly: `python forecasting/run_weather_baseline.py --city {seoul,washington,london}` (`--city` is required)
- `no_confirm=True` skips the interactive prompt for non-interactive/cluster use (direct runs ask for confirmation)
- Runs clean_only, then every scenario in `config.degradation_scales` (default: degraded only), for all models; `all_weather` is not run
- All horizons: [6, 24, 48, 168] hours over the same 5,880 h evaluation period (980 / 245 / 123 / 35 folds; the last 48 h fold is partial)
- Builds all models with `run_experiments.build_models(config)`, which loads the tuned hyperparameters from the JSON files named in the city config (ARIMA, SARIMAX, XGBoost, XGBoost_NoWeather, Prophet, NeuralProphet, NeuralProphet_NoWeather). A params file without `provenance` (written by older tuning code) raises `ValueError`; one tuned with uncommitted code prints a warning
- ARIMA/SARIMAX: `trend` is derived from `with_intercept` and the orders via `trend_from_intercept()`; a params file without `with_intercept` raises `KeyError` (re-run tuning)
- Auto-skips the degraded scenarios for models without covariates
- Displays degradation impact summary (gated behind `config.verbose`)
- Uses ForecastingExperiment class for W&B logging, checkpointing, and result saving
- Errors written to `errors_{version}.log` with full traceback before re-raising


## Key Design Decisions

### Weather Scenario Optimization
Models without covariates (`use_covariates=False`, e.g. Seasonal Naive, ARIMA, Prophet, XGBoost_NoWeather, NeuralProphet_NoWeather, TabPFN_NoWeather) automatically skip the degraded scenarios ('degraded' and any optional sensitivity scenario) since they ignore weather data. This avoids redundant computation.

Rationale: every degraded scenario = clean_only for models that don't use weather covariates.

### NeuralProphet Direct Multi-Step Forecasting
NeuralProphet is built with `n_forecasts = horizon` (in the forecasters and in tuning) and trained inside `predict()`.

Rationale:
- In neuralprophet 0.8.0 (`data/split.py`, `_make_future_dataframe`; the experiments use 0.9.0, guarded by the row-count check below), when `n_lags > 0` the `periods` argument of `make_future_dataframe` is overwritten with `n_forecasts`. With `n_forecasts=1`, only one future row was created, so scoring the last `horizon` rows mixed 1 real forecast with `horizon - 1` in-sample fitted values, and the covariate fill overwrote training-row covariates with test-period values.
- With `n_forecasts = horizon`, the future frame has exactly `horizon` rows; a row-count check raises an error if this assumption ever breaks (e.g. after a NeuralProphet upgrade).
- Results produced before this fix are invalid and need to be re-run.

### ARIMA/SARIMAX Intercept Consistency
The intercept selected by pmdarima during tuning is saved (`with_intercept`) and reproduced in statsmodels via `trend`.

Rationale:
- Previously the flag was not saved and `SARIMAXForecaster` passed no `trend`, so a SARIMAX tuned with an intercept was evaluated without one; statsmodels `ARIMA` added a constant only when d = 0, regardless of the tuning result.
- `tune_sarimax.py` previously passed covariates as `exogenous=`, which recent pmdarima versions do not accept as the covariate argument; it now uses `X=`.
- ARIMA/SARIMAX params files and results produced before this fix are invalid and need to be re-run.
- `trend_from_intercept()` first mapped an intercept with d + D = 1 to `"t"` for both models. For `SARIMAX`, `"t"` is a linear trend in the differenced series, i.e. a quadratic trend in levels, not pmdarima's drift. SARIMAX now gets `"c"` (`sarimax=True`); ARIMA keeps `"t"`. SARIMAX params files and results where the selected candidate had an intercept and d + D = 1 are invalid and need to be re-run.

### TimesFM Forecast Extraction and Configuration
The univariate TimesFM forecast is the last `horizon` values of `forecast()`, and the model uses the authors' recommended TimesFM 2.5 forecast config.

Rationale:
- The server compiles with `return_backcast=True` (needed for covariates). In timesfm 2.5, `forecast()` then returns the in-sample backcast (992 values for `max_context=1024`) followed by the forecast. The univariate branch took the first `horizon` values, i.e. backcast values, not the forecast. `forecast_with_covariates` was not affected (it slices the forecast part itself).
- The server used the library defaults (`normalize_inputs=False`, `use_continuous_quantile_head=False`, `fix_quantile_crossing=False`) instead of the configuration in the TimesFM 2.5 README. For the point forecast (median) only `normalize_inputs` matters; the other two affect the quantiles.
- TimesFM_NoWeather results produced before this fix are invalid; all TimesFM results need to be re-run.

### Measured NWP Error Model per City
The degraded scenarios use forecast errors measured for each city from real ECMWF IFS HRES forecasts (default `degradation_model = "nwp_measured"`), instead of one set of published error sizes for all cities.

Rationale:
- The published values did not match these cities: measured temperature error SD at 168 h is 2.5–3.3 °C (literature 3.8 °C), humidity 6–11 % at 24 h (13.6 %), wind 0.7–1.0 m/s at 24 h (2.0 m/s); visibility CV 119–137 % below the cap (25 %), precipitation amount CV of hits 140–342 % at 6–24 h (31–34 %), miss rate 0.58–0.71 and false-alarm ratio 0.65–0.82 at 168 h (both capped at 0.5)
- Errors are measured against the data each city's model is trained on (Seoul: the same station; London/Washington: ERA5, the source of their covariates), so the degraded covariates differ from the clean ones as a real forecast would differ from that data
- Default setting (revised 1 Oct 2026): fresh forecast and bias removed, i.e. an operator with an up-to-date, locally corrected forecast. The first version used the run available at the issue time (6–17 h old) and kept the biases. That gave (a) a forecast age fixed per city by its time zone (Seoul 14 h, London 10–11 h, Washington 15–16 h for horizons ≥ 24 h), so some hours of the day always had older forecasts and the cities differed for a reason unrelated to weather, and (b) a bias that appears only where the reference is a station (Seoul −1.5 °C; Washington's forecast is also about 1.5 °C too cold against its airport station, but its reference is ERA5). A real operator would use a regularly updated forecast corrected to local measurements. Both settings remain available
- Rain (2 Oct 2026): error rates per time of year (Washington's forecast misses 50 % of wet hours in July against 30 % in January; Seoul's monsoon), and Seoul rain measured against the station instead of ERA5. ERA5 had about twice the station's wet hours and fewer than half its downpours, so Seoul's rain errors were too small (amount error spread at 24 h: CV about 3.8 against the station vs 1.9 against ERA5 in the 2 Oct calibration; 3.4 against the station since the Seoul winter split of 3 Oct)
- Replaying whole runs gives the real persistence from hour to hour and the real links between temperature, humidity and wind; independent hourly noise (literature model) understated both (Known Limitations 1 and 15)
- All degraded results produced with the literature model must be re-run; the literature model stays available (`degradation_model = "literature"`) for comparison and to reproduce earlier results
- Forecast years (2024–26) differ from the data years (2011–18); forecasts were less accurate then (Known Limitations)

### Degradation Parameters from the Training Fold
The solar cap used by the degradation is computed from the clean training fold, not from the test window being degraded.

Rationale:
- Previously the cap was the 99.5th percentile of the test window itself; for short horizons this is about the window's own maximum, which clipped upward noise and biased degraded solar radiation downward.
- Results produced before this fix are invalid and need to be re-run.

### Precipitation False Alarms from the False Alarm Ratio
The false-alarm probability for a dry hour is derived from Sukovich et al.'s false alarm ratio (FAR) with the wet-hour share *p* of the training fold: P(false alarm | dry) = FAR / (1 − FAR) · POD · p / (1 − p). Precipitation is one variable with one event-detection draw per hour (since 2 Oct 2026 one total-precipitation column per city).

Rationale:
- FAR = false alarms / forecast events is conditional on a forecast event. It was used directly as the probability that a dry hour receives forecast precipitation, and false alarms were drawn separately for the rain and the snow column (both mapped to 'precipitation'). A dry test window then got forecast precipitation in 58% (6 h) to 68% (168 h) of its hours; the simulated FAR was about 0.92 instead of the cited 0.35-0.50.
- With the conversion, the simulated forecasts reproduce the cited FAR and POD (unit test `test_simulated_forecasts_reproduce_source_far_and_pod`); for 6% wet hours, P(false alarm | dry) is 1.6-3.2% (1 h to ≥60 h).
- Degraded results produced before this fix are invalid and need to be re-run.

### Precipitation Miss Rate by Intensity
Since 2 Oct 2026 the measured NWP model uses two precipitation miss-rate classes: light wet hours <1 mm/h and stronger wet hours ≥1 mm/h (`nwp_rain_intensity_dependent=True`; `False`: one miss-rate curve for all intensities).

Rationale:
- Calibration against the current truth sources shows a clear difference at 24 h: Seoul 0.281 / 0.155, London 0.433 / 0.111, Washington 0.481 / 0.224 (light / stronger, year-round). Light rain is missed more often in every month and lead time 1–168 h except London beyond 120 h in four months.
- A three-class split with a ≥5 mm/h class was rejected: 113 (Seoul), 1 (London) and 46 (Washington) observed hours ≥5 mm/h in 2024–26, below 200 pooled cases in every month even with a whole-year window; above 1 mm/h the miss rate barely differs (24 h, 1–5 / ≥5 mm/h: Seoul 0.16 / 0.14, Washington 0.22 / 0.23). The two-class split satisfies the existing minimum of 200 observed wet cases in every pooled lead window after seasonal widening.
- Rates keep the existing ±12 h lead pooling and monthly interpolation. Seasonal windows widen separately for each class; FAR and hit amount error are not split by intensity.
- Seoul Nov–Mar: see "Seoul winter precipitation in the calibration" below. Elsewhere the event threshold remains ≥0.1 mm/h.
- With `nwp_rain_frequency_unbiased=True`, the false-alarm probability uses the training fold's light/strong wet-hour mix and the corresponding two miss rates, so false alarms balance combined misses in expectation without using the test window's intensity distribution.

### Seoul Winter Precipitation in the Calibration
The Seoul station reports precipitation as 3-hour totals from November to March (00, 03, …, 21 h KST), as in the bike data. Since 3 Oct 2026 the calibration treats station and forecast alike (`weather/nwp/build_nwp_calibration.py`, `weather/nwp/analyze_nwp_errors.py`).

- Station format checked on the SYNOP reports used (Nov–Mar 2024–25; `weather/nwp/check_seoul_winter_split.py`): all 82 positive amounts sit at 00, 03, …, 21 KST, none at other hours; 345 of the 441 reports with precipitation in the present-weather group but no amount are at the other hours; the amounts add up to the station's 24-hour totals (median ratio 1.00, 72 totals).
- Station: each winter total is split evenly over its three hours (as `precipitation_mm`). Forecast: the run's three hourly values in the same window are summed and split evenly. Wet: 3-hour total ≥0.1 mm, for both. A window counts as winter if it ends in Nov–Mar, Korean time (the bike-data rule in `data/build_weather_v7.py`).
- Why the forecast is split too: the split station values carry no timing within the 3 hours, and the demand model is trained on split values, so the forecast is compared at the same 3-hour resolution. Test on Apr–Oct (truly hourly station) turned into split 3-hour totals, miss rate at 24 h (all / light / stronger): true hourly 0.22 / 0.27 / 0.18; split station vs hourly forecast 0.30 / 0.37 / 0.21; split station vs split forecast 0.18 / 0.24 / 0.11. Timing errors within the 3 hours are not counted (Limitation 21).
- Effect on the calibration files: London and Washington unchanged. Seoul precipitation only, e.g. miss rate at 24 h: year-round 0.254 (rain-intensity zip of 2 Oct, station split, forecast hourly with wet = >0 in winter) → 0.226, January 0.381 → 0.278; the 2 Oct production files (before the station split) had 0.228 and 0.22. The winter forecast rule of the zip also counted Open-Meteo's interpolated values below 0.1 mm (beyond 90 h only) as rain.
- The month of the station split was taken in UTC before (1 Nov 00–08 KST not split, 1 Apr 00–08 KST split); no effect on the current files (those 17 station hours were dry).

### Float Dtype for Degraded Columns
`degrade_weather_dataset()` stores every degraded column as float.

Rationale:
- `degrade_weather_forecast()` returns the int `0` for dry hours (precipitation) and night hours (solar). In a dry test window a precipitation column therefore came out as int64; the rain/snow correction (removed on 2 Oct 2026) then moved float amounts into it, which pandas >= 3.0 rejects with `TypeError` (the fold fails and is skipped) and pandas 2.x accepts with a deprecation warning. Values are unchanged.

### Precipitation and Snow: One Definition for All Cities
Since 2 Oct 2026 every city has two covariates with the same meaning: total precipitation incl. melted snow (mm/h; `precipitation_mm`) and snow depth on the ground (cm; `snow_depth_cm`), with the same column names in every city (Seoul: built from the original UCI columns `Rainfall` and `Snowfall`, which stay in the file). In the degraded scenarios precipitation gets the measured precipitation errors and snow depth the persistence forecast (value of the last training hour = issue time). The rain/snow phase correction was removed (it had been made config-driven for all cities earlier on 2 Oct 2026).

Rationale:
- The columns measured different things. Seoul `Snowfall` is the KMA ASOS snow depth (적설): e.g. it reaches 8.8 cm at 10:00 on 24 Nov 2018 (the depth KMA reported) and melts over two dry days up to 9.5 °C; identical values for up to 24 h at −10 °C. Seoul `Rainfall` is KMA precipitation (강수량), which includes melted snow. London/Washington had liquid rain only (`rainfall_mm` = Open-Meteo `rain`) and snow falling per hour (`snowfall_cm` = Open-Meteo `snowfall`). One covariate name stood for different quantities across cities.
- The phase correction treated Seoul's snow depth as falling snow: above 2 °C (degraded temperature) it moved the depth into `Rainfall`, i.e. 23–43 evaluation hours per horizon of rain that never fell (21–47 mm); the precipitation draw also zeroed or rescaled the depth and counted snow-covered hours as wet (wet-hour share 0.085 instead of 0.027 in the late-November training folds). In London/Washington it moved amounts between mm and cm without conversion and re-assigned observed phases (Washington: 4–19 clean hours per horizon).
- The precipitation errors were measured on total precipitation (Seoul: station amounts; London/Washington: ERA5 `precipitation`), which is now the degraded column in every city.
- Snow depth is kept, not dropped: snow on the ground affects cycling for days after the snowfall (Seoul 24–26 Nov 2018: snow fell for about 4 h and lay for 51 h). No snow-depth forecast errors are available, and persistence is the forecast an operator has without a snow model.
- The covariates changed in all three cities (Seoul: see "Seoul Winter Precipitation Spread over Three Hours"), so SARIMAX, XGBoost and NeuralProphet (`clean_only`) are re-tuned for all three. All results must come from this version.

### London/Washington Weather on the Local Clock
The Open-Meteo weather columns of London and Washington were re-aligned to the local clock with daylight saving (2 Oct 2026, data files; visibility unchanged).

Rationale:
- They had been stored on standard time all year (London UTC+0, Washington UTC−5), while the bike counts and visibility (Visual Crossing) follow the clock. During daylight saving (London 58 %, Washington 65 % of all hours; 73 % and 76 % of the evaluation hours) each row held the weather of the following hour. Evidence: temperature matched Open-Meteo best at the standard-time offset in every month; solar radiation peaked at the same clock hour in winter and summer; the commuter peaks of the counts stay at 08:00 and 17:00 all year.
- After the re-alignment the temperature differs from Open-Meteo at the row's true UTC hour by 0.22–0.38 °C (monthly means) in every month (before: up to 0.8 °C in London and 1.0 °C in Washington in summer), and the solar peak moves one clock hour later during daylight saving.
- Values were moved, not changed; outside daylight saving nothing changes.

### Seoul Winter Precipitation Spread over Three Hours
KMA reports Seoul precipitation as 3-hour totals from November to March (at 00, 03, …, 21 h; hourly from April to October). In `precipitation_mm` each of these totals is spread evenly over its three hours (the value at hour t covers the hours labelled t − 2, t − 1 and t); April–October values are unchanged, and the totals are kept (2 Oct 2026).

Rationale:
- In the 3-hour format the amount of three hours sits in one hour and the two hours before show 0. Seoul's whole tuning period (Dec 2017 – Mar 2018) is in this format, and folds whose training window spans March/April or October/November mix the two formats.
- Test on the hourly April–October data aggregated the same way: the 3-hour format keeps a correlation of 0.43 with the true hourly amounts and finds 34 % of the wet hours; the even split raises the correlation to 0.79 and recovers 100 % of the true wet hours.
- Cost: the split marks too many hours as wet (684 instead of 447 in that test; in 40 % of wet 3-hour blocks only one hour was wet) and spreads short showers at a third of their intensity. Seoul's wet hours rise from 528 to 690 (tuning period 64 → 192).
- The precipitation errors of the degradation are hourly; with the split the clean series is hourly too.

### Constant Covariates in SARIMAX
Covariates that are constant in a fold's training window are dropped for that fold (tuning and experiments).

Rationale:
- With a 720 h training window, season is usually constant and holiday often all zeros. A constant column duplicates the intercept and its coefficient cannot be estimated, so it would be arbitrary and would still be applied to the test values.
- In folds where season or holiday varies in the training window, the column is kept.

### On-the-Fly Degradation
Degradation applied fresh for each CV fold during data preparation, not pre-computed.

Rationale:
- Prevents data leakage between folds
- Different realistic errors per fold
- Saves disk space (no large degraded datasets)
- Computationally cheap (just percentile calculations + random sampling)

### Reproducible Error Simulation
Seed hierarchy: `horizon_seed = base_seed + 10000 * horizon + fold_idx` (+ `city_seed_term(dataset_name)` in the measured model, since 2 Oct 2026: before, fold k used the same random numbers in all three cities)

Rationale:
- Full reproducibility across runs
- Independent errors for different horizons
- Different errors per fold (realistic variability)
- Previously `base_seed + fold_idx + horizon` produced duplicate seeds across horizons (e.g. fold 18 at h=6 and fold 0 at h=24 both got `base_seed + 24`), so those folds shared the same random draws. With the horizon scaled by 10000 and `fold_idx < 10000`, seeds are unique. Degraded results produced before this fix (v6 and earlier) used the old seeds and are re-run in v7.

### Noise-Magnitude Sensitivity
Not run by default since 2 Oct 2026: the errors are measured per city from real forecasts, so the error size is no longer a guess, and the two extra scenarios would triple the degraded runs. The remaining uncertainty (forecast years 2024–26 vs data years 2011–18) is stated as a limitation. The mechanism stays available: add e.g. `"degraded_x150": 1.5` to `config.degradation_scales` and `weather_scenarios`.

Original rationale (1 Oct 2026):
- The measured model is calibrated on 2024–26 forecasts, while the data are from 2011–18, when forecasts were less accurate; the literature model was calibrated once, for all cities. The two extra scenarios test whether the conclusions under `degraded` hold if the real forecast error is lower or higher.
- Common random numbers (same seed, same number of draws for every scale): the scenarios differ only in error magnitude, not in the random draw, so the differences between them are not sampling noise.
- Precipitation event detection is not scaled: miss rate and FAR are probabilities capped at 50 % (no skill), not error magnitudes; at 1.5× they would reach the cap from 30 h instead of 60 h. Keeping them fixed also keeps the random draws aligned across scales.
- Comparisons against the baseline are computed per scenario, so the sensitivity scenarios do not change the clean_only/degraded win rates and skill scores.

### Dynamic CV Fold Calculation
First fold cutoff counted back from end of dataset: `data_end - (n_folds * max_horizon)`
Folds per horizon = `ceil(n_folds * max_horizon / horizon)`; all horizons cover the same evaluation period.

### Partial Last Fold (no "drop last")
If the evaluation period is not a multiple of a horizon (5,880 h / 48 h = 122.5), the last fold is kept with a shorter test window: it starts at its regular origin, forecasts only the remaining hours (`n_steps = len(test_df)`), and those hours are scored. For 48 h: 122 full folds + 1 fold of 24 h (lead times 1–24).

Rationale:
- Previously (v6) `split()` stopped at the last full fold, so for h=48 the last 24 h of the evaluation period were silently dropped and h=48 was evaluated on a different period than the other horizons.
- Standard rolling-origin evaluation (Tashman 2000; Hyndman's `tsCV`) scores all available target hours and does not score forecasts beyond the end of the data. Every evaluation hour is scored exactly once per horizon, for any dataset length and horizon, without changing the tuning/evaluation split.
- Recursive and single-model forecasts (Seasonal Naive, ARIMA, SARIMAX, XGBoost, Prophet) give the same first `n` steps for `predict(n)` as for `predict(horizon)`. NeuralProphet is a direct multi-step model (`n_forecasts = n_steps`), so its partial fold uses a model with the shorter output length.

### ARIMA/SARIMAX Order Selection
Candidate orders from `auto_arima` on 6 spread tune folds; selection by mean 24-h MAE of the experiment model (statsmodels) on all 90 tune folds.

Rationale:
- Previously the order was chosen by AIC on `splits[0]` only. For London and Washington that split lies about 5 months before the 90 tune folds all other models use; for all cities it was a single 30-day window.
- The previous validation re-fitted with pmdarima, which differs from the statsmodels models used in the experiments (e.g. stationarity/invertibility enforcement). Selection now uses the experiment models themselves.
- Same folds and same criterion (24-h MAE on all 90 tune folds) as XGBoost and Prophet.

### No Post-Processing of Forecasts
Forecasts are scored as each model produces them: no clipping of negative values or other post-processing, in tuning and in the experiments. TimesFM_NoWeather returns non-negative forecasts because of TimesFM's own inference setting (`infer_is_positive=True`: forecasts are floored at 0 when the whole context is non-negative), which is part of the model as published. TimesFM with covariates can return negative forecasts: TimesFM forecasts the residual of the in-context regression, which has negative values, so the floor does not apply. Checked against the timesfm 3.0.2 source (3 Oct 2026): `infer_is_positive` floors the forecast at 0 only when the whole input is non-negative, and `forecast_with_covariates` adds the regression part afterwards, without clipping.

### Imputed Data Tracking and Exclusion from Scoring
Uses the `Functioning Day` column (No = imputed), configured via `functioning_day_col` for all three cities
Tracked per fold: train_imputed, test_imputed (unchanged), plus test_scored (observed test hours)
Logged in detailed results for post-hoc analysis

Imputed hours are excluded from scoring (experiments and tuning): only observed test hours enter MAE/RMSE/MASE/sMAPE, and the MASE scale uses only fully observed seasonal-naive pairs. They remain in the data used for fitting and as lag inputs.

Rationale: imputed values are not real demand, so scoring forecasts against them measures agreement with the imputation, not forecast skill.

### Runtime Measurement
Each fold's `fit()` and `predict()` calls are timed separately with `time.perf_counter()`. Weather preparation, feature construction, `reset()` and scoring are excluded. Per-fold times are in the detailed CSV; per (model, horizon, scenario) mean, std and total are in the aggregated CSV.

Rationale:
- Makes the accuracy/cost trade-off between models reportable per horizon and scenario.
- Fit and predict are kept separate because the split differs by model: NeuralProphet trains inside `predict()` (its `fit_time_s` is ≈ 0 and `predict_time_s` includes training), and XGBoost's recursive forecasts make `predict()` cost grow with `n_steps`. `runtime_s` (fit + predict) is the comparable total across models.
- Totals depend on the number of folds (980 for h=6 vs 35 for h=168); compare `runtime_mean_s` for per-forecast cost and `runtime_total_s` for the cost of the full evaluation period.

### Rolling Window CV
Training window is fixed-size and advances by `horizon` hours with each fold. Test set is `horizon` hours (the last fold can be shorter, see Partial Last Fold).
Ensures a consistent lookback window across all folds.

### Code Provenance
`provenance.get_code_version()` returns the git commit of the repository and whether tracked files had uncommitted changes (`git_dirty`; untracked files such as data and results are ignored). `provenance.get_library_versions()` returns the versions of Python and the forecasting libraries (numpy, pandas, scikit-learn, statsmodels, pmdarima, xgboost, optuna, prophet, neuralprophet, torch, tabpfn, tabpfn-time-series, and timesfm from `.timesfm_venv`). Tuning JSONs get both (`provenance`); every results row gets `git_commit`/`git_dirty`; aggregated rows and the W&B config also get `library_versions`. Without git (e.g. on the cluster, where the repository is copied without `.git`), `git_commit` holds a code ID instead, `code:<16 hex>` = SHA-256 of all `.py` and `.npz` files in `forecasting/` (without `testing/`, `__pycache__` and hidden folders), and `git_dirty` is `None`; `python forecasting/provenance.py` prints the code ID of the current files.

Rationale:
- Every number in the paper maps to one code version; old params files and results can be told apart from current ones without relying on file dates.
- Paper runs must use committed code (`git_dirty = False`); a warning is printed when `git_dirty` is `True` (not with the code ID, where it is `None`).
- A checkpoint or results CSV written by older code (no provenance fields) is rejected, because resuming from it would silently skip experiments or append rows under a different header. Before a full re-run, move the old `results_master_{version}.csv`, `detailed_results_master_{version}.csv`, `forecasts_*_{version}.csv` and `checkpoint_*.json` away, or use a new `results_version`.

### Checkpoint Recovery
Saves completed `(dataset_name, model, horizon, scenario)` tuples to JSON after each successfully completed experiment, after its aggregated and fold-level rows are on disk (previously fold-level rows were written only at the end of a city's run, so an interrupted run lost them and the resume skipped the experiments). Failed model-horizon-scenario combinations are never checkpointed.
On restart, skips already-completed experiments; failed/uncheckpointed combinations are retried.
Prevents data loss from crashes during long runs — including mid-run failures when iterating over multiple datasets.
File format: `checkpoint_{experiment_name}.json` in `results/` directory.

## Data Requirements

**Input CSV must have:**
- A datetime column (hourly frequency) — column name set via `config.date_col`
- A target variable column — set via `config.target_col`
- Weather covariates as configured in `config.weather_covariates` and `config.weather_degradation_mapping`
- Optional: functioning day / imputation tracking column (`config.functioning_day_col`)
- Optional: holiday column (`config.holiday_col`) — normalized to 0/1 at load time
- Optional: season column (`config.season_col`) — normalized to 0–3 at load time via `config.season_mapping`

**Weather Column Mapping:**
Dataset columns mapped to degradation variable types via `config.weather_degradation_mapping`. Example (Seoul):
- Temperature → 'temperature'
- Humidity → 'humidity'
- Wind speed → 'wind_speed'
- Visibility → 'visibility'
- Solar Radiation → 'solar_radiation'
- precipitation_mm → 'precipitation' (total precipitation incl. melted snow, mm/h)
- snow_depth_cm → 'snow_depth' (snow on the ground, cm)
- Dew point temperature → (not mapped, excluded from degradation)

London and Washington use the same names for precipitation and snow depth (`precipitation_mm`, `snow_depth_cm`).

**Weather variables per city (2 Oct 2026):**
- Precipitation = total precipitation incl. melted snow (mm/h) in every city. Seoul `precipitation_mm` = KMA ASOS 강수량 (station 108; original column `Rainfall`), which KMA reports as 3-hour totals at 00, 03, …, 21 h from November to March (spread evenly over their three hours, see Key Design Decisions) and hourly from April to October. London/Washington `precipitation_mm` = `rainfall_mm` + `snowfall_cm` / 0.7 (Open-Meteo `rain` and `snowfall`, 0.7 cm of snow per mm of water; equals Open-Meteo `precipitation`)
- Snow = snow depth on the ground (cm) in every city. Seoul `snow_depth_cm` = KMA ASOS 적설 (original column `Snowfall`: snow depth, despite the name). London/Washington `snow_depth_cm` = Open-Meteo archive `snow_depth` (ERA5-Land, 1 cm steps) at the grid cell of the airport station (Heathrow 51.479, −0.449; Reagan National 38.8483, −77.0342)
- `rainfall_mm` and `snowfall_cm` (London/Washington) and `Rainfall` and `Snowfall` (Seoul) stay in the CSVs but are not used
- Visibility: Seoul `Visibility` = KMA station 108 (10 m units, × 0.01 → km at load time, capped at 20 km); London/Washington `visibility_km` = Visual Crossing (`data/add_visibility.py`, locations "London,UK" and "Washington,DC"; London up to 65 km, Washington capped at 16 km)
- Time alignment: all weather columns follow the local clock with daylight saving, like the bike counts (London/Washington Open-Meteo columns re-aligned on 2 Oct 2026, see Key Design Decisions). Seoul has no daylight saving

**Preprocessing:**
- Sort by date (stable sort)
- Drop duplicate timestamps (keep first: for a repeated autumn hour the daylight-saving occurrence)
- Apply `column_scale_factors`
- No other filtering (imputed hours stay in the data and are used for fitting; they are excluded from scoring only)

## W&B Integration

**Run config:** `dataset`, `horizons`, `n_folds`, `n_train_samples`, `git_commit`, `git_dirty`, `library_versions`

**Logged per experiment:**
- {model}_{scenario}_h{horizon}_MAE/RMSE/MASE/sMAPE (aggregated)
- {model}_{scenario}_h{horizon}_runtime_mean_s / _runtime_total_s (aggregated)

**Fold-level tables:**
- {model}_{scenario}_h{horizon}_folds (detailed per-fold results)

**Summary tables:**
- results_table (aggregated results across all experiments)
- results_summary (grouped by model and scenario)

**Errors:**
- `error` (message for each failed fold)

**Comparative metrics** (only when `compute_and_log_comparative_metrics(..., log_wandb=True)`; `main.py` uses `False`):
- `comparative_metrics` table
- {model}_{scenario}_vs_{baseline}_win_rate / _skill_score, each with _ci_lower / _ci_upper (one set per scenario)

**Projects:**
- Production: `bike-forecasting`
- Testing: `bike-forecasting-testing`

## Results Schema

**Columns in results_master_{version}.csv:**
- `dataset`, `run_name`, `timestamp`, `model`, `horizon`, `n_folds` (successful folds), `n_failed_folds` (expected − successful; saved complete experiments have 0; retained for schema compatibility and must be 0 for paper results)
- `total_test_hours`: test hours summed over folds (= evaluation period, 5,880); `partial_folds`: number of folds with fewer than `horizon` test hours (1 for h=48, else 0)
- `MAE_mean`/`MAE_std`, `RMSE_mean`/`RMSE_std`, `MASE_mean`/`MASE_std`, `sMAPE_mean`/`sMAPE_std`
- `total_test_imputed`, `total_train_imputed`, `folds_with_imputed_test` (counts unchanged)
- `total_test_scored`: observed (scored) test hours summed over folds; `folds_without_scored_test`: folds whose test window is fully imputed (NaN metrics, skipped in `*_mean`/`*_std`)
- `*_mean`/`*_std`: mean/std over folds of the fold metrics, which are computed on observed hours only
- `version`: Results version string (from `config.results_version`), stored as a separate column alongside `run_name`
- `git_commit`, `git_dirty`: code version that produced the row (see Code Provenance); also in the detailed CSV
- `folds_with_convergence_warnings`: folds whose `fit()`/`predict()` raised a warning containing "converge" (e.g. SARIMAX/ARIMA optimizer); per fold: `convergence_warnings` (count) in the detailed CSV
- `weather_scenario`: 'all_weather' (only used in Pilot project), 'clean_only', 'degraded' (or an optional sensitivity scenario such as 'degraded_x150')
- `model_uses_covariates`: Boolean (from model.use_covariates property)
- `degradation_seed`: Random seed used (42 by default)
- `degradation_model`: error model of the degraded scenarios and its settings (`config.degradation_label()`, e.g. `nwp_measured(fresh,no_bias,seasonal_rain,intensity_miss,rain_freq_unbiased,cap_mean)`, or `literature`); also in the detailed CSV
- `num_weather_vars`: Number of columns in `X_train` (taken from the first fold) (0 for models without covariates; for models with covariates: 7 degradable + holiday + season = 9, plus 4 calendar time features for XGBoost = 13 total features; XGBoost_NoWeather: 4, the calendar time features only; TabPFN creates its own calendar features, which is 17 additional features)
- Runtime in seconds (wall-clock, over successful folds): `fit_time_mean_s`, `predict_time_mean_s`, `runtime_mean_s`, `runtime_std_s` (per fold), and `fit_time_total_s`, `predict_time_total_s`, `runtime_total_s` (summed over folds)

**Additional columns in detailed_results_master_{version}.csv (per fold):** `MAE`, `RMSE`, `MASE`, `sMAPE` (observed hours only), `fold`, `test_imputed`, `train_imputed`, `test_hours`, `test_scored`, `fit_time_s`, `predict_time_s`, `runtime_s` (seconds)

**forecasts_{dataset_name}_{version}.csv (one row per forecast hour, one file per dataset):**
- `dataset`, `model`, `horizon`, `weather_scenario`, `fold`: same keys as the detailed CSV
- `lead_time`: hours ahead, 1 … `test_hours` of the fold
- `datetime`: the hour being forecast (from `config.date_col`)
- `y_true`, `y_pred`: actual value and point forecast, exactly as scored
- `observed`: `False` for imputed hours (`functioning_day_col == 'No'`), same mask as in scoring; imputed hours are kept, not dropped
- `version`, `git_commit`
- Written only for successfully completed model-horizon-scenario runs. If any fold fails, no forecast or detailed-result rows from that incomplete combination are written. Every evaluation hour appears once per successful (model, horizon, scenario): 5,880 rows each, about 1.8 million rows over the three cities (roughly 250–300 MB)
- Output only: nothing in the pipeline reads it, and the other results files are unchanged by it. The MAE of a fold recomputed from its `observed` rows equals the fold's `MAE`


## Known Limitations


1. Measured model: temperature, humidity and wind errors are real (persistent, linked to each other); solar radiation, precipitation and visibility errors are persistent but independent of each other and of the other variables. Literature model: independent errors across variables and from hour to hour (real forecast errors persist over many hours)
2. Measured model: the errors come from ECMWF forecasts of 2024–26 (model versions 49r1 and 50r1) and are applied to data from 2011–18, when forecasts were less accurate; the replayed errors belong to another day than the test window (same season and start hour), so they do not depend on the weather of the test window. Literature model: below the shortest lead time verified in the sources (12 h for temperature and humidity, 24 h for wind, solar radiation, precipitation and visibility) the error formulas are extrapolated
3. Measured model references: Seoul temperature, humidity, wind and visibility = station 47108 (the station of the Seoul data); London and Washington = ERA5 / ERA5-Land at the nearest grid cell, without the Open-Meteo height correction (their covariates are from the Open-Meteo archive at 51.5074, −0.1278 and 38.9073, −77.0369, see `data/provenance/`: temperature differs from the airport cell by 0.27 °C (London) and 0.30 °C (Washington) on average after the daylight-saving re-alignment); solar radiation = ERA5 for all cities (no usable free station data; Seoul's covariates are station data); precipitation = the station for Seoul (since 2 Oct 2026), ERA5 for London and Washington; visibility errors from ISD station reports, which end in Aug 2025. Seoul has 844 complete runs (station gaps), London and Washington 1,817. Literature model: error growth calibrated to published statistics, the same for every city
4. NeuralProphet training cost grows with the horizon (one output per forecast step with `n_forecasts = horizon`), and the model is retrained on every `predict()` call
5. NeuralProphet hyperparameters are tuned at `config.tune_horizon` only and reused for all evaluation horizons
6. Season and holiday can only be used by SARIMAX (and NeuralProphet) in folds where they vary within the 30-day training window
7. ARIMA/SARIMAX candidate orders come from `auto_arima`'s stepwise AIC search on 6 folds; orders it does not propose are not considered
8. NeuralProphet is searched on 6 of the 90 tune folds (run time); all other tuned models are selected on all 90
9. Literature model: precipitation detection errors use POD and FAR that Sukovich et al. (2014) verified for the top 1% of 24-hour events; they are applied to all hourly precipitation. Both models: a training fold without precipitation (wet-hour share 0) gives no false alarms in that fold
10. Snow depth has no measured forecast errors: the degraded scenarios hold it at its value at the issue time (persistence), so melting and new snow within the horizon are missed. London/Washington snow depth is ERA5-Land (1 cm steps; Open-Meteo notes it tends to be overestimated, e.g. 1–3 cm for days at 10–18 °C in Washington), Seoul's is measured at the station; both are snow on open ground, not on cleared streets. London has no snow in its evaluation period, Washington only 24–31 Dec 2012
11. The partial last fold (h=48) covers lead times 1–24 only and is averaged with equal weight to the full folds; for NeuralProphet it is forecast by a model with `n_forecasts = 24`
12. Fold metrics are averaged with equal weight per fold; a fold with few observed hours (partly imputed test window) counts as much as a fully observed one
13. Runtimes are wall-clock times on the machine that ran the experiment: they depend on hardware, CPU/GPU availability, thread settings (e.g. XGBoost `n_jobs=-1`, TabPFN `CPUParallelWorker`) and concurrent load, so they are only comparable within one run environment. The first fold of a model can include one-off costs (library warm-up, model loading for TabPFN/TimesFM). Failed folds are not timed
14. Sensitivity scenarios are not run by default (since 2 Oct 2026). If added, they scale error magnitudes only; precipitation event detection (miss rate, false-alarm probability, false-alarm amount) and, in the measured model, whether visibility falls below the cap stay at their calibrated values at every scale
15. Literature model only: consequence of the hour-to-hour independence (Limitation 1): for models that use each step's covariates only for that step (SARIMAX, NeuralProphet, TabPFN, TimesFM), the expected error per step is unchanged and mainly the fold-to-fold spread is affected. XGBoost forecasts recursively, so correlated covariate errors could compound through the fed-back lags; the independent-noise setting may therefore understate XGBoost's degradation relative to TabPFN. The measured model has persistent errors
16. Measured model: the fresh, locally corrected forecast (default) is approximated by the global 9 km model's errors at short lead times with its average bias removed; no free archive of past local forecasts exists. Local high-resolution forecasts are usually more accurate in the first hours, so these may be slightly pessimistic. The replayed run's start hour (00/12 UTC) can differ from the issue time of day by up to 6 h. With `nwp_fresh_forecast = False`: the issue time is the end of the training window and a run is assumed usable 6 h after its start; the 06/18 UTC runs (90 h only) are not used. Beyond 90 h ECMWF outputs 3-/6-hourly values; the replayed hourly errors there are those of Open-Meteo's interpolated hourly series
17. `testing/test_max_degradation.py` applies the literature model only
18. Measured model, precipitation: miss probability uses two broad intensity classes (<1 and ≥1 mm/h), not a continuous intensity relationship; the hit-amount error remains one distribution per lead time and season. Sparse intensity classes require wider seasonal windows (up to ±165 days for Seoul's stronger class). The rain-frequency bias is removed by default (`nwp_rain_frequency_unbiased`): false alarms balance the combined misses using the training fold's wet-hour and light/strong shares.
19. Seoul precipitation is reported as 3-hour totals from November to March (KMA); `precipitation_mm` spreads each total evenly over its three hours. This keeps the totals and most of the hourly signal (see Key Design Decisions) but marks too many hours as wet and spreads short showers at a third of their intensity. Affects Seoul's whole tuning period (Dec 2017 – Mar 2018) and 744 of the 5,880 evaluation hours (31 Mar and November 2018)
20. London/Washington: snow depth comes from the airport grid cell (download of 2 Oct 2026), the other Open-Meteo covariates from the original download (51.5074, −0.1278 and 38.9073, −77.0369; see `data/provenance/`)
21. Measured model, Seoul November–March precipitation: verified at 3-hour resolution (station and forecast split over the same windows), so timing errors within the 3 hours are not counted. On Apr–Oct data treated the same way the 24 h miss rate is 0.18 instead of the true hourly 0.22 (stronger rain 0.11 instead of 0.18)
22. Season covariate: the three source datasets define the seasons differently, and the season columns are used as published. Seoul (UCI `Seasons`) and London (Kaggle `season`) switch on the 1st of March, June, September and December (meteorological seasons); Washington (`season`) switches on 21 March, 21 June, 23 September and 21 December (astronomical seasons). After `season_mapping` all cities use the same codes (0 spring, 1 summer, 2 autumn, 3 winter), so the same code covers different calendar dates across cities (e.g. 1–20 March is spring in Seoul and London, winter in Washington). Washington's evaluation period (1 May – 31 Dec 2012) contains the switches on 21 June, 23 September and 21 December 2012
23. Visibility (measured model): errors are measured against station reports. Seoul's covariate is that station's own visibility. London's and Washington's covariates come from Visual Crossing (`data/add_visibility.py`, locations "London,UK" and "Washington,DC"). Washington's matches the Reagan National reports (82 % of hours within 0.1 km, 98 % within 1 km, 2011–12); London's does not match Heathrow (median difference 4.5 km; in hours with Heathrow below 9.9 km, 7 % within 0.1 km and correlation 0.51, 2015–17), so London's degraded visibility carries Heathrow forecast errors on a different series. The error is one lognormal distribution for all visibility levels below the cap, although the measured error depends on the level (London, mean / SD of log(forecast/observed): below 5 km +1.71 / 1.56, above 40 km −0.39 / 0.52)
24. Mean change of the degraded covariates after their caps and bounds (measured model, default setting; real data, experiment folds, all horizons; `weather/nwp/validate_on_real_data.py` prints these per city, each with its chance variation (±: 95 % range when the test windows are resampled)). **Since 4 Oct 2026 every degraded covariate keeps the clean value on average, and false alarms balance the missed rain amounts** (`nwp_mean_preserving_caps`, with `nwp_remove_bias`; `nwp_rain_amount_unbiased`, with `nwp_rain_frequency_unbiased`). Before, the errors were mean-preserving before the cuts, and the cuts removed only the errors beyond them: visibility at the city cap (Seoul 20 km, Washington 16 km) or London's training maximum, rain hits at the larger of the training maximum and the observed amount, humidity at 100 %, wind at 0, solar radiation at the training cap; and false alarms had the measured false-alarm amounts, much smaller than the missed amounts in Seoul (0.7 against 2.4 mm per hour). Method: visibility and rain hits: the log mean m is set per hour so that E[min(x·exp(m + σZ), c)] = x, Z ~ N(0, 1), for the hour's clean value x and cap c (closed form x·exp(m + σ²/2)·Φ(k − σ) + c·(1 − Φ(k)), k = (ln(c/x) − m)/σ; bisection on [−σ²/2, ln(c/x) + 8σ]; `capped_mean_log`); c = ∞ gives the earlier −σ²/2, c = x keeps the value; London's visibility cut becomes the larger of the training maximum and the hour's value. Humidity and wind: a shift d per hour so that the clipped value averages x over the candidate runs (exact, because the replayed run is drawn from them with equal probability; `clipped_mean_shift`). Solar radiation: a relative shift per hour from the closed-form mean of a clipped normal (`clipped_normal_shift`). False alarms: the measured amount distribution (log SD kept), rescaled so that its mean after the cap is the expected amount of a missed hour at that lead time from the training fold, (q·m_l·a_l + (1 − q)·m_s·a_s)/(q·m_l + (1 − q)·m_s) (q: light share, m: miss rates, a: mean amounts of light / stronger wet hours). Hours at a bound are not changed. The random numbers are drawn in the same order: which hours are missed or false alarms, the replayed run, temperature and snow depth are unchanged, and with both switches off the output is bit-identical to the earlier version. Seed 42, before → after (Seoul / London / Washington): visibility below the cap ×0.71 / ×0.80 / ×0.66 → ×0.99 / ×1.00 / ×0.99; rain-hit amounts ×0.65 / ×0.77 / ×0.78 → ×1.01 / ×0.99 / ×1.04; false-alarm amount 7.7 / 32.3 / 33.5 % → 19.7 / 35.6 / 36.2 % of the clean total (missed amount 24.8 / 32.5 / 33.7 %); total precipitation ×0.57 / ×0.84 / ×0.85 → ×0.96 / ×1.03 / ×1.05; humidity −0.17 / −0.32 / −0.83 → −0.01 / −0.18 / −0.52 %-points; wind +0.050 / −0.003 / +0.022 → +0.008 / −0.012 / +0.012 m/s; solar radiation in daylight ×0.98 / ×0.98 / ×0.99 → ×1.00 / ×0.99 / ×1.00. Average over 21 seeds after the change: humidity −0.01 / −0.02 / −0.04 %-points (before −0.16 / −0.17 / −0.35: the clip at 100 % accounted for −0.13 / −0.16 / −0.31, the rest of seed 42's −0.17 / −0.32 / −0.83 was chance), wind +0.007 / +0.001 / +0.003 m/s, solar ×0.998, visibility below the cap ×1.00, rain-hit amounts ×1.00 / ×0.99 / ×1.00, total precipitation ×1.01 / ×1.03 / ×1.00. **Remaining: chance variation of one seed.** The experiments use one seed (42). Its values differ from the averages by chance, mainly because a few heavy rain events and, at 168 h, only 35 replayed runs carry much weight: per seed the hit ratio varies by about ±0.04, total precipitation by ±0.04, humidity by ±0.2–0.3 %-points (Washington's −0.52 at seed 42 is the lowest of the 21 seeds, mostly the 168 h windows, −1.58). All models see the same degraded inputs, so comparisons between models are not affected; the absolute effect of the degradation carries this chance part. **Remaining: hours at a bound.** They cannot keep their value on average without losing their error: Seoul wind +0.007 m/s (calm hours, 0.8 % of hours), solar −0.2 % (hours at or above the training cap), and visibility at the cap falls below it with the measured probability (an event rate), so over all hours including those at the cap visibility is ×0.96 in Seoul and Washington. Hours close to a bound now sit at the bound more often, e.g. visibility below the cap reaches it in 43 % (Seoul) and 60 % (Washington) of hours (measured ECMWF forecasts: 47 % / 53 %; before: 17 % / 22 %). **Remaining: London rain frequency.** London's degraded data are wet about 4 % more often than the clean data (21 seeds: ×1.04; false alarms about 10 % more frequent than misses, because the training fold's wet share only estimates the test window's), so its total precipitation is ×1.03. Temperature (±0.03 °C) is unaffected. Cost: about 2 ms more per test window.



