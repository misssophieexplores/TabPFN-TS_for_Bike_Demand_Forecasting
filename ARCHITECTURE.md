# Shared Bike Demand Forecasting - Architecture

<!-- TODO: update! -->
## Project Structure
```
forecasting/
├── main.py                  # Multi-city orchestrator — runs all datasets sequentially
├── config.py                # Shared base configuration dataclass (wandb_project, results_version, horizons, etc.)
├── config_seoul.py          # Seoul-specific overrides (get_config())
├── config_london.py         # London-specific overrides (get_config())
├── config_washington.py     # Washington-specific overrides (get_config())
├── features.py              # Calendar time feature engineering (used by XGBoost)
├── provenance.py            # git commit + dirty flag + library versions for tuning JSONs and results
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
│       ├── tune_sarimax.py          # Auto-SARIMAX tuning (pmdarima)
│       ├── tune_xgboost.py          # XGBoost random search
│       ├── tune_prophet.py          # Prophet random search
│       └── tune_neuralprophet.py    # NeuralProphet random search
├── weather/
│   ├── nwp_error_model.py          # Measured NWP error model per city (default degradation model)
│   ├── weather_degradation.py      # Literature error model; training-fold parameters; rain/snow correction
│   ├── weather_processor.py        # Scenario orchestration
│   └── nwp/                        # Measured forecast errors: download, analysis, calibration
│       ├── fetch_nwp_data.py       # Downloads ECMWF forecasts, reanalysis and station data (normal internet access needed)
│       ├── analyze_nwp_errors.py   # Error statistics per city, variable and lead time (CSV for the paper)
│       ├── build_nwp_calibration.py # Builds the calibration files used by nwp_error_model.py
│       ├── expected_errors.py      # Errors applied by the default setting at 6/24/48/168 h (for the paper)
│       ├── validate_on_real_data.py # Runs the model on the real data with the experiment folds (check)
│       └── calibration/            # seoul.npz, london.npz, washington.npz (+ readable *_summary.json)
├── evaluation/
│   ├── cv.py                # TimeSeriesCV with dynamic fold calculation
│   └── metrics.py           # MAE, RMSE, MASE, sMAPE
├── testing/
│   ├── test_max_degradation.py
│   ├── test_weather_single_model.py
│   ├── test_weather_unit.py
│   ├── test_pipeline_unit.py   # CV folds, metrics, comparative metrics, XGBoost tuning = experiment (with and without weather)
│   └── preflight.py            # Pre-flight check before the paper runs: params files, build_models, CV, one fold per model (incl. TabPFN/TimesFM weights), W&B; writes nothing
├── run_experiments.py       # ForecastingExperiment class with W&B logging and checkpointing; load_and_prepare_data(); comparative metrics
└── run_weather_baseline.py  # Per-city experiment runner (called by main.py, or directly with --city)

data/
├── saving_data.py           # Data pre-processing and saving locally
├── SeoulBikeData.csv
├── LondonBikeData.csv
└── WashingtonBikeData.csv

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

1. **Load**: `load_and_prepare_data()` reads CSV, parses dates, sorts by time, drops duplicate timestamps (e.g. DST clock-back hours, keeps first), applies `config.column_scale_factors`, normalizes holiday and season columns and appends them to `weather_covariates` if not already listed
2. **Scenario Setup**: `WeatherProcessor` selects variables based on scenario
3. **Split**: `TimeSeriesCV.split(df, horizon, partial_last_fold=True)` creates rolling window train/test folds covering the full evaluation period
4. **Weather Preparation**: Per-fold weather preparation via `WeatherProcessor.prepare_weather_data(split=...)`. Training data always uses clean observed weather. In the degraded scenarios ('degraded'; optional noise-magnitude sensitivity scenarios if added to `degradation_scales`) the training fold also provides the degradation parameters (solar cap, wet-hour share), and test data receives per-row lead-time noise, scaled by the scenario's factor: row *i* is degraded using lead time *(i + 1)* hours, so error grows from the 1-hour error at the first step (not zero: every error formula has an intercept, e.g. temperature σ = 0.79 °C, humidity 13.0 %-points) up to full-horizon noise at the last step.
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
- `config.py`: Shared base dataclass. Contains the fields that are identical across all datasets: `wandb_project`, `results_version`, `horizons`, `n_folds`, `n_train_samples`, `seasonal_period`, `degradation_seed`, `degradation_model`, `nwp_fresh_forecast`, `nwp_remove_bias`, `nwp_seasonal_rain`, `nwp_rain_frequency_unbiased`, `nwp_run_hours_utc`, `nwp_availability_delay_h`, `nwp_season_window_days`, `degradation_scales`, `weather_scenarios`, `output_dir`, `verbose`, `experiment_name`, `tune_folds`, `tune_horizon`. Dataset-specific fields default to `None` (except `holiday_mapping` and `column_scale_factors`, see below).
- `config_seoul.py`, `config_london.py`, `config_washington.py`: Each exposes a `get_config()` function that instantiates `ForecastConfig` and overrides all dataset-specific fields. To update `wandb_project` or `results_version`, change `config.py` only — all cities pick it up automatically.
- Dataset-specific fields (set per city): `data_filename`, `dataset_name`, `date_col`, `target_col`, `functioning_day_col`, `holiday_col`, `holiday_mapping`, `season_col`, `season_mapping`, `weather_covariates`, `weather_degradation_mapping`, `rain_col`, `snow_col`, `timezone`, `nwp_calibration_file`, `column_scale_factors`, `arima_params_file`, `sarimax_params_file`, `xgb_params_file`, `xgb_noweather_params_file`, `prophet_params_file`, `neuralprophet_params_file`, `neuralprophet_noweather_params_file`
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
- `nwp_rain_frequency_unbiased` (`False`, option): measured model only; false alarms set equal to misses, so the degraded data are wet as often as the clean data (removes the rain-occurrence bias: Seoul's raw forecast is wet about 2.2× as often as the station)
- `degradation_label()`: error model and settings as written to the results, e.g. `nwp_measured(fresh,no_bias,seasonal_rain)`
- `timezone` (per city): IANA time zone of the date column (Seoul `Asia/Seoul`, London `Europe/London`, Washington `America/New_York`); the measured model converts the test hours to UTC. A repeated autumn hour is read as standard time, a non-existent spring hour is shifted forward (London and Washington have both in their data)
- `nwp_calibration_file` (per city): calibration file of the measured model, relative to `forecasting/` (`weather/nwp/calibration/<city>.npz`)
- `degradation_scales`: `{'degraded': 1.0}` (sensitivity factors removed 2 Oct 2026; add e.g. `'degraded_x150': 1.5` to run one) — every degraded scenario and the factor applied to the calibrated error magnitudes (see Noise-magnitude sensitivity). `config.is_degraded(scenario)` (True for its keys) is the single test for a degraded scenario, used by `WeatherProcessor` and by the skip logic in `run_experiments.py`
- `experiment_name`: Defaults to `{dataset_name}_{results_version}` (set in `__post_init__`; re-set in `run_weather_baseline.main()` if it starts with `None`)
- `tune_horizon`: Horizon used by all tuning scripts (24)
- `tune_folds`: Number of (last) CV folds used by the tuning scripts (90; `None` = all folds)
- `holiday_col`: Optional column name for public holidays (normalized to 0/1, appended to `weather_covariates` at load time)
- `holiday_mapping`: Dict mapping raw holiday string values → 0/1. Default `{'Yes': 1, 'No': 0}`; Seoul overrides with `{'Holiday': 1, 'No Holiday': 0}`. Only used when `holiday_col` has string (object) dtype; numeric holiday columns are coerced to int (NaN → 0).
- `column_scale_factors`: Dict `{column: factor}`; each listed column is multiplied by its factor at load time. Keys must match column names exactly (case-sensitive); non-matching keys are silently ignored. Default `{}`; Seoul sets `{"Visibility": 0.01}` (raw unit 10 m → km, the unit used for London and Washington).
- `rain_col`, `snow_col`: Rainfall and snowfall columns used by the rain/snow phase correction in the degraded scenarios (Seoul: `Rainfall`/`Snowfall`; London and Washington: `rainfall_mm`/`snowfall_cm`). Required for the degraded scenarios.
- `season_col`: Optional column name for season (normalized to 0–3 int via `season_mapping`, appended to `weather_covariates` at load time)
- `season_mapping`: Explicit per-dataset dict mapping raw season values to 0–3 integers (handles strings, 0-based, and 1-based encodings)
- `verbose`: If `True`, prints detailed progress (CV info, data loading, W&B URLs). Default `False` in `config.py` (cluster/server runs where stdout is captured in SLURM logs).
- Params files: each city config points to the tuned-parameter JSON files in `results/tuning/` (all tuned with `n_train_samples=720`). ARIMA and SARIMAX params files must contain `with_intercept`, i.e. they must come from the current tuning scripts; older files are rejected by `run_weather_baseline.py`. `neuralprophet_noweather_params_file` and `xgb_noweather_params_file` must be set (each tuned with `--scenario no_weather`); they are `None` until then and the runners raise `ValueError`. `build_models()` also raises if `xgb_noweather_params_file` was tuned with another scenario.

### Weather Degradation (`weather/`)
**WeatherProcessor**: Orchestrates weather data preparation for scenarios
- `prepare_weather_data(split='train'|'test')`: Main entry point, applies scenario logic. Training split always returns clean weather; test split applies degradation with per-row lead times in the degraded scenarios (`config.is_degraded`), with `noise_scale = config.degradation_scales[scenario]`. In a degraded scenario, the train split also computes `degradation_params` (solar cap = 99.5th percentile of the clean training fold; wet-hour share = share of training hours with rain or snow > 0; `visibility_max` = largest visibility of the training fold), which the test split then uses; the train split must be prepared first (otherwise `RuntimeError`). The test split passes the rows' timestamps (`config.date_col`) to the degradation.
- `get_weather_columns()`: Returns appropriate columns per scenario
- `degrade_dataframe(df, horizon, fold_idx, noise_scale=1.0, timestamps=None)`: On-the-fly degradation with proper seeding (the seed does not depend on `noise_scale`), with the model in `config.degradation_model`, followed by the rain/snow phase correction (temperature column taken from `weather_degradation_mapping`, rain/snow columns from `config.rain_col`/`config.snow_col`; raises `ValueError` if they are not set). The measured model needs `timestamps` (`ValueError` otherwise) and stores what it used in `last_degradation_info` (settings, forecast start, replayed start hour, first/last lead time, replayed run, number of candidate runs)

**Weather Scenarios:**
1. **all_weather**: All weather variables from `config.weather_covariates`, no degradation (original baseline)
2. **clean_only**: 7 degradable variables + holiday + season (excludes Dew point), no degradation
3. **degraded**: 7 degradable variables + holiday + season with realistic NWP forecast errors on degradable columns only (calibrated error model, scale 1.0)
4. Optional (not run by default since 2 Oct 2026): noise-magnitude sensitivity scenarios, e.g. degraded_x050 / degraded_x150

**Measured NWP error model** (default, `degradation_model = "nwp_measured"`, `weather/nwp_error_model.py`; full description in `weather/weather_methodology.md`):
- Data: 1,828 ECMWF IFS HRES 9 km runs per city (00/12 UTC, Mar 2024 – Sep 2026, Open-Meteo Single Runs API), compared with the source each city's covariates come from: Seoul = station 47108 (= KMA 108, the station of the Seoul data; Mar 2024 – Aug 2025, 844 complete runs); London and Washington = ERA5 / ERA5-Land (their covariates come from the Open-Meteo archive). Precipitation: Seoul = the station (hourly SYNOP amounts, Mar 2024 – Aug 2025, 1,049 runs; since 2 Oct 2026), London and Washington = ERA5. Solar radiation uses ERA5 for all cities, visibility the station
- Lead times (`nwp_fresh_forecast = True`, default): the weather forecast starts when the demand forecast is issued (first test hour − 1 h), so row i gets lead time i + 1 in every city; errors are replayed from the 00/12 UTC runs closest to the issue time of day. `False`: the newest run available at the issue time is used (6–17 h old), row i gets lead time run age + i + 1
- Temperature, humidity, wind speed: the actual errors of one real ECMWF run of the city are added (the selected start hour, start within ± 30 days of the test window's day of year, any year; drawn at random with the fold seed). With `nwp_remove_bias = True` (default) the average error of all these candidate runs is subtracted first (bias removed, e.g. Seoul −1.5 °C). Error growth, hour-to-hour persistence and the links between the three variables are those of the real forecasts. Humidity clipped to [0, 100], wind truncated at 0
- Solar radiation: X·(1 + b(lead) + s(lead)·z), b and s measured per lead time (b = 0 with the bias removed), z standard normal with the measured hour-to-hour correlation (AR(1)); 0 at night; capped at the solar cap of the training fold
- Rainfall and snowfall: one precipitation variable, one decision per hour; measured miss rate and false-alarm ratio per lead time and time of year (`nwp_seasonal_rain`; 12 monthly bins, interpolated by the forecast start date; wet = ≥ 0.1 mm/h); false-alarm ratio converted with the training fold's wet-hour share as in the literature model; misses and false alarms persist from hour to hour (latent AR(1) with the measured correlation); hits get the measured lognormal amount error (mean-preserving with the bias removed), false alarms the measured amount distribution
- Visibility: lognormal error in log space with measured mean (mean-preserving with the bias removed), SD and persistence. Seoul (cap 20 km) and Washington (cap 16 km): an hour at the cap stays there unless the forecast falls below it (measured probability 0.30 and 0.11); results are cut at the cap. London (no cap): cut at the training fold's maximum
- Calibration files: `weather/nwp/calibration/<city>.npz`, built by `weather/nwp/build_nwp_calibration.py` from the downloaded data (`weather/nwp/fetch_nwp_data.py`); `<city>_summary.json` lists the main numbers
- Checked on the real data with the experiment's CV folds (all horizons, every fold, `weather/nwp/validate_on_real_data.py`): simulated errors match the calibration (default setting: mean temperature error −0.06 to +0.04 °C, typical errors within 0.17 °C / 0.8 %-points / 0.09 m/s; first version: temperature error SD within 0.3 °C, visibility below the cap 0.27 vs 0.30 in Seoul and 0.11 vs 0.11 in Washington); about 8 ms per fold
- Typical errors applied by the default setting at 24 h: temperature 0.92 / 0.67 / 0.96 °C, humidity 7.4 / 4.4 / 6.4 %-points, wind 0.73 / 0.53 / 0.64 m/s (Seoul / London / Washington; `weather/nwp/expected_errors.py`)

**Literature error model** (`degradation_model = "literature"`, `weather_degradation.py`; same for all cities, row i has lead time i + 1). Degradation variables (7, all mapped in `weather_degradation_mapping` for Seoul, London and Washington):
- Temperature → Additive Gaussian error
- Humidity → Additive Gaussian error, clipped to [0, 100]
- Wind speed → Additive Gaussian, truncated at 0
- Solar Radiation → Heteroscedastic Gaussian error (σ ∝ value, so unit-independent), capped at the solar cap from the training fold
- Rainfall and Snowfall → one precipitation variable: one event-detection draw per hour (miss rate = FAR = 0.25 + 0.10·h/24, the line through the cited Day-1 and Day-2 values, capped at 50% from 60 h; miss rate = 1 − POD; false alarm for a dry hour with probability FAR / (1 − FAR) · POD · p / (1 − p), p = wet-hour share of the training fold; false-alarm amount written to the rain column) + one multiplicative lognormal magnitude error per detected hour
- Visibility → Multiplicative lognormal, constant CV = 25% (horizon-independent, unit-independent)

**Rain/snow phase correction** (all three cities, after degradation): precipitation is reassigned by the degraded temperature: above 2 °C snow is moved to rain, below 2 °C rain is moved to snow. Amounts are moved without unit conversion (rain in mm, snow in cm).

**Non-degradable Variables (excluded from degradation, always passed through as-is):**
- Dew point temperature (excluded from `weather_degradation_mapping`; only included in `all_weather`, via `weather_covariates`)
- Holiday (`config.holiday_col`) — passed through in clean_only and degraded
- Season (`config.season_col`) — passed through in clean_only and degraded

**Noise-magnitude sensitivity** (`noise_scale`, from `config.degradation_scales`):
- Scaled (multiplied by `noise_scale`, before clipping/capping). Measured model: the replayed temperature, humidity and wind errors (bias included); the relative solar error (b and s·z); the log error of precipitation amounts of hits; the log error of visibility and the depth below the cap. Literature model: Gaussian σ of temperature, humidity and wind speed; relative MAE of solar radiation; magnitude CV of detected precipitation; CV of visibility
- Not scaled: precipitation event detection (miss rate, false-alarm probability per dry hour, false-alarm amount), whether visibility falls below the cap (measured model), the replayed run, the wet-hour share, the solar cap, the physical bounds and the 2 °C rain/snow threshold
- `noise_scale = 1.0` ('degraded') is the calibrated error model, unchanged
- The seed does not depend on the scenario, and neither the detection outcomes nor the number of random draws per hour depend on `noise_scale`, so all degraded scenarios of a (horizon, fold) use the same random numbers (common random numbers): every additive error (before clipping) at 0.5× / 1.5× is 0.5 / 1.5 times the error at 1×, and precipitation is hit, missed and falsely forecast in the same hours (measured model: the same ECMWF run is replayed). The scenarios differ in error magnitude only
- `noise_scale < 0` raises `ValueError`

**Reproducibility:**
- Seed management: `horizon_seed = base_seed + 10000 * horizon + fold_idx` (`degrade_dataframe()` raises `ValueError` if `fold_idx` is outside [0, 10000))
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
- XGBoostForecaster: Uses lagged features (n_lags=24) + weather covariates (including holiday and season via `weather_covariates`) + calendar time features (hour, dayofweek, month, is_weekend). `use_time_features=True` — pipeline appends calendar features automatically at fold time. **Note: XGBoost must be re-tuned whenever `weather_covariates` changes (e.g. after adding holiday/season).**
- XGBoostForecaster_NoWeather (model name `XGBoost_NoWeather`, key `xgboost_noweather`): same model, without weather, holiday and season covariates: lagged demand + calendar time features (hour, dayofweek, month, is_weekend) only. `use_covariates=False`, `use_time_features=True`: the pipeline passes the calendar features only, and the degraded scenarios are skipped. Counterpart of XGBoost in the with/without-weather comparison (as TabPFN / TabPFN_NoWeather and NeuralProphet / NeuralProphet_NoWeather). Tuned with `tune_xgboost.py --scenario no_weather`
- TabPFNPipelineForecaster (model name `TabPFN`): `TabPFNTSPipeline` (tabpfn-time-series), zero-shot, TabPFN v2.5 pinned via `TABPFN_MODEL_CONFIG` (`tabpfn-v2.5-regressor-v2.5_default.ckpt`); all other settings at the pipeline defaults. Features: the pipeline's defaults (running index, calendar, auto-seasonal) + every covariate column (present in both context and future frame). Point forecast = median. Without the explicit pin the checkpoint would depend on the installed tabpfn-time-series version (1.0.10: v2; 1.1.0/1.2.0: v3; 1.3.0: v3.5)
- TabPFNPipelineForecaster_NoWeather (model name `TabPFN_NoWeather`): same pipeline and model, univariate (context and future frames contain only timestamps and target)
- TimesFMForecaster / TimesFMForecaster_NoWeather (`timesfm_model.py`, server `run_timesfm_server.py`): TimesFM 2.5 (200M, `google/timesfm-2.5-200m-pytorch`), zero-shot, no timestamps. Compiled with the authors' recommended forecast config (`normalize_inputs`, `use_continuous_quantile_head`, `force_flip_invariance`, `infer_is_positive`, `fix_quantile_crossing` all `True`), `max_context=1024` (context = the 720 training hours), `max_horizon=168`, `return_backcast=True` (required for covariates). Point forecast = median. With covariates: `forecast_with_covariates` in its default mode `"xreg + timesfm"` — an in-context linear regression on the covariates (standardized with context statistics, ridge 0, pseudo-inverse; constant covariates are harmless), TimesFM forecasts the residual; covariate values for the horizon come from `X_test` (clean or degraded). NoWeather: `forecast()`; with `return_backcast=True` this returns the backcast followed by the forecast, so the forecast is the last `horizon` values. The client checks the context has no NaN (the server finds the horizon rows by NaN target), checks each server reply and deletes its temp files
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
- SARIMAXForecaster: `SARIMAX(y, exog, order, seasonal_order, trend, enforce_stationarity=False, enforce_invertibility=False)`, fitted with `method='lbfgs'`, `maxiter=200`; a warning is raised if the optimizer does not converge. `y` and `X` get a synthetic hourly `DatetimeIndex` starting 2020-01-01 to silence statsmodels index warnings; the forecast index continues directly after the training index. The real timestamps are not used.
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
- `tune_prophet.py`: Prophet (changepoint / seasonality / holidays prior scales + seasonality mode)
- `tune_neuralprophet.py`: NeuralProphet (learning_rate + n_lags)

**Common to all tuning scripts (ARIMA, SARIMAX, XGBoost, Prophet, NeuralProphet):**
- Tuning data is selected with `df[date_col] <= TimeSeriesCV.get_cutoff_date(df)`. `<=` is correct: the cutoff timestamp is the last training hour of evaluation fold 0, not a test hour (test windows are `(test_start, test_end]`, so the held-out test period is `(cutoff, data_end]`). The held-out test period is never touched.
- The tuning period (first and last timestamp of the tuning data, from `TimeSeriesCV.get_tuning_period()`) is saved in every tuning JSON as `tuning_period`; `last_timestamp` equals the cutoff date.
- CV splits are built with `config.tune_horizon`; the tune folds are `TimeSeriesCV.tune_fold_indices(splits)`: the **last** `config.tune_folds` folds (`None` = all folds), without fully imputed test windows — one function for all tuning scripts
- `TimeSeriesCV.spread_fold_indices(tune_folds, n)`: n folds spread evenly over the tune folds, first and last included. Used for the NeuralProphet search (compute) and for the ARIMA/SARIMAX candidate orders; with the default n = 6 these are the same 6 folds
- Run for all cities by default, or one city with `--city {seoul,london,washington}`; results are saved to `--output-dir` (default `results/tuning`)
- The script prints the line to paste into the city config (e.g. `config.prophet_params_file = '...'`)
- Every params JSON contains `provenance`: `{"git_commit": str, "git_dirty": bool, "library_versions": {...}}` (see Code Provenance)
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
- Fold-level errors are logged to `errors_{version}.log` with full traceback and to W&B (`error`); `[ERROR]` is always printed regardless of `verbose`. A fold error aborts that entire model-horizon-scenario run immediately. `run_all_experiments()` catches the exception, records the failed experiment, leaves it unsaved and uncheckpointed, and continues with the remaining combinations. After all combinations have been attempted, it raises `RuntimeError` if any experiment failed, so `main.py` marks that city failed while preserving all successful completed work
- Runtime: `fit()` and `predict()` are timed per fold with `time.perf_counter()` (`fit_time_s`, `predict_time_s`, `runtime_s`) and aggregated per model, horizon and scenario (see Results Schema)

- `run_all_experiments(models, df, scenarios=None)`: `scenarios=None` uses `config.weather_scenarios`; failed model-horizon-scenario combinations are collected in `failed_experiments` and reported only after all remaining combinations have been attempted
- The module sets `warnings.filterwarnings('ignore')` globally, except for the SARIMAX "did not converge" warning (shown once per run)

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
- In neuralprophet 0.8.0 (`data/split.py`, `_make_future_dataframe`), when `n_lags > 0` the `periods` argument of `make_future_dataframe` is overwritten with `n_forecasts`. With `n_forecasts=1`, only one future row was created, so scoring the last `horizon` rows mixed 1 real forecast with `horizon - 1` in-sample fitted values, and the covariate fill overwrote training-row covariates with test-period values.
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
- The published values did not match these cities: measured temperature error SD at 168 h is 2.5–3.3 °C (literature 3.8 °C), humidity 6–11 % at 24 h (13.6 %), wind 0.7–1.0 m/s at 24 h (2.0 m/s); visibility CV is about 100 % where the data are not capped (25 %), precipitation amount CV of hits 110–170 % at 6–24 h (31–34 %), miss rate and false-alarm ratio reach 0.61–0.76 at 168 h (capped at 0.5)
- Errors are measured against the data each city's model is trained on (Seoul: the same station; London/Washington: ERA5, the source of their covariates), so the degraded covariates differ from the clean ones as a real forecast would differ from that data
- Default setting (revised 1 Oct 2026): fresh forecast and bias removed, i.e. an operator with an up-to-date, locally corrected forecast. The first version used the run available at the issue time (6–17 h old) and kept the biases. That gave (a) a forecast age fixed per city by its time zone (Seoul 14 h, London 10–11 h, Washington 15–16 h for horizons ≥ 24 h), so some hours of the day always had older forecasts and the cities differed for a reason unrelated to weather, and (b) a bias that appears only where the reference is a station (Seoul −1.5 °C; Washington's forecast is also about 1.5 °C too cold against its airport station, but its reference is ERA5). A real operator would use a regularly updated forecast corrected to local measurements. Both settings remain available
- Rain (2 Oct 2026): error rates per time of year (Washington's forecast misses 50 % of wet hours in July against 30 % in January; Seoul's monsoon), and Seoul rain measured against the station instead of ERA5. ERA5 had about twice the station's wet hours and fewer than half its downpours, so Seoul's rain errors were too small (amount error spread at 24 h: CV about 3.8 against the station vs 1.9 against ERA5)
- Replaying whole runs gives the real persistence from hour to hour and the real links between temperature, humidity and wind; independent hourly noise (literature model) understated both (Known Limitations 1 and 15)
- All degraded results produced with the literature model must be re-run; the literature model stays available (`degradation_model = "literature"`) for comparison and to reproduce earlier results
- Forecast years (2024–26) differ from the data years (2011–18); forecasts were less accurate then (Known Limitations)

### Degradation Parameters from the Training Fold
The solar cap used by the degradation is computed from the clean training fold, not from the test window being degraded.

Rationale:
- Previously the cap was the 99.5th percentile of the test window itself; for short horizons this is about the window's own maximum, which clipped upward noise and biased degraded solar radiation downward.
- Results produced before this fix are invalid and need to be re-run.

### Precipitation False Alarms from the False Alarm Ratio
The false-alarm probability for a dry hour is derived from Sukovich et al.'s false alarm ratio (FAR) with the wet-hour share *p* of the training fold: P(false alarm | dry) = FAR / (1 − FAR) · POD · p / (1 − p). Rainfall and snowfall are one precipitation variable with one event-detection draw per hour.

Rationale:
- FAR = false alarms / forecast events is conditional on a forecast event. It was used directly as the probability that a dry hour receives forecast precipitation, and false alarms were drawn separately for the rain and the snow column (both mapped to 'precipitation'). A dry test window then got forecast precipitation in 58% (6 h) to 68% (168 h) of its hours; the simulated FAR was about 0.92 instead of the cited 0.35-0.50.
- With the conversion, the simulated forecasts reproduce the cited FAR and POD (unit test `test_simulated_forecasts_reproduce_source_far_and_pod`); for 6% wet hours, P(false alarm | dry) is 1.6-3.2% (1 h to ≥60 h).
- Degraded results produced before this fix are invalid and need to be re-run.

### Float Dtype for Degraded Columns
`degrade_weather_dataset()` stores every degraded column as float.

Rationale:
- `degrade_weather_forecast()` returns the int `0` for dry hours (precipitation) and night hours (solar). In a dry test window a precipitation column therefore came out as int64; the rain/snow correction then moves float amounts into it, which pandas >= 3.0 rejects with `TypeError` (the fold fails and is skipped) and pandas 2.x accepts with a deprecation warning. Values are unchanged.

### Rain/Snow Phase Correction for All Cities
Column names come from the config (`rain_col`, `snow_col`, temperature column from `weather_degradation_mapping`).

Rationale:
- Previously the column names were hardcoded to Seoul's (`Temperature`, `Rainfall`, `Snowfall`), so the correction was silently skipped for London and Washington.

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
Seed hierarchy: `horizon_seed = base_seed + 10000 * horizon + fold_idx`

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
Forecasts are scored as each model produces them: no clipping of negative values or other post-processing, in tuning and in the experiments. TimesFM_NoWeather returns non-negative forecasts because of TimesFM's own inference setting (`infer_is_positive=True`: forecasts are floored at 0 when the whole context is non-negative), which is part of the model as published. TimesFM with covariates can return negative forecasts: TimesFM forecasts the residual of the in-context regression, which has negative values, so the floor does not apply.
<!-- TODO: check with code, please! Might be not correct -->

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
Training window is fixed-size and advances by `horizon` hours with each fold. Test set is always exactly `horizon` hours.
Ensures a consistent lookback window across all folds.

### Code Provenance
`provenance.get_code_version()` returns the git commit of the repository and whether tracked files had uncommitted changes (`git_dirty`; untracked files such as data and results are ignored). `provenance.get_library_versions()` returns the versions of Python and the forecasting libraries (numpy, pandas, scikit-learn, statsmodels, pmdarima, xgboost, optuna, prophet, neuralprophet, torch, tabpfn, tabpfn-time-series, and timesfm from `.timesfm_venv`). Tuning JSONs get both (`provenance`); every results row gets `git_commit`/`git_dirty`; aggregated rows and the W&B config also get `library_versions`.

Rationale:
- Every number in the paper maps to one code version; old params files and results can be told apart from current ones without relying on file dates.
- Paper runs must use committed code (`git_dirty = False`); a warning is printed otherwise.
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
- Rainfall → 'precipitation'
- Snowfall → 'precipitation'
- Dew point temperature → (not mapped, excluded from degradation)

**Preprocessing:**
- Sort by date
- Drop duplicate timestamps (keep first)
- Apply `column_scale_factors`
- No other filtering (imputed hours stay in the data and are used for fitting; they are excluded from scoring only)

## W&B Integration

**Run config:** `dataset`, `horizons`, `n_folds`, `n_train_samples`

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
- `degradation_model`: error model of the degraded scenarios and its settings (`config.degradation_label()`, e.g. `nwp_measured(fresh,no_bias,seasonal_rain)`, or `literature`); also in the detailed CSV
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
3. Measured model references: Seoul temperature, humidity, wind and visibility = station 47108 (the station of the Seoul data); London and Washington = ERA5 / ERA5-Land at the nearest grid cell, without the Open-Meteo height correction (their covariates are from the Open-Meteo archive but could not be reproduced exactly: median difference 0.3–0.5 °C); solar radiation and precipitation = ERA5 for all cities (no usable free station data; Seoul's covariates are station data); visibility errors from ISD station reports, which end in Aug 2025. Seoul has 844 complete runs (station gaps), London and Washington 1,817. Literature model: error growth calibrated to published statistics, the same for every city
4. NeuralProphet training cost grows with the horizon (one output per forecast step with `n_forecasts = horizon`), and the model is retrained on every `predict()` call
5. NeuralProphet hyperparameters are tuned at `config.tune_horizon` only and reused for all evaluation horizons
6. Season and holiday can only be used by SARIMAX (and NeuralProphet) in folds where they vary within the 30-day training window
7. ARIMA/SARIMAX candidate orders come from `auto_arima`'s stepwise AIC search on 6 folds; orders it does not propose are not considered
8. NeuralProphet is searched on 6 of the 90 tune folds (run time); all other tuned models are selected on all 90
9. Literature model: precipitation detection errors use POD and FAR that Sukovich et al. (2014) verified for the top 1% of 24-hour events; they are applied to all hourly precipitation. Both models: a training fold without precipitation (wet-hour share 0) gives no false alarms in that fold
10. The rain/snow phase correction uses a fixed 2 °C threshold (the real transition spans roughly 0–4 °C) and moves amounts between rain (mm) and snow (cm) without unit conversion
11. The partial last fold (h=48) covers lead times 1–24 only and is averaged with equal weight to the full folds; for NeuralProphet it is forecast by a model with `n_forecasts = 24`
12. Fold metrics are averaged with equal weight per fold; a fold with few observed hours (partly imputed test window) counts as much as a fully observed one
13. Runtimes are wall-clock times on the machine that ran the experiment: they depend on hardware, CPU/GPU availability, thread settings (e.g. XGBoost `n_jobs=-1`, TabPFN `CPUParallelWorker`) and concurrent load, so they are only comparable within one run environment. The first fold of a model can include one-off costs (library warm-up, model loading for TabPFN/TimesFM). Failed folds are not timed
14. Sensitivity scenarios are not run by default (since 2 Oct 2026). If added, they scale error magnitudes only; precipitation event detection (miss rate, false-alarm probability, false-alarm amount) and, in the measured model, whether visibility falls below the cap stay at their calibrated values at every scale
15. Literature model only: consequence of the hour-to-hour independence (Limitation 1): for models that use each step's covariates only for that step (SARIMAX, NeuralProphet, TabPFN, TimesFM), the expected error per step is unchanged and mainly the fold-to-fold spread is affected. XGBoost forecasts recursively, so correlated covariate errors could compound through the fed-back lags; the independent-noise setting may therefore understate XGBoost's degradation relative to TabPFN. The measured model has persistent errors
16. Measured model: the fresh, locally corrected forecast (default) is approximated by the global 9 km model's errors at short lead times with its average bias removed; no free archive of past local forecasts exists. Local high-resolution forecasts are usually more accurate in the first hours, so these may be slightly pessimistic. The replayed run's start hour (00/12 UTC) can differ from the issue time of day by up to 6 h. With `nwp_fresh_forecast = False`: the issue time is the end of the training window and a run is assumed usable 6 h after its start; the 06/18 UTC runs (90 h only) are not used. Beyond 90 h ECMWF outputs 3-/6-hourly values; the replayed hourly errors there are those of Open-Meteo's interpolated hourly series
17. `testing/test_max_degradation.py` applies the literature model only
18. Measured model, rain: Seoul is measured against the station (hourly amounts in wet hours; dry hours inferred from routine reports without an amount, hours with precipitation in the present-weather group but no amount left out); its seasonal values in winter rest on wide windows (up to ±120 days) because the station has few wet winter hours. The miss rate does not depend on rain intensity. By default the raw forecast's rain frequency is kept: Seoul's degraded test data get false rain in about 15 % of dry hours (London, Washington 4–5 %); option `nwp_rain_frequency_unbiased` removes this



