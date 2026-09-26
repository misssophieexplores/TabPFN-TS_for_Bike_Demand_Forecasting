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
├── models/
│   ├── base.py              # BaseForecaster abstract class
│   ├── statistical.py       # Seasonal Naive, ARIMA, SARIMAX, trend_from_intercept()
│   ├── ml_models.py         # XGBoost with lag features
│   ├── tabpfn_pipeline_model.py  # TabPFN pipeline models
│   ├── prophet_models.py    # Prophet, NeuralProphet, NeuralProphet_NoWeather
│   ├── timesfm_model.py     # TimesFMForecaster, TimesFMForecaster_NoWeather
│   └── tuning/              # Hyperparameter tuning scripts
│       ├── tune_arima.py            # Auto-ARIMA tuning (pmdarima)
│       ├── tune_sarimax.py          # Auto-SARIMAX tuning (pmdarima)
│       ├── tune_xgboost.py          # XGBoost random search
│       ├── tune_prophet.py          # Prophet random search
│       ├── tune_neuralprophet.py    # NeuralProphet random search (full)
│       └── tune_neuralprophet_fast.py  # NeuralProphet random search (speed-optimized for cluster limits)
├── weather/
│   ├── weather_degradation.py      # NWP forecast error simulation
│   └── weather_processor.py        # Scenario orchestration
├── evaluation/
│   ├── cv.py                # TimeSeriesCV with dynamic fold calculation
│   └── metrics.py           # MAE, RMSE, MASE, sMAPE
├── testing/
│   ├── test_max_degradation.py
│   ├── test_weather_single_model.py
│   └── test_weather_unit.py
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
├── checkpoint_{dataset_name}_{version}.json       # Per-run recovery checkpoints (file name = checkpoint_{experiment_name}.json)
└── errors_{version}.log                           # Fold-, city- and run-level error tracebacks
```

All scripts resolve `data/` and `results/` relative to the current working directory (`Path("data") / config.data_filename`), so run them from the repository root, e.g. `python forecasting/main.py`.

## Data Flow

1. **Load**: `load_and_prepare_data()` reads CSV, parses dates, sorts by time, drops duplicate timestamps (e.g. DST clock-back hours, keeps first), applies `config.column_scale_factors`, normalizes holiday and season columns and appends them to `weather_covariates` if not already listed
2. **Scenario Setup**: `WeatherProcessor` selects variables based on scenario
3. **Split**: `TimeSeriesCV.split()` creates rolling window train/test folds
4. **Weather Preparation**: Per-fold weather preparation via `WeatherProcessor.prepare_weather_data(split=...)`. Training data always uses clean observed weather. In the 'degraded' scenario the training fold also provides the degradation parameters (solar cap). For the 'degraded' scenario, test data receives per-row lead-time noise: row *i* is degraded using lead time *(i + 1)* hours, so error grows from near-zero at the first step up to full-horizon noise at the last step.
5. **Model inputs**: models with `use_time_features=True` get calendar features appended (`prepare_xgboost_features`); models with `needs_datetime=True` get a real `DatetimeIndex` on `X_train`/`X_test` (an empty DataFrame with that index if the model has no covariates)
6. **Fit**: `model.reset()`, then `model.fit(y_train, X_train)` on each fold (NeuralProphet only stores the data here — see Models)
7. **Predict**: `model.predict(horizon, X_test)` generates forecasts
8. **Evaluate**: `MetricsCalculator.calculate_all(y_test, y_pred, y_train)` computes metrics
9. **Log**: W&B logs aggregated metrics and a per-fold table
10. **Save**: Aggregated row appended to `results_master_{version}.csv` after each model-horizon-scenario run; fold-level rows written to `detailed_results_master_{version}.csv` at the end of the run
11. **Compare** (once, after all cities): `compute_and_log_comparative_metrics()` computes win rate and skill score vs `Seasonal_Naive`, pooled across cities

**Weather Data Flow:**
- **all_weather**: Use all weather columns from `config.weather_covariates` as-is (in `config.weather_scenarios`, but not run by `run_weather_baseline.py`)
- **clean_only**: 7 degradable columns + holiday + season (no degradation)
- **degraded**: 7 degradable columns + holiday + season + apply degradation to degradable columns only with seed(fold_idx, horizon). Degradation applied to **test split only** (training always uses clean observed weather). Each test row receives noise scaled to its own lead time (row *i* → lead time *i + 1* hours).

## Core Components

### Configuration (`config.py` + city configs)
- `config.py`: Shared base dataclass. Contains the fields that are identical across all datasets: `wandb_project`, `results_version`, `horizons`, `n_folds`, `n_train_samples`, `seasonal_period`, `degradation_seed`, `weather_scenarios`, `output_dir`, `verbose`, `experiment_name`, `tune_folds`, `tune_horizon`. Dataset-specific fields default to `None` (except `holiday_mapping` and `column_scale_factors`, see below).
- `config_seoul.py`, `config_london.py`, `config_washington.py`: Each exposes a `get_config()` function that instantiates `ForecastConfig` and overrides all dataset-specific fields. To update `wandb_project` or `results_version`, change `config.py` only — all cities pick it up automatically.
- Dataset-specific fields (set per city): `data_filename`, `dataset_name`, `date_col`, `target_col`, `functioning_day_col`, `holiday_col`, `holiday_mapping`, `season_col`, `season_mapping`, `weather_covariates`, `weather_degradation_mapping`, `rain_col`, `snow_col`, `column_scale_factors`, `arima_params_file`, `sarimax_params_file`, `xgb_params_file`, `prophet_params_file`, `neuralprophet_params_file`, `neuralprophet_noweather_params_file`
- Horizons: [6, 24, 48, 168] hours
- Training size (`n_train_samples`): 720 observations (30 days). The previous setting (4096 observations, 20 folds) is kept commented out in `config.py`.
- Number of folds (`n_folds`): 35
- Seasonal period: 24
- Results version: `v6`
- W&B project: `bike-forecasting`
- Weather scenarios: ['all_weather', 'clean_only', 'degraded']
- Degradation seed: 42 (reproducible error simulation)
- `experiment_name`: Defaults to `{dataset_name}_{results_version}` (set in `__post_init__`; re-set in `run_weather_baseline.main()` if it starts with `None`)
- `tune_horizon`: Horizon used by all tuning scripts (24)
- `tune_folds`: Number of (last) CV folds used by the tuning scripts (90; `None` = all folds)
- `holiday_col`: Optional column name for public holidays (normalized to 0/1, appended to `weather_covariates` at load time)
- `holiday_mapping`: Dict mapping raw holiday string values → 0/1. Default `{'Yes': 1, 'No': 0}`; Seoul overrides with `{'Holiday': 1, 'No Holiday': 0}`. Only used when `holiday_col` has string (object) dtype; numeric holiday columns are coerced to int (NaN → 0).
- `column_scale_factors`: Dict `{column: factor}`; each listed column is multiplied by its factor at load time. Keys must match column names exactly (case-sensitive); non-matching keys are silently ignored. Default `{}`; Seoul sets `{"Visibility": 0.01}` (raw unit 10 m → km, the unit used for London and Washington).
- `rain_col`, `snow_col`: Rainfall and snowfall columns used by the rain/snow phase correction in the 'degraded' scenario (Seoul: `Rainfall`/`Snowfall`; London and Washington: `rainfall_mm`/`snowfall_cm`). Required for 'degraded'.
- `season_col`: Optional column name for season (normalized to 0–3 int via `season_mapping`, appended to `weather_covariates` at load time)
- `season_mapping`: Explicit per-dataset dict mapping raw season values to 0–3 integers (handles strings, 0-based, and 1-based encodings)
- `verbose`: If `True`, prints detailed progress (CV info, data loading, W&B URLs). Default `False` in `config.py` (cluster/server runs where stdout is captured in SLURM logs).
- Params files: each city config points to the tuned-parameter JSON files in `results/tuning/` (all tuned with `n_train_samples=720`). ARIMA and SARIMAX params files must contain `with_intercept`, i.e. they must come from the current tuning scripts; older files are rejected by `run_weather_baseline.py`. `neuralprophet_noweather_params_file` must be set (tuned with `--scenario no_weather`); it is `None` until then and the runners raise `ValueError`.

### Weather Degradation (`weather/`)
**WeatherProcessor**: Orchestrates weather data preparation for scenarios
- `prepare_weather_data(split='train'|'test')`: Main entry point, applies scenario logic. Training split always returns clean weather; test split applies degradation with per-row lead times for the 'degraded' scenario. In 'degraded', the train split also computes `degradation_params` (solar cap = 99.5th percentile of the clean training fold), which the test split then uses; the train split must be prepared first (otherwise `RuntimeError`).
- `get_weather_columns()`: Returns appropriate columns per scenario
- `degrade_dataframe()`: On-the-fly degradation with proper seeding, followed by the rain/snow phase correction (temperature column taken from `weather_degradation_mapping`, rain/snow columns from `config.rain_col`/`config.snow_col`; raises `ValueError` if they are not set)

**Weather Scenarios:**
1. **all_weather**: All weather variables from `config.weather_covariates`, no degradation (original baseline)
2. **clean_only**: 7 degradable variables + holiday + season (excludes Dew point), no degradation
3. **degraded**: 7 degradable variables + holiday + season with realistic NWP forecast errors on degradable columns only

**Degradation Variables (7, all mapped in `weather_degradation_mapping` for Seoul, London and Washington):**
- Temperature → Additive Gaussian error
- Humidity → Additive Gaussian error, clipped to [0, 100]
- Wind speed → Additive Gaussian, truncated at 0
- Solar Radiation → Heteroscedastic Gaussian error (σ ∝ value, so unit-independent), capped at the solar cap from the training fold
- Rainfall → Multiplicative lognormal + event detection
- Snowfall → Multiplicative lognormal + event detection
- Visibility → Multiplicative lognormal, constant CV = 25% (horizon-independent, unit-independent)

**Rain/snow phase correction** (all three cities, after degradation): precipitation is reassigned by the degraded temperature: above 2 °C snow is moved to rain, below 2 °C rain is moved to snow. Amounts are moved without unit conversion (rain in mm, snow in cm).

**Non-degradable Variables (excluded from degradation, always passed through as-is):**
- Dew point temperature (excluded from `weather_degradation_mapping`; only included in `all_weather`, via `weather_covariates`)
- Holiday (`config.holiday_col`) — passed through in clean_only and degraded
- Season (`config.season_col`) — passed through in clean_only and degraded

**Error Growth:** Calibrated to verification statistics
- 6h: Small errors
- 24h: Moderate errors
- 48h: Larger errors
- 168h: Substantial errors

**Reproducibility:**
- Seed management: `horizon_seed = (base_seed + fold_idx) + horizon`
- Same seed → identical degradation
- Different folds → different realistic errors

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
- TabPFNPipelineForecaster: Uses TabPFNv2 with calendar + auto seasonal features + weather covariates
- TabPFNPipelineForecaster_NoWeather: TabPFNv2 with calendar + auto seasonal features only (univariate)
- TimesFMForecaster / TimesFMForecaster_NoWeather (`timesfm_model.py`): TimesFM, with and without weather covariates — TODO: document
- ProphetForecaster: Univariate Prophet (`use_covariates=False`). Daily, weekly and yearly seasonality on; defaults `seasonality_mode="multiplicative"`, `changepoint_prior_scale=0.05`, `seasonality_prior_scale=10.0`, `holidays_prior_scale=10.0`. Forecast timestamps come from `X_future.index` when available, otherwise an hourly range starting one hour after the last training timestamp.
- NeuralProphetForecaster: NeuralProphet with weather covariates (`use_covariates=True`). Every covariate column is added as a **lagged regressor** (`add_lagged_regressor`); columns that NeuralProphet silently drops (e.g. all-constant) are removed from the covariate list after fitting. Default `n_lags=24`, `seasonality_mode="multiplicative"`, all seasonalities on, `epochs=None` (NeuralProphet picks based on data size).
- NeuralProphetForecaster_NoWeather: Same as above, univariate (no covariates).

**Model list used for paper results** (`run_weather_baseline.py`; `run_experiments.main()` builds the same list): Seasonal Naive (`seasonal_period`), ARIMA and SARIMAX (orders and `with_intercept` from their params files, translated to a statsmodels `trend`), XGBoost (`n_lags` + `xgb_params`), Prophet (`prophet_params`), NeuralProphet (`n_lags` + `neuralprophet_params` from `neuralprophet_params_file`), NeuralProphet_NoWeather (same keys from its own `neuralprophet_noweather_params_file`), TabPFN and TabPFN_NoWeather, TimesFM and TimesFM_NoWeather (default arguments).

**ARIMA / SARIMAX specifics (`statistical.py`):**
- Tuning uses pmdarima, the experiments use statsmodels (`ARIMA`, `SARIMAX`). The two handle the constant differently: pmdarima's `with_intercept` is chosen during the search, statsmodels `SARIMAX` adds no constant unless `trend` is set, and statsmodels `ARIMA` adds one by default only when d = 0.
- Both forecasters therefore take a `trend` argument, and `trend_from_intercept(with_intercept, order, seasonal_order)` maps the tuned intercept to it:
  - no intercept → `"n"` (passed explicitly, so ARIMA does not add its default constant)
  - intercept, d + D = 0 → `"c"` (constant)
  - intercept, d + D = 1 → `"t"` (drift; a constant in the differenced series)
  - intercept, d + D ≥ 2 → `ValueError`
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
- Mode: LOCAL (CPU-based, no API rate limits)
- Worker: CPUParallelWorker
- Requires local source installation (see Usage Guide)

### Hyperparameter Tuning (`models/tuning/`)

**Tuning scripts:**
- `tune_arima.py`: Non-seasonal ARIMA (p,d,q)
- `tune_sarimax.py`: Seasonal ARIMA with exogenous variables (p,d,q)×(P,D,Q,s)
- `tune_xgboost.py`: XGBoost with lag features (n_lags + XGBoost hyperparameters)
- `tune_prophet.py`: Prophet (changepoint / seasonality / holidays prior scales + seasonality mode)
- `tune_neuralprophet.py`: NeuralProphet (learning_rate + n_lags), full search
- `tune_neuralprophet_fast.py`: NeuralProphet, speed-optimized variant of the above

**Common to the ARIMA, SARIMAX, Prophet and NeuralProphet tuning scripts:**
- Tune only on data up to `TimeSeriesCV.get_cutoff_date()` — the held-out test period is never touched
- CV splits are built with `config.tune_horizon`; `config.tune_folds` selects the **last** N folds (`None` = all folds)
- Run for all cities by default, or one city with `--city {seoul,london,washington}`; results are saved to `--output-dir` (default `results/tuning`)
- The script prints the line to paste into the city config (e.g. `config.prophet_params_file = '...'`)

**Common to the Prophet and NeuralProphet tuning scripts:**
- Random search optimizing mean MAE across folds; the best parameters are then re-evaluated fold by fold for the reported `mae_mean`/`mae_std`/`rmse_mean`/`rmse_std`
- Parameter sampling uses `np.random.default_rng(seed)` (`--seed`, default 42)
- A failed fold aborts that trial (traceback printed) and the search continues

**ARIMA/SARIMAX approach** (using pmdarima `auto_arima`):
- Stepwise search minimizing AIC, run once on the **first** CV split (`splits[0]`) of the pre-cutoff data
- ARIMA: non-seasonal, `max_p=7`, `max_q=3`, `max_order=8`, no covariates
- SARIMAX: seasonal with `m = --seasonal-period` (default 24), `max_p=5`, `max_q=3`, `max_P=2`, `max_Q=2`, `max_order=8`; covariates chosen by `--scenario` (`clean_only` default: keys of `weather_degradation_mapping` + holiday + season; `all_weather`: all of `config.weather_covariates`), passed as `X=` (in search, fold fits and `predict`); covariates constant in the respective training window are dropped (same rule as `SARIMAXForecaster`)
- Intercept: `with_intercept` is left at pmdarima's default (`'auto'`) during the search; the selected value is saved to the JSON and fixed for the validation folds
- Validation: on each of the last `tune_folds` folds, the found order and intercept are refit with pmdarima and scored on `tune_horizon` steps; failed folds are skipped (SARIMAX prints the traceback, ARIMA a short `FAILED` line)
- `--scenario` is also accepted by `tune_arima.py` but only labels the output file name; ARIMA uses no covariates

**ARIMA/SARIMAX Parameters:**
- **p/P**: Autoregressive order (past values)
- **d/D**: Differencing order (trend removal)
- **q/Q**: Moving average order (error correction)
- **s**: Seasonal period (24 for hourly data)
- **with_intercept**: Whether the model includes a constant (drift if differenced)

**XGBoost approach**:
- Wide random search optimizing MAE
- Jointly tunes n_lags and XGBoost hyperparameters
- Uses tune_folds for hyperparameter search, additional folds for validation
- n_lags options: [12, 24, 48, 168]

**XGBoost Parameters:**
- **n_lags**: Number of lagged target values as features
- **n_estimators**: Number of boosting trees
- **learning_rate**: Step size shrinkage
- **max_depth**: Maximum tree depth
- **min_child_weight**: Minimum sum of instance weight in child
- **subsample**: Fraction of samples for tree training
- **colsample_bytree**: Fraction of features for tree training
- **gamma**: Minimum loss reduction for split
- **reg_lambda**: L2 regularization
- **reg_alpha**: L1 regularization

**Prophet approach** (`tune_prophet.py`):
- Random search, default 50 trials (`--trials`)
- Daily, weekly and yearly seasonality always on
- Validation re-uses the same `tune_folds` folds as the search

**Prophet Parameters:**
- **changepoint_prior_scale**: log-uniform in [0.001, 0.5] — trend flexibility
- **seasonality_prior_scale**: log-uniform in [0.01, 10] — seasonality strength
- **holidays_prior_scale**: log-uniform in [0.01, 10] — only has an effect if holidays are added to the model (currently not); kept for forward-compatibility
- **seasonality_mode**: `multiplicative` or `additive`

**NeuralProphet approach** (`tune_neuralprophet.py`):
- Random search, default 30 trials (NeuralProphet is slow)
- Jointly tunes `learning_rate` and `n_lags`; `n_lags` options default to [12, 24, 48, 168] (`--n-lags-options`)
- Model: `n_forecasts = config.tune_horizon` (direct multi-step, same as the forecaster), multiplicative seasonality, all seasonalities on, `epochs=None`
- Covariates chosen by `--scenario`, added as lagged regressors:
  - `clean_only` (default): keys of `weather_degradation_mapping` + holiday + season
  - `all_weather`: all of `config.weather_covariates`
  - `no_weather`: no covariates; produces the params file for NeuralProphet_NoWeather (`config.neuralprophet_noweather_params_file`)
- `neuralprophet.set_random_seed(42)` before every model is created
- Validation re-uses the same `tune_folds` folds as the search

**NeuralProphet fast variant** (`tune_neuralprophet_fast.py`) — same model setup and covariates, with these speed-ups:
- Default 20 trials; `n_lags` options default to [24, 48], sampled with weight ∝ 1/√n_lags (favours cheaper models)
- Search phase: epochs capped (`--search-epochs`, default 50), `daily_seasonality` off, only the last `--search-folds` folds (default 2), optional `--max-train-rows` window per fold
- Validation phase: full epochs (`epochs=None`), `daily_seasonality` on, full training data, on the `tune_folds` folds (all folds if `None`)

**NeuralProphet Parameters:**
- **learning_rate**: log-uniform in [1e-4, 0.1]
- **n_lags**: Number of past target values fed to the autoregressive part (also the lag window for lagged regressors)

**Note on horizons:** NeuralProphet is tuned at a single horizon (`config.tune_horizon`), but at evaluation time `n_forecasts` equals each experiment horizon. The tuned `learning_rate`/`n_lags` are therefore reused for horizons they were not tuned on.

**Output format (ARIMA/SARIMAX)** — `arima_best_params_{city}_{scenario}_{n_train_samples}_{timestamp}.json` / `sarimax_best_params_{city}_{scenario}_{n_train_samples}_{timestamp}.json`:
```json
{
  "city": str,
  "scenario": "clean_only" | "all_weather",
  "n_train_samples": int,
  "order": [p, d, q],
  "seasonal_order": [P, D, Q, s],
  "with_intercept": bool,
  "aic": float,
  "bic": float,
  "mae_mean": float,
  "mae_std": float,
  "rmse_mean": float,
  "rmse_std": float,
  "validation_folds": int,
  "covariates_used": [str],
  "m": int
}
```
`seasonal_order`, `covariates_used` and `m` are SARIMAX only. `with_intercept` is required by `run_weather_baseline.py`.

**Output format (XGBoost):**
```json
{
  "n_lags": int,
  "xgb_params": {
    "n_estimators": int,
    "learning_rate": float,
    "max_depth": int,
    ...
  },
  "tuning": {
    "search_type": "wide_random_search_no_early_stopping_joint_n_lags",
    "trials": int,
    "tune_folds": int,
    "metric_optimized": "MAE",
    ...
  },
  "mae_mean": float,
  "mae_std": float,
  ...
}
```

**Output format (Prophet)** — `prophet_best_params_{city}_{n_train_samples}_{timestamp}.json`:
```json
{
  "city": str,
  "n_train_samples": int,
  "prophet_params": {
    "changepoint_prior_scale": float,
    "seasonality_prior_scale": float,
    "holidays_prior_scale": float,
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

**Output format (NeuralProphet)** — `neuralprophet_best_params_{city}_{scenario}_{n_train_samples}_{timestamp}.json` (same file name pattern for both scripts):
```json
{
  "city": str,
  "scenario": "clean_only" | "all_weather" | "no_weather",
  "n_train_samples": int,
  "n_lags": int,
  "neuralprophet_params": { "learning_rate": float },
  "tuning": {
    "search_type": "random_search_joint_n_lags" | "random_search_fast",
    "trials": int,
    "seed": int,
    "metric_optimized": "MAE",
    "best_tune_mae_mean": float,
    "best_tune_rmse_mean": float,
    "n_lags_options": [int],
    ...
  },
  "mae_mean": float,
  "mae_std": float,
  "rmse_mean": float,
  "rmse_std": float,
  "validation_folds": int,
  "covariates_used": [str]
}
```
The full script adds `tune_folds` to `tuning`; the fast script adds `search_folds`, `search_epochs_cap`, `validation_folds` and `max_train_rows`.

**Tuning frequency:** Once per dataset. Parameters describe data structure, not forecast length.

### Cross-Validation (`evaluation/cv.py`)
**TimeSeriesCV**:
- Rolling window split — fixed-size training window advances by `horizon` hours with each fold
- Dynamically calculates first fold date and actual number of folds
- Tracks imputed data per fold (train_imputed, test_imputed)
- Ensures minimum training size and exact horizon test size
- `get_cutoff_date(df)`: start of the held-out test period; tuning scripts only use data up to this date

### Metrics (`evaluation/metrics.py`)
**MetricsCalculator**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MASE: Mean Absolute Scaled Error (seasonal naive baseline)
- sMAPE: Symmetric Mean Absolute Percentage Error
- `compute_and_save_comparative_metrics(...)`: comparative metrics (win rate, skill score, with CIs) computed from the aggregated results file — called via `run_experiments.compute_and_log_comparative_metrics()`

### Multi-City Orchestrator (`main.py`)
Runs all datasets sequentially without manual intervention.
- Imports each city config via `get_config()` and passes it to `run_weather_baseline.main()`
- Bypasses the interactive confirmation prompt (`no_confirm=True`)
- Catches per-city failures, writes full traceback to `errors_{version}.log` (path taken from the first selected city's config), and continues to the next city
- After all cities: if at least one city succeeded, computes comparative metrics once, pooled across cities (`compute_and_log_comparative_metrics(first_config, log_wandb=False)`), and prints `model`, `n_tasks`, `win_rate`, `skill_score`
- Prints timestamped `STARTING`, `[OK]`, and `[FAILED]` lines to stdout; prints a pass/fail summary and total wall time at the end
- Exits with code 1 if any city failed
- Accepts `--cities` flag to run a subset (e.g. `python forecasting/main.py --cities seoul london`); default order: seoul, washington, london

### Experiment Runner (`run_experiments.py`)
**ForecastingExperiment**:
- Manages experiment lifecycle
- W&B initialization (`entity` from the `WANDB_ENTITY` environment variable, loaded from `.env`) and logging
- Checkpoint save/load for recovery — checkpoint keys include `dataset_name` so Seoul and Washington runs never conflict
- Runs all model-horizon-scenario combinations
- Saves aggregated and detailed results; on resume, reloads previously completed results from `results_master_{version}.csv`
- `save_results()` merges with the existing aggregated CSV and drops duplicates on (`dataset`, `model`, `horizon`, `weather_scenario`, `run_name`), keeping the latest; the detailed CSV is appended
- Automatic skip logic: models with `use_covariates=False` skip the 'degraded' scenario
- Fold-level errors logged to `errors_{version}.log` with full traceback and to W&B (`error`); `[ERROR]` line always printed to stdout regardless of `verbose`; the fold is skipped and the run continues

- `run_all_experiments(scenarios=None)` uses `config.weather_scenarios`
- The module sets `warnings.filterwarnings('ignore')` globally, except for the SARIMAX "did not converge" warning (shown once per run)

**`main(config=None)`**: run directly with `python forecasting/run_experiments.py --city {seoul,washington,london}`. Builds the same model list as `run_weather_baseline.py` but runs all of `config.weather_scenarios` (including `all_weather`) under the experiment name `baseline_models_{version}`.

**`load_and_prepare_data(config)`**: see Data Flow step 1. Returns `(df, dataset_name)`, where `dataset_name` is the CSV file stem. Note: it appends holiday/season to `config.weather_covariates` in place (only if not already listed).

**`compute_and_log_comparative_metrics(config, log_wandb=True)`**: tasks are (`dataset`, `horizon`, `weather_scenario`), pooled across all datasets in `results_master_{version}.csv`; baseline model `Seasonal_Naive`. Run once after all cities finish.

### Experiment Runner (`run_weather_baseline.py`)
**`main(config=None, no_confirm=False)`**: Main runner used for paper results
- Accepts an external config passed in from `main.py`; falls back to a default `ForecastConfig()` if `config` is `None`
- Run directly: `python forecasting/run_weather_baseline.py --city {seoul,washington,london}` (`--city` is required)
- `no_confirm=True` skips the interactive prompt for non-interactive/cluster use (direct runs ask for confirmation)
- Runs clean_only and degraded scenarios for all models (`all_weather` is commented out)
- All horizons: [6, 24, 48, 168] hours, 35 folds
- Loads tuned hyperparameters from the JSON files named in the city config (ARIMA, SARIMAX, XGBoost, Prophet, NeuralProphet, NeuralProphet_NoWeather)
- ARIMA/SARIMAX: `trend` is derived from `with_intercept` and the orders via `trend_from_intercept()`; a params file without `with_intercept` raises `KeyError` (re-run tuning)
- Auto-skips degraded scenario for models without covariates
- Displays degradation impact summary (gated behind `config.verbose`)
- Uses ForecastingExperiment class for W&B logging, checkpointing, and result saving
- Errors written to `errors_{version}.log` with full traceback before re-raising


## Key Design Decisions

### Weather Scenario Optimization
Models without covariates (`use_covariates=False`, e.g. Seasonal Naive, ARIMA, Prophet, NeuralProphet_NoWeather, TabPFN_NoWeather) automatically skip 'degraded' scenario since they ignore weather data. This avoids redundant computation (~50% savings for these models).

Rationale: degraded = clean_only for models that don't use weather covariates.

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

### Degradation Parameters from the Training Fold
The solar cap used by the degradation is computed from the clean training fold, not from the test window being degraded.

Rationale:
- Previously the cap was the 99.5th percentile of the test window itself; for short horizons this is about the window's own maximum, which clipped upward noise and biased degraded solar radiation downward.
- Results produced before this fix are invalid and need to be re-run.

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
Seed hierarchy: `horizon_seed = (base_seed + fold_idx) + horizon`

Rationale:
- Full reproducibility across runs
- Independent errors for different horizons
- Different errors per fold (realistic variability)

### Dynamic CV Fold Calculation
First fold cutoff counted back from end of dataset: `data_end - (n_folds * max_horizon)`
Actual folds = `min(requested_folds, available_data // longest_horizon)`

Rationale: Different horizons require different amounts of test data. Longer horizons = fewer possible folds.

### Imputed Data Tracking
Uses the `Functioning Day` column (No = imputed), configured via `functioning_day_col` for all three cities
Tracked per fold: train_imputed, test_imputed
Logged in detailed results for post-hoc analysis

### Rolling Window CV
Training window is fixed-size and advances by `horizon` hours with each fold. Test set is always exactly `horizon` hours.
Ensures a consistent lookback window across all folds.

### Checkpoint Recovery
Saves completed `(dataset_name, model, horizon, scenario)` tuples to JSON after each experiment.
On restart, skips already-completed experiments.
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
- No other filtering (includes imputed hours where present)

## W&B Integration

**Run config:** `dataset`, `horizons`, `n_folds`, `n_train_samples`

**Logged per experiment:**
- {model}_{scenario}_h{horizon}_MAE/RMSE/MASE/sMAPE (aggregated)

**Fold-level tables:**
- {model}_{scenario}_h{horizon}_folds (detailed per-fold results)

**Summary tables:**
- results_table (aggregated results across all experiments)
- results_summary (grouped by model and scenario)

**Errors:**
- `error` (message for each failed fold)

**Comparative metrics** (only when `compute_and_log_comparative_metrics(..., log_wandb=True)`; `main.py` uses `False`):
- `comparative_metrics` table
- {model}_vs_{baseline}_win_rate / _skill_score, each with _ci_lower / _ci_upper

**Projects:**
- Production: `bike-forecasting`
- Testing: `bike-forecasting-testing`

## Results Schema

**Columns in results_master_{version}.csv:**
- `dataset`, `run_name`, `timestamp`, `model`, `horizon`, `n_folds`
- `MAE_mean`/`MAE_std`, `RMSE_mean`/`RMSE_std`, `MASE_mean`/`MASE_std`, `sMAPE_mean`/`sMAPE_std`
- `total_test_imputed`, `total_train_imputed`, `folds_with_imputed_test`
- `version`: Results version string (from `config.results_version`), stored as a separate column alongside `run_name`
- `weather_scenario`: 'all_weather' (only used in Pilot project), 'clean_only', or 'degraded'
- `model_uses_covariates`: Boolean (from model.use_covariates property)
- `degradation_seed`: Random seed used (42 by default)
- `num_weather_vars`: Number of columns in `X_train` (taken from the first fold) (0 for models without covariates; for models with covariates: 7 degradable + holiday + season = 9, plus 4 calendar time features for XGBoost = 13 total features; TabPFN creates its own calendar features, which is 17 additional features)


## Known Limitations


1. Degradation assumes independent errors across variables (no cross-correlation)
2. Error growth calibrated to published statistics (ECMWF, KMA); may differ for specific locations
3. NeuralProphet training cost grows with the horizon (one output per forecast step with `n_forecasts = horizon`), and the model is retrained on every `predict()` call
4. NeuralProphet hyperparameters are tuned at `config.tune_horizon` only and reused for all evaluation horizons
5. Season and holiday can only be used by SARIMAX (and NeuralProphet) in folds where they vary within the 30-day training window
6. ARIMA/SARIMAX orders are searched on the first CV split only
7. ARIMA/SARIMAX tuning-time validation metrics (`mae_mean` etc. in the params files) come from pmdarima fits, not from `ARIMAForecaster`/`SARIMAXForecaster` (e.g. pmdarima enforces stationarity/invertibility, `SARIMAXForecaster` does not), so they are not directly comparable to experiment results
8. The rain/snow phase correction uses a fixed 2 °C threshold (the real transition spans roughly 0–4 °C) and moves amounts between rain (mm) and snow (cm) without unit conversion
