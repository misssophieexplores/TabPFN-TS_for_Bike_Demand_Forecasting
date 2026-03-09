# Seoul Bike Forecasting - Architecture


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
│   ├── statistical.py       # Seasonal Naive, ARIMA, SARIMAX
│   ├── ml_models.py         # XGBoost with lag features
│   ├── tabpfn_pipeline_model.py  # TabPFN pipeline models
│   ├── prophets.py          # TODO: Prophet and Neuralprophet
│   └── tuning/              # Hyperparameter tuning scripts
│       ├── tune_arima_auto.py       # Auto-ARIMA tuning
│       ├── tune_arimax_auto.py      # Auto-SARIMAX tuning
│       └── tune_xgboost.py          # XGBoost random search
├── weather/
│   ├── weather_degradation.py      # NWP forecast error simulation
│   └── weather_processor.py        # Scenario orchestration
├── evaluation/
│   ├── cv.py                # TimeSeriesCV with dynamic fold calculation
│   └── metrics.py           # MAE, RMSE, MASE, sMAPE
├── testing/
│   ├── wandb/                       # W&B artifacts and cache
│   ├── weather_partial_test/        # Partial weather test data
│   ├── tabpfn-v2-regressor-2noar4o2.ckpt  # TabPFN checkpoint
│   ├── test_experiment_runner.py
│   ├── test_full_experiment.py
│   ├── test_infrastructure.py
│   ├── test_max_degradation.py
│   ├── test_single_model.py
│   ├── test_tabpfn.py
│   ├── test_weather_all_scenarios.py
│   ├── test_weather_partial_baseline.py
│   ├── test_weather_single_model.py
│   └── test_weather_unit.py
├── run_experiments.py       # ForecastingExperiment class with W&B logging and checkpointing
└── run_weather_baseline.py  # Weather degradation baseline runner (callable programmatically)

data/
├── saving_data.py           # Data pre-processing and saving locally
├── SeoulBikeData.csv
├── LondonBikeData.csv
└── WashingtonBikeData.csv

results/                                    # Output directory 
├── figures/                 
├── tables/
├── tuning/                                 # Tuning results (JSON)
├── results_master_v4.csv                   # Aggregated results (all datasets)
├── detailed_results_master_v4.csv          # Fold-level results (all datasets)
└── checkpoint_{experiment_name}.json       # Per-run recovery checkpoints
```

## Data Flow

1. **Load**: `load_and_prepare_data()` reads CSV, parses dates, sorts by time, normalizes and appends holiday and season columns to `weather_covariates`
2. **Scenario Setup**: `WeatherProcessor` selects variables based on scenario
3. **Split**: `TimeSeriesCV.split()` creates expanding window train/test folds
4. **Weather Preparation**: Per-fold degradation (if scenario = 'degraded')
5. **Fit**: `model.fit(y_train, X_train)` trains on each fold
6. **Predict**: `model.predict(horizon, X_test)` generates forecasts
7. **Evaluate**: `MetricsCalculator.calculate_all()` computes metrics
8. **Log**: W&B logs fold-level and aggregated results
9. **Save**: Results appended to master CSVs with scenario tracking

**Weather Data Flow:**
- **all_weather**: Use all weather columns from `config.weather_covariates` as-is (not used for V3)
- **clean_only**: 6 degradable columns + holiday + season (no degradation)
- **degraded**: 6 degradable columns + holiday + season + apply degradation to degradable columns only with seed(fold_idx, horizon)

## Core Components

### Configuration (`config.py` + city configs)
- `config.py`: Shared base dataclass. Contains only fields that are identical across all datasets: `wandb_project`, `results_version`, `horizons`, `n_folds`, `n_train_samples`, `seasonal_period`, `degradation_seed`, `weather_scenarios`, `output_dir`. Dataset-specific fields default to `None`.
- `config_seoul.py`, `config_london.py`, `config_washington.py`: Each exposes a `get_config()` function that instantiates `ForecastConfig` and overrides all dataset-specific fields. To update `wandb_project` or `results_version`, change `config.py` only — all cities pick it up automatically.
- Dataset-specific fields (set per city): `data_filename`, `dataset_name`, `date_col`, `target_col`, `functioning_day_col`, `holiday_col`, `season_col`, `season_mapping`, `weather_covariates`, `weather_degradation_mapping`, `arima_params_file`, `sarimax_params_file`, `xgb_params_file`
- Horizons: [6, 24, 48, 168] hours
- Training size: 4096 observations
- Number of folds: 20
- Weather scenarios: ['all_weather', 'clean_only', 'degraded']
- Degradation seed: 42 (reproducible error simulation)
- `holiday_col`: Optional column name for public holidays (normalized to 0/1, appended to `weather_covariates` at load time)
- `season_col`: Optional column name for season (normalized to 0–3 int via `season_mapping`, appended to `weather_covariates` at load time)
- `season_mapping`: Explicit per-dataset dict mapping raw season values to 0–3 integers (handles strings, 0-based, and 1-based encodings)

### Weather Degradation (`weather/`)
**WeatherProcessor**: Orchestrates weather data preparation for scenarios
- `prepare_weather_data()`: Main entry point, applies scenario logic
- `get_weather_columns()`: Returns appropriate columns per scenario
- `degrade_dataframe()`: On-the-fly degradation with proper seeding

**Weather Scenarios:**
1. **all_weather**: All weather variables from `config.weather_covariates`, no degradation (original baseline)
2. **clean_only**: 6 degradable variables + holiday + season (excludes Dew point), no degradation
3. **degraded**: 6 degradable variables + holiday + season with realistic NWP forecast errors on degradable columns only

**Degradation Variables (6):**
- Temperature → Additive Gaussian error
- Humidity → Additive Gaussian error
- Wind speed → Additive Gaussian (reflected at 0)
- Visibility → Multiplicative lognormal error
- Solar Radiation → Heteroscedastic Gaussian error
- Rainfall → Multiplicative lognormal + event detection
- Snowfall → Multiplicative lognormal + event detection

**Non-degradable Variables (excluded from degradation, always passed through as-is):**
- Dew point temperature (excluded from `weather_degradation_mapping`, excluded from all scenarios)
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
- ARIMAForecaster: Tuned parameters via auto_arima (e.g., (2,1,2))
- SARIMAXForecaster: Tuned parameters via auto_arima (e.g., (4,0,0)×(1,0,1,24))
- XGBoostForecaster: Uses lagged features (n_lags=24) + weather covariates (including holiday and season via `weather_covariates`) + calendar time features (hour, dayofweek, month, is_weekend). `use_time_features=True` — pipeline appends calendar features automatically at fold time. **Note: XGBoost must be re-tuned whenever `weather_covariates` changes (e.g. after adding holiday/season).**
- TabPFNForecaster: Uses TabPFNv2 with calendar + auto seasonal features + weather covariates
- TabPFNForecaster_NoWeather: TabPFNv2 with calendar + auto seasonal features only (univariate)

**TabPFN Configuration:**
- Mode: LOCAL (CPU-based, no API rate limits)
- Worker: CPUParallelWorker
- Requires local source installation (see Usage Guide)

### Hyperparameter Tuning (`models/tuning/`)

**Tuning scripts:**
- `tune_arima_auto.py`: Non-seasonal ARIMA (p,d,q)
- `tune_sarimax_auto.py`: Seasonal ARIMA with exogenous variables (p,d,q)×(P,D,Q,s)
- `tune_xgboost.py`: XGBoost with lag features (n_lags + XGBoost hyperparameters)

**ARIMA/SARIMAX approach** (using pmdarima):
- Stepwise search minimizes AIC
- Validates on multiple folds
- Saves results to JSON with timestamp

**ARIMA/SARIMAX Parameters:**
- **p/P**: Autoregressive order (past values)
- **d/D**: Differencing order (trend removal)
- **q/Q**: Moving average order (error correction)
- **s**: Seasonal period (24 for hourly data)

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

**Output format (ARIMA/SARIMAX):**
```json
{
  "order": [p, d, q],
  "seasonal_order": [P, D, Q, s],
  "aic": float,
  "bic": float,
  "mae_mean": float,
  "mae_std": float,
  ...
}
```

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

**Tuning frequency:** Once per dataset. Parameters describe data structure, not forecast length.

### Cross-Validation (`evaluation/cv.py`)
**TimeSeriesCV**:
- Expanding window split
- Dynamically calculates first fold date and actual number of folds
- Tracks imputed data per fold (train_imputed, test_imputed)
- Ensures minimum training size and exact horizon test size

### Metrics (`evaluation/metrics.py`)
**MetricsCalculator**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MASE: Mean Absolute Scaled Error (seasonal naive baseline)
- sMAPE: Symmetric Mean Absolute Percentage Error

### Multi-City Orchestrator (`main.py`)
Runs all datasets sequentially without manual intervention.
- Imports each city config via `get_config()` and passes it to `run_weather_baseline.main()`
- Bypasses the interactive confirmation prompt (`no_confirm=True`)
- Catches per-city failures and continues to the next city
- Prints a summary of pass/fail at the end
- Accepts `--cities` flag to run a subset (e.g. `python main.py --cities seoul london`)

### Experiment Runner (`run_experiments.py`)
**ForecastingExperiment**:
- Manages experiment lifecycle
- W&B initialization and logging
- Checkpoint save/load for recovery — checkpoint keys include `dataset_name` so Seoul and Washington runs never conflict
- Runs all model-horizon-scenario combinations
- Saves aggregated and detailed results
- Automatic skip logic: models without covariates skip 'degraded' scenario

### Experiment Runner (`run_weather_baseline.py`)
**`main(config=None, no_confirm=False)`**: Main runner used for paper results
- Accepts an external config passed in from `main.py`, or creates a default `ForecastConfig()` when run directly — behaviour is unchanged for standalone use
- `no_confirm=True` skips the interactive prompt for non-interactive/cluster use
- Runs clean_only and degraded scenarios for all models
- All horizons: [6, 24, 48, 168] hours, 20 folds
- Loads tuned hyperparameters from JSON
- Auto-skips degraded scenario for models without covariates
- Displays degradation impact summary
- Uses ForecastingExperiment class for W&B logging, checkpointing, and result saving


## Key Design Decisions

### Weather Scenario Optimization
Models without covariates (Seasonal Naive, ARIMA, TabPFN_NoWeather) automatically skip 'degraded' scenario since they ignore weather data. This avoids redundant computation (~50% savings for these models).

Rationale: degraded = clean_only for models that don't use weather covariates.

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
First fold date calculated as: `data_start + n_train_samples`
Actual folds = `min(requested_folds, available_data // longest_horizon)`

Rationale: Different horizons require different amounts of test data. Longer horizons = fewer possible folds.

### Imputed Data Tracking
Uses existing `Functioning Day` column (No = imputed)
Tracked per fold: train_imputed, test_imputed
Logged in detailed results for post-hoc analysis

<!-- TODO: change to rolling windows for consistency with static lookback window implementation for TabPFN -->
### Expanding Window CV
Training set grows with each fold, test set is always exactly `horizon` hours.
Ensures models see increasing historical context.

### Checkpoint Recovery
Saves completed `(dataset_name, model, horizon, scenario)` tuples to JSON after each experiment.
On restart, skips already-completed experiments.
Prevents data loss from crashes during long runs — including mid-run failures when iterating over multiple datasets.
File format: `checkpoint_{experiment_name}.json` in `results/` directory.

## Data Requirements

**Input CSV must have:**
- `Date`: Datetime column (hourly frequency)
- `Rented Bike Count`: Target variable
- Weather covariates: Temperature, Humidity, Wind speed, Visibility, Dew point temperature, Solar Radiation, Rainfall, Snowfall
- `Functioning Day`: Yes/No (tracks imputed data)

**Weather Column Mapping:**
Dataset columns mapped to degradation variable types via `config.weather_degradation_mapping`:
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
- No filtering (includes imputed hours)
- 8760 total observations (Dec 2017 - Nov 2018)

## W&B Integration

**Logged per fold:**
- {model}_{scenario}_h{horizon}_fold{idx}_MAE/RMSE/MASE

**Logged per experiment:**
- {model}_{scenario}_h{horizon}_MAE/RMSE/MASE/sMAPE (aggregated)

**Fold-level tables:**
- {model}_{scenario}_h{horizon}_folds (detailed per-fold results)

**Summary tables:**
- results_table (aggregated results across all experiments)
- results_summary (grouped by model and scenario)

**Artifacts:**
- results_master.csv (aggregated metrics)
- detailed_results_master.csv (fold-level metrics)

**Projects:**
- Production: `seoul-bike-forecasting`
- Testing: `seoul-bike-testing`

## Results Schema

**New columns in results_master.csv:**
- `weather_scenario`: 'all_weather' (only used in Pilot project), 'clean_only', or 'degraded'
- `model_uses_covariates`: Boolean (from model.use_covariates property)
- `degradation_seed`: Random seed used (42 by default)
- `num_weather_vars`: Count of weather variables passed to model (0 for models without covariates; for models with covariates: 6 degradable + holiday + season = 8, plus 4 calendar time features for XGBoost = 12 total features; TabPFN creates its own calendar features, which is 17 additional features)


## Known Limitations

1. Weather degradation is applied fixed to all weather data and does not dynamically adapt within horizons.
2. Degradation assumes independent errors across variables (no cross-correlation)
3. Error growth calibrated to published statistics (ECMWF, KMA); may differ for specific locations