# Seoul Bike Forecasting - Architecture
<!-- TODO: check if updates are needed! -->

## Project Structure
```
forecasting/
├── config.py                 # Configuration dataclass (includes weather scenarios)
├── models/
│   ├── base.py              # BaseForecaster abstract class
│   ├── statistical.py       # Seasonal Naive, ARIMA, SARIMAX
│   ├── ml_models.py         # XGBoost with lag features
│   ├── tabpfn_model.py      # TabPFN-based models
│   ├── prophets.py          # TODO: Prophet and Neuralprophet
│   └── tuning/              # Hyperparameter tuning scripts
│       ├── tune_arima_auto.py       # Auto-ARIMA tuning
│       └── tune_sarimax_auto.py     # Auto-SARIMAX tuning
├── weather/
│   ├── __init__.py          # Package initialization
│   ├── weather_degradation.py      # NWP forecast error simulation
│   └── weather_processor.py        # Scenario orchestration
├── evaluation/
│   ├── cv.py                # TimeSeriesCV with dynamic fold calculation
│   └── metrics.py           # MAE, RMSE, MASE, sMAPE
├── testing/
│   ├── test_infrastructure.py
│   ├── test_single_model.py
│   ├── test_experiment_runner.py
│   ├── test_full_experiment.py
│   ├── test_weather_unit.py         # Weather degradation unit tests
│   ├── test_weather_single_model.py # Single model scenario tests
│   └── test_weather_all_scenarios.py # Cross-scenario validation
└── run_weather_baseline.py  # Weather degradation baseline runner

results/                     # Output directory (project root)
├── results_master.csv       # Aggregated results
├── detailed_results_master.csv  # Fold-level results
├── checkpoint_*.json        # Recovery checkpoints
└── tuning/                  # Tuning results (JSON)
```

## Data Flow

1. **Load**: `load_and_prepare_data()` reads CSV, parses dates, sorts by time
2. **Scenario Setup**: `WeatherProcessor` selects variables based on scenario
3. **Split**: `TimeSeriesCV.split()` creates expanding window train/test folds
4. **Weather Preparation**: Per-fold degradation (if scenario = 'degraded')
5. **Fit**: `model.fit(y_train, X_train)` trains on each fold
6. **Predict**: `model.predict(horizon, X_test)` generates forecasts
7. **Evaluate**: `MetricsCalculator.calculate_all()` computes metrics
8. **Log**: W&B logs fold-level and aggregated results
9. **Save**: Results appended to master CSVs with scenario tracking

**Weather Data Flow:**
- **all_weather**: Use all 8 weather columns as-is
- **clean_only**: Use 7 degradable columns as-is
- **degraded**: Use 7 degradable columns + apply degradation with seed(fold_idx, horizon)

## Core Components

### Configuration (`config.py`)
- `ForecastConfig`: Dataclass with all experiment parameters
- Horizons: [6, 24, 48, 168] hours
- Min training size: 4096 observations (TabPFN requirement)
- Number of folds: dynamically calculated based on data
- Weather scenarios: ['all_weather', 'clean_only', 'degraded']
- Degradation seed: 42 (reproducible error simulation)
- Weather degradation mapping: Maps dataset columns to degradation variable types

### Weather Degradation (`weather/`)
**WeatherProcessor**: Orchestrates weather data preparation for scenarios
- `prepare_weather_data()`: Main entry point, applies scenario logic
- `get_weather_columns()`: Returns appropriate columns per scenario
- `degrade_dataframe()`: On-the-fly degradation with proper seeding

**Weather Scenarios:**
1. **all_weather**: All 8 weather variables, no degradation (original baseline)
2. **clean_only**: 7 degradable variables (excludes Dew point), no degradation
3. **degraded**: 7 degradable variables with realistic NWP forecast errors

**Degradation Variables (7):**
- Temperature → Additive Gaussian error
- Humidity → Additive Gaussian error
- Wind speed → Additive Gaussian (reflected at 0)
- Visibility → Multiplicative lognormal error
- Solar Radiation → Heteroscedastic Gaussian error
- Rainfall → Multiplicative lognormal + event detection
- Snowfall → Multiplicative lognormal + event detection

**Non-degradable Variable (1):**
- Dew point temperature (excluded from degradation mapping)

**Error Growth:** Calibrated to ECMWF/KMA verification statistics
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
- `name`, `use_covariates`: Properties

**Implemented models:**
- SeasonalNaiveForecaster: Repeats last seasonal period
- ARIMAForecaster: Tuned parameters via auto_arima (e.g., (2,1,2))
- SARIMAXForecaster: Tuned parameters via auto_arima (e.g., (4,0,0)×(1,0,1,24))
- XGBoostForecaster: Uses lagged features (n_lags=24)
- TabPFNForecaster: Uses TabPFNv2 with calendar + auto seasonal features + weather covariates
- TabPFNForecaster_NoWeather: TabPFNv2 with calendar + auto seasonal features only (univariate)

**TabPFN Configuration:**
- Mode: LOCAL (CPU-based, no API rate limits)
- Worker: CPUParallelWorker (~2-5s per fold)
- Requires local source installation (see Usage Guide)

### Hyperparameter Tuning (`models/tuning/`)
**Auto-ARIMA approach** (using pmdarima):
- Stepwise search minimizes AIC
- Validates on multiple folds
- Saves results to JSON with timestamp

**Tuning scripts:**
- `tune_arima_auto.py`: Non-seasonal ARIMA (p,d,q)
- `tune_sarimax_auto.py`: Seasonal ARIMA with exogenous variables (p,d,q)×(P,D,Q,s)

**Parameters:**
- **p/P**: Autoregressive order (past values)
- **d/D**: Differencing order (trend removal)
- **q/Q**: Moving average order (error correction)
- **s**: Seasonal period (24 for hourly data)

**Output format:**
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

<!-- ### Experiment Runner (`run_experiments.py`)
**ForecastingExperiment**:
- Manages experiment lifecycle
- W&B initialization and logging
- Checkpoint save/load for recovery (now includes scenario)
- Runs all model-horizon-scenario combinations
- Saves aggregated and detailed results
- Automatic skip logic: models without covariates skip 'degraded' scenario -->

**run_weather_baseline.py**: <!--TODO: UPDATE -->
- Specialized runner for weather degradation experiments
- Runs clean_only and degraded scenarios
- Interactive confirmation before starting
- Displays degradation impact summary in results

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
First fold date calculated as: `data_start + min_train_size`
Actual folds = `min(requested_folds, available_data // longest_horizon)`

Rationale: Different horizons require different amounts of test data. Longer horizons = fewer possible folds.

### Imputed Data Tracking
Uses existing `Functioning Day` column (No = imputed)
Tracked per fold: train_imputed, test_imputed
Logged in detailed results for post-hoc analysis

### Expanding Window CV
Training set grows with each fold, test set is always exactly `horizon` hours.
Ensures models see increasing historical context.

### Checkpoint Recovery
Saves completed (model, horizon, scenario) combinations to JSON after each experiment.
On restart, skips already-completed experiments.
Prevents data loss from crashes during long runs.
File format: `checkpoint_{run_name}.json` in results/ directory.

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
<!-- TODO: update to exclude all_weather -->
- `weather_scenario`: 'all_weather' (only used in Pilot project), 'clean_only', or 'degraded'
- `model_uses_covariates`: Boolean (from model.use_covariates property)
- `degradation_seed`: Random seed used (42 by default)
- `num_weather_vars`: Count of weather variables (0 or 7; 8 only in Pilot project with 'clean_only' screnario)


## Known Limitations

1. SARIMAX convergence warnings during training (expected, doesn't affect predictions)
2. SARIMAX slower than other models (~30-60s per fold vs ~1s)
3. Hyperparameter tuning required before first run (one-time setup)
4. TabPFN requires 4096 training samples; runs on CPU in LOCAL mode 
5. XGBoost lag features may not capture complex seasonal patterns
6. Weather degradation is applied fixed to all weather data and does not dynamically adapt within horizons.
7. Weather degradation excludes Dew point temperature (not in degradation mapping)
8. Degradation assumes independent errors across variables (no cross-correlation)
9. Error growth calibrated to published statistics (ECMWF, KMA); may differ for specific locations