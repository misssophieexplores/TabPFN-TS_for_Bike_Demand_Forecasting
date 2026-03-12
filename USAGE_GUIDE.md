# Usage Guide - Forecasting Project

## Quick Start

### 1. Setup Environment
```bash
# Python 3.11 required
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your W&B credentials
echo "WANDB_API_KEY=your_key_here" > .env
echo "WANDB_ENTITY=your_entity" >> .env
```

### 2. Run Experiments
```bash
# Run all three datasets sequentially (no manual intervention)
python forecasting/main.py

# Run a subset of datasets
python forecasting/main.py --cities seoul london

# Run a single dataset directly (with interactive confirmation)
python forecasting/run_weather_baseline.py
```

## Configuration

Shared settings live in `config.py`. Dataset-specific settings live in `config_<city>.py`, each exposing a `get_config()` function.

**To update a shared setting** (e.g. bump version to `v6`, rename W&B project): edit `config.py` only — all city configs inherit it automatically.

**To add a new dataset**: create a new `config_<city>.py` following the existing city configs as a template, then add it to `get_configs()` in `main.py`.

### Shared fields (`config.py`)

| Field | Default | Description |
|---|---|---|
| `horizons` | `[6, 24, 48, 168]` | Forecast horizons in hours |
| `n_folds` | `20` | Max CV folds |
| `n_train_samples` | `4096` | Max training observations |
| `seasonal_period` | `24` | Seasonal period in hours |
| `degradation_seed` | `42` | Reproducibility seed for weather degradation |
| `output_dir` | `results` | Output directory |
| `results_version` | `v5` | Suffix for result CSV names |
| `wandb_project` | `bike-forecasting` | W&B project name |
| `verbose` | `True` | Print detailed progress. Set to `False` in city configs for cluster/server runs |

### Dataset-specific fields (set in each city config)

| Field | Description |
|---|---|
| `data_filename` | File in `data/` directory |
| `dataset_name` | Identifier used in checkpoints and results (e.g. `"seoul"`) |
| `date_col` | Datetime column name |
| `target_col` | Column to forecast |
| `functioning_day_col` | Imputation tracking column (set to `None` if not present) |
| `holiday_col` | Holiday column name (normalized to 0/1, appended to covariates). Set to `None` if not present. |
| `holiday_mapping` | Dict mapping raw holiday values → 0/1. Required if `holiday_col` contains strings (e.g. `{'Holiday': 1, 'No Holiday': 0}`). Washington/London (numeric) need no mapping. |
| `season_col` | Season column name (normalized to 0–3 via `season_mapping`, appended to covariates). Set to `None` if not present. |
| `season_mapping` | Explicit dict mapping raw season values → 0–3. Required if `season_col` is set. |
| `weather_covariates` | List of weather covariate column names |
| `weather_degradation_mapping` | Maps dataset columns to degradation variable types |
| `arima_params_file` | Path to tuned ARIMA JSON |
| `sarimax_params_file` | Path to tuned SARIMAX JSON |
| `xgb_params_file` | Path to tuned XGBoost JSON |

**Example: adding a new dataset**
```python
# config_washington.py
from config import ForecastConfig

def get_config() -> ForecastConfig:
    config = ForecastConfig()
    config.data_filename = "WashingtonBikeData.csv"
    config.dataset_name = "washington"
    config.date_col = "datetime"
    config.target_col = "cnt"
    config.functioning_day_col = None
    config.holiday_col = "holiday"
    config.holiday_mapping = None  # already numeric; use e.g. {'Holiday': 1, 'No Holiday': 0} for string values
    config.season_col = "season"
    config.season_mapping = {1: 0, 2: 1, 3: 2, 4: 3}  # 1-based → 0-based
    config.weather_covariates = ["temp", "hum", "windspeed", ...]
    config.weather_degradation_mapping = {"temp": "temperature", ...}
    config.xgb_params_file = "results/tuning/xgboost_best_params_<timestamp>.json"
    return config
```

Season mapping examples across datasets:
```python
#   Seoul (strings):      {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
#   Washington (1-based): {1: 0, 2: 1, 3: 2, 4: 3}
#   London (0-based):     {0: 0, 1: 1, 2: 2, 3: 3}
```

## Hyperparameter Tuning

Tune once per dataset. Parameters describe data structure, not forecast horizon.

### ARIMA
```bash
pip install pmdarima  # if needed
python forecasting/models/tuning/tune_arima_auto.py  # uses config defaults
# or override data file:
python tune_arima_auto.py --data data/OtherDataset.csv --folds 5
```
Output: `results/tuning/arima_best_params_<timestamp>.json`

### SARIMAX
```bash
python forecasting/models/tuning/tune_sarimax.py  # uses config defaults
python forecasting/models/tuning/tune_sarimax.py --data data/OtherDataset.csv --folds 5
```
Output: `results/tuning/sarimax_best_params_<timestamp>.json`

### XGBoost
```bash

# Default (all weather covariates):
python forecasting/models/tuning/tune_xgboost.py

# Degradable covariates only (for clean_only / degraded scenarios):
python forecasting/models/tuning/tune_xgboost.py --scenario clean_only

# Override data or other options:
python forecasting/models/tuning/tune_xgboost.py --data data/OtherDataset.csv --horizon 24 --trials 600
```
Output: `results/tuning/xgboost_best_params_<timestamp>.json`

**Note:** Calendar time features (hour, dayofweek, month, is_weekend) are always included in tuning automatically. Holiday and season are included via `weather_covariates` (not as time features). **Any tuning JSON produced before holiday/season were added to `weather_covariates` is stale and must be regenerated.**

**The `--scenario` flag controls which covariates are used:**
- `all_weather` (default): all `weather_covariates` from config, including Dew point temperature
- `clean_only`: only degradable covariates (keys of `weather_degradation_mapping`), excludes Dew point

### Applying Tuned Parameters

Update the params file paths in the relevant city config:
```python
# e.g. in config_london.py
config.arima_params_file = "results/tuning/arima_best_params_<timestamp>.json"
config.sarimax_params_file = "results/tuning/sarimax_best_params_<timestamp>.json"
config.xgb_params_file = "results/tuning/xgboost_best_params_<timestamp>.json"
```

## Weather Scenarios

| Scenario | Covariates | Degradation | Notes |
|---|---|---|---|
| `all_weather` | All vars in `weather_covariates` | No | Original baseline |
| `clean_only` | 6 degradable vars + holiday + season (excl. Dew point) | No | New clean baseline |
| `degraded` | 6 degradable vars + holiday + season | Yes (degradable cols only) | Robustness test |

Models without covariates are automatically skipped for the `degraded` scenario.

Covariate selection by scenario is handled automatically — controlled by `weather_covariates` and `weather_degradation_mapping` in `config.py`.

## TabPFN Models

**Installation (required before first use):**
```bash
# Clone tabpfn-time-series repo
git clone https://github.com/PriorLabs/tabpfn-time-series.git
cd tabpfn-time-series
pip install -e .  # includes CPU fallback
cd ../seoul-bike-demand
```

**Why from source:** Published pip version requires GPU for LOCAL mode. Source includes CPU fallback.

**Two variants:**
- `TabPFNForecaster`: With weather covariates
- `TabPFNForecaster_NoWeather`: Univariate only

Requires `n_train_samples` (4096) observations — early folds are automatically skipped.


## Adding a New Model

1. Create model class in `models/statistical.py` or `models/ml_models.py`:
```python
from models.base import BaseForecaster

class MyForecaster(BaseForecaster):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        self.model = None

    @property
    def name(self) -> str:
        return "MyModel"

    @property
    def use_covariates(self) -> bool:
        return True  # or False

    def fit(self, y, X=None):
        self.model = SomeModel(self.param1, self.param2)
        self.model.fit(y, X)

    def predict(self, horizon, X=None):
        return self.model.predict(horizon, X)

    def reset(self):
        self.model = None
```

2. Add to `all_models` in `run_weather_baseline.py` (parameters come from config):
```python
all_models = [
    SeasonalNaiveForecaster(seasonal_period=config.seasonal_period),
    MyForecaster(param1=value1, param2=value2),
]
```

3. Test individually:
```bash
# Add to MODEL_MAP in test_single_model.py, then:
python testing/test_single_model.py --model my_model
```

## Results Files

**Aggregated results** (`results/results_master_{version}.csv`): one row per model-horizon-scenario combination.
- `dataset`, `run_name`, `version`, `timestamp`, `model`, `horizon`, `weather_scenario`
- `MAE_mean`, `MAE_std`, `RMSE_mean`, `RMSE_std`, `MASE_mean`, `MASE_std`, `sMAPE_mean`, `sMAPE_std`

**Detailed results** (`results/detailed_results_master_{version}.csv`): one row per fold.
- All columns above plus `fold`, `test_imputed`, `train_imputed`

Both files accumulate across runs and datasets (append mode). Get latest results for a specific dataset only:
```python
df = pd.read_csv('results/detailed_results_master_{version}.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
latest = (df[df['dataset'] == 'seoul']
          .sort_values('timestamp')
          .groupby(['model', 'horizon', 'fold'])
          .last()
          .reset_index())
```

The version suffix is controlled by `config.results_version` in `config.py`.

## Checkpointing & Recovery

Checkpoints are saved after every completed `(dataset, model, horizon, scenario)` combination. If a run is interrupted — including mid-way through a multi-city run — restart with the same experiment name and it will skip everything already done:

```bash
# Resume a multi-city run
python forecasting/main.py --cities seoul washington london

# Resume a single-city run
python forecasting/run_weather_baseline.py  # uses same experiment_name from config
```

The experiment name is derived automatically as `{dataset_name}_{results_version}` (e.g. `seoul_v4`). Checkpoint files are at `results/checkpoint_{experiment_name}.json`.

Because `dataset_name` is part of the checkpoint key, Seoul and Washington completions never interfere with each other even if they share the same results file.

## W&B Dashboard

Access at: `https://wandb.ai/{entity}/bike-forecasting/runs`

All datasets log to the same W&B project (`bike-forecasting`), distinguished by the `dataset` field logged per run. The project name is controlled by `config.wandb_project`. Test scripts can override it: `config.wandb_project = "bike-forecasting-testing"`.

