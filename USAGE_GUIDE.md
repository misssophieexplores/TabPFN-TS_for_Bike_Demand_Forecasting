# Usage Guide - Seoul Bike Forecasting
<!-- TODO: check if updates are needed! -->

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Set W&B credentials in .env
echo "WANDB_API_KEY=your_key_here" > .env
echo "WANDB_ENTITY=your_entity" >> .env
```

### 2. Run Tests
```bash
cd forecasting/testing

# Test individual models
python test_single_model.py

# Test full pipeline (minimal)
python test_full_experiment.py
```

### 3. Run Experiments
```bash
cd forecasting
python run_weather_baseline.py
```

## Configuration

<!-- TODO: Horizons and folds currently hardcoded in run_weather_baseline.py -->
<!-- Should be moved to config.py for consistency -->

Current settings (in `run_weather_baseline.py`):
- Horizons: `[6, 24, 48, 168]`
- Folds: `20`

Edit `config.py` for:
- Training size: `max_train_samples = 4096`
- Weather covariates: `weather_covariates = [...]`
- Weather scenarios: `weather_scenarios = ['clean_only', 'degraded']`

## Hyperparameter Tuning

### ARIMA Tuning
Find optimal (p, d, q) parameters:
```bash
cd forecasting/models/tuning
pip install pmdarima  # Install if needed
python tune_arima_auto.py --data ../../../data/SeoulBikeData.csv --folds 5
```

**Output:** `results/tuning/arima_best_params_<timestamp>.json`

**Parameters explained:**
- **p**: Number of autoregressive lags (past hours used)
- **d**: Differencing order (removes trend)
- **q**: Moving average terms (forecast error correction)

### SARIMAX Tuning
Find optimal (p,d,q)×(P,D,Q,s) parameters:
```bash
cd forecasting/models/tuning
python tune_sarimax_auto.py --data ../../../data/SeoulBikeData.csv --folds 5
```

**Output:** `results/tuning/sarimax_best_params_<timestamp>.json`

**Parameters explained:**
- **p, d, q**: Same as ARIMA
- **P**: Seasonal autoregressive terms (e.g., yesterday's pattern)
- **D**: Seasonal differencing
- **Q**: Seasonal moving average terms
- **s**: Seasonal period (24 for daily cycle in hourly data)



### XGBoost Tuning
Find optimal n_lags and XGBoost hyperparameters:
```bash
cd forecasting/models/tuning
python tune_xgboost.py --data ../../../data/SeoulBikeData.csv --horizon 24 --trials 600 --folds 10 --tune-folds 5
```

**Output:** `results/tuning/xgboost_best_params_<timestamp>.json`

**Parameters explained:**
- **n_lags**: Number of past target values used as features (options: 24, 168)
- **n_estimators**: Number of boosting trees
- **learning_rate**: Step size for gradient descent
- **max_depth**: Maximum tree depth (controls model complexity)
- Regularization: **gamma**, **reg_lambda**, **reg_alpha**

### Applying Tuned Parameters

**ARIMA/SARIMAX:** Update `run_weather_baseline.py` with found parameters:
```python
models = [
    ARIMAForecaster(order=(2, 1, 2)),  # From tune_arima_auto.py
    SARIMAXForecaster(
        order=(4, 0, 0),
        seasonal_order=(1, 0, 1, 24)  # From tune_sarimax_auto.py
    ),
]
```

**XGBoost:** Update JSON file path in `run_weather_baseline.py`:
```python
# Line ~20: Update with your timestamp
with open("results/tuning/xgboost_best_params_20260110_042536.json") as f:
    xgb_cfg = json.load(f)
```

**Note:** Tune once per dataset. Parameters describe data structure, not forecast horizon.




## TabPFN Models

**Installation (required before first use):**
```bash
# Clone tabpfn-time-series repo
git clone https://github.com/PriorLabs/tabpfn-time-series.git
cd tabpfn-time-series

# Install from source (includes CPU fallback)
pip install -e .

# Return to your project
cd ../seoul-bike-demand
```

**Why from source:** Published pip version requires GPU for LOCAL mode. Source includes CPU fallback.

**Configuration:**
- Mode: LOCAL (CPU-based, no API rate limits)
<!-- - Speed: ~2-5 seconds per fold -->
- Number samples: 4096 (early folds auto-skipped)

**Two variants:**
- `TabPFNForecaster`: With weather covariates
- `TabPFNForecaster_NoWeather`: Univariate only

**Test installation:**
```bash
cd forecasting/testing
python test_single_model.py --model tabpfn
```

## Adding a New Model

1. Create model class in appropriate file (`models/statistical.py` or `models/ml_models.py`)
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
        # Train model
        self.model = SomeModel(self.param1, self.param2)
        self.model.fit(y, X)
    
    def predict(self, horizon, X=None):
        # Generate forecasts
        return self.model.predict(horizon, X)
    
    def reset(self):
        # Clear state
        self.model = None
```

2. Add to model list in `run_weather_baseline.py`:
```python
all_models = [
    SeasonalNaiveForecaster(seasonal_period=24),
    MyForecaster(param1=value1, param2=value2),
    # ... other models
]
```

3. Test individually:
```python
# In testing/test_single_model.py
models = [MyForecaster(param1=value1, param2=value2)]
```

### Master Results Files

**Aggregated Results (`results_master.csv`)**
Accumulates all runs. Columns include:
- `run_name`, `timestamp`: Track which run produced results
- `model`, `horizon`, `n_folds`
- `MAE_mean`, `MAE_std`, `RMSE_mean`, `RMSE_std`, etc.
- Imputation tracking columns


**Detailed Results (`detailed_results_master.csv`)**
Fold-level data accumulating all runs. One row per fold:
- `run_name`, `timestamp`: Track which run produced this fold
- `fold`: Fold number
- `model`, `horizon`
- `MAE`, `RMSE`, `MASE`, `sMAPE`: Metrics for this fold
- `test_imputed`: Imputed observations in this fold's test set
- `train_imputed`: Imputed observations in this fold's train set

**Get latest fold results:**
```python
df = pd.read_csv('detailed_results_master.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
latest = df.sort_values('timestamp').groupby(['model', 'horizon', 'fold']).last().reset_index()
```

## Checkpointing & Recovery

If run is interrupted:
```python
# Restart will automatically skip completed experiments
experiment = ForecastingExperiment(
    config=config,
    experiment_name="same_run_name"  # Use same name as interrupted run
)
```

Checkpoints stored in: `results/checkpoint_{run_name}.json`

## W&B Dashboard

Access at: `https://wandb.ai/{entity}/seoul-bike-forecasting/runs`

**View:**
- Real-time fold-level metrics
- Aggregated results tables
- Download artifacts (CSV files)
- Compare across runs

**Test runs:** Use project `seoul-bike-testing` (set in test scripts)

## Common Issues

### "AttributeError: 'ForecastConfig' object has no attribute 'first_fold_date'"
**Fix:** Remove `first_fold_date` from W&B config in `run_experiments.py`

### SARIMAX convergence warnings
**Issue:** "Maximum Likelihood optimization failed to converge"
**Fix:** Increase `maxiter` in `statistical.py` (line ~164): change to `maxiter=200`
**Note:** This is expected during hyperparameter tuning - auto_arima handles it

### detailed_results CSV not created
**Fix:** Ensure `detailed_results` attribute exists in experiment runner
**Check:** Console output for "Results saved" message

### TabPFN requires 4096+ samples
**Issue:** "TabPFN requires min 4096 samples, got XXXX"
**Fix:** Early folds may have <4096 samples - they're automatically skipped
**Note:** This is expected behavior, not an error

## Tips

1. **Start small**: Test with 1 model, 1 horizon, 3 folds before full runs
2. **Monitor W&B**: Check logs during runs for errors
3. **Use checkpoints**: Long runs should use unique experiment names for recovery
4. **Tune parameters first**: Run tuning scripts before full experiments
5. **Check imputation**: Review imputed fold counts in results