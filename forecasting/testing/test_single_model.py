"""
Test individual models with minimal folds.

Usage: 

python forecasting/testing/test_single_model.py --model seasonal_naive
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from models.tabpfn_model import TabPFNForecaster_Custom #, TabPFNForecaster_NoWeather
from models.tabpfn_pipeline_model import TabPFNPipelineForecaster, TabPFNPipelineForecaster_NoWeather
from models.prophet_models import ProphetForecaster, NeuralProphetForecaster, NeuralProphetForecaster_NoWeather
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator

# Module-level config so MODEL_MAP and argparse defaults can reference it
config = ForecastConfig()
# Load tuned model parameters:

with open(config.arima_params_file) as f:
    arima_cfg = json.load(f)

with open(config.sarimax_params_file) as f:
    sarimax_cfg = json.load(f)
with open(config.xgb_params_file) as f:
    _xgb_cfg = json.load(f)

MODEL_MAP = {
    'seasonal_naive':         SeasonalNaiveForecaster(seasonal_period=config.seasonal_period),
    'arima':                  ARIMAForecaster(order=tuple(arima_cfg["order"])),
    'sarimax':                SARIMAXForecaster(order=tuple(sarimax_cfg["order"]), seasonal_order=tuple(sarimax_cfg["seasonal_order"])),
    'xgboost':                XGBoostForecaster(n_lags=_xgb_cfg["n_lags"], **_xgb_cfg["xgb_params"]),
    'tabpfn_custom':          TabPFNForecaster_Custom(),
    'tabpfn':                 TabPFNPipelineForecaster(),
    'tabpfn_noweather':       TabPFNPipelineForecaster_NoWeather(),
    'prophet':                ProphetForecaster(),
    'neuralprophet':          NeuralProphetForecaster(),
    'neuralprophet_noweather': NeuralProphetForecaster_NoWeather(),
}


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data"""
    df = pd.read_csv(data_path)
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.sort_values(config.date_col).reset_index(drop=True)
    return df


def test_model(model_name: str, data_path: str = None):
    """Test single model on limited folds"""

    if model_name not in MODEL_MAP:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_MAP.keys())}")
        return

    if data_path is None:
        data_path = str(Path(__file__).parent.parent.parent / "data" / config.data_filename)

    df = load_data(data_path)
    print(f"Data loaded: {len(df)} observations")

    model = MODEL_MAP[model_name]
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()

    test_horizons = [min(config.horizons), max(config.horizons)]

    print(f"\n{'='*60}")
    print(f"Testing: {model.name}")
    print(f"Uses covariates: {model.use_covariates}")
    print(f"{'='*60}")

    all_results = []

    for horizon in test_horizons:
        print(f"\n--- Horizon: {horizon}h ---")

        splits = cv.split(df, horizon)

        if len(splits) == 0:
            print(f"ERROR: No splits created for horizon {horizon}h")
            continue

        test_fold_indices = [0, len(splits)//2, len(splits)-1]

        for fold_idx in test_fold_indices:
            if fold_idx >= len(splits):
                continue

            train_df, test_df = splits[fold_idx]
            y_train = train_df[config.target_col].values
            y_test = test_df[config.target_col].values

            X_train = None
            X_test = None
            if model.use_covariates:
                X_train = train_df[config.weather_covariates]
                X_test = test_df[config.weather_covariates]

            # Attach real DatetimeIndex for models that need timestamps (Prophet, TabPFN)
            if getattr(model, "needs_datetime", False):
                train_dates = pd.DatetimeIndex(train_df[config.date_col].values)
                test_dates  = pd.DatetimeIndex(test_df[config.date_col].values)
                if X_train is None:
                    X_train = pd.DataFrame(index=train_dates)
                else:
                    X_train = X_train.set_index(train_dates)
                if X_test is None:
                    X_test = pd.DataFrame(index=test_dates)
                else:
                    X_test = X_test.set_index(test_dates)

            try:
                if 'TabPFN' in model.name and len(y_train) < config.n_train_samples:
                    print(f"  Fold {fold_idx:2d}: SKIP - TabPFN needs {config.n_train_samples}+ samples, got {len(y_train)}")
                    continue

                model.reset()
                model.fit(y_train, X_train)
                y_pred = model.predict(horizon, X_test)

                metrics = calc.calculate_all(y_test, y_pred, y_train)

                print(f"  Fold {fold_idx:2d}: "
                      f"MAE={metrics['MAE']:6.1f} | "
                      f"RMSE={metrics['RMSE']:6.1f} | "
                      f"MASE={metrics['MASE']:5.2f} | "
                      f"sMAPE={metrics['sMAPE']:5.1f}%")

                all_results.append({'model': model.name, 'horizon': horizon, 'fold': fold_idx, **metrics})

            except Exception as e:
                print(f"  Fold {fold_idx:2d}: ERROR - {str(e)[:80]}")
                import traceback
                traceback.print_exc()

    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY (averaged across test folds)")
        print(f"{'='*60}")
        results_df = pd.DataFrame(all_results)
        summary = results_df.groupby('horizon')[['MAE', 'RMSE', 'MASE', 'sMAPE']].mean()
        print(summary.round(2))
        print(f"\nModel test complete: {model.name}")
    else:
        print(f"\nERROR: No successful runs for {model.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test individual forecasting model')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_MAP.keys()),
                        help='Model to test')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data file (optional, defaults to config)')

    args = parser.parse_args()
    test_model(args.model, args.data)