"""
Test single model with weather scenarios.

Usage:
  python forecasting/testing/test_weather_single_model.py --city seoul --model tabpfn --scenario degraded
  python forecasting/testing/test_weather_single_model.py --city seoul --model xgboost --scenario clean_only
  python forecasting/testing/test_weather_single_model.py --city seoul --model arima --scenario clean_only
  python forecasting/testing/test_weather_single_model.py --city seoul --model prophet --scenario clean_only
  python forecasting/testing/test_weather_single_model.py --city seoul --model neuralprophet --scenario degraded
  
Tests with 3 folds (quick validation)
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from models.tabpfn_pipeline_model import TabPFNPipelineForecaster, TabPFNPipelineForecaster_NoWeather
from models.prophet_models import ProphetForecaster, NeuralProphetForecaster, NeuralProphetForecaster_NoWeather
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from weather.weather_processor import WeatherProcessor
from run_experiments import load_and_prepare_data


def build_model_map(config, arima_cfg, sarimax_cfg, xgb_cfg):
    return {
        'seasonal_naive':   SeasonalNaiveForecaster(seasonal_period=config.seasonal_period),
        'arima':            ARIMAForecaster(order=tuple(arima_cfg["order"])),
        'sarimax':          SARIMAXForecaster(order=tuple(sarimax_cfg["order"]), seasonal_order=tuple(sarimax_cfg["seasonal_order"])),
        'xgboost':          XGBoostForecaster(n_lags=xgb_cfg["n_lags"], **xgb_cfg["xgb_params"]),
        'tabpfn':                  TabPFNPipelineForecaster(),
        'tabpfn_noweather':        TabPFNPipelineForecaster_NoWeather(),
        'prophet':                 ProphetForecaster(),
        'neuralprophet':           NeuralProphetForecaster(),
        'neuralprophet_noweather': NeuralProphetForecaster_NoWeather(),
    }


def test_single_model_scenario(config, model_map, model_name: str, scenario: str, n_folds: int = 3):
    if model_name not in model_map:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(model_map.keys())}")
        return

    df, _ = load_and_prepare_data(config)

    config.n_folds = n_folds
    model = model_map[model_name]
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    weather_proc = WeatherProcessor(config)

    scenario_info = weather_proc.get_scenario_summary(scenario)

    print(f"\n{'='*60}")
    print(f"Test Configuration")
    print(f"{'='*60}")
    print(f"Model: {model.name}")
    print(f"Uses covariates: {model.use_covariates}")
    print(f"Scenario: {scenario}")
    print(f"Variables: {scenario_info['num_vars']} - {scenario_info['variables']}")
    print(f"Degraded: {scenario_info['degraded']}")
    print(f"Folds: {n_folds}")
    print(f"{'='*60}\n")

    if not model.use_covariates and scenario == "degraded":
        print(f"[SKIP] Model doesn't use covariates, skipping degraded scenario")
        print("This is expected behavior!")
        return

    test_horizon = min(config.horizons)
    print(f"Testing horizon: {test_horizon}h")

    splits = cv.split(df, test_horizon)
    fold_results = []

    for fold_idx in range(min(n_folds, len(splits))):
        train_df, test_df = splits[fold_idx]
        y_train = train_df[config.target_col].values
        y_test = test_df[config.target_col].values

        X_train = None
        X_test = None
        if model.use_covariates:
            X_train = weather_proc.prepare_weather_data(train_df, scenario, test_horizon, fold_idx, split='train')
            X_test = weather_proc.prepare_weather_data(test_df, scenario, test_horizon, fold_idx, split='test')
            print(f"Fold {fold_idx}: Weather vars = {X_train.columns.tolist()}")

            print(f"Fold {fold_idx}: ALL columns in X_train: {X_train.columns.tolist()}")
            print(f"  holiday present: {config.holiday_col in X_train.columns if config.holiday_col else 'N/A (not configured)'}")
            print(f"  season present:  {config.season_col in X_train.columns if config.season_col else 'N/A (not configured)'}")

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
            model.reset()
            model.fit(y_train, X_train)
            y_pred = model.predict(test_horizon, X_test)

            metrics = calc.calculate_all(y_test, y_pred, y_train)
            metrics['fold'] = fold_idx
            metrics['scenario'] = scenario
            metrics['num_weather_vars'] = len(X_train.columns) if X_train is not None else 0

            fold_results.append(metrics)

            print(f"  Fold {fold_idx:2d}: "
                  f"MAE={metrics['MAE']:6.1f} | "
                  f"RMSE={metrics['RMSE']:6.1f} | "
                  f"MASE={metrics['MASE']:5.2f}")

        except Exception as e:
            print(f"  Fold {fold_idx:2d}: ERROR - {str(e)[:80]}")

    if fold_results:
        results_df = pd.DataFrame(fold_results)
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
        print(f"MAE:  {results_df['MAE'].mean():6.1f} ± {results_df['MAE'].std():5.1f}")
        print(f"RMSE: {results_df['RMSE'].mean():6.1f} ± {results_df['RMSE'].std():5.1f}")
        print(f"MASE: {results_df['MASE'].mean():5.2f} ± {results_df['MASE'].std():4.2f}")
        print(f"{'='*60}\n")
        print("TEST PASSED: Single model + scenario test completed successfully!")
    else:
        print("\nWARNING: No successful folds!")


if __name__ == "__main__":
    from config_seoul import get_config as seoul_config
    from config_washington import get_config as washington_config
    from config_london import get_config as london_config

    city_configs = {
        "seoul":      seoul_config,
        "washington": washington_config,
        "london":     london_config,
    }

    # Parse --city first to build config and model map before other args
    parser = argparse.ArgumentParser(description='Test single forecasting model with weather scenario')
    parser.add_argument('--city', choices=list(city_configs.keys()), required=True,
                        help="City dataset to use.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True,
                        choices=['all_weather', 'clean_only', 'degraded'])
    parser.add_argument('--folds', type=int, default=3)
    args = parser.parse_args()

    config = city_configs[args.city]()

    with open(config.arima_params_file) as f:
        arima_cfg = json.load(f)
    with open(config.sarimax_params_file) as f:
        sarimax_cfg = json.load(f)
    with open(config.xgb_params_file) as f:
        xgb_cfg = json.load(f)

    model_map = build_model_map(config, arima_cfg, sarimax_cfg, xgb_cfg)

    if args.model not in model_map:
        print(f"Unknown model: {args.model}. Available: {list(model_map.keys())}")
        sys.exit(1)

    test_single_model_scenario(config, model_map, args.model, args.scenario, args.folds)