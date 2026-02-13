"""
Test single model with weather scenarios.

Usage:
  python test_weather_single_model.py --model tabpfn --scenario degraded
  python test_weather_single_model.py --model xgboost --scenario clean_only
  python test_weather_single_model.py --model arima --scenario all_weather

Tests with 3 folds (quick validation)
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from models.tabpfn_model import TabPFNForecaster, TabPFNForecaster_NoWeather
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from weather.weather_processor import WeatherProcessor


# Model registry
MODEL_MAP = {
    'seasonal_naive': SeasonalNaiveForecaster(seasonal_period=24),
    'arima': ARIMAForecaster(order=(2, 1, 2)),
    'sarimax': SARIMAXForecaster(order=(4, 0, 0), seasonal_order=(1, 0, 1, 24)),
    'xgboost': XGBoostForecaster(n_lags=24),
    'tabpfn': TabPFNForecaster(),
    'tabpfn_noweather': TabPFNForecaster_NoWeather(),
}


def test_single_model_scenario(
    model_name: str,
    scenario: str,
    data_path: str = "../../data/SeoulBikeData.csv",
    n_folds: int = 3
):
    """
    Test single model with one scenario.
    
    Parameters
    ----------
    model_name : str
        Model identifier (key in MODEL_MAP)
    scenario : str
        Weather scenario: 'all_weather', 'clean_only', or 'degraded'
    data_path : str
        Path to data file
    n_folds : int
        Number of CV folds to run (default: 3 for quick test)
    """
    
    if model_name not in MODEL_MAP:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_MAP.keys())}")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Setup
    config = ForecastConfig()
    config.n_folds = n_folds  # Use fewer folds for testing
    model = MODEL_MAP[model_name]
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    weather_proc = WeatherProcessor(config)
    
    # Get scenario info
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
    
    # Check if should skip
    if not model.use_covariates and scenario == "degraded":
        print(f"[SKIP] Model doesn't use covariates, skipping degraded scenario")
        print("This is expected behavior!")
        return
    
    # Test on shortest horizon only
    test_horizon = min(config.horizons)
    print(f"Testing horizon: {test_horizon}h")
    
    splits = cv.split(df, test_horizon)
    fold_results = []
    
    for fold_idx in range(min(n_folds, len(splits))):
        train_df, test_df = splits[fold_idx]
        y_train = train_df[config.target_col].values
        y_test = test_df[config.target_col].values
        
        # Prepare weather data using WeatherProcessor
        X_train = None
        X_test = None
        if model.use_covariates:
            X_train = weather_proc.prepare_weather_data(
                train_df, scenario, test_horizon, fold_idx
            )
            X_test = weather_proc.prepare_weather_data(
                test_df, scenario, test_horizon, fold_idx
            )
            
            print(f"Fold {fold_idx}: Weather vars = {X_train.columns.tolist()}")
        
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
    
    # Summary
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
    parser = argparse.ArgumentParser(
        description='Test single forecasting model with weather scenario'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=list(MODEL_MAP.keys()),
        help='Model to test'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        choices=['all_weather', 'clean_only', 'degraded'],
        help='Weather scenario to test'
    )
    parser.add_argument(
        '--data',
        type=str,
        default="../../data/SeoulBikeData.csv",
        help='Path to data file'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=3,
        help='Number of CV folds (default: 3)'
    )
    
    args = parser.parse_args()
    test_single_model_scenario(
        args.model,
        args.scenario,
        args.data,
        args.folds
    )
