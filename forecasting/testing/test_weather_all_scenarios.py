"""
Test single model across all weather scenarios.

Usage:
  python test_weather_all_scenarios.py --model tabpfn --folds 5
  python test_weather_all_scenarios.py --model seasonal_naive --folds 5

Tests one model with all applicable scenarios.
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


def test_all_scenarios_single_model(
    model_name: str,
    data_path: str = "../../data/SeoulBikeData.csv",
    n_folds: int = 5
):
    """
    Test one model across all applicable scenarios.
    
    Parameters
    ----------
    model_name : str
        Model identifier
    data_path : str
        Path to data file
    n_folds : int
        Number of CV folds (default: 5)
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
    config.n_folds = n_folds
    model = MODEL_MAP[model_name]
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    weather_proc = WeatherProcessor(config)
    
    print(f"\n{'='*70}")
    print(f"Testing Model: {model.name}")
    print(f"Uses covariates: {model.use_covariates}")
    print(f"Folds: {n_folds}")
    print(f"{'='*70}\n")
    
    # Determine applicable scenarios
    if model.use_covariates:
        scenarios = ['all_weather', 'clean_only', 'degraded']
    else:
        scenarios = ['all_weather', 'clean_only']  # Skip degraded
        print("Note: Skipping 'degraded' scenario (model doesn't use covariates)\n")
    
    # Test on shortest horizon only
    test_horizon = min(config.horizons)
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*70}")
        
        scenario_info = weather_proc.get_scenario_summary(scenario)
        print(f"Variables: {scenario_info['num_vars']} - {scenario_info['variables'][:3]}...")
        print(f"Degraded: {scenario_info['degraded']}\n")
        
        splits = cv.split(df, test_horizon)
        fold_results = []
        
        for fold_idx in range(min(n_folds, len(splits))):
            train_df, test_df = splits[fold_idx]
            y_train = train_df[config.target_col].values
            y_test = test_df[config.target_col].values
            
            # Prepare weather data
            X_train = None
            X_test = None
            if model.use_covariates:
                X_train = weather_proc.prepare_weather_data(
                    train_df, scenario, test_horizon, fold_idx
                )
                X_test = weather_proc.prepare_weather_data(
                    test_df, scenario, test_horizon, fold_idx
                )
            
            try:
                model.reset()
                model.fit(y_train, X_train)
                y_pred = model.predict(test_horizon, X_test)
                
                metrics = calc.calculate_all(y_test, y_pred, y_train)
                metrics['fold'] = fold_idx
                metrics['scenario'] = scenario
                
                fold_results.append(metrics)
                
                if (fold_idx + 1) % 2 == 0:
                    print(f"  Completed {fold_idx + 1}/{n_folds} folds", end="\r")
                    
            except Exception as e:
                print(f"  Fold {fold_idx}: ERROR - {str(e)[:60]}")
        
        if fold_results:
            results_df = pd.DataFrame(fold_results)
            summary = {
                'scenario': scenario,
                'MAE_mean': results_df['MAE'].mean(),
                'MAE_std': results_df['MAE'].std(),
                'RMSE_mean': results_df['RMSE'].mean(),
                'MASE_mean': results_df['MASE'].mean(),
                'folds': len(fold_results)
            }
            all_results.append(summary)
            
            print(f"\n  MAE:  {summary['MAE_mean']:6.1f} ± {summary['MAE_std']:5.1f}")
            print(f"  RMSE: {summary['RMSE_mean']:6.1f}")
            print(f"  MASE: {summary['MASE_mean']:5.2f}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("CROSS-SCENARIO COMPARISON")
    print(f"{'='*70}")
    
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        print(comparison_df.to_string(index=False))
        
        # Check degradation impact (if applicable)
        if 'clean_only' in comparison_df['scenario'].values and \
           'degraded' in comparison_df['scenario'].values:
            clean_mae = comparison_df[comparison_df['scenario'] == 'clean_only']['MAE_mean'].values[0]
            deg_mae = comparison_df[comparison_df['scenario'] == 'degraded']['MAE_mean'].values[0]
            impact = deg_mae - clean_mae
            impact_pct = (impact / clean_mae) * 100
            
            print(f"\nDegradation Impact:")
            print(f"  MAE increase: {impact:+.1f} bikes/hour ({impact_pct:+.1f}%)")
            
            if impact > 0:
                print(f"  ✓ Degradation causes error increase (expected!)")
            else:
                print(f"  ⚠ Warning: Degradation should increase error!")
    
    print(f"\n{'='*70}")
    print("TEST PASSED: All scenarios completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test single model across all weather scenarios'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help='Model to test'
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
        default=5,
        help='Number of CV folds (default: 5)'
    )
    
    args = parser.parse_args()
    test_all_scenarios_single_model(
        args.model,
        args.data,
        args.folds
    )
