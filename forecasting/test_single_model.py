"""
Test individual models with minimal folds.
Usage: python test_single_model.py --model seasonal_naive
"""
import argparse
import pandas as pd
from pathlib import Path

from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator


MODEL_MAP = {
    'seasonal_naive': SeasonalNaiveForecaster(seasonal_period=24),
    'arima': ARIMAForecaster(order=(1, 1, 1)),
    'sarimax': SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 1, 0, 24)),
    'xgboost': XGBoostForecaster(n_lags=24),
}


def test_model(model_name: str, data_path: str = "data/SeoulBikeData.csv"): #TODO: maybe change to config path?
    """Test single model on limited folds"""
    
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
    model = MODEL_MAP[model_name]
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    
    # Test horizons: shortest and longest
    test_horizons = [min(config.horizons), max(config.horizons)]
    
    print(f"\n{'='*60}")
    print(f"Testing: {model.name}")
    print(f"{'='*60}")
    
    for horizon in test_horizons:
        print(f"\nHorizon: {horizon}h")
        
        splits = cv.split(df, horizon)
        # Test first, middle, last fold
        test_folds = [0, len(splits)//2, len(splits)-1]
        
        for fold_idx in test_folds:
            train_df, test_df = splits[fold_idx]
            y_train = train_df[config.target_col].values
            y_test = test_df[config.target_col].values
            
            X_train = None
            X_test = None
            if model.use_covariates:
                X_train = train_df[config.weather_covariates]
                X_test = test_df[config.weather_covariates]
            
            try:
                model.reset()
                model.fit(y_train, X_train)
                y_pred = model.predict(horizon, X_test)
                
                metrics = calc.calculate_all(y_test, y_pred, y_train)
                print(f"  Fold {fold_idx:2d}: "
                      f"MAE={metrics['MAE']:6.1f} | "
                      f"RMSE={metrics['RMSE']:6.1f} | "
                      f"MASE={metrics['MASE']:5.2f}")
                      
            except Exception as e:
                print(f"  Fold {fold_idx:2d}: ERROR - {str(e)[:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test individual forecasting model')
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(MODEL_MAP.keys()),
                       help='Model to test')
    parser.add_argument('--data', type=str, default="data/SeoulBikeData.csv",
                       help='Path to data file')
    
    args = parser.parse_args()
    test_model(args.model, args.data)