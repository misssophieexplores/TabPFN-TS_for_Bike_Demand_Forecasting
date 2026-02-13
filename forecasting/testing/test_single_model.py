"""
Test individual models with minimal folds.
Usage: python test_single_model.py --model seasonal_naive
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from models.tabpfn_model import TabPFNForecaster, TabPFNForecaster_NoWeather
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator

MODEL_MAP = {
    'seasonal_naive': SeasonalNaiveForecaster(seasonal_period=24),
    'arima': ARIMAForecaster(order=(1, 1, 1)),
    'sarimax': SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(5, 1, 0, 24)),
    'xgboost': XGBoostForecaster(n_lags=24),
    'tabpfn': TabPFNForecaster(),
    'tabpfn_noweather': TabPFNForecaster_NoWeather(),
}


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data"""
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
        
    return df


def test_model(model_name: str, data_path: str = None):
    """Test single model on limited folds"""
    
    if model_name not in MODEL_MAP:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_MAP.keys())}")
        return
    
    # Determine data path
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "SeoulBikeData.csv"
    
    # Load data
    df = load_data(str(data_path))
    print(f"Data loaded: {len(df)} observations")
    
    # Setup
    config = ForecastConfig()
    model = MODEL_MAP[model_name]
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    
    # Test horizons: shortest and longest
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
        
        # Test first, middle, last fold
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
            
            try:

                # Check minimum samples for TabPFN
                if 'TabPFN' in model.name and len(y_train) < 4096:
                    print(f"  Fold {fold_idx:2d}: SKIP - TabPFN needs 4096+ samples, got {len(y_train)}")
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
                
                all_results.append({
                    'model': model.name,
                    'horizon': horizon,
                    'fold': fold_idx,
                    **metrics
                })
                      
            except Exception as e:
                print(f"  Fold {fold_idx:2d}: ERROR - {str(e)[:80]}")
                import traceback
                traceback.print_exc()
    
    # Summary
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
                       help='Path to data file (optional)')
    
    args = parser.parse_args()
    test_model(args.model, args.data)