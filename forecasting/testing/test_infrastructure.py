"""
Test core infrastructure: config, CV splits, metrics, data loading.
Run this before testing individual models.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import ForecastConfig
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator


def test_config():
    """Test configuration loads correctly"""
    print("\n" + "="*60)
    print("TEST 1: Configuration")
    print("="*60)
    
    try:
        config = ForecastConfig()
        print(f"Horizons: {config.horizons}")
        print(f"Folds: {config.n_folds}")
        print(f"Min train size: {config.max_train_samples}")
        print(f"Target column: {config.target_col}")
        print(f"Weather covariates: {len(config.weather_covariates)} variables")
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_data_loading():
    """Test data loading and preprocessing"""
    print("\n" + "="*60)
    print("TEST 2: Data Loading")
    print("="*60)
    
    try:
        data_path = Path(__file__).parent.parent.parent / "data" / "SeoulBikeData.csv" #TODO: change from hardcoded path to config!
        if not data_path.exists():
            print(f"FAILED: Data file not found at {data_path}")
            return False, None
        
        df = pd.read_csv(data_path)
        print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        # DON'T filter - data is imputed
        print(f"Imputed hours (Functioning Day = No): {(df['Functioning Day'] == 'No').sum()}")
        
        config = ForecastConfig()
        if config.target_col not in df.columns:
            print(f"FAILED: Target column '{config.target_col}' not found")
            return False, None
        
        missing_covariates = [col for col in config.weather_covariates if col not in df.columns]
        if missing_covariates:
            print(f"FAILED: Missing covariates: {missing_covariates}")
            return False, None
        
        print("PASSED")
        return True, df
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_cv_splits(df):
    """Test cross-validation split generation"""
    print("\n" + "="*60)
    print("TEST 3: Cross-Validation Splits")
    print("="*60)
    
    if df is None:
        print("SKIPPED: No data available")
        return False
    
    try:
        config = ForecastConfig()
        cv = TimeSeriesCV(config)
        
        # Test with one horizon
        test_horizon = 24
        splits = cv.split(df, horizon=test_horizon)
        
        if len(splits) == 0:
            print("FAILED: No splits created")
            return False
        
        print(f"\nValidating splits...")
        
        # Check first split
        train_df, test_df = splits[0]
        print(f"  First split - Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Verify test size matches horizon
        if len(test_df) != test_horizon:
            print(f"FAILED: Test set size {len(test_df)} != horizon {test_horizon}")
            return False
        
        # Verify temporal ordering
        if train_df['Date'].max() >= test_df['Date'].min():
            print("FAILED: Train and test sets overlap temporally")
            return False
        
        # Check minimum train size
        if len(train_df) < config.max_train_samples:
            print(f"FAILED: Train size {len(train_df)} < min {config.max_train_samples}")
            return False
        
        print("PASSED")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_metrics():
    """Test metric calculations"""
    print("\n" + "="*60)
    print("TEST 4: Metrics Calculation")
    print("="*60)
    
    try:
        calc = MetricsCalculator()
        
        # Create simple test data
        y_true = np.array([100, 200, 150, 180, 220])
        y_pred = np.array([110, 190, 160, 170, 210])
        y_train = np.random.randint(50, 250, 100)
        
        # Test individual metrics
        mae = calc.mae(y_true, y_pred)
        rmse = calc.rmse(y_true, y_pred)
        mase = calc.mase(y_true, y_pred, y_train)
        smape = calc.smape(y_true, y_pred)
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MASE: {mase:.2f}")
        print(f"sMAPE: {smape:.2f}%")
        
        # Sanity checks
        if mae < 0 or np.isnan(mae) or np.isinf(mae):
            print(f"FAILED: Invalid MAE value: {mae}")
            return False
        
        if rmse < 0 or np.isnan(rmse) or np.isinf(rmse):
            print(f"FAILED: Invalid RMSE value: {rmse}")
            return False
        
        if np.isnan(mase) or np.isinf(mase):
            print(f"FAILED: Invalid MASE value: {mase}")
            return False
        
        if smape < 0 or smape > 100 or np.isnan(smape):
            print(f"FAILED: Invalid sMAPE value: {smape}")
            return False
        
        # Test calculate_all
        all_metrics = calc.calculate_all(y_true, y_pred, y_train)
        expected_keys = ['MAE', 'RMSE', 'MASE', 'sMAPE']
        if not all(key in all_metrics for key in expected_keys):
            print(f"FAILED: Missing keys in calculate_all output")
            return False
        
        print("PASSED")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all infrastructure tests"""
    print("\n" + "="*70)
    print("INFRASTRUCTURE TESTING")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['config'] = test_config()
    results['data'], df = test_data_loading()
    results['cv'] = test_cv_splits(df)
    results['metrics'] = test_metrics()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED - Ready to test models")
    else:
        print("SOME TESTS FAILED - Fix issues before proceeding")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)