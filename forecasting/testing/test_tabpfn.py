"""
Test script for TabPFN forecasters.
Run with: python test_tabpfn.py
"""
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ForecastConfig
from models.tabpfn_model import TabPFNForecaster, TabPFNForecaster_NoWeather
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator


def test_tabpfn_models(data_path: str = None):
    """Test both TabPFN variants"""
    
    # Determine data path
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "SeoulBikeData.csv"
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Setup
    config = ForecastConfig()
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    
    # Test both models
    models = [
        TabPFNForecaster(),
        TabPFNForecaster_NoWeather()
    ]
    
    # Test on single horizon
    horizon = 168
    print(f"\nTesting horizon: {horizon}h")
    
    splits = cv.split(df, horizon)
    
    # Test on first fold only
    train_df, test_df = splits[0]
    y_train = train_df[config.target_col].values
    y_test = test_df[config.target_col].values
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model.name}")
        print(f"{'='*60}")
        
        # Prepare covariates
        X_train = None
        X_test = None
        if model.use_covariates:
            X_train = train_df[config.weather_covariates]
            X_test = test_df[config.weather_covariates]
        
        try:
            # Check training size
            print(f"Training size: {len(y_train)} samples")
            if len(y_train) < 4096:
                print(f"WARNING: Need 4096+ samples, skipping")
                continue
            
            # Fit and predict
            print("Fitting...")
            model.reset()
            model.fit(y_train, X_train)
            
            print("Predicting...")
            y_pred = model.predict(horizon, X_test)
            
            # Calculate metrics
            metrics = calc.calculate_all(y_test, y_pred, y_train)
            
            print(f"\nResults:")
            print(f"  MAE:  {metrics['MAE']:6.1f}")
            print(f"  RMSE: {metrics['RMSE']:6.1f}")
            print(f"  MASE: {metrics['MASE']:5.2f}")
            print(f"  sMAPE: {metrics['sMAPE']:5.2f}")
            
            print(f"\nSUCCESS: {model.name} works!")
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_tabpfn_models()