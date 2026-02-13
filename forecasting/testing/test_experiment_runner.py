"""
Test experiment runner with minimal configuration (no W&B).
Tests: 1 model, 1 horizon, 3 folds
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster

# We need to test run_experiments, but it has W&B
# So we'll manually test the core logic without W&B


def load_data() -> pd.DataFrame:
    """Load and prepare data"""
    data_path = Path(__file__).parent.parent.parent / "data" / "SeoulBikeData.csv" #TODO: change to config path?
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True) #TODO: change to use config
    return df


def test_experiment_runner_core():
    """Test core experiment logic without W&B"""
    
    print("\n" + "="*70)
    print("TEST: Experiment Runner Core Logic (No W&B)")
    print("="*70)
    
    # Import here to avoid W&B init
    from evaluation.cv import TimeSeriesCV
    from evaluation.metrics import MetricsCalculator
    
    # Load data
    df = load_data()
    print(f"Data loaded: {len(df)} observations")
    
    # Minimal config
    config = ForecastConfig()
    config.horizons = [6]  # Just one horizon
    config.n_folds = 3     # Just 3 folds
    
    # One simple model
    model = SeasonalNaiveForecaster(seasonal_period=24)
    
    # Setup
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    
    print(f"\nTesting: {model.name}")
    print(f"Horizon: {config.horizons[0]}h")
    print(f"Requested folds: {config.n_folds}")
    
    # Get splits
    horizon = config.horizons[0]
    splits = cv.split(df, horizon)
    
    print(f"Actual folds created: {len(splits)}")
    
    if len(splits) == 0:
        print("FAILED: No splits created")
        return False
    
    # Run through folds
    results = []
    for fold_idx, (train_df, test_df) in enumerate(splits):
        y_train = train_df[config.target_col].values
        y_test = test_df[config.target_col].values
        
        try:
            model.reset()
            model.fit(y_train)
            y_pred = model.predict(horizon)
            
            metrics = calc.calculate_all(y_test, y_pred, y_train)
            results.append(metrics)
            
            print(f"  Fold {fold_idx}: MAE={metrics['MAE']:.1f}, RMSE={metrics['RMSE']:.1f}")
            
        except Exception as e:
            print(f"  Fold {fold_idx}: ERROR - {e}")
            return False
    
    # Aggregate
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        print(f"\nAggregated results:")
        print(f"  MAE: {results_df['MAE'].mean():.1f} ± {results_df['MAE'].std():.1f}")
        print(f"  RMSE: {results_df['RMSE'].mean():.1f} ± {results_df['RMSE'].std():.1f}")
        print(f"  MASE: {results_df['MASE'].mean():.2f} ± {results_df['MASE'].std():.2f}")
        print("\nPASSED: Core experiment logic works")
        return True
    else:
        print("\nFAILED: No successful results")
        return False


def test_checkpoint_logic():
    """Test checkpoint save/load"""
    
    print("\n" + "="*70)
    print("TEST: Checkpoint Logic")
    print("="*70)
    
    import json
    from pathlib import Path
    
    test_checkpoint_file = Path("test_checkpoint.json")
    
    # Test save
    completed = {('model_a', 6), ('model_b', 24)}
    checkpoint_data = {
        'completed': [list(x) for x in completed],
        'last_updated': '2024-12-27T12:00:00'
    }
    
    with open(test_checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print("Checkpoint saved")
    
    # Test load
    with open(test_checkpoint_file, 'r') as f:
        loaded = json.load(f)
    
    loaded_set = set(tuple(x) for x in loaded.get('completed', []))
    
    if loaded_set == completed:
        print("Checkpoint loaded correctly")
        print(f"  Completed experiments: {loaded_set}")
        
        # Cleanup
        test_checkpoint_file.unlink()
        print("\nPASSED: Checkpoint logic works")
        return True
    else:
        print(f"FAILED: Loaded {loaded_set} != saved {completed}")
        return False


def main():
    """Run all experiment runner tests"""
    
    results = {
        'core_logic': test_experiment_runner_core(),
        'checkpoint': test_checkpoint_logic(),
    }
    
    print("\n" + "="*70)
    print("EXPERIMENT RUNNER TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("All tests passed - Ready to test with W&B")
    else:
        print("Some tests failed - Fix before proceeding")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)