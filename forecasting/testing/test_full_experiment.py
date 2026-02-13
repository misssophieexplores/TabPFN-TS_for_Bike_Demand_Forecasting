"""
Test full ForecastingExperiment with W&B (Phase 0.4 + 0.5 combined)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['WANDB_PROJECT'] = 'seoul-bike-testing'

from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from run_experiments import ForecastingExperiment, load_and_prepare_data


def main():
    print("\n" + "="*70)
    print("TEST: Full Experiment Runner with W&B")
    print("="*70)
    
    # Minimal config
    config = ForecastConfig()
    config.horizons = [6]
    config.n_folds = 3
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "SeoulBikeData.csv"
    df = load_and_prepare_data(str(data_path))
    
    # All models with optimized parameters
    models = [
        SeasonalNaiveForecaster(seasonal_period=24),
        ARIMAForecaster(order=(2, 1, 2)),
        SARIMAXForecaster(order=(4, 0, 0), seasonal_order=(1, 0, 1, 24)),
        XGBoostForecaster(n_lags=24)
    ]
    
    # Run experiment
    experiment = ForecastingExperiment(
        config=config,
        experiment_name="test_minimal"
    )
    
    try:
        results_df = experiment.run_all_experiments(models, df, verbose=True)
        
        print(f"\n{len(results_df)} results collected")
        print(results_df)
        
        # Save
        experiment.save_results(results_df)
        
        print("\nPASSED: Check results/ folder and W&B dashboard")
        
    finally:
        experiment.finish()


if __name__ == "__main__":
    main()