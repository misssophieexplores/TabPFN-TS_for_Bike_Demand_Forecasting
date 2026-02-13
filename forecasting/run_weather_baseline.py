"""
Weather Degradation Baseline Experiments

Runs complete baseline with weather scenarios:
- clean_only: 7 degradable variables, no degradation (NEW baseline)
- degraded: 7 degradable variables, with forecast errors (robustness test)

Optimizations:
- Models without covariates automatically skipped in degraded scenario
- On-the-fly degradation (no pre-computed files)
- Full reproducibility with seed=42

Usage:
  python run_weather_baseline.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

import json
with open("results/tuning/xgboost_best_params_20260110_042536.json") as f:
    xgb_cfg = json.load(f)

xgb_params = xgb_cfg["xgb_params"]
n_lags = xgb_cfg["n_lags"]


from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from models.tabpfn_model import TabPFNForecaster, TabPFNForecaster_NoWeather
from run_experiments import ForecastingExperiment, load_and_prepare_data


def main():
    """Main execution for weather degradation baseline"""
    
    print("\n" + "="*80)
    print("WEATHER DEGRADATION BASELINE EXPERIMENTS")
    print("="*80 + "\n")
    
    # Configuration
    config = ForecastConfig()
    config.horizons = [6, 24, 48, 168]  # All horizons
    config.n_folds = 20  # Full baseline
    
    # Load data
    df, dataset_name = load_and_prepare_data("data/SeoulBikeData.csv")
    
    if config.dataset_name is None:
        config.dataset_name = dataset_name
    
    # All models
    all_models = [
        SeasonalNaiveForecaster(seasonal_period=24),
        ARIMAForecaster(order=(2, 1, 2)),
        SARIMAXForecaster(order=(4, 0, 0), seasonal_order=(1, 0, 1, 24)),
        XGBoostForecaster(
            n_lags=n_lags,
            **xgb_params,
        ),
        TabPFNForecaster(),
        TabPFNForecaster_NoWeather(),
    ]
    
    # Scenarios to run
    scenarios = [
        "clean_only",   # NEW baseline (7 vars, no degradation)
        "degraded"      # Degraded weather (7 vars, with degradation)
        # "all_weather" # Optional: can reuse existing baseline results
    ]
    
    print("Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Models: {len(all_models)}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Horizons: {config.horizons}")
    print(f"  Folds: {config.n_folds}")
    print(f"  Degradation seed: {config.degradation_seed}")
    print()
    
    # Model summary
    models_with_cov = [m for m in all_models if m.use_covariates]
    models_without_cov = [m for m in all_models if not m.use_covariates]
    
    print("Model Summary:")
    print(f"  With weather covariates: {[m.name for m in models_with_cov]}")
    print(f"  Without weather covariates: {[m.name for m in models_without_cov]}")
    print()
    
    print("Execution Plan:")
    print(f"  clean_only: ALL {len(all_models)} models")
    print(f"  degraded: {len(models_with_cov)} models (skip {len(models_without_cov)} without covariates)")
    print()
    
    # Confirm before starting
    response = input("Start baseline run? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    experiment = ForecastingExperiment(
        config=config,
        output_dir="results",
        experiment_name="seoul_bike_baseline_v3"
    )
    
    try:
        results_df = experiment.run_all_experiments(
            models=all_models,
            df=df,
            scenarios=scenarios,
            verbose=True
        )
        
        # Display summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        if len(results_df) > 0:
            print("\nBy Model and Scenario:")
            summary = results_df.groupby(['model', 'weather_scenario'])[
                ['MAE_mean', 'RMSE_mean', 'MASE_mean']
            ].mean().round(2)
            print(summary)
            
            print("\nBy Scenario:")
            scenario_summary = results_df.groupby('weather_scenario')[
                ['MAE_mean', 'RMSE_mean', 'MASE_mean']
            ].mean().round(2)
            print(scenario_summary)
            
            # Degradation impact analysis (for models with covariates)
            cov_results = results_df[results_df['model_uses_covariates'] == True]
            
            if len(cov_results) > 0:
                print("\nDegradation Impact (models with covariates only):")
                
                for model_name in cov_results['model'].unique():
                    model_data = cov_results[cov_results['model'] == model_name]
                    
                    clean_data = model_data[model_data['weather_scenario'] == 'clean_only']
                    deg_data = model_data[model_data['weather_scenario'] == 'degraded']
                    
                    if len(clean_data) > 0 and len(deg_data) > 0:
                        # Average across horizons
                        clean_mae = clean_data['MAE_mean'].mean()
                        deg_mae = deg_data['MAE_mean'].mean()
                        impact = deg_mae - clean_mae
                        impact_pct = (impact / clean_mae) * 100
                        
                        print(f"  {model_name:20s}: {clean_mae:6.1f} → {deg_mae:6.1f} "
                              f"({impact:+6.1f}, {impact_pct:+5.1f}%)")
        
        # Save results
        output_file = experiment.save_results(results_df)
        print(f"\nResults saved to: {output_file}")
        print(f"Detailed results saved to: results/detailed_results_master_v3.csv")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted - Progress saved to checkpoint")
        print("Restart with same experiment name to resume")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        experiment.finish()
        print("\n" + "="*80)
        print("WEATHER BASELINE COMPLETE")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
