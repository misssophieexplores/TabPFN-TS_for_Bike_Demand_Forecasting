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
import json

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from models.tabpfn_pipeline_model import TabPFNPipelineForecaster, TabPFNPipelineForecaster_NoWeather
from run_experiments import ForecastingExperiment, load_and_prepare_data


def main(config=None, no_confirm=False):
    """Main execution for weather degradation baseline.

    Parameters
    ----------
    config : ForecastConfig, optional
        Config to use. If None, creates a default ForecastConfig() — preserving
        the original behaviour when the script is run directly.
    no_confirm : bool, optional
        Skip the interactive confirmation prompt. Set to True when called
        programmatically from main.py so cluster jobs don't hang.
    """

    print("\n" + "="*80)
    print("WEATHER DEGRADATION BASELINE EXPERIMENTS")
    print("="*80 + "\n")

    if config is None:
        config = ForecastConfig()


    # Load data
    df, dataset_name = load_and_prepare_data(config)
    if config.dataset_name is None:
        config.dataset_name = dataset_name

    print("weather_covariates:", config.weather_covariates)
    from weather.weather_processor import WeatherProcessor
    wp = WeatherProcessor(config)
    print("clean_only cols:", wp.get_weather_columns("clean_only"))

    # Load tuned model parameters
    with open(config.arima_params_file) as f:
        arima_cfg = json.load(f)

    with open(config.sarimax_params_file) as f:
        sarimax_cfg = json.load(f)

    with open(config.xgb_params_file) as f:
        xgb_cfg = json.load(f)
    xgb_params = xgb_cfg["xgb_params"]
    n_lags = xgb_cfg["n_lags"]

    # All models
    all_models = [
        SeasonalNaiveForecaster(seasonal_period=config.seasonal_period),
        ARIMAForecaster(order=tuple(arima_cfg["order"])),
        SARIMAXForecaster(order=tuple(sarimax_cfg["order"]), seasonal_order=tuple(sarimax_cfg["seasonal_order"])),
        XGBoostForecaster(n_lags=n_lags, **xgb_params),

        # TabPFNForecaster_Custom(),
        # TabPFNForecaster_NoWeather(),
        TabPFNPipelineForecaster(),
        TabPFNPipelineForecaster_NoWeather()
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
    
    # Confirm before starting (skipped when called programmatically)
    if no_confirm:
        print("Confirmation skipped (no_confirm=True).")
    else:
        response = input("Start baseline run? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run experiments
    experiment = ForecastingExperiment(
        config=config,
        output_dir=config.output_dir,
        experiment_name=config.experiment_name
    )
    
    try:
        results_df = experiment.run_all_experiments(
            models=all_models,
            df=df,
            scenarios=scenarios,
            verbose=config.verbose
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
        detailed = f"{config.output_dir}/detailed_results_master_{config.results_version}.csv"
        print(f"Detailed results saved to: {detailed}")
        
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