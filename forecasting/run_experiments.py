"""
Experiment runner with W&B logging and checkpointing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import wandb
import json

warnings.filterwarnings('ignore')

from config import ForecastConfig
from models.base import BaseForecaster
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster
from models.ml_models import XGBoostForecaster
from models.tabpfn_pipeline_model import TabPFNPipelineForecaster, TabPFNPipelineForecaster_NoWeather
from models.prophet_models import ProphetForecaster, NeuralProphetForecaster, NeuralProphetForecaster_NoWeather
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from weather.weather_processor import WeatherProcessor
from features import add_time_features

# Load environment variables
load_dotenv()

class ForecastingExperiment:
    """Manages and runs forecasting experiments with W&B logging"""
    
    def __init__(
        self, 
        config: ForecastConfig, 
        output_dir: str = None,
        experiment_name: Optional[str] = None
    ):
        self.config = config
        self.cv = TimeSeriesCV(config)
        self.metrics_calc = MetricsCalculator()
        self.results = []

        
        # Setup output directory
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        
        # Initialize W&B with credentials from .env
        self.run_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            entity=os.getenv('WANDB_ENTITY'),
            name=self.run_name,
            config={
                "dataset": config.dataset_name,
                "horizons": config.horizons,
                "n_folds": config.n_folds,
                "n_train_samples": config.n_train_samples,
            }
        )
        
        # Checkpoint file for recovery
        self.checkpoint_file = self.output_dir / f"checkpoint_{self.run_name}.json"
        self.completed_experiments = self._load_checkpoint()
        
        if config.verbose:
            print(f"W&B run: {wandb.run.url}")
            print(f"Run name: {self.run_name}")
        
    def _load_checkpoint(self) -> set:
        """Load completed experiments from checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(tuple(x) for x in data.get('completed', []))
        return set()
    
    def _save_checkpoint(self, model_name: str, horizon: int, scenario: str):
        """Save checkpoint after completing experiment"""
        self.completed_experiments.add((self.config.dataset_name, model_name, horizon, scenario))
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'completed': [list(x) for x in self.completed_experiments],
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def run_single_experiment(
        self,
        model: BaseForecaster,
        df: pd.DataFrame,
        horizon: int,
        weather_scenario: str = "all_weather",
        verbose: bool = True
    ) -> Optional[Dict]:
        """
        Run CV for one model-horizon-scenario combination with W&B logging.
        
        Parameters
        ----------
        model : BaseForecaster
            Forecasting model to evaluate
        df : pd.DataFrame
            Full dataset
        horizon : int
            Forecast horizon in hours
        weather_scenario : str, default='all_weather'
            Weather scenario: 'all_weather', 'clean_only', or 'degraded'
        verbose : bool, default=True
            Print progress messages
            
        Returns
        -------
        Optional[Dict]
            Aggregated results dictionary, or None if skipped/failed
        """
        
        # Skip if model doesn't use covariates and scenario is degraded
        if not model.use_covariates and weather_scenario == "degraded":
            if verbose:
                print(f"[SKIP] {model.name} | h={horizon} | {weather_scenario} "
                      f"(model doesn't use covariates, equivalent to clean_only)")
            return None
        
        # Skip if already completed
        if (self.config.dataset_name, model.name, horizon, weather_scenario) in self.completed_experiments:
            if verbose:
                print(f"[SKIP] {model.name} | h={horizon} | {weather_scenario} (already completed)")
            return None
        
        if verbose:
            print(f"[RUN] {model.name} | h={horizon} | {weather_scenario}", end="", flush=True)
        
        # Initialize weather processor
        weather_proc = WeatherProcessor(self.config)
        
        splits = self.cv.split(df, horizon)
        fold_results = []
        
        for fold_idx, (train_df, test_df) in enumerate(splits):
            y_train = train_df[self.config.target_col].values
            y_test = test_df[self.config.target_col].values
            
            # Prepare weather data using WeatherProcessor
            X_train = None
            X_test = None
            if model.use_covariates:
                X_train = weather_proc.prepare_weather_data(
                    train_df, weather_scenario, horizon, fold_idx, split="train"
                )
                X_test = weather_proc.prepare_weather_data(
                    test_df, weather_scenario, horizon, fold_idx, split="test"
                )

            # Append calendar time features for models that need them (e.g. XGBoost).
            # Done here so timestamps are available — models never see raw datetimes.
            if model.use_time_features:
                time_train = add_time_features(train_df, self.config.date_col)
                time_test = add_time_features(test_df, self.config.date_col)
                X_train = pd.concat([X_train.reset_index(drop=True), time_train], axis=1) if X_train is not None else time_train
                X_test = pd.concat([X_test.reset_index(drop=True), time_test], axis=1) if X_test is not None else time_test

            # Attach real DatetimeIndex for models that need timestamps (Prophet, TabPFN).
            # This replaces any previously hardcoded date ranges in those models.
            if getattr(model, "needs_datetime", False):
                train_dates = pd.DatetimeIndex(train_df[self.config.date_col].values)
                test_dates  = pd.DatetimeIndex(test_df[self.config.date_col].values)
                if X_train is None:
                    X_train = pd.DataFrame(index=train_dates)
                else:
                    X_train = X_train.set_index(train_dates)
                if X_test is None:
                    X_test = pd.DataFrame(index=test_dates)
                else:
                    X_test = X_test.set_index(test_dates)



            try:
                model.reset()
                model.fit(y_train, X_train)
                y_pred = model.predict(horizon, X_test)
                
                # Calculate metrics
                metrics = self.metrics_calc.calculate_all(y_test, y_pred, y_train)
                metrics['dataset'] = self.config.dataset_name
                metrics['run_name'] = self.run_name
                metrics['version'] = self.config.results_version
                metrics['timestamp'] = datetime.now().isoformat()
                metrics['fold'] = fold_idx
                metrics['model'] = model.name
                metrics['horizon'] = horizon

                # Track weather scenario information
                metrics['weather_scenario'] = weather_scenario
                metrics['model_uses_covariates'] = model.use_covariates
                metrics['degradation_seed'] = self.config.degradation_seed
                metrics['num_weather_vars'] = len(X_train.columns) if X_train is not None else 0

                # Track imputation info (no printing)
                fday = self.config.functioning_day_col
                test_imputed = (test_df[fday] == 'No').sum() if fday and fday in test_df.columns else 0
                train_imputed = (train_df[fday] == 'No').sum() if fday and fday in train_df.columns else 0
                metrics['test_imputed'] = test_imputed
                metrics['train_imputed'] = train_imputed

                fold_results.append(metrics)

                # Log fold results to W&B
                # wandb.log({
                #     f"{model.name}_h{horizon}_fold{fold_idx}_MAE": metrics['MAE'],
                #     f"{model.name}_h{horizon}_fold{fold_idx}_RMSE": metrics['RMSE'],
                #     f"{model.name}_h{horizon}_fold{fold_idx}_MASE": metrics['MASE'],
                #     "fold": fold_idx,
                #     "horizon": horizon,
                # })
                
                
                if verbose and (fold_idx + 1) % 5 == 0:
                    print(".", end="", flush=True)
                    
            except Exception as e:
                error_msg = f"Error in {model.name} h={horizon} fold={fold_idx}: {str(e)}"
                print(f"\n[ERROR] {error_msg}")
                import traceback
                error_log_path = Path(self.config.output_dir) / f"errors_{self.config.results_version}.log"
                with open(error_log_path, "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {error_msg}\n")
                    f.write(f"{'='*80}\n")
                    traceback.print_exc(file=f)
                wandb.log({"error": error_msg})
                continue
        
        if len(fold_results) == 0:
            print(f" [FAILED] All folds failed")
            return None
        
        # Aggregate results
        results_df = pd.DataFrame(fold_results)
        aggregated = {
            'dataset': self.config.dataset_name,
            'run_name': self.run_name,
            'version': self.config.results_version,
            'timestamp': datetime.now().isoformat(),
            'model': model.name,
            'horizon': horizon,
            'weather_scenario': weather_scenario,
            'model_uses_covariates': model.use_covariates,
            'degradation_seed': self.config.degradation_seed,
            'num_weather_vars': results_df['num_weather_vars'].iloc[0] if len(results_df) > 0 else 0,
            'n_folds': len(fold_results),
            'MAE_mean': results_df['MAE'].mean(),
            'MAE_std': results_df['MAE'].std(),
            'RMSE_mean': results_df['RMSE'].mean(),
            'RMSE_std': results_df['RMSE'].std(),
            'MASE_mean': results_df['MASE'].mean(),
            'MASE_std': results_df['MASE'].std(),
            'sMAPE_mean': results_df['sMAPE'].mean(),
            'sMAPE_std': results_df['sMAPE'].std(),
            'total_test_imputed': results_df['test_imputed'].sum(),
            'total_train_imputed': results_df['train_imputed'].sum(),
            'folds_with_imputed_test': (results_df['test_imputed'] > 0).sum(),
        }
        
        # Log aggregated results to W&B
        wandb.log({
            f"{model.name}_{weather_scenario}_h{horizon}_MAE": aggregated['MAE_mean'],
            f"{model.name}_{weather_scenario}_h{horizon}_RMSE": aggregated['RMSE_mean'],
            f"{model.name}_{weather_scenario}_h{horizon}_MASE": aggregated['MASE_mean'],
            f"{model.name}_{weather_scenario}_h{horizon}_sMAPE": aggregated['sMAPE_mean'],
        })
        
        self.results.append(aggregated)
        self._save_checkpoint(model.name, horizon, weather_scenario)
        
        if verbose:
            print(f" [DONE] ({len(fold_results)} folds) | MAE: {aggregated['MAE_mean']:.1f}")
        
        # Log all folds as table
        fold_table = wandb.Table(dataframe=pd.DataFrame(fold_results))
        wandb.log({f"{model.name}_{weather_scenario}_h{horizon}_folds": fold_table})

        return aggregated, fold_results
    
    def run_all_experiments(
        self,
        models: List[BaseForecaster],
        df: pd.DataFrame,
        scenarios: List[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run all model-horizon-scenario combinations.
        
        Parameters
        ----------
        models : List[BaseForecaster]
            List of forecasting models to evaluate
        df : pd.DataFrame
            Full dataset
        scenarios : List[str], optional
            List of weather scenarios to test. If None, uses config.weather_scenarios
        verbose : bool, default=True
            Print progress messages
            
        Returns
        -------
        pd.DataFrame
            Results dataframe with all completed experiments
        """
        
        if scenarios is None:
            scenarios = self.config.weather_scenarios
        
        # Calculate total experiments (accounting for skipped combinations)
        total = 0
        for scenario in scenarios:
            for model in models:
                # Count only if model will run
                if not (not model.use_covariates and scenario == "degraded"):
                    total += len(self.config.horizons)
        
        completed = len(self.completed_experiments)
        
        if verbose:
            print(f"Experiments: {len(models)} models x {len(self.config.horizons)} horizons x {len(scenarios)} scenarios")
            print(f"Total: {total} (some skipped for models without covariates)")
            if completed > 0:
                print(f"Resuming from checkpoint: {completed} already completed")

        all_fold_results = []
        
        for scenario in scenarios:
            for model in models:
                # Skip models without covariates for degraded scenario
                if not model.use_covariates and scenario == "degraded":
                    if verbose:
                        print(f"[SKIP] {model.name} for {scenario} (no covariates)\n")
                    continue
                
                for horizon in self.config.horizons:
                    result = self.run_single_experiment(
                        model, df, horizon, scenario, verbose
                    )

                    if result is not None:  
                        aggregated, fold_results = result
                        all_fold_results.extend(fold_results)
                    completed += 1
                    
                    if verbose:
                        print(f"Progress: {completed}/{total}\n")
        
        results_df = pd.DataFrame(self.results)
        
        # Log final results table to W&B
        if len(results_df) > 0:
            summary = results_df.groupby(['model', 'weather_scenario'])[['MAE_mean', 'RMSE_mean', 'MASE_mean']].mean()
            wandb.log({
                "results_table": wandb.Table(dataframe=results_df),
                "results_summary": wandb.Table(dataframe=summary.reset_index())
            })

        # Save detailed fold results
        self.detailed_results = all_fold_results

        return results_df
    def save_results(self, results_df: pd.DataFrame) -> str:
        """Append results to master CSV"""
        
        # Aggregated results - APPEND mode
        filename_agg = self.output_dir / f"results_master_{self.config.results_version}.csv"
        if filename_agg.exists():
            existing = pd.read_csv(filename_agg)
            results_df = pd.concat([existing, results_df], ignore_index=True)
        results_df.to_csv(filename_agg, index=False)
        
        # Detailed results - APPEND mode
        filename_detailed = self.output_dir / f"detailed_results_master_{self.config.results_version}.csv"
        if hasattr(self, 'detailed_results') and self.detailed_results:
            detailed_df = pd.DataFrame(self.detailed_results)
            if filename_detailed.exists():
                existing = pd.read_csv(filename_detailed)
                detailed_df = pd.concat([existing, detailed_df], ignore_index=True)
            detailed_df.to_csv(filename_detailed, index=False)
        
        if self.config.verbose:
            print(f"Results appended to: {filename_agg}")
        return str(filename_agg)

    def finish(self):
        """Cleanup and finish W&B run"""
        wandb.finish()
        if self.config.verbose:
            print("W&B run finished")



def load_and_prepare_data(config: ForecastConfig) -> tuple[pd.DataFrame, str]:
    """
    Load and prepare dataset using paths and column names from config.

    Returns
    --------
    tuple[pd.DataFrame, str]
        Prepared dataset and dataset name extracted from filename
    """
    filepath = Path("data") / config.data_filename
    df = pd.read_csv(filepath)

    # Parse datetime and sort
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.sort_values(config.date_col).reset_index(drop=True)

    # Drop duplicate timestamps (e.g. DST clock-back hours)
    n_dupes = df[config.date_col].duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates(subset=config.date_col, keep="first").reset_index(drop=True)
        if config.verbose:
            print(f"Dropped {n_dupes} duplicate timestamps (DST)")

    # Extract dataset name from filename (stem without extension)
    dataset_name = Path(config.data_filename).stem

    # Apply column-specific scale factors if defined in config
    for col, factor in config.column_scale_factors.items():
        if col in df.columns:
            df[col] = df[col] * factor

    # Normalize and append holiday column (→ 0/1 int)
    if config.holiday_col and config.holiday_col in df.columns:
        col = df[config.holiday_col]
        if col.dtype == object:
            df[config.holiday_col] = col.map(config.holiday_mapping).astype(int)
        else:
            df[config.holiday_col] = pd.to_numeric(col, errors='coerce').fillna(0).astype(int)
        if config.holiday_col not in config.weather_covariates:
            config.weather_covariates.append(config.holiday_col)
            if config.verbose:
                print(f"Holiday column '{config.holiday_col}' added to weather_covariates")
    elif config.holiday_col:
        if config.verbose:
            print(f"Warning: holiday_col '{config.holiday_col}' not found in dataset — skipping")

    # Normalize and append season column (→ 0–3 int) using explicit per-dataset mapping
    if config.season_col and config.season_col in df.columns:
        if config.season_mapping is None:
            raise ValueError(
                f"season_col '{config.season_col}' is set but season_mapping is None. "
                f"Please define season_mapping in your config for this dataset."
            )
        mapped = df[config.season_col].map(config.season_mapping)
        if mapped.isna().any():
            unmapped = df[config.season_col][mapped.isna()].unique().tolist()
            raise ValueError(
                f"season_mapping produced NaN values — these raw values are not covered: "
                f"{unmapped}. Check config.season_mapping."
            )
        df[config.season_col] = mapped.astype(int)
        if config.season_col not in config.weather_covariates:
            config.weather_covariates.append(config.season_col)
            if config.verbose:
                print(f"Season column '{config.season_col}' added to weather_covariates")
    elif config.season_col:
        if config.verbose:
            print(f"Warning: season_col '{config.season_col}' not found in dataset — skipping")

    if config.verbose:
        print(f"Loaded data: {len(df)} observations")
        print(f"Date range: {df[config.date_col].min()} to {df[config.date_col].max()}")
        print(f"Dataset name: {dataset_name}")

    return df, dataset_name


def main():
    """Main execution"""

    config = ForecastConfig()

    # Load data using config (no separate filepath argument)
    df, dataset_name = load_and_prepare_data(config)
    if config.dataset_name is None:
        config.dataset_name = dataset_name
    if config.experiment_name is None or config.experiment_name.startswith("None"):
        config.experiment_name = f"{config.dataset_name}_{config.results_version}"

    # Load tuned model parameters
    with open(config.arima_params_file) as f:
        arima_cfg = json.load(f)

    with open(config.sarimax_params_file) as f:
        sarimax_cfg = json.load(f)

    with open(config.xgb_params_file) as f:
        xgb_cfg = json.load(f)
    xgb_params = xgb_cfg["xgb_params"]
    n_lags = xgb_cfg["n_lags"]

    models = [
        SeasonalNaiveForecaster(seasonal_period=config.seasonal_period),
        ARIMAForecaster(order=tuple(arima_cfg["order"])),
        SARIMAXForecaster(order=tuple(sarimax_cfg["order"]), seasonal_order=tuple(sarimax_cfg["seasonal_order"])),
        XGBoostForecaster(n_lags=n_lags, **xgb_params),
        ProphetForecaster(),
        NeuralProphetForecaster(),
        NeuralProphetForecaster_NoWeather(),
        TabPFNPipelineForecaster(),
        TabPFNPipelineForecaster_NoWeather(),
    ]

    experiment = ForecastingExperiment(
        config=config,
        output_dir=config.output_dir,
        experiment_name=f"baseline_models_{config.results_version}"
    )

    try:
        results_df = experiment.run_all_experiments(models, df, verbose=True)

        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print("\nBy Model:")
        print(results_df.groupby('model')[['MAE_mean', 'RMSE_mean', 'MASE_mean']].mean().round(2))
        print("\nBy Horizon:")
        print(results_df.groupby('horizon')[['MAE_mean', 'RMSE_mean', 'MASE_mean']].mean().round(2))

        experiment.save_results(results_df)

    except KeyboardInterrupt:
        print("\n\nInterrupted - Progress saved to checkpoint")
    except Exception as e:
        import traceback
        error_log_path = Path(config.output_dir) / f"errors_{config.results_version}.log"
        error_log_path.parent.mkdir(exist_ok=True)
        print(f"\n[FAILED] {config.dataset_name}: {e}")
        print(f"  See {error_log_path} for details")
        with open(error_log_path, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FAILED: {config.dataset_name}\n")
            f.write(f"{'='*80}\n")
            traceback.print_exc(file=f)
        raise
    finally:
        experiment.finish()


if __name__ == "__main__":
    main()