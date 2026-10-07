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
import time
from datetime import datetime
from dotenv import load_dotenv
import wandb
import json


warnings.filterwarnings('ignore')

from config import ForecastConfig
from features import prepare_xgboost_features
from models.base import BaseForecaster
from models.statistical import SeasonalNaiveForecaster, ARIMAForecaster, SARIMAXForecaster, trend_from_intercept
from models.ml_models import XGBoostForecaster, XGBoostForecaster_NoWeather
from models.tabpfn_pipeline_model import TabPFNPipelineForecaster, TabPFNPipelineForecaster_NoWeather
from models.prophet_models import ProphetForecaster, NeuralProphetForecaster, NeuralProphetForecaster_NoWeather
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from weather.weather_processor import WeatherProcessor
from features import add_time_features
from models.timesfm_model import TimesFMForecaster, TimesFMForecaster_NoWeather
from provenance import get_code_version, get_library_versions
# prepare_fold_inputs is imported from here by the tuning scripts and tests
from fold_runner import prepare_fold_inputs, run_fold, run_folds_parallel, fold_workers  # noqa: F401


# Load environment variables
load_dotenv()

# Name used as baseline for skill_score / win_rate comparisons
BASELINE_MODEL = "Seasonal_Naive"

# Model keys accepted by build_models(), in paper order
MODEL_KEYS = [
    "seasonal_naive", "arima", "sarimax", "xgboost", "xgboost_noweather", "prophet",
    "neuralprophet", "neuralprophet_noweather",
    "tabpfn", "tabpfn_noweather", "timesfm", "timesfm_noweather",
]


def _append_rows(path: Path, rows: pd.DataFrame) -> None:
    """Append rows to a CSV; refuse if the existing header has other columns."""
    if path.exists():
        existing_cols = list(pd.read_csv(path, nrows=0).columns)
        if existing_cols != list(rows.columns):
            raise RuntimeError(
                f"{path} has different columns than the rows being written. "
                f"Move it away or use a new results_version."
            )
    rows.to_csv(path, mode='a', header=not path.exists(), index=False)


def _load_params(path: Optional[str], field: str, how_to_create: str) -> dict:
    if not path:  # None or "" (open("") would only say "No such file: ''")
        raise ValueError(f"config.{field} is not set ({path!r}). {how_to_create}")
    with open(path) as f:
        params = json.load(f)
    # Params files from older tuning code (no provenance) were tuned with a
    # different procedure and must not be used for paper runs.
    if "provenance" not in params:
        raise ValueError(
            f"config.{field} = {path} was written by older tuning code (no "
            f"'provenance'). Re-tune. {how_to_create}"
        )
    if params["provenance"].get("git_dirty"):
        print(f"WARNING: {path} was tuned with uncommitted code changes "
              f"(commit {params['provenance'].get('git_commit')})")
    return params


def build_models(config: ForecastConfig, keys: Optional[List[str]] = None) -> List[BaseForecaster]:
    """
    Build the forecasters with the tuned parameters from the params files named
    in the city config. Single source of truth for run_weather_baseline.py,
    run_experiments.main() and the tests. keys=None builds all MODEL_KEYS;
    only the params files of the requested models are read.
    Checked here: provenance present, and the scenario of
    xgb_noweather_params_file. City, n_train_samples, tuning period and the
    other scenarios are checked by testing/preflight.py.
    """
    keys = list(MODEL_KEYS) if keys is None else list(keys)
    unknown = [k for k in keys if k not in MODEL_KEYS]
    if unknown:
        raise ValueError(f"Unknown model keys {unknown}; available: {MODEL_KEYS}")

    models = []
    for key in keys:
        if key == "seasonal_naive":
            models.append(SeasonalNaiveForecaster(seasonal_period=config.seasonal_period))
        elif key == "arima":
            cfg = _load_params(config.arima_params_file, "arima_params_file", "Run tune_arima.py.")
            order = tuple(cfg["order"])
            # Intercept selected during tuning -> statsmodels trend. A params
            # file without with_intercept raises KeyError: re-run tuning.
            trend = trend_from_intercept(cfg["with_intercept"], order, (0, 0, 0, 0))
            models.append(ARIMAForecaster(order=order, trend=trend))
        elif key == "sarimax":
            cfg = _load_params(config.sarimax_params_file, "sarimax_params_file", "Run tune_sarimax.py.")
            order, seasonal_order = tuple(cfg["order"]), tuple(cfg["seasonal_order"])
            trend = trend_from_intercept(cfg["with_intercept"], order, seasonal_order, sarimax=True)
            models.append(SARIMAXForecaster(order=order, seasonal_order=seasonal_order, trend=trend))
        elif key == "xgboost":
            cfg = _load_params(config.xgb_params_file, "xgb_params_file", "Run tune_xgboost.py.")
            models.append(XGBoostForecaster(n_lags=cfg["n_lags"], **cfg["xgb_params"]))
        elif key == "xgboost_noweather":
            cfg = _load_params(config.xgb_noweather_params_file, "xgb_noweather_params_file",
                               "Run tune_xgboost.py --scenario no_weather for this city first.")
            if cfg.get("scenario") != "no_weather":
                raise ValueError(f"config.xgb_noweather_params_file = {config.xgb_noweather_params_file} "
                                 f"was tuned with scenario '{cfg.get('scenario')}', not 'no_weather'.")
            models.append(XGBoostForecaster_NoWeather(n_lags=cfg["n_lags"], **cfg["xgb_params"]))
        elif key == "prophet":
            cfg = _load_params(config.prophet_params_file, "prophet_params_file", "Run tune_prophet.py.")
            models.append(ProphetForecaster(**cfg["prophet_params"]))
        elif key == "neuralprophet":
            cfg = _load_params(config.neuralprophet_params_file, "neuralprophet_params_file",
                               "Run tune_neuralprophet.py.")
            models.append(NeuralProphetForecaster(n_lags=cfg["n_lags"], **cfg["neuralprophet_params"]))
        elif key == "neuralprophet_noweather":
            cfg = _load_params(config.neuralprophet_noweather_params_file,
                               "neuralprophet_noweather_params_file",
                               "Run tune_neuralprophet.py --scenario no_weather for this city first.")
            models.append(NeuralProphetForecaster_NoWeather(n_lags=cfg["n_lags"], **cfg["neuralprophet_params"]))
        elif key == "tabpfn":
            models.append(TabPFNPipelineForecaster())
        elif key == "tabpfn_noweather":
            models.append(TabPFNPipelineForecaster_NoWeather())
        elif key == "timesfm":
            models.append(TimesFMForecaster())
        elif key == "timesfm_noweather":
            models.append(TimesFMForecaster_NoWeather())
    return models


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
        self.failed_experiments = []

        
        # Setup output directory
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        
        # Git commit of the code, written to every results row
        self.code_version = get_code_version()
        self.library_versions = get_library_versions()

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
                **self.code_version,
                "library_versions": self.library_versions,
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
            # A checkpoint from older code would silently skip experiments
            if 'code_version' not in data:
                raise RuntimeError(
                    f"{self.checkpoint_file} was written by older code (no code_version). "
                    f"Move it away (with the results files) or use a new results_version."
                )
            if data['code_version'].get('git_commit') != self.code_version['git_commit']:
                print(f"WARNING: resuming checkpoint from commit "
                      f"{data['code_version'].get('git_commit')}; current commit "
                      f"{self.code_version['git_commit']}. Completed experiments keep their old commit.")
            return set(tuple(x) for x in data.get('completed', []))
        return set()
    
    def _save_checkpoint(self, model_name: str, horizon: int, scenario: str):
        """Save checkpoint after completing experiment"""
        self.completed_experiments.add((self.config.dataset_name, model_name, horizon, scenario))
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'completed': [list(x) for x in self.completed_experiments],
                'last_updated': datetime.now().isoformat(),
                'code_version': self.code_version,
            }, f, indent=2)
    

    def _log_fold_error(self, model, horizon, fold_idx, error, traceback_text):
        error_msg = f"Error in {model.name} h={horizon} fold={fold_idx}: {error}"
        print(f"\n[ERROR] {error_msg}")
        error_log_path = Path(self.config.output_dir) / f"errors_{self.config.results_version}.log"
        with open(error_log_path, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {error_msg}\n")
            f.write(f"{'='*80}\n")
            f.write(traceback_text)
        wandb.log({"error": error_msg})

    def run_single_experiment(
        self,
        model: BaseForecaster,
        df: pd.DataFrame,
        horizon: int,
        weather_scenario: str = "all_weather",
        verbose: bool = True
    ) -> Optional[tuple]:
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
            Weather scenario: 'all_weather', 'clean_only', or a degraded
            scenario (a key of config.degradation_scales)
        verbose : bool, default=True
            Print progress messages
            
        Returns
        -------
        Optional[tuple]
            (aggregated results dict, list of fold-level dicts), or None if
            skipped. A failed fold raises (the whole run is aborted).
        """

        # Skip if model doesn't use covariates and scenario is degraded
        # ('degraded' or a noise-magnitude sensitivity scenario)
        if not model.use_covariates and self.config.is_degraded(weather_scenario):
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

        # partial_last_fold=True: if the evaluation period is not a multiple of
        # the horizon, the last fold is scored on the remaining hours instead of
        # being dropped, so every horizon covers the same evaluation period.
        splits = self.cv.split(df, horizon, partial_last_fold=True)
        expected_folds = self.cv.expected_n_folds(horizon)
        eval_hours = self.cv.get_eval_hours()
        covered_hours = sum(len(test_df) for _, test_df in splits)
        if len(splits) != expected_folds or covered_hours != eval_hours:
            raise RuntimeError(
                f"{self.config.dataset_name} h={horizon}: {len(splits)} CV folds covering "
                f"{covered_hours} h, expected {expected_folds} folds covering {eval_hours} h. "
                f"Check the data for missing hourly timestamps."
            )
        fold_results = []
        fold_forecasts = []  # hourly forecasts, one DataFrame per successful fold

        # Models with parallel_folds (SARIMAX) fit their folds in parallel
        # worker processes, one math thread each; the folds are independent,
        # so the results are the same as one after another (fold_runner.py).
        n_workers = fold_workers() if getattr(model, "parallel_folds", False) else 1
        t_start = time.time()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START {self.config.dataset_name} | "
              f"{model.name} | h={horizon} | {weather_scenario} | {len(splits)} folds"
              + (f" | {n_workers} workers" if n_workers > 1 else ""), flush=True)

        if n_workers > 1:
            outcomes = run_folds_parallel(
                self.config, model, weather_proc, splits, weather_scenario,
                horizon, self.run_name, self.code_version, n_workers,
            )
            for fold_idx, outcome in enumerate(outcomes):
                if outcome[0] == "error":
                    # First failing fold, as in a sequential run
                    self._log_fold_error(model, horizon, fold_idx, outcome[1], outcome[2])
                    raise RuntimeError(outcome[1])
                fold_results.append(outcome[1])
                fold_forecasts.append(outcome[2])
        else:
            for fold_idx, (train_df, test_df) in enumerate(splits):
                try:
                    # Inside the try: an error in the input preparation (e.g. the
                    # weather degradation) is written to the errors log with its
                    # traceback, like a model error.
                    metrics, forecasts = run_fold(
                        self.config, model, weather_proc, train_df, test_df,
                        weather_scenario, horizon, fold_idx, self.run_name, self.code_version,
                    )
                except Exception as e:
                    import traceback
                    self._log_fold_error(model, horizon, fold_idx, str(e), traceback.format_exc())
                    # Abort only this model/horizon/scenario. run_all_experiments()
                    # catches the exception, leaves it uncheckpointed, and continues.
                    raise
                fold_results.append(metrics)
                fold_forecasts.append(forecasts)

                if verbose and (fold_idx + 1) % 5 == 0:
                    print(".", end="", flush=True)

        # Not reachable at present: a fold error re-raises above, so every
        # completed run has all folds (n_failed_folds = 0; the column is kept
        # for schema compatibility).
        if len(fold_results) == 0:
            print(f" [FAILED] All folds failed")
            return None
        n_failed_folds = expected_folds - len(fold_results)
        if n_failed_folds > 0:
            # Would mean fewer folds than the other models (not comparable).
            print(f"\n[WARN] {model.name} | h={horizon} | {weather_scenario}: "
                  f"{n_failed_folds}/{expected_folds} folds failed (see errors log)")

        # Aggregate fold results. Folds whose test window is fully imputed
        # have NaN metrics; pandas mean/std skip them (skipna).
        results_df = pd.DataFrame(fold_results)
        aggregated = {
            'dataset': self.config.dataset_name,
            'run_name': self.run_name,
            'version': self.config.results_version,
            **self.code_version,
            'library_versions': json.dumps(self.library_versions, sort_keys=True),
            'timestamp': datetime.now().isoformat(),
            'model': model.name,
            'horizon': horizon,
            'weather_scenario': weather_scenario,
            'model_uses_covariates': model.use_covariates,
            'degradation_seed': self.config.degradation_seed,
            'degradation_model': self.config.degradation_label(),
            'num_weather_vars': results_df['num_weather_vars'].iloc[0] if len(results_df) > 0 else 0,
            'n_folds': len(fold_results),
            'n_failed_folds': n_failed_folds,
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
            'total_test_hours': results_df['test_hours'].sum(),
            'partial_folds': (results_df['test_hours'] < horizon).sum(),
            'total_test_scored': results_df['test_scored'].sum(),
            'folds_without_scored_test': (results_df['test_scored'] == 0).sum(),
            'folds_with_convergence_warnings': int((results_df['convergence_warnings'] > 0).sum()),
            # Runtime (seconds) over successful folds
            'fit_time_mean_s': results_df['fit_time_s'].mean(),
            'predict_time_mean_s': results_df['predict_time_s'].mean(),
            'runtime_mean_s': results_df['runtime_s'].mean(),
            'runtime_std_s': results_df['runtime_s'].std(),
            'fit_time_total_s': results_df['fit_time_s'].sum(),
            'predict_time_total_s': results_df['predict_time_s'].sum(),
            'runtime_total_s': results_df['runtime_s'].sum(),
        }

        # Log aggregated results to W&B
        wandb.log({
            f"{model.name}_{weather_scenario}_h{horizon}_MAE": aggregated['MAE_mean'],
            f"{model.name}_{weather_scenario}_h{horizon}_RMSE": aggregated['RMSE_mean'],
            f"{model.name}_{weather_scenario}_h{horizon}_MASE": aggregated['MASE_mean'],
            f"{model.name}_{weather_scenario}_h{horizon}_sMAPE": aggregated['sMAPE_mean'],
            f"{model.name}_{weather_scenario}_h{horizon}_runtime_mean_s": aggregated['runtime_mean_s'],
            f"{model.name}_{weather_scenario}_h{horizon}_runtime_total_s": aggregated['runtime_total_s'],
        })

        # Write results after every experiment, BEFORE the checkpoint: an
        # interrupted run (e.g. cluster time limit) then loses nothing, and a
        # checkpointed experiment always has its rows on disk. The forecasts
        # file is written first, so a failure there leaves the results files
        # untouched.
        self.results.append(aggregated)
        filename_agg = self.output_dir / f"results_master_{self.config.results_version}.csv"
        filename_detailed = self.output_dir / f"detailed_results_master_{self.config.results_version}.csv"
        filename_forecasts = (self.output_dir /
                              f"forecasts_{self.config.dataset_name}_{self.config.results_version}.csv")
        _append_rows(filename_forecasts, pd.concat(fold_forecasts, ignore_index=True))
        _append_rows(filename_detailed, pd.DataFrame(fold_results))
        _append_rows(filename_agg, pd.DataFrame([aggregated]))
        self._save_checkpoint(model.name, horizon, weather_scenario)

        # Log all folds as table
        fold_table = wandb.Table(dataframe=pd.DataFrame(fold_results))
        wandb.log({f"{model.name}_{weather_scenario}_h{horizon}_folds": fold_table})

        if verbose:
            print(f" [DONE] ({len(fold_results)} folds) | MAE: {aggregated['MAE_mean']:.1f}")
        n_conv = aggregated['folds_with_convergence_warnings']
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE  {self.config.dataset_name} | "
              f"{model.name} | h={horizon} | {weather_scenario} | MAE {aggregated['MAE_mean']:.1f} | "
              f"{(time.time() - t_start) / 60:.1f} min"
              + (f" | not converged in {n_conv} folds" if n_conv else ""), flush=True)

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

        # On resume, reload previously completed results from CSV
        filename_agg = self.output_dir / f"results_master_{self.config.results_version}.csv"
        if filename_agg.exists():
            existing = pd.read_csv(filename_agg)
            self.results = existing.to_dict('records')
            # Rows from other code versions must not be mixed into one paper run
            if 'git_commit' not in existing.columns:
                raise RuntimeError(
                    f"{filename_agg} has no git_commit column (written by older code). "
                    f"Move it and its checkpoint/detailed files away, or use a new results_version."
                )
            other = sorted(set(existing['git_commit'].dropna()) - {self.code_version['git_commit']})
            if other:
                print(f"WARNING: {filename_agg} contains rows from other commits: {other}. "
                      f"Current commit: {self.code_version['git_commit']}")

        
        # Calculate total experiments (accounting for skipped combinations)
        total = 0
        for scenario in scenarios:
            for model in models:
                # Count only if model will run
                if not (not model.use_covariates and self.config.is_degraded(scenario)):
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
                # Skip models without covariates for the degraded scenarios
                if not model.use_covariates and self.config.is_degraded(scenario):
                    if verbose:
                        print(f"[SKIP] {model.name} for {scenario} (no covariates)\n")
                    continue
                
                for horizon in self.config.horizons:
                    try:
                        result = self.run_single_experiment(
                            model, df, horizon, scenario, verbose
                        )
                    except Exception as e:
                        # Fail only this model/horizon/scenario. Successful
                        # experiments are already saved/checkpointed; this one is
                        # deliberately left uncheckpointed so a rerun retries it.
                        self.failed_experiments.append(
                            (model.name, horizon, scenario, str(e))
                        )
                        print(
                            f"[FAILED EXPERIMENT] {model.name} | h={horizon} | "
                            f"{scenario}: {e}"
                        )
                        result = None

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

        if self.failed_experiments:
            failed = ", ".join(
                f"{model}/h{horizon}/{scenario}"
                for model, horizon, scenario, _ in self.failed_experiments
            )
            raise RuntimeError(
                f"{len(self.failed_experiments)} experiment(s) failed: {failed}. "
                "Successful experiments were saved and checkpointed; rerun main.py "
                "to retry the failed combinations."
            )

        return results_df
    

    def save_results(self, results_df: pd.DataFrame) -> str:
        """
        Rewrite the aggregated results file without duplicate rows. Aggregated
        and fold-level rows are already written after every experiment
        (run_single_experiment), so nothing is lost if this is never reached.
        """

        # Aggregated results - APPEND mode
        filename_agg = self.output_dir / f"results_master_{self.config.results_version}.csv"
        if filename_agg.exists():
            existing = pd.read_csv(filename_agg)
            results_df = pd.concat([existing, results_df], ignore_index=True)
            results_df = results_df.drop_duplicates(
                subset=['dataset', 'model', 'horizon', 'weather_scenario', 'run_name'],
                keep='last'
            )
        results_df.to_csv(filename_agg, index=False)

        if self.config.verbose:
            print(f"Results appended to: {filename_agg}")
        return str(filename_agg)

    def finish(self):
        """Cleanly close the W&B run. Safe to call multiple times."""
        if wandb.run is not None:
            wandb.finish()
            

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
    # Stable sort: keep="first" below then keeps the file's first occurrence
    # of a repeated autumn hour (daylight-saving time, as to_utc reads it);
    # the default quicksort does not guarantee which one survives
    df = df.sort_values(config.date_col, kind="stable").reset_index(drop=True)

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
        if not pd.api.types.is_numeric_dtype(col):
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


def compute_and_log_comparative_metrics(config, log_wandb=True):
    """Compute + save comparative metrics from the AGGREGATED results file.

    Tasks are (dataset, horizon, weather_scenario), pooled across all
    datasets; one comparison per (model, weather_scenario). Run this ONCE
    after all cities have finished (main.py calls it when all *selected*
    cities succeeded; it uses whatever rows results_master_{version}.csv
    holds at that moment).
    """
    filename_agg = Path(config.output_dir) / f"results_master_{config.results_version}.csv"
    comparative_df = MetricsCalculator.compute_and_save_comparative_metrics(
        filename_agg, config.output_dir, config.results_version
    )
    if log_wandb and wandb.run is not None:
        wandb.log({"comparative_metrics": wandb.Table(dataframe=comparative_df)})
        for _, row in comparative_df.iterrows():
            # scenario in the key: one comparison per (model, scenario)
            key = f"{row['model']}_{row['weather_scenario']}_vs_{row['baseline']}"
            wandb.log({
                f"{key}_win_rate": row['win_rate'],
                f"{key}_win_rate_ci_lower": row['win_rate_ci_lower'],
                f"{key}_win_rate_ci_upper": row['win_rate_ci_upper'],
                f"{key}_skill_score": row['skill_score'],
                f"{key}_skill_score_ci_lower": row['skill_score_ci_lower'],
                f"{key}_skill_score_ci_upper": row['skill_score_ci_upper'],
            })
    if config.verbose:
        print("\nComparative metrics vs", BASELINE_MODEL)
        print(comparative_df[['model', 'weather_scenario', 'n_tasks', 'win_rate', 'skill_score']].to_string(index=False))
    return comparative_df


def main(config: Optional[ForecastConfig] = None):
    """
    Main execution. Without a config, the city is taken from --city:
        python forecasting/run_experiments.py --city {seoul,washington,london}
    Runs every scenario in config.weather_scenarios (all_weather included)
    under the run name baseline_models_{version}. Not used for the v7 paper
    runs: use main.py (run_weather_baseline.main()).
    """
    if config is None:
        import argparse
        parser = argparse.ArgumentParser(description="Run all baseline experiments for one city")
        parser.add_argument("--city", type=str, required=True,
                            choices=["seoul", "washington", "london"])
        args = parser.parse_args()
        if args.city == "seoul":
            from config_seoul import get_config
        elif args.city == "washington":
            from config_washington import get_config
        else:
            from config_london import get_config
        config = get_config()

    df, dataset_name = load_and_prepare_data(config)
    if config.dataset_name is None:
        config.dataset_name = dataset_name
    if config.experiment_name is None or config.experiment_name.startswith("None"):
        config.experiment_name = f"{config.dataset_name}_{config.results_version}"

    models = build_models(config)

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
        raise
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