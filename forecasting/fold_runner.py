"""
One CV fold of one experiment: model inputs, fit, predict, metrics and hourly
forecasts. Used by run_experiments.ForecastingExperiment.run_single_experiment.

Folds are independent: each fold refits the model from scratch on its own
720-hour window, and the weather degradation of a fold is seeded by
(horizon, fold, city) only. So a fold gives the same result whether the folds
run one after another or in parallel. Models with parallel_folds = True
(SARIMAX) run their folds in parallel worker processes, one math thread per
worker (run_folds()); all other models run them one after another, as before.

Only light modules are imported here, so a worker process starts quickly and
needs little memory (no torch, TabPFN, TimesFM or W&B).
"""
import os
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from features import prepare_xgboost_features
from evaluation.metrics import MetricsCalculator


def prepare_fold_inputs(
    config,
    model,
    weather_proc,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weather_scenario: str,
    horizon: int,
    fold_idx: int,
):
    """
    Model inputs for one fold, exactly as in the experiments:
    (y_train, X_train, y_test, X_test).
    - covariates from WeatherProcessor (train: clean; test: degraded in the
      degraded scenarios, see config.degradation_scales)
    - calendar features appended for use_time_features models (XGBoost)
    - real DatetimeIndex for needs_datetime models (Prophet, NeuralProphet, TabPFN);
      an empty frame with that index if the model has no covariates
    """
    y_train = train_df[config.target_col].values
    y_test = test_df[config.target_col].values

    X_train = None
    X_test = None
    if model.use_covariates:
        X_train = weather_proc.prepare_weather_data(
            train_df, weather_scenario, horizon, fold_idx, split="train"
        )
        X_test = weather_proc.prepare_weather_data(
            test_df, weather_scenario, horizon, fold_idx, split="test"
        )

    if model.use_time_features:
        X_train = prepare_xgboost_features(train_df, config.date_col, X_train)
        X_test = prepare_xgboost_features(test_df, config.date_col, X_test)

    if getattr(model, "needs_datetime", False):
        train_dates = pd.DatetimeIndex(train_df[config.date_col].values)
        test_dates = pd.DatetimeIndex(test_df[config.date_col].values)
        X_train = pd.DataFrame(index=train_dates) if X_train is None else X_train.set_index(train_dates)
        X_test = pd.DataFrame(index=test_dates) if X_test is None else X_test.set_index(test_dates)

    return y_train, X_train, y_test, X_test


def run_fold(config, model, weather_proc, train_df, test_df, weather_scenario,
             horizon, fold_idx, run_name, code_version):
    """
    Fit and score the model on one fold. Returns (metrics dict, hourly
    forecasts DataFrame); raises on any error.
    """
    metrics_calc = MetricsCalculator()
    # Not timed (see Runtime).
    y_train, X_train, y_test, X_test = prepare_fold_inputs(
        config, model, weather_proc, train_df, test_df,
        weather_scenario, horizon, fold_idx,
    )
    model.reset()
    # Warnings are ignored globally; record them here to count
    # convergence warnings (e.g. SARIMAX/ARIMA optimizer) per fold.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Wall-clock runtime of fit() and predict() only (data and
        # feature preparation excluded).
        t0 = time.perf_counter()
        model.fit(y_train, X_train)
        fit_time = time.perf_counter() - t0
        # Forecast as many steps as the fold has test hours: horizon,
        # or fewer for a partial last fold (no data exist beyond it).
        n_steps = len(test_df)
        t0 = time.perf_counter()
        y_pred = model.predict(n_steps, X_test)
        predict_time = time.perf_counter() - t0
    n_convergence_warnings = sum(
        "converge" in str(w.message).lower() for w in caught
    )

    # Calculate metrics on observed hours only: imputed hours
    # (Functioning Day == 'No') are excluded from scoring, in the
    # test window and in the MASE scaling of the training window.
    fday = config.functioning_day_col
    test_observed = metrics_calc.observed_mask(test_df, fday)
    train_observed = metrics_calc.observed_mask(train_df, fday)
    metrics = metrics_calc.calculate_all(
        y_test, y_pred, y_train,
        test_mask=test_observed, train_mask=train_observed,
    )
    metrics['dataset'] = config.dataset_name
    metrics['run_name'] = run_name
    metrics['version'] = config.results_version
    metrics.update(code_version)
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['fold'] = fold_idx
    metrics['model'] = model.name
    metrics['horizon'] = horizon
    metrics['weather_scenario'] = weather_scenario
    metrics['model_uses_covariates'] = model.use_covariates
    metrics['degradation_seed'] = config.degradation_seed
    metrics['degradation_model'] = config.degradation_label()
    metrics['num_weather_vars'] = len(X_train.columns) if X_train is not None else 0

    # Track imputation info
    test_imputed = (test_df[fday] == 'No').sum() if fday and fday in test_df.columns else 0
    train_imputed = (train_df[fday] == 'No').sum() if fday and fday in train_df.columns else 0
    metrics['test_imputed'] = test_imputed
    metrics['train_imputed'] = train_imputed
    metrics['test_hours'] = n_steps
    metrics['test_scored'] = int(test_observed.sum())

    metrics['convergence_warnings'] = n_convergence_warnings

    # Runtime (seconds)
    metrics['fit_time_s'] = fit_time
    metrics['predict_time_s'] = predict_time
    metrics['runtime_s'] = fit_time + predict_time

    # Hourly forecasts of this fold, for forecasts_{dataset}_{version}.csv.
    # Only written, never used for scoring. Imputed hours are kept and
    # flagged with the same mask the metrics use (observed=False).
    # calculate_all() has already checked len(y_pred) == n_steps.
    forecasts = pd.DataFrame({
        'dataset': config.dataset_name,
        'model': model.name,
        'horizon': horizon,
        'weather_scenario': weather_scenario,
        'fold': fold_idx,
        'lead_time': np.arange(1, n_steps + 1),
        'datetime': test_df[config.date_col].values,
        'y_true': y_test,
        'y_pred': np.asarray(y_pred, dtype=float).ravel(),
        'observed': np.asarray(test_observed, dtype=bool),
        'version': config.results_version,
        'git_commit': code_version['git_commit'],
    })
    return metrics, forecasts


def run_fold_safe(*args):
    """run_fold() in a worker process: ("ok", metrics, forecasts), or
    ("error", message, traceback text) instead of raising, so the main
    process can log the error exactly like a sequential run."""
    # Same global warning filter as run_experiments.py (fresh worker process)
    warnings.filterwarnings('ignore')
    try:
        metrics, forecasts = run_fold(*args)
        return ("ok", metrics, forecasts)
    except Exception as e:
        return ("error", str(e), traceback.format_exc())


def fold_workers() -> int:
    """
    Worker processes for parallel folds: FOLD_WORKERS if set, otherwise the
    job's CPUs (SLURM_CPUS_PER_TASK, never the whole node), otherwise the CPUs
    this process may run on.
    """
    for var in ("FOLD_WORKERS", "SLURM_CPUS_PER_TASK"):
        if os.environ.get(var):
            return max(1, int(os.environ[var]))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on macOS
        return max(1, os.cpu_count() or 1)


def run_folds_parallel(config, model, weather_proc, splits, weather_scenario,
                       horizon, run_name, code_version, n_workers):
    """
    All folds in n_workers worker processes (joblib/loky, one math thread
    each). Returns one ("ok", metrics, forecasts) or ("error", message,
    traceback) per fold, in fold order.
    """
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_workers, backend="loky", inner_max_num_threads=1)(
        delayed(run_fold_safe)(config, model, weather_proc, train_df, test_df,
                               weather_scenario, horizon, fold_idx, run_name, code_version)
        for fold_idx, (train_df, test_df) in enumerate(splits)
    )
