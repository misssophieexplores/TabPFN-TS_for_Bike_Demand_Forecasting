"""
NeuralProphet Hyperparameter Tuning (random search)

Jointly tunes: learning_rate, n_lags

The model is set up exactly as in evaluation (models/prophet_models.py):
  - n_forecasts = config.tune_horizon (direct multi-step forecast)
  - covariates as future regressors (add_future_regressor)
  - daily and weekly seasonality on, yearly off (30-day training window)
  - epochs=None (NeuralProphet picks the number of epochs)
  - neuralprophet.set_random_seed(42) before every model is created

Search: random search on --search-folds folds (default 6), spread evenly over
the last config.tune_folds folds (all folds if None), so the search covers the
whole tuning period. There is no separate validation re-run (NeuralProphet is
slow; the evaluation period is the out-of-sample test).

Covariate set, selected by --scenario:
    clean_only  : degradable covariates (keys of weather_degradation_mapping)
                  + holiday + season                    -> NeuralProphetForecaster
    all_weather : all weather_covariates from config
    no_weather  : no covariates                          -> NeuralProphetForecaster_NoWeather

Usage:
    python forecasting/models/tuning/tune_neuralprophet.py --city seoul
    python forecasting/models/tuning/tune_neuralprophet.py --city seoul --scenario no_weather
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import contextlib
import io
import json
import logging
import os
import traceback
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ForecastConfig
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from run_experiments import load_and_prepare_data
from provenance import check_output_dir, save_params_json


# Fixed seed applied immediately before every NeuralProphet model is created.
NP_SEED = 42


# ----------------------------------------------------------------------
# Silencing
# ----------------------------------------------------------------------
for _name in [
    "NP", "NP.config_model", "NP.utils_torch", "NP.df_utils", "NP.config",
    "NP.forecaster", "NP.data.processing", "NP.data.splitting", "neuralprophet",
    "lightning", "lightning.pytorch", "lightning.pytorch.utilities.rank_zero",
    "lightning.pytorch.accelerators.cuda", "pytorch_lightning",
    "pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.cuda",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)
    logging.getLogger(_name).propagate = False

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_LIGHTNING_SEED_WORKERS"] = "0"
os.environ["LIGHTNING_LOGGER_LEVEL"] = "ERROR"


@contextlib.contextmanager
def _silence_all_output():
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def select_covariates(config: ForecastConfig, df: pd.DataFrame, scenario: str) -> List[str]:
    if scenario == "no_weather":
        return []
    if scenario == "all_weather":
        return [c for c in config.weather_covariates if c in df.columns]
    # clean_only
    missing = [c for c in config.weather_degradation_mapping if c not in df.columns]
    if missing:
        raise ValueError(f"data file has no column(s) {missing}: replace the data folder (2 Oct 2026)")
    covariates = [c for c in config.weather_degradation_mapping.keys() if c in df.columns]
    for col in [config.holiday_col, config.season_col]:
        if col and col in df.columns:
            covariates.append(col)
    return covariates


def sample_params(rng: np.random.Generator, n_lags_options: List[int]) -> Dict:
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(1e-4), np.log(0.1)))),
        "n_lags": int(rng.choice(n_lags_options)),
    }


def _build_model(n_lags: int, n_forecasts: int, learning_rate: float):
    from neuralprophet import NeuralProphet, set_random_seed

    kwargs = dict(
        n_lags=n_lags,
        n_forecasts=n_forecasts,
        learning_rate=learning_rate,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        epochs=None,
        drop_missing=True,
    )
    try:
        set_random_seed(NP_SEED)
        return NeuralProphet(**kwargs, trainer_config={"enable_model_summary": False})
    except TypeError:
        set_random_seed(NP_SEED)
        model = NeuralProphet(**kwargs)
        if hasattr(model, "config_train") and hasattr(model.config_train, "trainer_kwargs"):
            model.config_train.trainer_kwargs = {
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        return model


def evaluate_params_on_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig,
    params: Dict,
    covariate_cols: List[str],
) -> Tuple[float, float]:
    n_lags = params["n_lags"]
    h = config.tune_horizon

    if len(train_df) < n_lags + h + 1:
        raise ValueError(
            f"Insufficient training rows ({len(train_df)}) for n_lags={n_lags}."
        )

    model = _build_model(n_lags, h, params["learning_rate"])

    np_train = pd.DataFrame({
        "ds": train_df[config.date_col].values,
        "y": train_df[config.target_col].values,
    })

    active_covariates: List[str] = []
    for col in covariate_cols:
        if col not in train_df.columns:
            continue
        np_train[col] = train_df[col].values
        model.add_future_regressor(col)
        active_covariates.append(col)

    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        with warnings.catch_warnings(), _silence_all_output():
            warnings.filterwarnings("ignore")
            model.fit(np_train, freq="h", progress="none")
    finally:
        torch.load = _orig_load

    # Sync to cols NP actually kept (it drops regressors that are constant
    # in the training window)
    kept = model.config_regressors.regressors
    active_covariates = [c for c in active_covariates if kept and c in kept]

    np_train = np_train[["ds", "y"] + active_covariates]

    # Future covariate values for the h forecast steps
    regressors_df = None
    if active_covariates:
        regressors_df = pd.DataFrame(
            {col: test_df[col].values[:h] for col in active_covariates}
        )
        if len(regressors_df) != h:
            raise RuntimeError(f"test_df has {len(regressors_df)} rows, expected {h}.")

    with _silence_all_output():
        future_df = model.make_future_dataframe(
            np_train, regressors_df=regressors_df, periods=h, n_historic_predictions=True
        )

    # n_forecasts == h, so exactly h future rows are expected
    last_train_ds = pd.Timestamp(np_train["ds"].iloc[-1])
    n_future = int((future_df["ds"] > last_train_ds).sum())
    if n_future != h:
        raise RuntimeError(
            f"NeuralProphet future frame has {n_future} future rows, expected {h}."
        )

    with _silence_all_output(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        forecast = model.predict(future_df)

    # Step i (1-based) is in column yhat{i} of row -(h - i + 1)
    y_pred = np.asarray(
        [forecast[f"yhat{i+1}"].iloc[-h + i] for i in range(h)], dtype=float
    )
    y_test = test_df[config.target_col].values[:h]

    # Imputed hours (Functioning Day == 'No') are excluded from scoring
    calc = MetricsCalculator()
    metrics = calc.calculate_all(
        y_test, y_pred, train_df[config.target_col].values,
        test_mask=calc.observed_mask(test_df, config.functioning_day_col)[:h],
        train_mask=calc.observed_mask(train_df, config.functioning_day_col),
    )
    return float(metrics["MAE"]), float(metrics["RMSE"])


def tune_neuralprophet(
    df: pd.DataFrame,
    config: ForecastConfig,
    city: str,
    scenario: str = "clean_only",
    trials: int = 20,
    seed: int = 42,
    n_lags_options: Optional[List[int]] = None,
    search_folds: int = 6,
    verbose: bool = True,
) -> Dict:
    if not n_lags_options:
        n_lags_options = [12, 24, 48, 168]

    covariate_cols = select_covariates(config, df, scenario)
    cv = TimeSeriesCV(config)

    # Only tune on pre-cutoff data — never touch the held-out test period
    cutoff_date = cv.get_cutoff_date(df)
    tune_df = df[df[config.date_col] <= cutoff_date].copy()
    splits = cv.split(tune_df, config.tune_horizon)

    if len(splits) == 0:
        raise RuntimeError("No CV splits available for the given horizon/config.")

    # Last config.tune_folds folds, without fully imputed test windows
    tune_fold_range = cv.tune_fold_indices(splits)
    if not tune_fold_range:
        raise RuntimeError("No scored tune folds available.")

    # Search folds spread evenly over the tune folds (first and last included);
    # the ARIMA/SARIMAX order search uses the same folds
    search_fold_range = cv.spread_fold_indices(tune_fold_range, search_folds)

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 70)
        print(
            f"NEURALPROPHET TUNING | city={city} "
            f"| horizon={config.tune_horizon}h | scenario={scenario}"
        )
        print("=" * 70)
        print(f"Trials: {trials}")
        print(f"Search folds: {len(search_fold_range)} spread over {len(tune_fold_range)} "
              f"tune folds: {search_fold_range}")
        print(f"n_lags_options: {n_lags_options}")
        print(f"Cutoff date: {cutoff_date}")
        print(f"Tune obs: {len(tune_df)}")
        print(f"Covariates ({len(covariate_cols)}): {covariate_cols}")
        print("=" * 70)

    best_params: Optional[Dict] = None
    best_mae = float("inf")
    best_rmse = float("inf")

    # --- SEARCH PHASE ---
    for t in range(1, trials + 1):
        params = sample_params(rng, n_lags_options)
        maes, rmses, failed = [], [], False

        for fold_idx in search_fold_range:
            train_df_, test_df_ = splits[fold_idx]
            try:
                mae, rmse = evaluate_params_on_fold(
                    train_df_, test_df_, config, params, covariate_cols
                )
                maes.append(mae)
                rmses.append(rmse)
            except Exception:
                failed = True
                print(f"\n[{t:>4}/{trials}] FAILED fold {fold_idx} "
                      f"(n_lags={params['n_lags']} lr={params['learning_rate']:.5f}):")
                traceback.print_exc()
                break

        if failed or not maes:
            continue

        mae_mean = float(np.mean(maes))
        rmse_mean = float(np.mean(rmses))

        if verbose:
            marker = " *" if mae_mean < best_mae else ""
            print(f"[{t:>4}/{trials}] MAE={mae_mean:.2f} RMSE={rmse_mean:.2f} | "
                  f"lr={params['learning_rate']:.5f} n_lags={params['n_lags']}{marker}")

        if mae_mean < best_mae:
            best_mae = mae_mean
            best_rmse = rmse_mean
            best_params = params

    if best_params is None:
        raise RuntimeError("No successful trials.")

    if verbose:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS FROM SEARCH")
        print("=" * 70)
        print(f"Search MAE={best_mae:.2f}  RMSE={best_rmse:.2f}")
        print(json.dumps(best_params, indent=2))

    return {
        "city": city,
        "scenario": scenario,
        "n_train_samples": config.n_train_samples,
        "tuning_period": cv.get_tuning_period(tune_df),
        "n_lags": int(best_params["n_lags"]),
        "neuralprophet_params": {"learning_rate": best_params["learning_rate"]},
        "tuning": {
            "search_type": "random_search",
            "trials": int(trials),
            "search_folds": len(search_fold_range),
            "search_fold_indices": [int(i) for i in search_fold_range],
            "tune_folds": len(tune_fold_range),
            "seed": int(seed),
            "np_seed": NP_SEED,
            "metric_optimized": "MAE",
            "best_tune_mae_mean": float(best_mae),
            "best_tune_rmse_mean": float(best_rmse),
            "n_lags_options": list(map(int, n_lags_options)),
        },
        "covariates_used": covariate_cols,
    }


def save_results(params: dict, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city, scenario, n_train = params["city"], params["scenario"], params["n_train_samples"]
    output_file = (
        out_dir / f"neuralprophet_best_params_{city}_{scenario}_{n_train}_{timestamp}.json"
    )
    # complete JSON printed to the log first (restorable if the write fails)
    return save_params_json(params, output_file)


def run_city(city: str, args) -> None:
    if city == "seoul":
        from config_seoul import get_config
    elif city == "london":
        from config_london import get_config
    elif city == "washington":
        from config_washington import get_config
    config = get_config()

    n_lags_options = [int(x.strip()) for x in args.n_lags_options.split(",") if x.strip()]

    print(f"\nLoading data for {city}...")
    df, _ = load_and_prepare_data(config)
    print(f"Loaded {len(df)} observations")

    params = tune_neuralprophet(
        df=df,
        config=config,
        city=city,
        scenario=args.scenario,
        trials=args.trials,
        seed=args.seed,
        n_lags_options=n_lags_options,
        search_folds=args.search_folds,
        verbose=True,
    )
    output_file = save_results(params, args.output_dir)
    field = (
        "neuralprophet_noweather_params_file"
        if args.scenario == "no_weather"
        else "neuralprophet_params_file"
    )
    print(f"  --> config.{field} = '{output_file}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune NeuralProphet (random search)")
    parser.add_argument("--city", type=str,
                        choices=["seoul", "london", "washington"], default=None,
                        help="City to tune (default: all cities)")
    parser.add_argument("--scenario", type=str,
                        choices=["clean_only", "all_weather", "no_weather"],
                        default="clean_only",
                        help="Covariate set (default: clean_only)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Random search trials (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for parameter sampling (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results/tuning")
    parser.add_argument("--n-lags-options", type=str, default="12,24,48,168",
                        help="Comma-separated n_lags candidates (default: 12,24,48,168)")
    parser.add_argument("--search-folds", type=int, default=6,
                        help="Number of search folds, spread evenly over the tune folds (default: 6)")
    args = parser.parse_args()
    # A read-only output folder must stop the job now, not after hours of tuning
    check_output_dir(args.output_dir)

    cities = [args.city] if args.city else ["seoul", "london", "washington"]
    for city in cities:
        print("\n" + "=" * 70)
        print(f"TUNING NEURALPROPHET FOR: {city.upper()}")
        print("=" * 70)
        run_city(city, args)

    print("\n" + "=" * 70)
    print("ALL TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
