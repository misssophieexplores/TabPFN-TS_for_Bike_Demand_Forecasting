"""
NeuralProphet Hyperparameter Tuning (random search) — FAST VARIANT

Speed-optimized for shared cluster time limits. Key differences vs. the
original tuning script:
  - epochs capped during tuning (default 50)
  - n_lags=168 removed from default options
  - fewer trials (20) but biased toward cheap n_lags
  - 2 search folds, then validation on all available folds at the end
  - daily_seasonality off during search (AR with n_lags>=24 covers it)
  - optional fixed-window training subset per fold

Usage:
    python forecasting/models/tuning/tune_neuralprophet_fast.py --city seoul
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


# ----------------------------------------------------------------------
# Silencing (same as before)
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
    if scenario == "all_weather":
        return [c for c in config.weather_covariates if c in df.columns]
    else:
        covariates = [c for c in config.weather_degradation_mapping.keys() if c in df.columns]
        for col in [config.holiday_col, config.season_col]:
            if col and col in df.columns:
                covariates.append(col)
        return covariates


# SPEEDUP: bias n_lags sampling toward cheaper options
def sample_params(rng: np.random.Generator, n_lags_options: List[int]) -> Dict:
    # Weight smaller n_lags higher: each option's weight ∝ 1/sqrt(n_lags)
    weights = np.array([1.0 / np.sqrt(n) for n in n_lags_options])
    weights = weights / weights.sum()
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(1e-4), np.log(0.1)))),
        "n_lags": int(rng.choice(n_lags_options, p=weights)),
    }


def evaluate_params_on_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig,
    params: Dict,
    covariate_cols: List[str],
    epochs_override: Optional[int] = None,  # SPEEDUP: cap epochs during tuning
    daily_seasonality: bool = True,         # SPEEDUP: disable during search
    max_train_rows: Optional[int] = None,   # SPEEDUP: fixed-window subsample
) -> Tuple[float, float]:
    from neuralprophet import NeuralProphet

    n_lags = params["n_lags"]
    horizon = config.tune_horizon

    # SPEEDUP: optionally subsample training to a fixed recent window
    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.iloc[-max_train_rows:].copy()

    if len(train_df) < n_lags + horizon + 1:
        raise ValueError(
            f"Insufficient training rows ({len(train_df)}) for n_lags={n_lags}."
        )

    try:
        model = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=1,
            learning_rate=params["learning_rate"],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=daily_seasonality,  # SPEEDUP
            seasonality_mode="multiplicative",
            epochs=epochs_override,  # SPEEDUP
            drop_missing=True,
            trainer_config={
                "enable_model_summary": False,
            },
        )
    except TypeError:
        model = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=1,
            learning_rate=params["learning_rate"],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=daily_seasonality,
            seasonality_mode="multiplicative",
            epochs=epochs_override,
            drop_missing=True,
        )
        if hasattr(model, "config_train") and hasattr(model.config_train, "trainer_kwargs"):
            model.config_train.trainer_kwargs = {
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }

    np_train = pd.DataFrame({
        "ds": train_df[config.date_col].values,
        "y": train_df[config.target_col].values,
    })

    active_covariates: List[str] = []
    for col in covariate_cols:
        if col not in train_df.columns:
            continue
        np_train[col] = train_df[col].values
        model.add_lagged_regressor(col)
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

    if model.config_lagged_regressors:
        active_covariates = [c for c in active_covariates if c in model.config_lagged_regressors]
    else:
        active_covariates = []

    keep_cols = ["ds", "y"] + active_covariates
    np_train = np_train[keep_cols]

    with _silence_all_output():
        future_df = model.make_future_dataframe(
            np_train, periods=horizon, n_historic_predictions=True
        )
        for col in active_covariates:
            if col in test_df.columns:
                future_df.iloc[-horizon:, future_df.columns.get_loc(col)] = (
                    test_df[col].values[:horizon]
                )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            forecast = model.predict(future_df)

    yhat_cols = sorted([c for c in forecast.columns if c.startswith("yhat")])
    y_pred = forecast[yhat_cols].iloc[-horizon:].values.flatten()[:horizon]
    y_test = test_df[config.target_col].values[:horizon]

    calc = MetricsCalculator()
    metrics = calc.calculate_all(y_test, y_pred, train_df[config.target_col].values)
    return float(metrics["MAE"]), float(metrics["RMSE"])


def tune_neuralprophet(
    df: pd.DataFrame,
    config: ForecastConfig,
    city: str,
    scenario: str = "clean_only",
    trials: int = 20,                  # SPEEDUP: down from 30
    seed: int = 42,
    n_lags_options: Optional[List[int]] = None,
    search_epochs: int = 50,           # SPEEDUP
    search_folds: int = 2,             # SPEEDUP: fewer folds during search
    max_train_rows: Optional[int] = None,  # SPEEDUP: e.g. 8760 = 1 year
    verbose: bool = True,
) -> Dict:
    # SPEEDUP: default options exclude 168
    if not n_lags_options:
        n_lags_options = [24, 48]

    covariate_cols = select_covariates(config, df, scenario)
    cv = TimeSeriesCV(config)

    cutoff_date = cv.get_cutoff_date(df)
    tune_df = df[df[config.date_col] <= cutoff_date].copy()
    splits = cv.split(tune_df, config.tune_horizon)

    if len(splits) == 0:
        raise RuntimeError("No CV splits available for the given horizon/config.")

    # SPEEDUP: split into search folds (subset, last N) and validation folds (all)
    n_avail = len(splits)
    n_search = min(search_folds, n_avail)
    search_fold_range = range(n_avail - n_search, n_avail)

    if config.tune_folds is None:
        validation_fold_range = range(n_avail)
    else:
        validation_fold_range = range(
            n_avail - min(config.tune_folds, n_avail), n_avail
        )

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 70)
        print(
            f"NEURALPROPHET TUNING (FAST) | city={city} "
            f"| horizon={config.tune_horizon}h | scenario={scenario}"
        )
        print("=" * 70)
        print(f"Trials: {trials}")
        print(f"Search folds: {len(search_fold_range)} "
              f"(last {n_search} of {n_avail})")
        print(f"Validation folds: {len(validation_fold_range)}")
        print(f"Search epochs cap: {search_epochs}")
        print(f"n_lags_options: {n_lags_options} (weighted toward smaller)")
        print(f"max_train_rows: {max_train_rows or 'unlimited'}")
        print(f"Cutoff date: {cutoff_date}")
        print(f"Tune obs: {len(tune_df)}")
        print(f"Covariates ({len(covariate_cols)}): {covariate_cols}")
        print("=" * 70)

    best_params: Optional[Dict] = None
    best_mae = float("inf")
    best_rmse = float("inf")

    # --- SEARCH PHASE (fast: capped epochs, fewer folds, daily_seas off) ---
    for t in range(1, trials + 1):
        params = sample_params(rng, n_lags_options)
        maes, rmses, failed = [], [], False

        for fold_idx in search_fold_range:
            train_df_, test_df_ = splits[fold_idx]
            try:
                mae, rmse = evaluate_params_on_fold(
                    train_df_, test_df_, config, params, covariate_cols,
                    epochs_override=search_epochs,    # SPEEDUP
                    daily_seasonality=False,          # SPEEDUP
                    max_train_rows=max_train_rows,    # SPEEDUP
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
        print(f"\nValidating with FULL epochs on {len(validation_fold_range)} folds...")

    # --- VALIDATION PHASE (full epochs, full folds, full daily_seas) ---
    mae_values, rmse_values = [], []
    for fold_idx in validation_fold_range:
        train_df_, test_df_ = splits[fold_idx]
        try:
            mae, rmse = evaluate_params_on_fold(
                train_df_, test_df_, config, best_params, covariate_cols,
                epochs_override=None,        # full convergence
                daily_seasonality=True,
                max_train_rows=None,         # full data for final validation
            )
            mae_values.append(mae)
            rmse_values.append(rmse)
            if verbose:
                print(f"  Fold {fold_idx}: MAE={mae:.1f}, RMSE={rmse:.1f}")
        except Exception:
            print(f"  Fold {fold_idx}: FAILED")
            traceback.print_exc()

    mae_mean = float(np.mean(mae_values)) if mae_values else float("nan")
    mae_std = float(np.std(mae_values)) if mae_values else float("nan")
    rmse_mean = float(np.mean(rmse_values)) if rmse_values else float("nan")
    rmse_std = float(np.std(rmse_values)) if rmse_values else float("nan")

    if verbose:
        print(f"\nValidation results:")
        print(f"  MAE:  {mae_mean:.2f} ± {mae_std:.2f}")
        print(f"  RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}")

    return {
        "city": city,
        "scenario": scenario,
        "n_train_samples": config.n_train_samples,
        "n_lags": int(best_params["n_lags"]),
        "neuralprophet_params": {"learning_rate": best_params["learning_rate"]},
        "tuning": {
            "search_type": "random_search_fast",
            "trials": int(trials),
            "search_folds": len(search_fold_range),
            "search_epochs_cap": int(search_epochs),
            "validation_folds": len(validation_fold_range),
            "seed": int(seed),
            "metric_optimized": "MAE",
            "best_tune_mae_mean": float(best_mae),
            "best_tune_rmse_mean": float(best_rmse),
            "n_lags_options": list(map(int, n_lags_options)),
            "max_train_rows": max_train_rows,
        },
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "validation_folds": len(mae_values),
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
    with open(output_file, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    return output_file


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
        search_epochs=args.search_epochs,
        search_folds=args.search_folds,
        max_train_rows=args.max_train_rows,
        verbose=True,
    )
    output_file = save_results(params, args.output_dir)
    print(f"  --> config.neuralprophet_params_file = '{output_file}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune NeuralProphet — fast variant for cluster time limits"
    )
    parser.add_argument("--city", type=str,
                        choices=["seoul", "london", "washington"], default=None)
    parser.add_argument("--scenario", type=str,
                        choices=["clean_only", "all_weather"], default="clean_only")
    parser.add_argument("--trials", type=int, default=20,
                        help="Random search trials (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/tuning")
    parser.add_argument("--n-lags-options", type=str, default="24,48",
                        help="Comma-separated n_lags candidates (default: 24,48)")
    parser.add_argument("--search-epochs", type=int, default=50,
                        help="Epoch cap during search phase (default: 50)")
    parser.add_argument("--search-folds", type=int, default=2,
                        help="Folds during search phase (default: 2)")
    parser.add_argument("--max-train-rows", type=int, default=None,
                        help="Optional cap on training rows per fold (e.g. 8760 = 1y)")
    args = parser.parse_args()

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