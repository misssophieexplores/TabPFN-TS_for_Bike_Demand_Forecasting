"""
Prophet Hyperparameter Tuning (random search)

Tunes: changepoint_prior_scale, seasonality_prior_scale,
       holidays_prior_scale, seasonality_mode

Note: holidays_prior_scale only affects inference if country holidays are
      added to the model (e.g. via model.add_country_holidays). It is included
      in the search space for completeness and forward-compatibility.

Usage:
    # Tune all cities:
    python forecasting/models/tuning/tune_prophet.py

    # Tune a specific city only:
    python forecasting/models/tuning/tune_prophet.py --city seoul

    # Override trials or seed:
    python forecasting/models/tuning/tune_prophet.py --city seoul --trials 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # forecasting/

import argparse
import contextlib
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

# Silence Prophet / cmdstanpy sampler output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect stdout/stderr to /dev/null (suppresses Stan sampler noise)."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def sample_params(rng: np.random.Generator) -> Dict:
    return {
        "changepoint_prior_scale": float(
            np.exp(rng.uniform(np.log(0.001), np.log(0.5)))
        ),
        "seasonality_prior_scale": float(
            np.exp(rng.uniform(np.log(0.01), np.log(10.0)))
        ),
        "holidays_prior_scale": float(
            np.exp(rng.uniform(np.log(0.01), np.log(10.0)))
        ),
        "seasonality_mode": str(rng.choice(["multiplicative", "additive"])),
    }


def evaluate_params_on_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig,
    params: Dict,
) -> Tuple[float, float]:
    from prophet import Prophet

    prophet_train = pd.DataFrame({
        "ds": train_df[config.date_col].values,
        "y": train_df[config.target_col].values,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        **params,
    )

    with warnings.catch_warnings(), _suppress_stdout():
        warnings.filterwarnings("ignore")
        model.fit(prophet_train)

    last_train_date = prophet_train["ds"].iloc[-1]
    future_dates = pd.date_range(
        start=last_train_date + pd.Timedelta(hours=1),
        periods=config.tune_horizon,
        freq="h",
    )
    forecast = model.predict(pd.DataFrame({"ds": future_dates}))
    y_pred = forecast["yhat"].values

    y_test = test_df[config.target_col].values[: config.tune_horizon]
    calc = MetricsCalculator()
    metrics = calc.calculate_all(y_test, y_pred, train_df[config.target_col].values)
    return float(metrics["MAE"]), float(metrics["RMSE"])


def tune_prophet(
    df: pd.DataFrame,
    config: ForecastConfig,
    city: str,
    trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    cv = TimeSeriesCV(config)

    # Only tune on pre-cutoff data — never touch the held-out test period
    cutoff_date = cv.get_cutoff_date(df)
    tune_df = df[df[config.date_col] <= cutoff_date].copy()
    splits = cv.split(tune_df, config.tune_horizon)

    if len(splits) == 0:
        raise RuntimeError("No CV splits available for the given horizon/config.")

    if config.tune_folds is None:
        fold_range = range(len(splits))
    else:
        n = min(config.tune_folds, len(splits))
        fold_range = range(len(splits) - n, len(splits))

    if len(fold_range) == 0:
        raise RuntimeError(
            f"fold_range is empty — splits={len(splits)}, tune_folds={config.tune_folds}. "
            "Check config.tune_folds is not 0 and that enough data exists for CV splits."
        )

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 70)
        print(
            f"PROPHET TUNING (random search) | city={city} "
            f"| horizon={config.tune_horizon}h"
        )
        print("=" * 70)
        print(f"Trials: {trials}")
        print(
            f"Folds: {len(fold_range)} "
            f"({'all' if config.tune_folds is None else config.tune_folds})"
        )
        print(f"Cutoff date (held-out test start): {cutoff_date}")
        print(f"Tuning on {len(tune_df)} observations (pre-cutoff)")
        print(f"n_train_samples: {config.n_train_samples}")
        print("=" * 70)

    best_params: Optional[Dict] = None
    best_mae = float("inf")
    best_rmse = float("inf")
    n_failed = 0

    for t in range(1, trials + 1):
        params = sample_params(rng)
        maes: List[float] = []
        rmses: List[float] = []
        failed = False

        for fold_idx in fold_range:
            train_df, test_df = splits[fold_idx]
            try:
                mae, rmse = evaluate_params_on_fold(
                    train_df, test_df, config, params
                )
                maes.append(mae)
                rmses.append(rmse)
            except Exception:
                failed = True
                n_failed += 1
                print(
                    f"\n[{t:>4}/{trials}] FAILED fold {fold_idx} "
                    f"(params: cps={params['changepoint_prior_scale']:.4f} "
                    f"sps={params['seasonality_prior_scale']:.4f} "
                    f"mode={params['seasonality_mode']}):"
                )
                traceback.print_exc()
                break

        if failed or not maes:
            continue

        mae_mean = float(np.mean(maes))
        rmse_mean = float(np.mean(rmses))

        if verbose and (t <= 10 or t % 10 == 0):
            print(
                f"[{t:>4}/{trials}] MAE={mae_mean:.2f} RMSE={rmse_mean:.2f} | "
                f"cps={params['changepoint_prior_scale']:.4f} "
                f"sps={params['seasonality_prior_scale']:.4f} "
                f"mode={params['seasonality_mode']}"
            )

        if mae_mean < best_mae:
            best_mae = mae_mean
            best_rmse = rmse_mean
            best_params = params

    if best_params is None:
        raise RuntimeError(
            f"No successful trials out of {trials} attempted ({n_failed} fold failures). "
            "Check tracebacks above for the root cause."
        )

    if verbose:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS FOUND")
        print("=" * 70)
        print(f"Tuning MAE={best_mae:.2f}  RMSE={best_rmse:.2f}")
        print(json.dumps(best_params, indent=2))
        print(f"\nValidating on {len(fold_range)} folds...")

    mae_values: List[float] = []
    rmse_values: List[float] = []

    for fold_idx in fold_range:
        train_df, test_df = splits[fold_idx]
        try:
            mae, rmse = evaluate_params_on_fold(
                train_df, test_df, config, best_params
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
        "n_train_samples": config.n_train_samples,
        "prophet_params": best_params,
        "tuning": {
            "search_type": "random_search",
            "trials": int(trials),
            "tune_folds": len(fold_range),
            "seed": int(seed),
            "metric_optimized": "MAE",
            "best_tune_mae_mean": float(best_mae),
            "best_tune_rmse_mean": float(best_rmse),
        },
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "validation_folds": len(mae_values),
    }


def save_results(params: dict, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city, n_train = params["city"], params["n_train_samples"]
    output_file = out_dir / f"prophet_best_params_{city}_{n_train}_{timestamp}.json"
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

    print(f"\nLoading data for {city}...")
    df, _ = load_and_prepare_data(config)
    print(f"Loaded {len(df)} observations")

    params = tune_prophet(
        df=df,
        config=config,
        city=city,
        trials=args.trials,
        seed=args.seed,
        verbose=True,
    )
    output_file = save_results(params, args.output_dir)
    print(f"  --> config.prophet_params_file = '{output_file}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Prophet (random search)")
    parser.add_argument(
        "--city",
        type=str,
        choices=["seoul", "london", "washington"],
        default=None,
        help="City to tune (default: all cities)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Random search trials (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="results/tuning", help="Directory to save results"
    )
    args = parser.parse_args()

    cities = [args.city] if args.city else ["seoul", "london", "washington"]

    for city in cities:
        print("\n" + "=" * 70)
        print(f"TUNING PROPHET FOR: {city.upper()}")
        print("=" * 70)
        run_city(city, args)

    print("\n" + "=" * 70)
    print("ALL TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()