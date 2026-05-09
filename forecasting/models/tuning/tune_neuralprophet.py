"""
NeuralProphet Hyperparameter Tuning (random search)

Jointly tunes: learning_rate, n_lags, num_hidden_layers, d_hidden
Covariate selection driven by --scenario (default: clean_only):
    clean_only  : degradable covariates only (keys of weather_degradation_mapping)
    all_weather : all weather_covariates from config

Note: NeuralProphet is slow. Default trials=30 is a conservative starting point;
      increase with --trials if compute budget allows.

Usage:
    # Tune all cities:
    python forecasting/models/tuning/tune_neuralprophet.py

    # Tune a specific city only:
    python forecasting/models/tuning/tune_neuralprophet.py --city seoul

    # Override scenario or other options:
    python forecasting/models/tuning/tune_neuralprophet.py --city seoul --scenario all_weather --trials 60
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # forecasting/

import argparse
import json
import logging
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

# Silence NeuralProphet training output
logging.getLogger("NP").setLevel(logging.WARNING)
logging.getLogger("NP.config_model").setLevel(logging.WARNING)
logging.getLogger("NP.utils_torch").setLevel(logging.WARNING)
logging.getLogger("neuralprophet").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def select_covariates(config: ForecastConfig, df: pd.DataFrame, scenario: str) -> List[str]:
    if scenario == "all_weather":
        return [c for c in config.weather_covariates if c in df.columns]
    else:  # clean_only
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


def evaluate_params_on_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig,
    params: Dict,
    covariate_cols: List[str],
) -> Tuple[float, float]:
    from neuralprophet import NeuralProphet

    n_lags = params["n_lags"]
    horizon = config.tune_horizon

    # NeuralProphet needs at least n_lags + n_forecasts rows
    if len(train_df) < n_lags + horizon + 1:
        raise ValueError(
            f"Insufficient training rows ({len(train_df)}) for n_lags={n_lags}."
        )

    model = NeuralProphet(
        n_lags=n_lags,
        n_forecasts=1,
        learning_rate=params["learning_rate"],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        epochs=None,    # let NeuralProphet choose based on data size
        drop_missing=True,
    )

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

    # Monkey-patch torch.load to fix PyTorch 2.4+ checkpoint loading incompatibility
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.fit(np_train, freq="h")
    finally:
        torch.load = _orig_load

    # Sync to cols NeuralProphet actually kept (silently drops e.g. all-zero cols)
    if model.config_lagged_regressors:
        active_covariates = [
            c for c in active_covariates if c in model.config_lagged_regressors
        ]
    else:
        active_covariates = []

    keep_cols = ["ds", "y"] + active_covariates
    np_train = np_train[keep_cols]

    # make_future_dataframe returns full training df + horizon future rows;
    # fill only the tail rows with held-out covariate values from test_df.
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
    trials: int = 30,
    seed: int = 42,
    n_lags_options: Optional[List[int]] = None,
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

    fold_range = (
        range(len(splits))
        if config.tune_folds is None
        else range(len(splits) - min(config.tune_folds, len(splits)), len(splits))
    )

    if len(fold_range) == 0:
        raise RuntimeError(
            f"fold_range is empty — splits={len(splits)}, tune_folds={config.tune_folds}. "
            "Check config.tune_folds is not 0 and that enough data exists for CV splits."
        )

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 70)
        print(
            f"NEURALPROPHET TUNING (random search) | city={city} "
            f"| horizon={config.tune_horizon}h | scenario={scenario}"
        )
        print("=" * 70)
        print(f"Trials: {trials}")
        print(
            f"Folds: {len(fold_range)} "
            f"({'all' if config.tune_folds is None else config.tune_folds})"
        )
        print(f"n_lags_options: {n_lags_options}")
        print(f"Cutoff date (held-out test start): {cutoff_date}")
        print(f"Tuning on {len(tune_df)} observations (pre-cutoff)")
        print(f"n_train_samples: {config.n_train_samples}")
        print(f"Covariates ({len(covariate_cols)}): {covariate_cols}")
        print("=" * 70)

    best_params: Optional[Dict] = None
    best_mae = float("inf")
    best_rmse = float("inf")
    n_failed = 0

    for t in range(1, trials + 1):
        params = sample_params(rng, n_lags_options)
        maes: List[float] = []
        rmses: List[float] = []
        failed = False

        for fold_idx in fold_range:
            train_df, test_df = splits[fold_idx]
            try:
                mae, rmse = evaluate_params_on_fold(
                    train_df, test_df, config, params, covariate_cols
                )
                maes.append(mae)
                rmses.append(rmse)
            except Exception:
                failed = True
                n_failed += 1
                print(
                    f"\n[{t:>4}/{trials}] FAILED fold {fold_idx} "
                    f"(params: n_lags={params['n_lags']} "
                    f"lr={params['learning_rate']:.5f}):"
                )
                traceback.print_exc()
                break

        if failed or not maes:
            continue

        mae_mean = float(np.mean(maes))
        rmse_mean = float(np.mean(rmses))

        if verbose and (t <= 10 or t % 5 == 0):
            print(
                f"[{t:>4}/{trials}] MAE={mae_mean:.2f} RMSE={rmse_mean:.2f} | "
                f"lr={params['learning_rate']:.5f} n_lags={params['n_lags']}"
            )

        if mae_mean < best_mae:
            best_mae = mae_mean
            best_rmse = rmse_mean
            best_params = params

    if best_params is None:
        raise RuntimeError("No successful trials.")

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
                train_df, test_df, config, best_params, covariate_cols
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
        # top-level n_lags mirrors the XGBoost output convention
        "n_lags": int(best_params["n_lags"]),
        "neuralprophet_params": {
            "learning_rate": best_params["learning_rate"],
        },
        "tuning": {
            "search_type": "random_search_joint_n_lags",
            "trials": int(trials),
            "tune_folds": len(fold_range),
            "seed": int(seed),
            "metric_optimized": "MAE",
            "best_tune_mae_mean": float(best_mae),
            "best_tune_rmse_mean": float(best_rmse),
            "n_lags_options": list(map(int, n_lags_options)),
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
        verbose=True,
    )
    output_file = save_results(params, args.output_dir)
    print(f"  --> config.neuralprophet_params_file = '{output_file}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune NeuralProphet (random search, joint n_lags + architecture)"
    )
    parser.add_argument(
        "--city",
        type=str,
        choices=["seoul", "london", "washington"],
        default=None,
        help="City to tune (default: all cities)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["clean_only", "all_weather"],
        default="clean_only",
        help="Covariate set (default: clean_only)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Random search trials (default: 30; NeuralProphet is slow)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="results/tuning", help="Directory to save results"
    )
    parser.add_argument(
        "--n-lags-options",
        type=str,
        default="12,24,48,168",
        help="Comma-separated n_lags candidates (default: 12,24,48,168)",
    )
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