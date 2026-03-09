"""
XGBoost Hyperparameter Tuning (random search, NO early stopping)

What this does:
- Jointly tunes n_lags AND XGBoost hyperparameters in ONE run.
- Covariate selection is driven by --scenario:
    all_weather  : all weather_covariates from config (incl. Dew point)
    clean_only   : degradable covariates only (keys of weather_degradation_mapping)
- Uses wide search space (less restrictive) for a "final good" tuning.
- Saves best config to results/tuning/xgboost_best_params_<timestamp>.json

Run:
    python forecasting/models/tuning/tune_xgboost.py --city seoul
    python forecasting/models/tuning/tune_xgboost.py --city london
    python forecasting/models/tuning/tune_xgboost.py --city washington

    # Specify scenario (default: clean_only):
    python forecasting/models/tuning/tune_xgboost.py --city seoul --scenario clean_only

    # Override other options:
    python forecasting/models/tuning/tune_xgboost.py --city seoul --horizon 24 --trials 600
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # forecasting/

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from config import ForecastConfig
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from features import add_time_features


def load_data(filepath: str, config: ForecastConfig) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.sort_values(config.date_col).reset_index(drop=True)
    return df  # no drop — covariate selection happens in tune_xgboost


def select_covariates(config: ForecastConfig, df: pd.DataFrame, scenario: str) -> List[str]:
    """
    Return the covariate columns to use for a given scenario.

    all_weather : all weather_covariates present in df
    clean_only  : only degradable covariates (weather_degradation_mapping keys) present in df
    """
    if scenario == "all_weather":
        candidates = config.weather_covariates
    else:
        # clean_only / degraded — exclude non-degradable covariates (e.g. Dew point)
        candidates = list(config.weather_degradation_mapping.keys())

    return [c for c in candidates if c in df.columns]


def create_lagged_features(
    y: np.ndarray,
    X: Optional[pd.DataFrame],
    n_lags: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(y)

    lag_mat = np.column_stack([np.roll(y, lag) for lag in range(1, n_lags + 1)])
    lag_mat = lag_mat[n_lags:, :]

    lag_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
    feats = pd.DataFrame(lag_mat, columns=lag_cols)

    if X is not None:
        X_part = X.reset_index(drop=True).iloc[n_lags:].reset_index(drop=True)
        feats = pd.concat([feats, X_part], axis=1)

    y_target = y[n_lags:]
    return feats, y_target


def iterative_forecast(
    model: xgb.XGBRegressor,
    y_history: np.ndarray,
    X_future: pd.DataFrame,
    n_lags: int,
) -> np.ndarray:
    preds: List[float] = []
    hist = y_history.astype(float).tolist()
    Xf = X_future.reset_index(drop=True)

    for t in range(len(Xf)):
        lag_vals = list(reversed(hist[-n_lags:]))  # lag_1 is most recent
        row = {f"lag_{i + 1}": float(lag_vals[i]) for i in range(n_lags)}
        for col in Xf.columns:
            row[col] = float(Xf.loc[t, col])

        X_row = pd.DataFrame([row])
        yhat = float(model.predict(X_row)[0])
        preds.append(yhat)
        hist.append(yhat)

    return np.array(preds, dtype=float)


def sample_params(rng: np.random.Generator) -> Dict:
    """
    Random-search space sized to be realistic (not extreme) and still meaningful.

    Goals:
    - Avoid very slow configs (no DART, no 8000 trees, no depth 14).
    - Avoid trivial configs (not capped at tiny models only).
    - Use fast training implementation (hist) and all CPU cores.
    """
    learning_rate = float(np.exp(rng.uniform(np.log(0.005), np.log(0.2))))
    reg_lambda = float(np.exp(rng.uniform(np.log(1e-3), np.log(50.0))))
    reg_alpha = float(np.exp(rng.uniform(np.log(1e-10), np.log(5.0))))

    return {
        "objective": "reg:squarederror",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "n_estimators": int(rng.choice([400, 800, 1200, 1600, 2000, 3000, 5000])),
        "learning_rate": learning_rate,
        "max_depth": int(rng.integers(2, 13)),
        "min_child_weight": float(rng.uniform(1.0, 80.0)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "gamma": float(rng.uniform(0.0, 20.0)),
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
    }


def evaluate_params_on_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig,
    params: Dict,
    n_lags: int,
    max_train_size: Optional[int],
) -> Tuple[float, float]:
    y_train = train_df[config.target_col].values
    X_train = train_df[config.weather_covariates].reset_index(drop=True)
    y_test = test_df[config.target_col].values
    X_test = test_df[config.weather_covariates].reset_index(drop=True)

    # Append calendar time features — must match what run_experiments.py does at inference
    X_train = pd.concat([X_train, add_time_features(train_df, config.date_col)], axis=1)
    X_test = pd.concat([X_test, add_time_features(test_df, config.date_col)], axis=1)

    if max_train_size is not None and len(y_train) > max_train_size:
        y_train = y_train[-max_train_size:]
        X_train = X_train.iloc[-max_train_size:].reset_index(drop=True)

    X_sup, y_sup = create_lagged_features(y_train, X_train, n_lags=n_lags)

    model = xgb.XGBRegressor(**params)
    model.fit(X_sup, y_sup)

    y_pred = iterative_forecast(
        model=model,
        y_history=y_train,
        X_future=X_test,
        n_lags=n_lags,
    )

    calc = MetricsCalculator()
    metrics = calc.calculate_all(y_test, y_pred, y_train)
    return float(metrics["MAE"]), float(metrics["RMSE"])


def tune_xgboost(
    df: pd.DataFrame,
    config: ForecastConfig,
    scenario: str = "clean_only",
    horizon: int = 24,
    folds: int = 10,
    tune_folds: int = 5,
    trials: int = 600,
    seed: int = 42,
    n_lags_options: Optional[List[int]] = None,
    max_train_size: int = 8000,
    verbose: bool = True,
) -> Dict:
    config.weather_covariates = select_covariates(config, df, scenario)

    cv = TimeSeriesCV(config)
    splits = cv.split(df, horizon)
    if len(splits) == 0:
        raise RuntimeError("No CV splits available for the given horizon/config.")

    tune_folds = min(tune_folds, len(splits))
    folds = min(folds, len(splits))

    if not n_lags_options:
        n_lags_options = [12, 24, 48, 168]

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 70)
        print("XGBOOST FINAL TUNING (wide random search, NO early stopping)")
        print("=" * 70)
        print(f"Scenario: {scenario}")
        print(f"Horizon: {horizon}h")
        print(f"Trials: {trials}")
        print(f"Tune folds: {tune_folds}")
        print(f"Validate folds: {folds}")
        print(f"n_lags_options: {n_lags_options}")
        print(f"Max train size: {max_train_size}")
        print(f"Covariates used ({len(config.weather_covariates)}): {config.weather_covariates}")
        print("=" * 70)

    best_params: Optional[Dict] = None
    best_n_lags: Optional[int] = None
    best_mae = float("inf")
    best_rmse = float("inf")

    for t in range(1, trials + 1):
        params = sample_params(rng)
        trial_n_lags = int(rng.choice(n_lags_options))

        maes: List[float] = []
        rmses: List[float] = []
        failed = False


        tune_split_indices = range(len(splits) - tune_folds, len(splits))
        for fold_idx in tune_split_indices:
            train_df, test_df = splits[fold_idx]
            try:
                mae, rmse = evaluate_params_on_fold(
                    train_df=train_df,
                    test_df=test_df,
                    config=config,
                    params=params,
                    n_lags=trial_n_lags,
                    max_train_size=max_train_size,
                )
                maes.append(mae)
                rmses.append(rmse)
            except Exception as e:
                failed = True
                if verbose and (t <= 10 or t % 25 == 0):
                    print(f"[{t:>4}/{trials}] FAILED fold {fold_idx}: {str(e)[:160]}")
                break

        if failed or not maes:
            continue

        mae_mean = float(np.mean(maes))
        rmse_mean = float(np.mean(rmses))

        if verbose and (t <= 10 or t % 25 == 0):
            print(
                f"[{t:>4}/{trials}] "
                f"MAE={mae_mean:.2f} RMSE={rmse_mean:.2f} "
                f"n_lags={trial_n_lags} "
                f"n_estimators={params['n_estimators']} "
                f"depth={params['max_depth']}"
            )

        if mae_mean < best_mae:
            best_mae = mae_mean
            best_rmse = rmse_mean
            best_params = params
            best_n_lags = trial_n_lags

    if best_params is None or best_n_lags is None:
        raise RuntimeError("No successful trials.")

    if verbose:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS FOUND")
        print("=" * 70)
        print(f"Tuning MAE={best_mae:.2f} RMSE={best_rmse:.2f}")
        print(f"Best n_lags={best_n_lags}")
        print(json.dumps(best_params, indent=2))

    # Validation on full fold set
    mae_values: List[float] = []
    rmse_values: List[float] = []

    for fold_idx in range(len(splits) - folds, len(splits)):
        train_df, test_df = splits[fold_idx]
        try:
            mae, rmse = evaluate_params_on_fold(
                train_df=train_df,
                test_df=test_df,
                config=config,
                params=best_params,
                n_lags=best_n_lags,
                max_train_size=max_train_size,
            )
            mae_values.append(mae)
            rmse_values.append(rmse)
            if verbose:
                print(f"  Fold {fold_idx}: MAE={mae:.1f}, RMSE={rmse:.1f}")
        except Exception as e:
            if verbose:
                print(f"  Fold {fold_idx}: FAILED - {str(e)[:160]}")
            continue

    mae_mean = float(np.mean(mae_values)) if mae_values else float("nan")
    mae_std = float(np.std(mae_values)) if mae_values else float("nan")
    rmse_mean = float(np.mean(rmse_values)) if rmse_values else float("nan")
    rmse_std = float(np.std(rmse_values)) if rmse_values else float("nan")

    if verbose:
        print("\nValidation results:")
        print(f"  MAE: {mae_mean:.2f} ± {mae_std:.2f}")
        print(f"  RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}")

    return {
        "n_lags": int(best_n_lags),
        "xgb_params": best_params,
        "tuning": {
            "search_type": "wide_random_search_no_early_stopping_joint_n_lags",
            "scenario": scenario,
            "trials": int(trials),
            "tune_folds": int(tune_folds),
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
        "validation_folds": int(len(mae_values)),
        "max_train_size": int(max_train_size),
        "horizon": int(horizon),
        "covariates_used": list(config.weather_covariates),
        "time_features_used": ["hour", "dayofweek", "month", "is_weekend"],
    }


def save_results(params: dict, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"xgboost_best_params_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost (wide random search, no early stopping)")
    parser.add_argument("--city", type=str, required=True, choices=["seoul", "london", "washington"])
    parser.add_argument("--scenario", type=str, default="clean_only", choices=["all_weather", "clean_only"])
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon for validation")
    parser.add_argument("--folds", type=int, default=10, help="Number of validation folds")
    parser.add_argument("--tune-folds", type=int, default=5, help="Number of folds used during tuning")
    parser.add_argument("--trials", type=int, default=600, help="Random search trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-size", type=int, default=8000, help="Maximum training observations")
    parser.add_argument("--output-dir", type=str, default="results/tuning", help="Directory to save results")
    parser.add_argument("--n-lags-options", type=str, default="12,24,48,168")
    args = parser.parse_args()

    if args.city == "seoul":
        from config_seoul import get_config
    elif args.city == "london":
        from config_london import get_config
    elif args.city == "washington":
        from config_washington import get_config
    config = get_config()

    n_lags_options = [int(x.strip()) for x in args.n_lags_options.split(",") if x.strip()]

    data_path = str(Path("data") / config.data_filename)
    print(f"Loading data from {data_path}")
    df = load_data(data_path, config)
    print(f"Loaded {len(df)} observations\n")

    params = tune_xgboost(
        df=df,
        config=config,
        scenario=args.scenario,
        horizon=args.horizon,
        folds=args.folds,
        tune_folds=args.tune_folds,
        trials=args.trials,
        seed=args.seed,
        n_lags_options=n_lags_options,
        max_train_size=args.max_train_size,
        verbose=True,
    )

    output_file = save_results(params, args.output_dir)
    print(f"\nUpdate config_{args.city}.py with:")
    print(f"  config.xgb_params_file = '{output_file}'")


if __name__ == "__main__":
    main()