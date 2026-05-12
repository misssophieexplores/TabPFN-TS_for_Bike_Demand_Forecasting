"""
XGBoost Hyperparameter Tuning (Optuna TPE + early stopping)

What this does:
- Jointly tunes n_lags AND XGBoost hyperparameters in ONE run.
- Uses Optuna TPE (Tree-structured Parzen Estimator) instead of random search.
- n_estimators is NOT searched — determined automatically via XGBoost early stopping
  on a held-out slice of each training fold (last 15%, min 24 obs).
- Covariate selection is driven by --scenario (default: clean_only):
    clean_only   : degradable covariates only (keys of weather_degradation_mapping)
    all_weather  : all weather_covariates from config
- Saves best config to results/tuning/xgboost_best_params_<city>_<scenario>_<n_train>_<timestamp>.json

Run:
    # Tune all cities:
    python forecasting/models/tuning/tune_xgboost.py

    # Tune a specific city only:
    python forecasting/models/tuning/tune_xgboost.py --city seoul

    # Override scenario or other options:
    python forecasting/models/tuning/tune_xgboost.py --city seoul --scenario clean_only --trials 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # forecasting/

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import ForecastConfig
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from features import add_time_features
from run_experiments import load_and_prepare_data

EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 3000
EVAL_FRAC = 0.15
MIN_EVAL_SIZE = 24


def select_covariates(config: ForecastConfig, df: pd.DataFrame, scenario: str) -> List[str]:
    if scenario == "all_weather":
        return [c for c in config.weather_covariates if c in df.columns]
    else:  # clean_only
        covariates = [c for c in config.weather_degradation_mapping.keys() if c in df.columns]
        for col in [config.holiday_col, config.season_col]:
            if col and col in df.columns:
                covariates.append(col)
        return covariates


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
        lag_vals = list(reversed(hist[-n_lags:]))
        row = {f"lag_{i + 1}": float(lag_vals[i]) for i in range(n_lags)}
        for col in Xf.columns:
            row[col] = float(Xf.loc[t, col])
        X_row = pd.DataFrame([row])
        yhat = float(model.predict(X_row)[0])
        preds.append(yhat)
        hist.append(yhat)
    return np.array(preds, dtype=float)


def evaluate_params_on_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig,
    params: Dict,
    n_lags: int,
    max_train_size: Optional[int],
    use_early_stopping: bool = True,
) -> Tuple[float, float, int]:
    """Returns (MAE, RMSE, best_n_estimators)."""
    y_train = train_df[config.target_col].values
    X_train = train_df[config.weather_covariates].reset_index(drop=True)
    y_test = test_df[config.target_col].values
    X_test = test_df[config.weather_covariates].reset_index(drop=True)

    X_train = pd.concat([X_train, add_time_features(train_df, config.date_col)], axis=1)
    X_test = pd.concat([X_test, add_time_features(test_df, config.date_col)], axis=1)

    if max_train_size is not None and len(y_train) > max_train_size:
        y_train = y_train[-max_train_size:]
        X_train = X_train.iloc[-max_train_size:].reset_index(drop=True)

    X_sup, y_sup = create_lagged_features(y_train, X_train, n_lags=n_lags)

    model = xgb.XGBRegressor(**params)

    if use_early_stopping:
        eval_size = max(MIN_EVAL_SIZE, int(len(X_sup) * EVAL_FRAC))
        X_fit = X_sup.iloc[:-eval_size]
        y_fit = y_sup[:-eval_size]
        X_eval_es = X_sup.iloc[-eval_size:]
        y_eval_es = y_sup[-eval_size:]
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_eval_es, y_eval_es)],
            verbose=False,
        )
        best_iteration = int(model.best_iteration) + 1
    else:
        model.fit(X_sup, y_sup)
        best_iteration = params.get("n_estimators", MAX_ESTIMATORS)

    y_pred = iterative_forecast(model=model, y_history=y_train, X_future=X_test, n_lags=n_lags)

    calc = MetricsCalculator()
    metrics = calc.calculate_all(y_test, y_pred, y_train)
    return float(metrics["MAE"]), float(metrics["RMSE"]), best_iteration


def tune_xgboost(
    df: pd.DataFrame,
    config: ForecastConfig,
    city: str,
    scenario: str = "clean_only",
    trials: int = 100,
    seed: int = 42,
    n_lags_options: Optional[List[int]] = None,
    max_train_size: int = 8000,
    verbose: bool = True,
) -> Dict:
    config.weather_covariates = select_covariates(config, df, scenario)
    cv = TimeSeriesCV(config)

    cutoff_date = cv.get_cutoff_date(df)
    tune_df = df[df[config.date_col] <= cutoff_date].copy()

    splits = cv.split(tune_df, config.tune_horizon)
    if len(splits) == 0:
        raise RuntimeError("No CV splits available for the given horizon/config.")

    tune_split_indices = (
        range(len(splits)) if config.tune_folds is None
        else range(len(splits) - min(config.tune_folds, len(splits)), len(splits))
    )
    val_split_indices = tune_split_indices

    if not n_lags_options:
        n_lags_options = [12, 24, 48, 168]

    if verbose:
        print("=" * 70)
        print(f"XGBOOST TUNING (Optuna TPE + early stopping) | city={city} | horizon={config.tune_horizon}h | scenario={scenario}")
        print("=" * 70)
        print(f"Trials: {trials}")
        print(f"Tune folds: {len(tune_split_indices)} ({'all' if config.tune_folds is None else config.tune_folds})")
        print(f"n_lags_options: {n_lags_options}")
        print(f"Max train size: {max_train_size}")
        print(f"Early stopping rounds: {EARLY_STOPPING_ROUNDS} | Max estimators: {MAX_ESTIMATORS}")
        print(f"Cutoff date (held-out test start): {cutoff_date}")
        print(f"Tuning on {len(tune_df)} observations (pre-cutoff)")
        print(f"n_train_samples: {config.n_train_samples}")
        print(f"Covariates used ({len(config.weather_covariates)}): {config.weather_covariates}")
        print("=" * 70)

    # Store best_iterations per trial for final n_estimators selection
    trial_best_iterations: Dict[int, List[int]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
            "n_estimators": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 80.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 20.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 5.0, log=True),
        }
        trial_n_lags = trial.suggest_categorical("n_lags", n_lags_options)

        maes: List[float] = []
        best_iters: List[int] = []

        for fold_idx in tune_split_indices:
            train_df, test_df = splits[fold_idx]
            try:
                mae, rmse, best_iter = evaluate_params_on_fold(
                    train_df=train_df, test_df=test_df, config=config,
                    params=params, n_lags=trial_n_lags, max_train_size=max_train_size,
                    use_early_stopping=True,
                )
                maes.append(mae)
                best_iters.append(best_iter)
            except Exception as e:
                if verbose:
                    print(f"  Trial {trial.number} FAILED fold {fold_idx}: {str(e)[:160]}")
                raise optuna.exceptions.TrialPruned()

        trial_best_iterations[trial.number] = best_iters
        mae_mean = float(np.mean(maes))

        if verbose:
            print(
                f"  Trial {trial.number:>3}/{trials} | MAE={mae_mean:.2f} | "
                f"n_lags={trial_n_lags} | depth={params['max_depth']} | "
                f"lr={params['learning_rate']:.4f} | avg_n_estimators={int(np.mean(best_iters))}"
            )

        return mae_mean

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials)

    best_trial = study.best_trial
    best_params_optuna = best_trial.params
    best_n_lags = int(best_params_optuna.pop("n_lags"))
    best_n_estimators = int(np.mean(trial_best_iterations[best_trial.number]))

    best_params = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "n_estimators": best_n_estimators,
        **best_params_optuna,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS FOUND")
        print("=" * 70)
        print(f"Tuning MAE={best_trial.value:.2f}")
        print(f"Best n_lags={best_n_lags}")
        print(f"Best n_estimators (from early stopping)={best_n_estimators}")
        print(json.dumps(best_params, indent=2))

    # Validation using fixed n_estimators (no early stopping)
    mae_values: List[float] = []
    rmse_values: List[float] = []

    for fold_idx in val_split_indices:
        train_df, test_df = splits[fold_idx]
        try:
            mae, rmse, _ = evaluate_params_on_fold(
                train_df=train_df, test_df=test_df, config=config,
                params=best_params, n_lags=best_n_lags, max_train_size=max_train_size,
                use_early_stopping=False,
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
        "city": city,
        "scenario": scenario,
        "n_train_samples": config.n_train_samples,
        "n_lags": best_n_lags,
        "xgb_params": best_params,
        "tuning": {
            "search_type": "optuna_tpe_multivariate_with_early_stopping",
            "trials": int(trials),
            "tune_folds": len(list(tune_split_indices)),
            "seed": int(seed),
            "metric_optimized": "MAE",
            "best_tune_mae_mean": float(best_trial.value),
            "n_lags_options": list(map(int, n_lags_options)),
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "max_estimators_cap": MAX_ESTIMATORS,
        },
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "validation_folds": int(len(mae_values)),
        "max_train_size": int(max_train_size),
        "covariates_used": list(config.weather_covariates),
        "time_features_used": ["hour", "dayofweek", "month", "is_weekend"],
    }


def save_results(params: dict, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city, scenario, n_train = params['city'], params['scenario'], params['n_train_samples']
    output_file = out_dir / f"xgboost_best_params_{city}_{scenario}_{n_train}_{timestamp}.json"
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

    df, _ = load_and_prepare_data(config)
    print(f"Loaded {len(df)} observations for {city}")

    params = tune_xgboost(
        df=df,
        config=config,
        city=city,
        scenario=args.scenario,
        trials=args.trials,
        seed=args.seed,
        n_lags_options=n_lags_options,
        max_train_size=args.max_train_size,
        verbose=True,
    )
    output_file = save_results(params, args.output_dir)
    print(f"  --> config.xgb_params_file = '{output_file}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost (Optuna TPE + early stopping)")
    parser.add_argument('--city', type=str, choices=["seoul", "london", "washington"],
                        default=None, help='City to tune (default: all cities)')
    parser.add_argument('--scenario', type=str, choices=['clean_only', 'all_weather'],
                        default='clean_only', help='Covariate set (default: clean_only)')
    parser.add_argument("--trials", type=int, default=100, help="Optuna TPE trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-size", type=int, default=8000, help="Maximum training observations")
    parser.add_argument("--output-dir", type=str, default="results/tuning", help="Directory to save results")
    parser.add_argument("--n-lags-options", type=str, default="12,24,48,168")
    args = parser.parse_args()

    cities = [args.city] if args.city else ["seoul", "london", "washington"]

    for city in cities:
        print("\n" + "=" * 70)
        print(f"TUNING XGBOOST FOR: {city.upper()}")
        print("=" * 70)
        run_city(city, args)

    print("\n" + "=" * 70)
    print("ALL TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
