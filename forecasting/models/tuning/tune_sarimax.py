"""
SARIMAX Hyperparameter Tuning using auto_arima

Uses pmdarima's auto_arima for automatic parameter search with:
- Seasonal components
- Exogenous variables (weather covariates)

Usage:
    # Tune all cities:
    python forecasting/models/tuning/tune_sarimax.py

    # Tune a specific city only:
    python forecasting/models/tuning/tune_sarimax.py --city seoul

    # Override scenario (default: clean_only):
    python forecasting/models/tuning/tune_sarimax.py --scenario all_weather
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # forecasting/

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import traceback
from datetime import datetime
import warnings

from config import ForecastConfig
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from run_experiments import load_and_prepare_data

try:
    from pmdarima import auto_arima
except ImportError:
    print("ERROR: pmdarima not installed")
    print("Install with: pip install pmdarima")
    sys.exit(1)


def select_covariates(config: ForecastConfig, df: pd.DataFrame, scenario: str) -> list:
    if scenario == "all_weather":
        return [c for c in config.weather_covariates if c in df.columns]
    else:  # clean_only
        covariates = [c for c in config.weather_degradation_mapping.keys() if c in df.columns]
        for col in [config.holiday_col, config.season_col]:
            if col and col in df.columns:
                covariates.append(col)
        return covariates


def tune_sarimax(
    df: pd.DataFrame,
    config: ForecastConfig,
    city: str,
    scenario: str = "clean_only",
    seasonal: bool = True,
    m: int = 24,
    verbose: bool = True
) -> dict:
    covariates = select_covariates(config, df, scenario)
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()

    # Only tune on pre-cutoff data — never touch the held-out test period
    cutoff_date = cv.get_cutoff_date(df)
    tune_df = df[df[config.date_col] <= cutoff_date].copy()

    if verbose:
        print("="*70)
        print(f"SARIMAX AUTO-TUNING (pmdarima) | city={city} | horizon={config.tune_horizon}h | scenario={scenario}")
        print("="*70)
        print(f"Covariates ({len(covariates)}): {covariates}")
        print(f"Seasonal: {seasonal}, m={m}")
        print(f"Cutoff date (held-out test start): {cutoff_date}")
        print(f"Tuning on {len(tune_df)} observations (pre-cutoff)")
        print(f"n_train_samples: {config.n_train_samples}")
        print(f"Validation folds: {'all available' if config.tune_folds is None else config.tune_folds}")
        print("="*70)

    splits = cv.split(tune_df, config.tune_horizon)
    train_df, test_df = splits[0]
    y_train = train_df[config.target_col].values
    X_train = train_df[covariates].values

    if verbose:
        print(f"\nSearching optimal parameters on {len(y_train)} observations...")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = auto_arima(
            y_train,
            exogenous=X_train,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5, max_q=3,
            max_P=2, max_Q=2,
            max_order=8,
            trace=verbose,
            information_criterion='aic',
            n_jobs=-1
        )

    order = model.order
    seasonal_order = model.seasonal_order
    aic = model.aic()
    bic = model.bic()

    if verbose:
        print("\n" + "="*70)
        print("BEST PARAMETERS FOUND")
        print("="*70)
        print(f"order: {order}")
        print(f"seasonal_order: {seasonal_order}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")

    fold_range = range(len(splits)) if config.tune_folds is None else range(min(config.tune_folds, len(splits)))

    if verbose:
        print(f"\nValidating on {len(fold_range)} folds...")

    mae_values = []
    rmse_values = []

    for fold_idx in fold_range:
        train_df, test_df = splits[fold_idx]
        y_train_fold = train_df[config.target_col].values
        y_test_fold = test_df[config.target_col].values
        X_train_fold = train_df[covariates].values
        X_test_fold = test_df[covariates].values

        try:
            model_fold = auto_arima(
                y_train_fold,
                exogenous=X_train_fold,
                start_p=order[0], start_q=order[2], start_P=seasonal_order[0], start_Q=seasonal_order[2],
                max_p=order[0], max_q=order[2], max_P=seasonal_order[0], max_Q=seasonal_order[2],
                d=order[1], D=seasonal_order[1],
                seasonal=seasonal,
                m=m,
                suppress_warnings=True,
                error_action='ignore'
            )
            y_pred = model_fold.predict(n_periods=config.tune_horizon, exogenous=X_test_fold)
            metrics = calc.calculate_all(y_test_fold, y_pred, y_train_fold)
            mae_values.append(metrics['MAE'])
            rmse_values.append(metrics['RMSE'])
            if verbose:
                print(f"  Fold {fold_idx}: MAE={metrics['MAE']:.1f}, RMSE={metrics['RMSE']:.1f}")
        except Exception as e:
            traceback.print_exc()
            continue

    if mae_values:
        mae_mean, mae_std = np.mean(mae_values), np.std(mae_values)
        rmse_mean, rmse_std = np.mean(rmse_values), np.std(rmse_values)
    else:
        mae_mean = mae_std = rmse_mean = rmse_std = np.nan

    if verbose:
        print(f"\nValidation results:")
        print(f"  MAE: {mae_mean:.2f} ± {mae_std:.2f}")
        print(f"  RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}")

    return {
        'city': city,
        'scenario': scenario,
        'n_train_samples': config.n_train_samples,
        'order': order,
        'seasonal_order': seasonal_order,
        'aic': float(aic),
        'bic': float(bic),
        'mae_mean': float(mae_mean),
        'mae_std': float(mae_std),
        'rmse_mean': float(rmse_mean),
        'rmse_std': float(rmse_std),
        'validation_folds': len(mae_values),
        'covariates_used': list(covariates),
        'm': m
    }


def save_results(params: dict, output_dir: str = '.') -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    city, scenario, n_train = params['city'], params['scenario'], params['n_train_samples']
    output_file = output_dir / f'sarimax_best_params_{city}_{scenario}_{n_train}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    return output_file


def run_city(city: str, args) -> None:
    if city == 'seoul':
        from config_seoul import get_config
    elif city == 'london':
        from config_london import get_config
    elif city == 'washington':
        from config_washington import get_config
    config = get_config()

    df, _ = load_and_prepare_data(config)
    print(f"Loaded {len(df)} observations for {city}")

    params = tune_sarimax(df=df, config=config, city=city, scenario=args.scenario,
                          seasonal=True, m=args.seasonal_period, verbose=True)
    output_file = save_results(params, args.output_dir)
    print(f"  --> config.sarimax_params_file = '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description='Tune SARIMAX using auto_arima')
    parser.add_argument('--city', type=str, choices=['seoul', 'london', 'washington'],
                        default=None, help='City to tune (default: all cities)')
    parser.add_argument('--scenario', type=str, choices=['clean_only', 'all_weather'],
                        default='clean_only', help='Covariate set (default: clean_only)')
    parser.add_argument('--seasonal-period', type=int, default=24, help='Seasonal period (default: 24)')
    parser.add_argument('--output-dir', type=str, default='results/tuning', help='Directory to save results')
    args = parser.parse_args()

    cities = [args.city] if args.city else ['seoul', 'london', 'washington']

    for city in cities:
        print("\n" + "="*70)
        print(f"TUNING SARIMAX FOR: {city.upper()}")
        print("="*70)
        run_city(city, args)

    print("\n" + "="*70)
    print("ALL TUNING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()