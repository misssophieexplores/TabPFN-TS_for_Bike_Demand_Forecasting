"""
SARIMAX order tuning (seasonal, with weather covariates).

Procedure (shared with tune_arima.py, see arima_search.py):
  1. pmdarima auto_arima (stepwise, AIC, seasonal period m) with the covariates
     on the training window of each of the --search-folds candidate folds
     (default 6, spread evenly over the tune folds) -> candidate orders
  2. every candidate is fitted with SARIMAXForecaster (statsmodels, as in the
     experiments, same covariates) on ALL tune folds (same folds as XGBoost
     and Prophet) and scored on the next tune_horizon hours; the lowest mean
     MAE is selected

Covariates are the experiment's columns for --scenario (WeatherProcessor):
    clean_only  (default): degradable covariates + holiday + season
    all_weather          : all weather_covariates
Covariates constant in a training window are dropped (same rule as
SARIMAXForecaster).

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
sys.path.insert(0, str(Path(__file__).resolve().parent))      # models/tuning/

import argparse
import json
from datetime import datetime

import pandas as pd

from config import ForecastConfig
from run_experiments import load_and_prepare_data
from provenance import get_provenance
from arima_search import search_orders


def auto_arima_kwargs(m: int) -> dict:
    return dict(
        seasonal=True,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_p=5, max_q=3,
        max_P=2, max_Q=2,
        max_order=8,
        information_criterion='aic',
    )


def tune_sarimax(
    df: pd.DataFrame,
    config: ForecastConfig,
    city: str,
    scenario: str = "clean_only",
    m: int = 24,
    search_folds: int = 6,
    verbose: bool = True
) -> dict:
    if verbose:
        print("=" * 70)
        print(f"SARIMAX TUNING | city={city} | horizon={config.tune_horizon}h | scenario={scenario} | m={m}")
        print("=" * 70)

    result = search_orders(
        df, config, seasonal=True, scenario=scenario,
        auto_arima_kwargs={**auto_arima_kwargs(m), "trace": False},
        search_folds=search_folds, verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 70)
        print(f"SELECTED: order={result['order']} seasonal_order={result['seasonal_order']} "
              f"with_intercept={result['with_intercept']} (trend={result['trend']}) | "
              f"search MAE={result['tuning']['best_tune_mae_mean']:.2f}")
        print("=" * 70)

    return {
        'city': city,
        'scenario': scenario,
        'n_train_samples': config.n_train_samples,
        'tuning_period': result['tuning_period'],
        'order': result['order'],
        'seasonal_order': result['seasonal_order'],
        'with_intercept': result['with_intercept'],
        'trend': result['trend'],
        'covariates_used': result['covariates_used'],
        'm': m,
        'tuning': result['tuning'],
    }


def save_results(params: dict, output_dir: str = '.') -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    city, scenario, n_train = params['city'], params['scenario'], params['n_train_samples']
    output_file = output_dir / f'sarimax_best_params_{city}_{scenario}_{n_train}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump({**params, "provenance": get_provenance()}, f, indent=2)
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
                          m=args.seasonal_period, search_folds=args.search_folds, verbose=True)
    output_file = save_results(params, args.output_dir)
    print(f"  --> config.sarimax_params_file = '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description='Tune SARIMAX order (auto_arima candidates, MAE selection)')
    parser.add_argument('--city', type=str, choices=['seoul', 'london', 'washington'],
                        default=None, help='City to tune (default: all cities)')
    parser.add_argument('--scenario', type=str, choices=['clean_only', 'all_weather'],
                        default='clean_only', help='Covariate set (default: clean_only)')
    parser.add_argument('--seasonal-period', type=int, default=24, help='Seasonal period (default: 24)')
    parser.add_argument('--search-folds', type=int, default=6,
                        help='Folds for auto_arima candidate orders, spread evenly over the tune '
                             'folds (default: 6); candidates are scored on all tune folds')
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
