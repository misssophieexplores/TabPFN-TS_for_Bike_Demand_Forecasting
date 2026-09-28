"""
ARIMA order tuning (non-seasonal, no covariates).

Procedure (shared with tune_sarimax.py, see arima_search.py):
  1. pmdarima auto_arima (stepwise, AIC) on the training window of each of the
     --search-folds search folds (default 6, spread evenly over the tune folds,
     same folds as tune_neuralprophet.py) -> candidate orders
  2. every candidate is fitted with ARIMAForecaster (statsmodels, as in the
     experiments) on every search fold and scored on the next tune_horizon
     hours; the lowest mean MAE is selected

Usage:
    # Tune all cities:
    python forecasting/models/tuning/tune_arima.py

    # Tune a specific city only:
    python forecasting/models/tuning/tune_arima.py --city seoul
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

AUTO_ARIMA_KWARGS = dict(
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    max_p=7, max_q=3,
    max_order=8,
    information_criterion='aic',
)


def tune_arima(
    df: pd.DataFrame,
    config: ForecastConfig,
    city: str,
    search_folds: int = 6,
    verbose: bool = True,
) -> dict:
    if verbose:
        print("=" * 70)
        print(f"ARIMA TUNING | city={city} | horizon={config.tune_horizon}h")
        print("=" * 70)

    result = search_orders(
        df, config, seasonal=False, scenario="clean_only",
        auto_arima_kwargs={**AUTO_ARIMA_KWARGS, "trace": False},
        search_folds=search_folds, verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 70)
        print(f"SELECTED: order={result['order']} with_intercept={result['with_intercept']} "
              f"(trend={result['trend']}) | search MAE={result['tuning']['best_tune_mae_mean']:.2f}")
        print("=" * 70)

    return {
        'city': city,
        'n_train_samples': config.n_train_samples,
        'tuning_period': result['tuning_period'],
        'order': result['order'],
        'with_intercept': result['with_intercept'],
        'trend': result['trend'],
        'tuning': result['tuning'],
    }


def save_results(params: dict, output_dir: str = '.') -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    city, n_train = params['city'], params['n_train_samples']
    output_file = output_dir / f'arima_best_params_{city}_{n_train}_{timestamp}.json'
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

    params = tune_arima(df=df, config=config, city=city, search_folds=args.search_folds, verbose=True)
    output_file = save_results(params, args.output_dir)
    print(f"  --> config.arima_params_file = '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description='Tune ARIMA order (auto_arima candidates, MAE selection)')
    parser.add_argument('--city', type=str, choices=['seoul', 'london', 'washington'],
                        default=None, help='City to tune (default: all cities)')
    parser.add_argument('--search-folds', type=int, default=6,
                        help='Search folds, spread evenly over the tune folds (default: 6)')
    parser.add_argument('--output-dir', type=str, default='results/tuning', help='Directory to save results')
    args = parser.parse_args()

    cities = [args.city] if args.city else ['seoul', 'london', 'washington']

    for city in cities:
        print("\n" + "="*70)
        print(f"TUNING ARIMA FOR: {city.upper()}")
        print("="*70)
        run_city(city, args)

    print("\n" + "="*70)
    print("ALL TUNING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
