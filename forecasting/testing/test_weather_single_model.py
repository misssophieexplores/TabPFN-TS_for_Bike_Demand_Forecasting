"""
Test single model with weather scenarios — same setup as the paper runs.

Models are built with run_experiments.build_models() (tuned parameters from the
params files in the city config) and each fold's inputs with
run_experiments.prepare_fold_inputs(), exactly as in run_weather_baseline.py.
Scoring excludes imputed hours, as in the experiments.

Usage:
  python forecasting/testing/test_weather_single_model.py --city seoul --model tabpfn --scenario degraded
  python forecasting/testing/test_weather_single_model.py --city seoul --model xgboost --scenario clean_only
  python forecasting/testing/test_weather_single_model.py --city seoul --model arima --scenario clean_only
  python forecasting/testing/test_weather_single_model.py --city seoul --model prophet --scenario clean_only
  python forecasting/testing/test_weather_single_model.py --city seoul --model neuralprophet --scenario degraded
  python forecasting/testing/test_weather_single_model.py --city seoul --model timesfm_noweather --scenario clean_only

Tests the first --folds folds (default 3) of the evaluation period at the
shortest horizon (quick validation).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from weather.weather_processor import WeatherProcessor
from run_experiments import MODEL_KEYS, build_models, load_and_prepare_data, prepare_fold_inputs


def test_single_model_scenario(config, model_key: str, scenario: str, n_folds: int = 3):
    df, _ = load_and_prepare_data(config)
    model = build_models(config, [model_key])[0]

    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    weather_proc = WeatherProcessor(config)
    scenario_info = weather_proc.get_scenario_summary(scenario)

    print(f"\n{'='*60}")
    print("Test Configuration")
    print(f"{'='*60}")
    print(f"Model: {model.name}")
    print(f"Uses covariates: {model.use_covariates}")
    print(f"Scenario: {scenario}")
    print(f"Variables: {scenario_info['num_vars']} - {scenario_info['variables']}")
    print(f"Degraded: {scenario_info['degraded']}")
    print(f"Folds: {n_folds}")
    print(f"{'='*60}\n")

    if not model.use_covariates and config.is_degraded(scenario):
        print(f"[SKIP] Model doesn't use covariates, skipping degraded scenario {scenario}")
        print("This is expected behavior!")
        return

    test_horizon = min(config.horizons)
    print(f"Testing horizon: {test_horizon}h")

    # Same folds as the experiments (evaluation period, partial last fold kept)
    splits = cv.split(df, test_horizon, partial_last_fold=True)
    fold_results = []
    fday = config.functioning_day_col

    for fold_idx in range(min(n_folds, len(splits))):
        train_df, test_df = splits[fold_idx]
        y_train, X_train, y_test, X_test = prepare_fold_inputs(
            config, model, weather_proc, train_df, test_df, scenario, test_horizon, fold_idx
        )
        if X_train is not None:
            print(f"Fold {fold_idx}: columns in X_train: {X_train.columns.tolist()}")
            if model.use_covariates:
                print(f"  holiday present: {config.holiday_col in X_train.columns if config.holiday_col else 'N/A'}")
                print(f"  season present:  {config.season_col in X_train.columns if config.season_col else 'N/A'}")
        try:
            model.reset()
            model.fit(y_train, X_train)
            y_pred = model.predict(len(test_df), X_test)

            metrics = calc.calculate_all(
                y_test, y_pred, y_train,
                test_mask=calc.observed_mask(test_df, fday),
                train_mask=calc.observed_mask(train_df, fday),
            )
            metrics['fold'] = fold_idx
            metrics['scenario'] = scenario
            metrics['num_weather_vars'] = len(X_train.columns) if X_train is not None else 0
            fold_results.append(metrics)

            print(f"  Fold {fold_idx:2d}: "
                  f"MAE={metrics['MAE']:6.1f} | "
                  f"RMSE={metrics['RMSE']:6.1f} | "
                  f"MASE={metrics['MASE']:5.2f}")

        except Exception as e:
            print(f"  Fold {fold_idx:2d}: ERROR - {str(e)[:200]}")

    if len(fold_results) == min(n_folds, len(splits)):
        results_df = pd.DataFrame(fold_results)
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
        print(f"MAE:  {results_df['MAE'].mean():6.1f} ± {results_df['MAE'].std():5.1f}")
        print(f"RMSE: {results_df['RMSE'].mean():6.1f} ± {results_df['RMSE'].std():5.1f}")
        print(f"MASE: {results_df['MASE'].mean():5.2f} ± {results_df['MASE'].std():4.2f}")
        print(f"{'='*60}\n")
        print("TEST PASSED: all folds completed")
    else:
        print(f"\nTEST FAILED: {len(fold_results)}/{min(n_folds, len(splits))} folds completed")
        sys.exit(1)


if __name__ == "__main__":
    from config_seoul import get_config as seoul_config
    from config_washington import get_config as washington_config
    from config_london import get_config as london_config

    city_configs = {
        "seoul":      seoul_config,
        "washington": washington_config,
        "london":     london_config,
    }

    parser = argparse.ArgumentParser(description='Test single forecasting model with weather scenario')
    parser.add_argument('--city', choices=list(city_configs.keys()), required=True,
                        help="City dataset to use.")
    parser.add_argument('--model', type=str, required=True, choices=MODEL_KEYS)
    parser.add_argument('--scenario', type=str, required=True,
                        choices=seoul_config().weather_scenarios)  # same for every city (config.py)
    parser.add_argument('--folds', type=int, default=3)
    args = parser.parse_args()

    test_single_model_scenario(city_configs[args.city](), args.model, args.scenario, args.folds)
