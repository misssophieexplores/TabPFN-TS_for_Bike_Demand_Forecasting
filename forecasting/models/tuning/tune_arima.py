"""
ARIMA Hyperparameter Tuning using auto_arima

Uses pmdarima's auto_arima for automatic parameter search.
Non-seasonal ARIMA only (no covariates).

Usage:
    python forecasting/models/tuning/tune_arima.py --city seoul
    python forecasting/models/tuning/tune_arima.py --city london
    python forecasting/models/tuning/tune_arima.py --city washington
"""

import sys
from pathlib import Path

# robust import path regardless of current working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # forecasting/

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings

from config import ForecastConfig
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator

try:
    from pmdarima import auto_arima
except ImportError:
    print("ERROR: pmdarima not installed")
    print("Install with: pip install pmdarima")
    sys.exit(1)


def load_data(filepath: str, config: ForecastConfig) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.sort_values(config.date_col).reset_index(drop=True)
    return df  # no drop — covariate selection happens below


def tune_arima(
    df: pd.DataFrame,
    config: ForecastConfig,
    horizon: int = 24,
    max_folds: int = 20,
    verbose: bool = True
) -> dict:
    """
    Find optimal ARIMA parameters using auto_arima.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    horizon : int
        Forecast horizon for validation
    max_folds : int
        Number of CV folds for validation
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict
        Best parameters and validation metrics
    """
    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    
    if verbose:
        print("="*70)
        print("ARIMA AUTO-TUNING (pmdarima)")
        print("="*70)
        print(f"Horizon: {horizon}h")
        print(f"Validation folds: {max_folds}")
        print("="*70)
    
    # Use first fold for tuning
    splits = cv.split(df, horizon)
    train_df, test_df = splits[0]
    
    y_train = train_df[config.target_col].values
    
    if verbose:
        print(f"\nSearching optimal parameters on {len(y_train)} observations...")
    
    # Run auto_arima (non-seasonal)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        model = auto_arima(
            y_train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=7, max_q=3,
            max_order=8,
            trace=verbose,
            information_criterion='aic',
            n_jobs=-1
        )

    # Extract parameters
    order = model.order
    aic = model.aic()
    bic = model.bic()
    
    if verbose:
        print("\n" + "="*70)
        print("BEST PARAMETERS FOUND")
        print("="*70)
        print(f"order: {order}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
    
    # Validate on multiple folds
    if verbose:
        print(f"\nValidating on {max_folds} folds...")
    
    mae_values = []
    rmse_values = []
    
    for fold_idx in range(min(max_folds, len(splits))):
        train_df, test_df = splits[fold_idx]
        
        y_train_fold = train_df[config.target_col].values
        y_test_fold = test_df[config.target_col].values
        
        try:
            # Fit model with found parameters
            model_fold = auto_arima(
                y_train_fold,
                start_p=order[0], start_q=order[2],
                max_p=order[0], max_q=order[2],
                d=order[1],
                seasonal=False,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            # Predict
            y_pred = model_fold.predict(n_periods=horizon)
            
            # Metrics
            metrics = calc.calculate_all(y_test_fold, y_pred, y_train_fold)
            mae_values.append(metrics['MAE'])
            rmse_values.append(metrics['RMSE'])
            
            if verbose:
                print(f"  Fold {fold_idx}: MAE={metrics['MAE']:.1f}, RMSE={metrics['RMSE']:.1f}")
        
        except Exception as e:
            if verbose:
                print(f"  Fold {fold_idx}: FAILED - {str(e)[:50]}")
            continue
    
    # Aggregate validation results
    if len(mae_values) > 0:
        mae_mean = np.mean(mae_values)
        mae_std = np.std(mae_values)
        rmse_mean = np.mean(rmse_values)
        rmse_std = np.std(rmse_values)
    else:
        mae_mean = mae_std = rmse_mean = rmse_std = np.nan
    
    if verbose:
        print(f"\nValidation results:")
        print(f"  MAE: {mae_mean:.2f} ± {mae_std:.2f}")
        print(f"  RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}")
    
    return {
        'order': order,
        'aic': float(aic),
        'bic': float(bic),
        'mae_mean': float(mae_mean),
        'mae_std': float(mae_std),
        'rmse_mean': float(rmse_mean),
        'rmse_std': float(rmse_std),
        'validation_folds': len(mae_values)
    }


def save_results(params: dict, output_dir: str = '.') -> Path:
    """Save tuning results to JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'arima_best_params_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Tune ARIMA using auto_arima')
    parser.add_argument(
        '--city',
        type=str,
        required=True,
        choices=['seoul', 'london', 'washington'],
        help='Which city config to use'
    )
    parser.add_argument('--horizon', type=int, default=24, help='Forecast horizon for validation')
    parser.add_argument('--folds', type=int, default=20, help='Number of validation folds')
    parser.add_argument('--output-dir', type=str, default='results/tuning', help='Directory to save results')

    args = parser.parse_args()

    if args.city == 'seoul':
        from config_seoul import get_config
    elif args.city == 'london':
        from config_london import get_config
    elif args.city == 'washington':
        from config_washington import get_config
    config = get_config()

    data_path = str(Path("data") / config.data_filename)
    print(f"Loading data from {data_path}...")
    df = load_data(data_path, config)
    print(f"Loaded {len(df)} observations\n")

    params = tune_arima(
        df=df,
        config=config,
        horizon=args.horizon,
        max_folds=args.folds,
        verbose=True
    )

    output_file = save_results(params, args.output_dir)

    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    print(f"\nUpdate config_{args.city}.py with:")
    print(f"  config.arima_params_file = '{output_file}'")


if __name__ == '__main__':
    main()