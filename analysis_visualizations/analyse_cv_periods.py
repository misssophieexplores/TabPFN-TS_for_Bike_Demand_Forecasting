import math
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'forecasting' / 'evaluation'))
sys.path.insert(0, str(project_root / 'forecasting'))

from cv import TimeSeriesCV
import config_london
import config_seoul
import config_washington

DATA_DIR = project_root / "data"

datasets = [
    ('London', config_london, DATA_DIR / 'LondonBikeData.csv'),
    ('Seoul', config_seoul, DATA_DIR / 'SeoulBikeData.csv'),
    ('Washington', config_washington, DATA_DIR / 'WashingtonBikeData.csv')
]

for city_name, config_module, filepath in datasets:
    if not filepath.exists():
        continue

    config = config_module.get_config()
    config.verbose = False
    df = pd.read_csv(filepath)
    df[config.date_col] = pd.to_datetime(df[config.date_col])

    data_start = df[config.date_col].min()
    data_end = df[config.date_col].max()

    earliest_train_start = None

    print(f"\n{'='*80}")
    print(f"{city_name}: {data_start} → {data_end}")
    print(f"{'='*80}")
    print(f"{'Horizon':>10} | {'Folds':>5} | {'Earliest train_start':>22} | {'Latest train_end':>22} | {'Earliest test_start':>22} | {'Latest test_end':>22} |{'Unused data before':>20}")

    for horizon in config.horizons:
        cv = TimeSeriesCV(config)
        splits = cv.split(df, horizon)
        if not splits:
            continue

        info = cv.get_split_info(splits)
        fold0_train_start = info['train_start'].min()
        fold0_train_end = info['train_end'].max()
        fold0_test_start = info['test_start'].min()
        fold0_test_end = info['test_end'].max()
        unused = fold0_train_start - data_start

        if earliest_train_start is None or fold0_train_start < earliest_train_start:
            earliest_train_start = fold0_train_start

        print(f"{horizon:>9}h | {len(splits):>5} | {str(fold0_train_start):>22} | {str(fold0_train_end):>22} |  | {str(fold0_test_start):>22} | {str(fold0_test_end):>22} |{str(unused):>20}")

    unused_total = earliest_train_start - data_start
    print(f"\n  Earliest date used in ANY training fold: {earliest_train_start}")
    print(f"  Data never used:                         {data_start} → {earliest_train_start} ({unused_total})")
    print(f"  Data used in at least one fold:          {earliest_train_start} → {data_end} ({data_end - earliest_train_start})")
    print(f"  Total number of data points:              {len(df)}")
    lookback_windows = [4096, 60*24, 30*24]
    for n_train_samples  in lookback_windows:
        for h in [max(config.horizons), min(config.horizons)]:
            n_total_folds = math.floor((len(df) - n_train_samples) / h)
            print(f"  Max #folds for horizon {h:>3} and lookback window {n_train_samples:>5}: {n_total_folds:>5}")
