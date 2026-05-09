"""
Time series cross-validation with dynamic fold calculation (rolling window).
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from config import ForecastConfig


class TimeSeriesCV:
    """
    Time series cross-validation with rolling window.
    
    Automatically calculates first fold date and actual number of folds
    based on available data and minimum training requirements.
    Each fold uses a fixed-size training window that advances with the fold.
    Tracks which folds contain imputed data.
    """
    
    def __init__(self, config: ForecastConfig):
        """
        Initialize CV splitter.
        
        Parameters:
        -----------
        config : ForecastConfig
            Configuration object with CV parameters
        """
        self.config = config
        self._first_fold_date = None
        self._actual_n_folds = None
        self._imputed_fold_info = []

    def split(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        # ... (validation and setup unchanged) ...

        # Calculate first fold date dynamically (unchanged)
        data_start = df[self.config.date_col].min()
        data_end = df[self.config.date_col].max()

        n_eval_hours = self.config.n_folds * max(self.config.horizons)
        first_fold_date = data_end - pd.Timedelta(hours=n_eval_hours)

        # For reporting only — no longer used to cap the loop
        total_hours = len(df)
        available_for_testing = total_hours - self.config.n_train_samples
        longest_horizon = max(self.config.horizons)
        max_possible_folds = int(np.floor(available_for_testing / longest_horizon))
        actual_n_folds = max_possible_folds  # will be updated after loop

        self._first_fold_date = first_fold_date
        self._actual_n_folds = None  # set after loop
        self._imputed_fold_info = []

        if self.config.verbose:
            print(f"CV Info for horizon={horizon}h:")
            print(f"  Data: {data_start} to {data_end} ({total_hours} hours)")
            print(f"  First fold date: {first_fold_date}")
            print(f"  Available for testing: {available_for_testing} hours")
            print(f"  Requested folds: {self.config.n_folds} (cutoff anchor only)")
            print(f"  Using all available test data from first fold date onward")

        # Generate splits — run until data is exhausted
        splits = []
        fold = 0

        while True:
            test_start = first_fold_date + pd.Timedelta(hours=fold * horizon)
            test_end = test_start + pd.Timedelta(hours=horizon)

            if test_end > data_end:
                break

            train_end = test_start
            train_start = train_end - pd.Timedelta(hours=self.config.n_train_samples)
            train_mask = (df[self.config.date_col] > train_start) & (df[self.config.date_col] <= train_end)
            test_mask = (df[self.config.date_col] > test_start) & (df[self.config.date_col] <= test_end)

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            # ... (imputation tracking unchanged) ...

            if len(train_df) >= self.config.n_train_samples and len(test_df) == horizon:
                splits.append((train_df, test_df))

            fold += 1

        self._actual_n_folds = len(splits)

        if self.config.verbose:
            print(f"  Actual folds generated: {self._actual_n_folds}")

        # ... (imputation reporting unchanged) ...

        return splits
    
    
    def get_split_info(self, splits: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
        """
        Get summary information about the splits including imputation info.
        
        Parameters:
        -----------
        splits : List[Tuple[pd.DataFrame, pd.DataFrame]]
            List of (train_df, test_df) tuples
            
        Returns:
        --------
        pd.DataFrame
            Summary with train/test sizes, date ranges, and imputation info
        """
        info = []
        for i, (train_df, test_df) in enumerate(splits):
            fold_info = {
                'fold': i,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_start': train_df[self.config.date_col].min(),
                'train_end': train_df[self.config.date_col].max(),
                'test_start': test_df[self.config.date_col].min(),
                'test_end': test_df[self.config.date_col].max()
            }
            
            # Add imputation info if available
            if i < len(self._imputed_fold_info):
                fold_info['train_imputed'] = self._imputed_fold_info[i]['train_imputed']
                fold_info['test_imputed'] = self._imputed_fold_info[i]['test_imputed']
            
            info.append(fold_info)
            
        return pd.DataFrame(info)
    
    def get_imputation_summary(self) -> pd.DataFrame:
        """
        Get summary of imputed data across all folds.
        
        Returns:
        --------
        pd.DataFrame
            Summary of imputation by fold
        """
        if not self._imputed_fold_info:
            return pd.DataFrame()
        
        return pd.DataFrame(self._imputed_fold_info)