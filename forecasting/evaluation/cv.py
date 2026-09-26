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

    def get_cutoff_date(self, df: pd.DataFrame) -> pd.Timestamp:
        """
        Return the first fold cutoff date (start of held-out test period).

        This is the single source of truth for the cutoff calculation.
        split() calls this method internally to guarantee consistency.

        The cutoff timestamp itself is NOT part of the test period: split()
        builds test windows as (test_start, test_end] with test_start = cutoff
        for fold 0, so the cutoff is the last training hour of fold 0 and the
        held-out test period is (cutoff, data_end]. Tuning data is therefore
        selected with `<= cutoff`.

        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset with datetime index or column

        Returns:
        --------
        pd.Timestamp
            Cutoff date separating tuning/training data from held-out test period
        """
        df = df.copy()
        df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
        data_end = df[self.config.date_col].max()
        n_eval_hours = self.get_eval_hours()
        return data_end - pd.Timedelta(hours=n_eval_hours)

    def get_eval_hours(self) -> int:
        """Length of the evaluation period in hours: n_folds * max(horizons)."""
        return self.config.n_folds * max(self.config.horizons)

    def expected_n_folds(self, horizon: int) -> int:
        """
        Number of evaluation folds for this horizon with partial_last_fold=True:
        ceil(eval_hours / horizon). If eval_hours is not a multiple of horizon,
        the last fold is partial (eval_hours % horizon test hours).
        """
        return int(np.ceil(self.get_eval_hours() / horizon))

    def get_tuning_period(self, tune_df: pd.DataFrame) -> dict:
        """
        First and last timestamp of the tuning data (ISO strings, for the
        tuning JSON). The last timestamp equals get_cutoff_date() of the full data.
        """
        dates = pd.to_datetime(tune_df[self.config.date_col])
        return {
            "first_timestamp": dates.min().isoformat(),
            "last_timestamp": dates.max().isoformat(),
        }

    def split(
        self,
        df: pd.DataFrame,
        horizon: int,
        partial_last_fold: bool = False,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits for time series cross-validation.
        
        Dynamically calculates the first fold date and number of folds
        based on available data. Tracks which folds contain imputed data.
        Uses all available data from the first fold date onward.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset with datetime index or column
        horizon : int
            Forecast horizon (number of hours)
        partial_last_fold : bool, default=False
            If the period from the first fold date to the end of the data is
            not a multiple of horizon, keep the last fold with a shorter test
            window (only the remaining hours, from the regular fold origin)
            instead of dropping it. Used for evaluation (run_experiments.py),
            so every horizon covers the full evaluation period. Tuning scripts
            use the default (full-horizon folds only).
            
        Returns:
        --------
        List[Tuple[pd.DataFrame, pd.DataFrame]]
            List of (train_df, test_df) tuples, one per fold
        """
        # Validate input
        if self.config.date_col not in df.columns:
            raise ValueError(f"Column '{self.config.date_col}' not found in dataframe")
        
        df = df.copy()
        df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
        df = df.sort_values(self.config.date_col).reset_index(drop=True)
        
        data_start = df[self.config.date_col].min()
        data_end = df[self.config.date_col].max()

        # Use get_cutoff_date as single source of truth for first fold date
        first_fold_date = self.get_cutoff_date(df)
        
        # Calculate available testing hours
        total_hours = len(df)
        available_for_testing = total_hours - self.config.n_train_samples
        
        # Calculate maximum possible folds for this horizon
        longest_horizon = max(self.config.horizons)
        max_possible_folds = int(np.floor(available_for_testing / longest_horizon))
        
        # Store for reporting
        self._first_fold_date = first_fold_date
        self._actual_n_folds = None  # set after loop
        self._imputed_fold_info = []
        
        if self.config.verbose:
            print(f"CV Info for horizon={horizon}h:")
            print(f"  Data: {data_start} to {data_end} ({total_hours} hours)")
            print(f"  First fold date: {first_fold_date}")
            print(f"  Available for testing: {available_for_testing} hours")
            print(f"  Max possible folds: {max_possible_folds}")
            print(f"  Requested folds: {self.config.n_folds}")
            print(f"  Using all available data from first fold date onward")
  
        # Generate splits — run until data is exhausted
        splits = []
        fold = 0

        while True:
            # Calculate fold boundaries
            test_start = first_fold_date + pd.Timedelta(hours=fold * horizon)
            test_end = test_start + pd.Timedelta(hours=horizon)
            
            # Break if we've run out of data. With partial_last_fold, a last
            # fold that extends past data_end is truncated to data_end.
            if test_start >= data_end:
                break
            if test_end > data_end:
                if not partial_last_fold:
                    break
                test_end = data_end
            n_test_hours = int((test_end - test_start) / pd.Timedelta(hours=1))
            
            # Create train/test split (rolling: fixed-size window ending at test_start)
            train_end = test_start
            train_start = train_end - pd.Timedelta(hours=self.config.n_train_samples)
            train_mask = (df[self.config.date_col] > train_start) & (df[self.config.date_col] <= train_end)
            test_mask = (df[self.config.date_col] > test_start) & (df[self.config.date_col] <= test_end)
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            # Check for imputed data in this fold
            train_imputed = 0
            test_imputed = 0
            fday = self.config.functioning_day_col
            if fday and fday in train_df.columns:
                train_imputed = (train_df[fday] == 'No').sum()
                test_imputed = (test_df[fday] == 'No').sum()
            
            # Store imputation info
            self._imputed_fold_info.append({
                'fold': fold,
                'train_imputed': train_imputed,
                'test_imputed': test_imputed,
                'train_total': len(train_df),
                'test_total': len(test_df)
            })
            
            # Verify training size and valid test size (horizon hours, or the
            # remaining hours for a partial last fold)
            if len(train_df) >= self.config.n_train_samples and len(test_df) == n_test_hours:
                splits.append((train_df, test_df))

            fold += 1

        self._actual_n_folds = len(splits)

        if self.config.verbose:
            print(f"  Actual folds generated: {self._actual_n_folds}")

        # Report imputation summary
        imputed_folds = [info for info in self._imputed_fold_info if info['test_imputed'] > 0]
        if imputed_folds and self.config.verbose:
            print(f"\n  Imputation info:")
            print(f"    Folds with imputed test data: {len(imputed_folds)}/{len(splits)}")
            for info in imputed_folds:
                print(f"      Fold {info['fold']}: {info['test_imputed']}/{info['test_total']} test observations imputed")
                
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