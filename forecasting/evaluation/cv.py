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
        """
        Generate train/test splits for time series cross-validation.
        
        Dynamically calculates the first fold date and number of folds
        based on available data. Tracks which folds contain imputed data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset with datetime index or column
        horizon : int
            Forecast horizon (number of hours)
            
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
        
        # Calculate first fold date dynamically
        data_start = df[self.config.date_col].min()
        data_end = df[self.config.date_col].max()
        
        # First fold cutoff date counted back from end of dataset
        n_eval_hours = self.config.n_folds * max(self.config.horizons)  # total evaluation window based on folds and longest horizon
        first_fold_date = data_end - pd.Timedelta(hours=n_eval_hours)  # count back from end for consistent cutoff across datasets
        
        # Calculate available testing hours
        total_hours = len(df)
        available_for_testing = total_hours - self.config.n_train_samples
        
        # Calculate maximum possible folds for this horizon
        longest_horizon = max(self.config.horizons)
        max_possible_folds = int(np.floor(available_for_testing / longest_horizon))
        
        # Actual folds is minimum of requested and possible
        actual_n_folds = min(self.config.n_folds, max_possible_folds)
        
        # Store for reporting
        self._first_fold_date = first_fold_date
        self._actual_n_folds = actual_n_folds
        self._imputed_fold_info = []
        
        # Print info on first call
        print(f"CV Info for horizon={horizon}h:")
        print(f"  Data: {data_start} to {data_end} ({total_hours} hours)")
        print(f"  First fold date: {first_fold_date}")        
        print(f"  Available for testing: {available_for_testing} hours")
        print(f"  Max possible folds: {max_possible_folds}")
        print(f"  Requested folds: {self.config.n_folds}")
        print(f"  Actual folds: {actual_n_folds}")
        
        if actual_n_folds < self.config.n_folds:
            print(f"  WARNING: Using {actual_n_folds} folds instead of requested {self.config.n_folds}")
  
        # Generate splits
        splits = []
        
        for fold in range(actual_n_folds):
            # Calculate fold boundaries
            test_start = first_fold_date + pd.Timedelta(hours=fold * horizon)
            test_end = test_start + pd.Timedelta(hours=horizon)
            
            # Break if we've run out of data
            if test_end > data_end:
                break
            
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
            
            # Verify training size and valid test size
            if len(train_df) >= self.config.n_train_samples and len(test_df) == horizon:
                splits.append((train_df, test_df))
        
        # Report imputation summary
        imputed_folds = [info for info in self._imputed_fold_info if info['test_imputed'] > 0]
        if imputed_folds:
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