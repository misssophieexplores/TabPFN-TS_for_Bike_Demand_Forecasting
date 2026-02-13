"""
TabPFN-TS forecasting models.
Implements two variants: with and without weather covariates.
"""

import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'  # Add to run locally

import torch
torch.set_num_threads(1)



import numpy as np
import pandas as pd
from typing import Optional
from models.base import BaseForecaster

from tabpfn_time_series import TimeSeriesDataFrame, TabPFNTimeSeriesPredictor, TabPFNMode
from tabpfn_time_series import FeatureTransformer
from tabpfn_time_series.features import RunningIndexFeature, CalendarFeature, AutoSeasonalFeature


class TabPFNForecaster(BaseForecaster):
    """
    TabPFN-TS forecaster WITH weather covariates.
    
    Uses TabPFNv2 with calendar + auto seasonal features + weather covariates.
    Requires minimum 4096 training samples.
    """
    
    def __init__(self, tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL):
        """
        Initialize TabPFN forecaster with covariates.
        
        Parameters:
        -----------
        tabpfn_mode : TabPFNMode
            CLIENT (uses API) or LOCAL (uses local GPU)
        """
        super().__init__("TabPFN", use_covariates=True)
        self.tabpfn_mode = tabpfn_mode
        self.predictor = None
        # self.predictor = TabPFNTimeSeriesPredictor(
        #     tabpfn_mode=TabPFNMode.LOCAL,
        #     n_jobs=1  # Force single process
        # )
        self.feature_transformer = None
        self.train_tsdf = None
        self.last_timestamp = None
        
    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Fit TabPFN model.
        
        Parameters:
        -----------
        y_train : np.ndarray
            Training target values (min 4096 samples)
        X_train : pd.DataFrame, optional
            Training weather covariates
        """
        # Validate minimum sample size
        if len(y_train) < 4096:
            raise ValueError(f"TabPFN requires min 4096 samples, got {len(y_train)}")
        
         # Force CPU device explicitly (disable MPS)
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False

        # Create timestamps (TabPFN uses calendar features)
        timestamps = pd.date_range(start='2017-12-01', periods=len(y_train), freq='h')
        
        # Build time series dataframe
        ts_data = {
            'timestamp': timestamps,
            'target': y_train
        }
        
        # Add weather covariates if provided
        if X_train is not None:
            X_train_reset = X_train.reset_index(drop=True)
            for col in X_train_reset.columns:
                ts_data[col] = X_train_reset[col].values
        
        # Create TabPFN TimeSeriesDataFrame
        ts_df = pd.DataFrame(ts_data)
        ts_df['item_id'] = 'series_1'
        ts_df = ts_df.set_index(['item_id', 'timestamp'])
        
        self.train_tsdf = TimeSeriesDataFrame(ts_df)
        self.last_timestamp = timestamps[-1]
        
        # Setup feature transformer: calendar + auto seasonal features
        self.feature_transformer = FeatureTransformer([
            RunningIndexFeature(),
            CalendarFeature(),
            AutoSeasonalFeature(),
        ])
        
        # Setup predictor
        self.predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=self.tabpfn_mode)
        self._is_fitted = True
        
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecast.
        
        Parameters:
        -----------
        horizon : int
            Number of steps ahead to forecast
        X_future : pd.DataFrame, optional
            Future weather covariates
            
        Returns:
        --------
        np.ndarray
            Forecast values of shape (horizon,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        
        # Create future timestamps
        future_timestamps = pd.date_range(
            start=self.last_timestamp + pd.Timedelta(hours=1), 
            periods=horizon, 
            freq='h'
        )
        
        # Build test dataframe
        test_data = {
            'timestamp': future_timestamps,
            'target': np.nan  # Will be predicted
        }
        
        # Add future weather covariates if provided
        if X_future is not None:
            X_future_reset = X_future.reset_index(drop=True)
            for col in X_future_reset.columns:
                # Handle case where X_future might be longer than horizon
                test_data[col] = X_future_reset[col].values[:horizon]
        
        # Create TabPFN TimeSeriesDataFrame
        test_df = pd.DataFrame(test_data)
        test_df['item_id'] = 'series_1'
        test_df = test_df.set_index(['item_id', 'timestamp'])
        test_tsdf = TimeSeriesDataFrame(test_df)
        
        # Transform features (calendar + auto seasonal)
        train_transformed, test_transformed = self.feature_transformer.transform(
            self.train_tsdf, test_tsdf
        )
        
        # Predict
        pred = self.predictor.predict(train_transformed, test_transformed)
        
        return pred['target'].values
    
    def reset(self) -> None:
        """Reset model state between CV folds."""
        super().reset()
        self.predictor = None
        self.feature_transformer = None
        self.train_tsdf = None
        self.last_timestamp = None


class TabPFNForecaster_NoWeather(BaseForecaster):
    """
    TabPFN-TS forecaster WITHOUT weather covariates.
    
    Uses only calendar + auto seasonal features (no exogenous variables).
    Requires minimum 4096 training samples.
    """
    
    def __init__(self, tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL):
        """
        Initialize TabPFN forecaster without covariates.
        
        Parameters:
        -----------
        tabpfn_mode : TabPFNMode
            CLIENT (uses API) or LOCAL (uses local GPU)
        """
        super().__init__("TabPFN_NoWeather", use_covariates=False)
        self.tabpfn_mode = tabpfn_mode
        self.predictor = None
        # self.predictor = TabPFNTimeSeriesPredictor(
        #     tabpfn_mode=TabPFNMode.LOCAL,
        #     n_jobs=1  # Force single process
        # )
        self.feature_transformer = None
        self.train_tsdf = None
        self.last_timestamp = None
        
    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Fit TabPFN model (ignores X_train).
        
        Parameters:
        -----------
        y_train : np.ndarray
            Training target values (min 4096 samples)
        X_train : pd.DataFrame, optional
            Ignored (model doesn't use covariates)
        """
        # Validate minimum sample size
        if len(y_train) < 4096:
            raise ValueError(f"TabPFN requires min 4096 samples, got {len(y_train)}")

         # Force CPU device explicitly (disable MPS)
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False

        # Create timestamps
        timestamps = pd.date_range(start='2017-12-01', periods=len(y_train), freq='h')
        
        # Build time series dataframe (target only, no covariates)
        ts_df = pd.DataFrame({
            'timestamp': timestamps,
            'target': y_train
        })
        ts_df['item_id'] = 'series_1'
        ts_df = ts_df.set_index(['item_id', 'timestamp'])
        
        self.train_tsdf = TimeSeriesDataFrame(ts_df)
        self.last_timestamp = timestamps[-1]
        
        # Setup feature transformer: calendar + auto seasonal features
        self.feature_transformer = FeatureTransformer([
            RunningIndexFeature(),
            CalendarFeature(),
            AutoSeasonalFeature(),
        ])
        
        # Setup predictor
        self.predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=self.tabpfn_mode)
        self._is_fitted = True
        
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecast (ignores X_future).
        
        Parameters:
        -----------
        horizon : int
            Number of steps ahead to forecast
        X_future : pd.DataFrame, optional
            Ignored (model doesn't use covariates)
            
        Returns:
        --------
        np.ndarray
            Forecast values of shape (horizon,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        
        # Create future timestamps
        future_timestamps = pd.date_range(
            start=self.last_timestamp + pd.Timedelta(hours=1), 
            periods=horizon, 
            freq='h'
        )
        
        # Build test dataframe (target only, no covariates)
        test_df = pd.DataFrame({
            'timestamp': future_timestamps,
            'target': np.nan  # Will be predicted
        })
        test_df['item_id'] = 'series_1'
        test_df = test_df.set_index(['item_id', 'timestamp'])
        test_tsdf = TimeSeriesDataFrame(test_df)
        
        # Transform features
        train_transformed, test_transformed = self.feature_transformer.transform(
            self.train_tsdf, test_tsdf
        )
        
        # Predict
        pred = self.predictor.predict(train_transformed, test_transformed)
        
        return pred['target'].values
    
    def reset(self) -> None:
        """Reset model state between CV folds."""
        super().reset()
        self.predictor = None
        self.feature_transformer = None
        self.train_tsdf = None
        self.last_timestamp = None