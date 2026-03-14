"""
TabPFN-TS Pipeline forecasting models (new API).

Uses TabPFNTSPipeline introduced in the updated tabpfn-time-series release.
Covariates are passed directly as extra columns — no manual FeatureTransformer needed.
Context length and truncation are handled internally by the pipeline.

Two variants:
  - TabPFNPipelineForecaster:          WITH weather covariates
  - TabPFNPipelineForecaster_NoWeather: WITHOUT weather covariates (univariate)
"""

import os
# #TODO: comment out before running on GPU
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd
from typing import Optional

from models.base import BaseForecaster
from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode


class TabPFNPipelineForecaster(BaseForecaster):
    """
    TabPFN-TS forecaster using the new TabPFNTSPipeline API, WITH weather covariates.

    Covariates are included as extra columns in context_df and future_df.
    Context length is handled internally by the pipeline.
    """

    def __init__(self, tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL):
        """
        Parameters
        ----------
        tabpfn_mode : TabPFNMode
            LOCAL (CPU/GPU on your machine) or CLIENT (cloud API).
        """
        super().__init__("TabPFN", use_covariates=True)
        self.tabpfn_mode = tabpfn_mode
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Store training context. Pipeline is zero-shot — no actual training occurs.
        Context length is managed internally by TabPFNTSPipeline.

        Parameters
        ----------
        y_train : np.ndarray
            Training target values.
        X_train : pd.DataFrame, optional
            Training weather covariates (aligned with y_train).
        """
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False

        timestamps = pd.date_range(start='2017-12-01', periods=len(y_train), freq='h')

        context_data = {
            'item_id': 'series_1',
            'timestamp': timestamps,
            'target': y_train,
        }

        context_df = pd.DataFrame(context_data)

        if X_train is not None:
            X_reset = X_train.reset_index(drop=True)
            for col in X_reset.columns:
                context_df[col] = X_reset[col].values

        self.context_df = context_df
        self.last_timestamp = timestamps[-1]
        self.pipeline = TabPFNTSPipeline(tabpfn_mode=self.tabpfn_mode)
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecast using the pipeline.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X_future : pd.DataFrame, optional
            Future weather covariates for the forecast horizon.

        Returns
        -------
        np.ndarray
            Forecast values of shape (horizon,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        future_timestamps = pd.date_range(
            start=self.last_timestamp + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h'
        )

        future_data = {
            'item_id': 'series_1',
            'timestamp': future_timestamps,
        }

        future_df = pd.DataFrame(future_data)

        if X_future is not None:
            X_reset = X_future.reset_index(drop=True)
            for col in X_reset.columns:
                future_df[col] = X_reset[col].values[:horizon]

        pred_df = self.pipeline.predict_df(
            context_df=self.context_df,
            future_df=future_df,
        )

        return pred_df['target'].values

    def reset(self) -> None:
        """Reset model state between CV folds."""
        super().reset()
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None


class TabPFNPipelineForecaster_NoWeather(BaseForecaster):
    """
    TabPFN-TS forecaster using the new TabPFNTSPipeline API, WITHOUT weather covariates.

    Univariate only — X_train and X_future are ignored.
    Context length is handled internally by the pipeline.
    """

    def __init__(self, tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL):
        """
        Parameters
        ----------
        tabpfn_mode : TabPFNMode
            LOCAL (CPU/GPU on your machine) or CLIENT (cloud API).
        """
        super().__init__("TabPFN_NoWeather", use_covariates=False)
        self.tabpfn_mode = tabpfn_mode
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Store training context. X_train is ignored.

        Parameters
        ----------
        y_train : np.ndarray
            Training target values.
        X_train : pd.DataFrame, optional
            Ignored.
        """
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False

        timestamps = pd.date_range(start='2017-12-01', periods=len(y_train), freq='h')

        self.context_df = pd.DataFrame({
            'item_id': 'series_1',
            'timestamp': timestamps,
            'target': y_train,
        })

        self.last_timestamp = timestamps[-1]
        self.pipeline = TabPFNTSPipeline(tabpfn_mode=self.tabpfn_mode)
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecast. X_future is ignored.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X_future : pd.DataFrame, optional
            Ignored.

        Returns
        -------
        np.ndarray
            Forecast values of shape (horizon,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        future_timestamps = pd.date_range(
            start=self.last_timestamp + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h'
        )

        future_df = pd.DataFrame({
            'item_id': 'series_1',
            'timestamp': future_timestamps,
        })

        pred_df = self.pipeline.predict_df(
            context_df=self.context_df,
            future_df=future_df,
        )

        return pred_df['target'].values

    def reset(self) -> None:
        """Reset model state between CV folds."""
        super().reset()
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None