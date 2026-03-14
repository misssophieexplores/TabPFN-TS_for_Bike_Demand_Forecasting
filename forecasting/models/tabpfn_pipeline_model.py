"""
TabPFN-TS Pipeline forecasting models (new API).

Uses TabPFNTSPipeline introduced in the updated tabpfn-time-series release.
Covariates are passed directly as extra columns — no manual FeatureTransformer needed.
Context length and truncation are handled internally by the pipeline.

Two variants:
  - TabPFNPipelineForecaster:            WITH weather covariates
  - TabPFNPipelineForecaster_NoWeather:  WITHOUT weather covariates (univariate)

Both set needs_datetime = True so run_experiments.py attaches a real
DatetimeIndex to X_train/X_test from the dataset's date column.
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

    needs_datetime = True

    def __init__(self, tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL):
        super().__init__("TabPFN", use_covariates=True)
        self.tabpfn_mode = tabpfn_mode
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Store training context. Pipeline is zero-shot — no actual training occurs.

        Parameters
        ----------
        y_train : np.ndarray
            Training target values.
        X_train : pd.DataFrame
            Must have a DatetimeIndex (set by run_experiments.py).
            Weather covariates are included as extra columns.
        """
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False

        if X_train is None or not isinstance(X_train.index, pd.DatetimeIndex):
            raise ValueError(
                "TabPFNPipelineForecaster requires X_train with a DatetimeIndex. "
                "Ensure needs_datetime=True is handled in run_experiments.py."
            )

        context_df = pd.DataFrame({
            'item_id': 'series_1',
            'timestamp': X_train.index,
            'target': y_train,
        })

        for col in X_train.columns:
            context_df[col] = X_train[col].values

        self.context_df = context_df
        self.last_timestamp = X_train.index[-1]
        self.pipeline = TabPFNTSPipeline(tabpfn_mode=self.tabpfn_mode)
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Parameters
        ----------
        horizon : int
        X_future : pd.DataFrame
            Must have a DatetimeIndex. Weather covariates as columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "TabPFNPipelineForecaster requires X_future with a DatetimeIndex."
            )

        future_df = pd.DataFrame({
            'item_id': 'series_1',
            'timestamp': X_future.index[:horizon],
        })

        for col in X_future.columns:
            future_df[col] = X_future[col].values[:horizon]

        pred_df = self.pipeline.predict_df(
            context_df=self.context_df,
            future_df=future_df,
        )

        return pred_df['target'].values

    def reset(self) -> None:
        super().reset()
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None


class TabPFNPipelineForecaster_NoWeather(BaseForecaster):
    """
    TabPFN-TS forecaster using the new TabPFNTSPipeline API, WITHOUT weather covariates.

    Univariate only — X_train columns are ignored, but DatetimeIndex is still used
    for timestamps.
    """

    needs_datetime = True

    def __init__(self, tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL):
        super().__init__("TabPFN_NoWeather", use_covariates=False)
        self.tabpfn_mode = tabpfn_mode
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Parameters
        ----------
        y_train : np.ndarray
        X_train : pd.DataFrame
            Must have a DatetimeIndex. Columns are ignored (univariate).
        """
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False

        if X_train is None or not isinstance(X_train.index, pd.DatetimeIndex):
            raise ValueError(
                "TabPFNPipelineForecaster_NoWeather requires X_train with a DatetimeIndex. "
                "Ensure needs_datetime=True is handled in run_experiments.py."
            )

        self.context_df = pd.DataFrame({
            'item_id': 'series_1',
            'timestamp': X_train.index,
            'target': y_train,
        })

        self.last_timestamp = X_train.index[-1]
        self.pipeline = TabPFNTSPipeline(tabpfn_mode=self.tabpfn_mode)
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Parameters
        ----------
        horizon : int
        X_future : pd.DataFrame
            Must have a DatetimeIndex. Columns are ignored (univariate).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "TabPFNPipelineForecaster_NoWeather requires X_future with a DatetimeIndex."
            )

        future_df = pd.DataFrame({
            'item_id': 'series_1',
            'timestamp': X_future.index[:horizon],
        })

        pred_df = self.pipeline.predict_df(
            context_df=self.context_df,
            future_df=future_df,
        )

        return pred_df['target'].values

    def reset(self) -> None:
        super().reset()
        self.pipeline = None
        self.context_df = None
        self.last_timestamp = None