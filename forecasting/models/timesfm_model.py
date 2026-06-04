"""
TimesFM forecaster — univariate, no covariates.
forecasting/models/timesfm_model.py
"""
import numpy as np
import pandas as pd
from typing import Optional
import timesfm

from models.base import BaseForecaster


class TimesFMForecaster_NoWeather(BaseForecaster):

    needs_datetime = False

    def __init__(self):
        super().__init__("TimesFM_NoWeather", use_covariates=False)
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="torch", horizon_len=128),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
            ),
        )
        self._y_train = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        self._y_train = y_train
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        point_forecast, _ = self.model.forecast(
            inputs=[self._y_train.astype(np.float32)],
            freq=[0],
        )
        return point_forecast[0, :horizon]

    def reset(self) -> None:
        super().reset()
        self._y_train = None