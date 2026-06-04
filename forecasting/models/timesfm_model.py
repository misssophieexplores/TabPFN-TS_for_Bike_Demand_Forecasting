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
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=168,
            )
        )
        self._y_train = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        self._y_train = y_train
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        point_forecast, _ = self.model.forecast(
            horizon=horizon,
            inputs=[self._y_train.astype(np.float32)],
        )
        return point_forecast[0, :horizon]

    def reset(self) -> None:
        super().reset()
        self._y_train = None