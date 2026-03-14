"""
Prophet and NeuralProphet forecasting models.

Both set needs_datetime = True so run_experiments.py attaches a real
DatetimeIndex to X_train/X_test from the dataset's date column.
This ensures Prophet's seasonality decomposition uses correct calendar positions.

  ProphetForecaster                 — univariate, no covariates
  NeuralProphetForecaster           — with weather covariates
  NeuralProphetForecaster_NoWeather — univariate variant
"""

import numpy as np
import pandas as pd
from typing import Optional
from models.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """
    Univariate Prophet forecaster (no covariates).

    Real timestamps are required for correct daily/weekly/yearly seasonality.
    Provided via X_train.index (DatetimeIndex) set by run_experiments.py.
    """

    needs_datetime = True

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
    ):
        super().__init__("Prophet", use_covariates=False, use_time_features=False)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.model = None
        self._last_timestamp = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        from prophet import Prophet

        if X_train is None or not isinstance(X_train.index, pd.DatetimeIndex):
            raise ValueError(
                "ProphetForecaster requires X_train with a DatetimeIndex. "
                "Ensure needs_datetime=True is handled in run_experiments.py."
            )

        self._last_timestamp = X_train.index[-1]

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
        )
        train_df = pd.DataFrame({"ds": X_train.index, "y": y_train})
        self.model.fit(train_df)
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if X_future is not None and isinstance(X_future.index, pd.DatetimeIndex):
            future_dates = X_future.index[:horizon]
        else:
            future_dates = pd.date_range(
                start=self._last_timestamp + pd.Timedelta(hours=1),
                periods=horizon,
                freq="h",
            )

        forecast = self.model.predict(pd.DataFrame({"ds": future_dates}))
        return forecast["yhat"].values

    def reset(self) -> None:
        super().reset()
        self.model = None
        self._last_timestamp = None


class NeuralProphetForecaster(BaseForecaster):
    """
    NeuralProphet forecaster with weather covariates.

    All columns in X_train are added as lagged regressors.
    Real timestamps from X_train.index are used for seasonality.
    """

    needs_datetime = True

    def __init__(
        self,
        n_lags: int = 24,
        epochs: Optional[int] = None,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
    ):
        super().__init__("NeuralProphet", use_covariates=True, use_time_features=False)
        self.n_lags = n_lags
        self.epochs = epochs
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self._last_train_df = None
        self._covariate_cols: list[str] = []

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        from neuralprophet import NeuralProphet

        if X_train is None or not isinstance(X_train.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster requires X_train with a DatetimeIndex. "
                "Ensure needs_datetime=True is handled in run_experiments.py."
            )

        self.model = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=1,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            epochs=self.epochs,
        )

        train_df = pd.DataFrame({"ds": X_train.index, "y": y_train})

        self._covariate_cols = list(X_train.columns)
        for col in self._covariate_cols:
            train_df[col] = X_train[col].values
            self.model.add_lagged_regressor(col)

        self._last_train_df = train_df
        self.model.fit(train_df, freq="h")
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster requires X_future with a DatetimeIndex."
            )

        future_df = pd.DataFrame({"ds": X_future.index[:horizon]})
        for col in self._covariate_cols:
            future_df[col] = X_future[col].values[:horizon]

        # NeuralProphet needs training tail as context for AR lags
        context_df = pd.concat(
            [self._last_train_df.tail(self.n_lags), future_df], ignore_index=True
        )

        forecast = self.model.predict(context_df)
        yhat_cols = sorted([c for c in forecast.columns if c.startswith("yhat")])
        return forecast[yhat_cols].values.flatten()[:horizon]

    def reset(self) -> None:
        super().reset()
        self.model = None
        self._last_train_df = None
        self._covariate_cols = []


class NeuralProphetForecaster_NoWeather(BaseForecaster):
    """
    NeuralProphet forecaster, univariate (no covariates).
    X_train columns are ignored, but DatetimeIndex is still used for seasonality.
    """

    needs_datetime = True

    def __init__(
        self,
        n_lags: int = 24,
        epochs: Optional[int] = None,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
    ):
        super().__init__("NeuralProphet_NoWeather", use_covariates=False, use_time_features=False)
        self.n_lags = n_lags
        self.epochs = epochs
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self._last_train_df = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        from neuralprophet import NeuralProphet

        if X_train is None or not isinstance(X_train.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster_NoWeather requires X_train with a DatetimeIndex. "
                "Ensure needs_datetime=True is handled in run_experiments.py."
            )

        self.model = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=1,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            epochs=self.epochs,
        )

        train_df = pd.DataFrame({"ds": X_train.index, "y": y_train})
        self._last_train_df = train_df
        self.model.fit(train_df, freq="h")
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster_NoWeather requires X_future with a DatetimeIndex."
            )

        future_df = pd.DataFrame({"ds": X_future.index[:horizon]})

        context_df = pd.concat(
            [self._last_train_df.tail(self.n_lags), future_df], ignore_index=True
        )

        forecast = self.model.predict(context_df)
        yhat_cols = sorted([c for c in forecast.columns if c.startswith("yhat")])
        return forecast[yhat_cols].values.flatten()[:horizon]

    def reset(self) -> None:
        super().reset()
        self.model = None
        self._last_train_df = None