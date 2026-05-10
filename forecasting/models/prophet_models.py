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
        changepoint_prior_scale: float = 0.05, # model defaults 
        seasonality_prior_scale: float = 10.0, # model defaults
        holidays_prior_scale: float = 10.0, # model defaults
    ):
        super().__init__("Prophet", use_covariates=False, use_time_features=False)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
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
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
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
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
    ):
        super().__init__("NeuralProphet", use_covariates=True, use_time_features=False)
        self.n_lags = n_lags
        self.learning_rate = learning_rate
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
            learning_rate=self.learning_rate,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            epochs=self.epochs,
            drop_missing=True,
        )

        train_df = pd.DataFrame({"ds": X_train.index, "y": y_train})

        self._covariate_cols = list(X_train.columns)
        for col in self._covariate_cols:
            train_df[col] = X_train[col].values
            self.model.add_lagged_regressor(col)

        self._last_train_df = train_df

        # Monkey-patch torch.load to fix PyTorch 2.4+ checkpoint loading incompatibility
        import torch
        _original_load = torch.load
        torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})
        try:
            self.model.fit(train_df, freq="h")
        finally:
            torch.load = _original_load

        # Sync to cols NeuralProphet actually kept (it silently drops e.g. all-zero cols)
        if self.model.config_lagged_regressors:
            self._covariate_cols = [c for c in self._covariate_cols
                                    if c in self.model.config_lagged_regressors]
        else:
            self._covariate_cols = []

        # Drop removed cols from training context used in predict()
        keep_cols = ["ds", "y"] + self._covariate_cols
        self._last_train_df = self._last_train_df[keep_cols]

        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster requires X_future with a DatetimeIndex."
            )

        # make_future_dataframe returns the full training df + horizon future rows.
        # Covariates are already correct for training rows; we only need to fill
        # the last `horizon` rows with future covariate values.
        # We then take iloc[-horizon:] from predictions to get only the forecast.
        future_df = self.model.make_future_dataframe(
            self._last_train_df,
            periods=horizon,
            n_historic_predictions=True,
        )

        for col in self._covariate_cols:
            future_df.iloc[-horizon:, future_df.columns.get_loc(col)] = (
                X_future[col].values[:horizon]
            )

        forecast = self.model.predict(future_df)
        yhat_cols = sorted([c for c in forecast.columns if c.startswith("yhat")])
        return forecast[yhat_cols].iloc[-horizon:].values.flatten()[:horizon]

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
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
    ):
        super().__init__("NeuralProphet_NoWeather", use_covariates=False, use_time_features=False)
        self.n_lags = n_lags
        self.learning_rate = learning_rate
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
            learning_rate=self.learning_rate,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            epochs=self.epochs,
            drop_missing=True,
        )

        train_df = pd.DataFrame({"ds": X_train.index, "y": y_train})
        self._last_train_df = train_df

        # Monkey-patch torch.load to fix PyTorch 2.4+ checkpoint loading incompatibility
        import torch
        _original_load = torch.load
        torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})
        try:
            self.model.fit(train_df, freq="h")
        finally:
            torch.load = _original_load

        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster_NoWeather requires X_future with a DatetimeIndex."
            )

        # make_future_dataframe returns training df + horizon future rows.
        # No covariates to fill. Take iloc[-horizon:] from predictions.
        future_df = self.model.make_future_dataframe(
            self._last_train_df,
            periods=horizon,
            n_historic_predictions=True,
        )

        forecast = self.model.predict(future_df)
        yhat_cols = sorted([c for c in forecast.columns if c.startswith("yhat")])
        return forecast[yhat_cols].iloc[-horizon:].values.flatten()[:horizon]

    def reset(self) -> None:
        super().reset()
        self.model = None
        self._last_train_df = None