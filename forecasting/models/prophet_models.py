"""
Prophet and NeuralProphet forecasting models.

Both set needs_datetime = True so run_experiments.py attaches a real
DatetimeIndex to X_train/X_test from the dataset's date column.
This ensures Prophet's seasonality decomposition uses correct calendar positions.

  ProphetForecaster                 — univariate, no covariates
  NeuralProphetForecaster           — with weather covariates (future regressors)
  NeuralProphetForecaster_NoWeather — univariate variant

Yearly seasonality is off by default: the training window (n_train_samples=720,
30 days) is far shorter than one year. Daily and weekly seasonality stay on.
"""

import contextlib
import io
import logging
import os
import warnings

import numpy as np
import pandas as pd
from typing import Optional
from models.base import BaseForecaster


# Fixed seed applied immediately before every NeuralProphet model is created.
_NP_SEED = 42


# ---------------------------------------------------------------------------
# Silencing helpers
# ---------------------------------------------------------------------------
def _silence_neuralprophet():
    """
    Aggressively silence NeuralProphet + PyTorch Lightning logging.
    Call before instantiating NeuralProphet. Idempotent.
    """
    for name in (
        "NP", "neuralprophet",
        "NP.forecaster", "NP.config", "NP.config_model",
        "NP.utils", "NP.utils_torch",
        "NP.plotting", "NP.time_dataset", "NP.df_utils",
        "NP.data.processing", "NP.data.splitting",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False

    for name in (
        "pytorch_lightning", "lightning", "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.pytorch.accelerators.cuda",
        "pytorch_lightning.utilities.rank_zero",
        "pytorch_lightning.accelerators.cuda",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False

    # Disable tqdm progress bars globally (covers Lightning's TQDMProgressBar output)
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("LIGHTNING_LOGGER_LEVEL", "ERROR")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _silence_prophet():
    """Silence cmdstanpy chatter emitted by Prophet during fit."""
    for name in ("cmdstanpy", "prophet", "prophet.plot"):
        logging.getLogger(name).setLevel(logging.WARNING)


@contextlib.contextmanager
def _silence_all_output():
    """Capture stdout/stderr to suppress residual prints from NP/Lightning."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


def _np_fit(model, train_df):
    """
    Call NeuralProphet.fit() telling it NOT to install a progress-bar callback.
    Falls back across NP versions whose `progress` arg accepts different values.
    """
    with warnings.catch_warnings(), _silence_all_output():
        warnings.filterwarnings("ignore")
        for progress_arg in ("none", None, "off", "bar"):
            try:
                if progress_arg == "bar":
                    # Last-resort: no progress kwarg at all
                    return model.fit(train_df, freq="h")
                return model.fit(train_df, freq="h", progress=progress_arg)
            except TypeError:
                continue
        # If everything failed, re-raise by calling without the kwarg
        return model.fit(train_df, freq="h")


def _np_predict(model, future_df):
    """Run NeuralProphet inference silently."""
    with warnings.catch_warnings(), _silence_all_output():
        warnings.filterwarnings("ignore")
        return model.predict(future_df)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
class ProphetForecaster(BaseForecaster):
    """
    Univariate Prophet forecaster (no covariates).

    Real timestamps are required for correct daily/weekly/yearly seasonality.
    Provided via X_train.index (DatetimeIndex) set by run_experiments.py.
    """

    needs_datetime = True

    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
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
        _silence_prophet()
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
        with _silence_all_output():
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

        with _silence_all_output():
            forecast = self.model.predict(pd.DataFrame({"ds": future_dates}))
        return forecast["yhat"].values

    def reset(self) -> None:
        super().reset()
        self.model = None
        self._last_timestamp = None


# ---------------------------------------------------------------------------
# NeuralProphet with covariates
# ---------------------------------------------------------------------------
class NeuralProphetForecaster(BaseForecaster):
    """
    NeuralProphet forecaster with weather covariates.

    Covariates are future regressors (add_future_regressor): the forecast for
    each step uses the covariate values of that step, taken from X_future.

    The model is built with n_forecasts=horizon, and the horizon is only known
    in predict(). fit() therefore stores the training data; predict(horizon)
    creates and fits the NeuralProphet model.
    """

    needs_datetime = True

    def __init__(
        self,
        n_lags: int = 24,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = False,
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
        if X_train is None or not isinstance(X_train.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster requires X_train with a DatetimeIndex. "
                "Ensure needs_datetime=True is handled in run_experiments.py."
            )

        # Store training data only; the model is created and fitted in
        # predict(), where the horizon (= n_forecasts) is known.
        train_df = pd.DataFrame({"ds": X_train.index, "y": y_train})

        self._covariate_cols = list(X_train.columns)
        for col in self._covariate_cols:
            train_df[col] = X_train[col].values

        self._last_train_df = train_df
        self.model = None
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster requires X_future with a DatetimeIndex."
            )

        _silence_neuralprophet()
        from neuralprophet import NeuralProphet, set_random_seed

        h = int(horizon)
        train_df = self._last_train_df.copy()
        covariate_cols = list(self._covariate_cols)

        # NOTE: do NOT pass trainer_config={"enable_progress_bar": False} —
        # NeuralProphet installs its own ProgressBar callback, which conflicts
        # with that flag and raises MisconfigurationException. Disable the
        # progress bar via `progress="none"` in .fit() instead (see _np_fit).
        set_random_seed(_NP_SEED)
        self.model = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=h,
            learning_rate=self.learning_rate,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            epochs=self.epochs,
            drop_missing=True,
        )

        for col in covariate_cols:
            self.model.add_future_regressor(col)

        # PyTorch 2.4+ compat shim for NP's checkpoint loader
        import torch
        _original_load = torch.load
        torch.load = lambda *args, **kwargs: _original_load(
            *args, **{**kwargs, "weights_only": False}
        )
        try:
            _np_fit(self.model, train_df)
        finally:
            torch.load = _original_load

        # Sync to cols NP actually kept (it drops regressors that are constant
        # in the training window)
        kept = self.model.config_regressors.regressors
        covariate_cols = [c for c in covariate_cols if kept and c in kept]
        self._covariate_cols = covariate_cols

        keep_cols = ["ds", "y"] + covariate_cols
        train_df = train_df[keep_cols]

        # Future covariate values for the h forecast steps
        regressors_df = None
        if covariate_cols:
            regressors_df = pd.DataFrame(
                {col: X_future[col].values[:h] for col in covariate_cols}
            )
            if len(regressors_df) != h:
                raise RuntimeError(
                    f"X_future has {len(regressors_df)} rows, expected {h}."
                )

        with _silence_all_output():
            future_df = self.model.make_future_dataframe(
                train_df,
                regressors_df=regressors_df,
                periods=h,
                n_historic_predictions=True,
            )

        # The frame must contain exactly h future rows (after the training window)
        last_train_ds = pd.Timestamp(train_df["ds"].iloc[-1])
        n_future = int((future_df["ds"] > last_train_ds).sum())
        if n_future != h:
            raise RuntimeError(
                f"NeuralProphet future frame has {n_future} future rows, expected {h}."
            )

        forecast = _np_predict(self.model, future_df)
        # Step i (1-based) is in column yhat{i} of row -(h - i + 1)
        y_pred = [forecast[f"yhat{i+1}"].iloc[-h + i] for i in range(h)]
        return np.asarray(y_pred, dtype=float)

    def reset(self) -> None:
        super().reset()
        self.model = None
        self._last_train_df = None
        self._covariate_cols = []


# ---------------------------------------------------------------------------
# NeuralProphet univariate
# ---------------------------------------------------------------------------
class NeuralProphetForecaster_NoWeather(BaseForecaster):
    """
    NeuralProphet forecaster, univariate (no covariates).

    The model is built with n_forecasts=horizon, and the horizon is only known
    in predict(). fit() therefore stores the training data; predict(horizon)
    creates and fits the NeuralProphet model.
    """

    needs_datetime = True

    def __init__(
        self,
        n_lags: int = 24,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = False,
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
        if X_train is None or not isinstance(X_train.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster_NoWeather requires X_train with a DatetimeIndex. "
                "Ensure needs_datetime=True is handled in run_experiments.py."
            )

        # Store training data only; the model is created and fitted in
        # predict(), where the horizon (= n_forecasts) is known.
        self._last_train_df = pd.DataFrame({"ds": X_train.index, "y": y_train})
        self.model = None
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if X_future is None or not isinstance(X_future.index, pd.DatetimeIndex):
            raise ValueError(
                "NeuralProphetForecaster_NoWeather requires X_future with a DatetimeIndex."
            )

        _silence_neuralprophet()
        from neuralprophet import NeuralProphet, set_random_seed

        h = int(horizon)
        train_df = self._last_train_df.copy()

        set_random_seed(_NP_SEED)
        self.model = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=h,
            learning_rate=self.learning_rate,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            epochs=self.epochs,
            drop_missing=True,
        )

        import torch
        _original_load = torch.load
        torch.load = lambda *args, **kwargs: _original_load(
            *args, **{**kwargs, "weights_only": False}
        )
        try:
            _np_fit(self.model, train_df)
        finally:
            torch.load = _original_load

        with _silence_all_output():
            future_df = self.model.make_future_dataframe(
                train_df,
                periods=h,
                n_historic_predictions=True,
            )

        # The frame must contain exactly h future rows (after the training window)
        last_train_ds = pd.Timestamp(train_df["ds"].iloc[-1])
        n_future = int((future_df["ds"] > last_train_ds).sum())
        if n_future != h:
            raise RuntimeError(
                f"NeuralProphet future frame has {n_future} future rows, expected {h}."
            )

        forecast = _np_predict(self.model, future_df)
        # Step i (1-based) is in column yhat{i} of row -(h - i + 1)
        y_pred = [forecast[f"yhat{i+1}"].iloc[-h + i] for i in range(h)]
        return np.asarray(y_pred, dtype=float)

    def reset(self) -> None:
        super().reset()
        self.model = None
        self._last_train_df = None
