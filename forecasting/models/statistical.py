"""
Statistical forecasting models - FIXED VERSION.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.base import BaseForecaster


def trend_from_intercept(
    with_intercept: bool,
    order: tuple,
    seasonal_order: tuple = (0, 0, 0, 0),
    sarimax: bool = False,
) -> str:
    """
    Translate pmdarima's `with_intercept` into the equivalent statsmodels `trend`.

    pmdarima fits every model with statsmodels SARIMAX and passes trend="c"
    when with_intercept=True, whatever d and D are. The two statsmodels classes
    used in the experiments place the trend differently:
    - SARIMAX (SARIMAXForecaster, sarimax=True): the trend enters the
      (seasonally) differenced equation, so pmdarima's model is exactly
      trend="c" (a constant for d + D = 0, a drift for d + D = 1). trend="t"
      there would be a linear trend in the differenced series, i.e. a
      quadratic trend in levels.
    - ARIMA (ARIMAForecaster, sarimax=False): trend terms are regressors in the
      levels equation, so the same model is "c" for d + D = 0 and "t" (linear
      trend in levels = constant drift after differencing) for d + D = 1.
    Without an intercept, "n" is passed explicitly, because statsmodels ARIMA
    otherwise adds a constant by default when d == 0.
    """
    if not with_intercept:
        return "n"
    if sarimax:
        return "c"
    n_diff = order[1] + seasonal_order[1]
    if n_diff == 0:
        return "c"
    if n_diff == 1:
        return "t"
    raise ValueError(
        f"with_intercept=True with d + D = {n_diff} has no supported statsmodels equivalent "
        f"(order={order}, seasonal_order={seasonal_order})"
    )


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive baseline.
    
    Forecasts by repeating the pattern from seasonal_period steps ago.
    For hourly data with daily seasonality, uses lag 24.
    """
    
    def __init__(self, seasonal_period: int = 24):
        """
        Initialize seasonal naive forecaster.
        
        Parameters:
        -----------
        seasonal_period : int
            Seasonal period (24 for daily pattern in hourly data)
        """
        super().__init__("Seasonal_Naive", use_covariates=False)
        self.seasonal_period = seasonal_period
        self.y_train = None
        
    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """Store training data for naive forecast"""
        self.y_train = y_train.copy()
        self._is_fitted = True
        
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Repeat last seasonal_period values to cover horizon"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
            
        seasonal_pattern = self.y_train[-self.seasonal_period:]
        n_repeats = int(np.ceil(horizon / self.seasonal_period))
        forecast = np.tile(seasonal_pattern, n_repeats)[:horizon]
        
        return forecast


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA model wrapper.
    
    Autoregressive Integrated Moving Average model for univariate forecasting.
    Does not use covariates.
    """
    
    def __init__(self, order: tuple = (2, 1, 2), freq: str = 'h', trend: Optional[str] = None):
        """
        Initialize ARIMA forecaster.
        
        Parameters:
        -----------
        order : tuple
            ARIMA order (p, d, q) where:
            - p: number of AR terms
            - d: degree of differencing
            - q: number of MA terms
        freq : str
            Pandas frequency string (e.g., 'h' for hourly, 'D' for daily)
        trend : str, optional
            statsmodels trend ("n", "c", "t"). None keeps the statsmodels default.
            Use trend_from_intercept() to match a pmdarima-tuned model.
        """
        super().__init__("ARIMA", use_covariates=False)
        self.order = order
        self.freq = freq
        self.trend = trend
        self.model = None
        self.model_fit = None
        
    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """Fit ARIMA model"""
        self.model = ARIMA(y_train, order=self.order, trend=self.trend)
        self.model_fit = self.model.fit()
        self._is_fitted = True
        
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate ARIMA forecast"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
            
        forecast = self.model_fit.forecast(steps=horizon)
        return np.array(forecast)
    
    def reset(self) -> None:
        """Reset model state"""
        super().reset()
        self.model = None
        self.model_fit = None


class SARIMAXForecaster(BaseForecaster):
    """
    SARIMAX model wrapper.
    
    Seasonal ARIMA with eXogenous regressors.
    Supports weather covariates.
    
    FIXED: Proper datetime index handling to eliminate statsmodels warnings.
    """

    # One fit per fold, single-threaded statsmodels: the folds are fitted in
    # parallel worker processes (fold_runner.py)
    parallel_folds = True

    def __init__(
        self, 
        order: tuple = (4, 0, 0), 
        seasonal_order: tuple = (1, 0, 1, 24),
        freq: str = 'h',
        trend: Optional[str] = None
    ):
        """
        Initialize SARIMAX forecaster.
        
        Parameters:
        -----------
        order : tuple
            ARIMA order (p, d, q)
        seasonal_order : tuple
            Seasonal order (P, D, Q, s) where:
            - P: seasonal AR order
            - D: seasonal differencing
            - Q: seasonal MA order
            - s: seasonal period (24 for hourly data)
        freq : str
            Pandas frequency string (e.g., 'h' for hourly, 'D' for daily)
        trend : str, optional
            statsmodels trend ("n", "c", "t"). None means no trend term.
            Use trend_from_intercept() to match a pmdarima-tuned model.
        """
        super().__init__("SARIMAX", use_covariates=True)
        self.order = order
        self.seasonal_order = seasonal_order
        self.freq = freq
        self.trend = trend
        self.model = None
        self.model_fit = None
        self.exog_cols = None
        self._exog_sd = None
        self._train_index = None

    def _create_datetime_index(self, n_obs: int) -> pd.DatetimeIndex:
        """
        Create a proper DatetimeIndex for the data.
        
        This satisfies statsmodels' requirement for datetime-indexed data.
        Uses an arbitrary start date with specified frequency.
        """
        return pd.date_range(start='2020-01-01', periods=n_obs, freq=self.freq)
        
    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Fit SARIMAX model with covariates.
        
        Creates proper datetime index to eliminate statsmodels warnings.
        """
        datetime_index = self._create_datetime_index(len(y_train))
        self._train_index = datetime_index
        y_series = pd.Series(y_train, index=datetime_index, name='y')

        if X_train is not None:
            # Drop covariates that are constant in this training window (e.g. season
            # within 30 days, holiday with no holiday): their effect cannot be estimated
            # and a constant column duplicates the intercept. Same rule as tune_sarimax.py.
            self.exog_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) > 1]
            if self.exog_cols:
                # Each covariate divided by its training-window SD: same model
                # (rescaled coefficients), better-conditioned optimisation.
                X_train = X_train[self.exog_cols].astype(float)
                self._exog_sd = X_train.std(ddof=0)
                X_train = X_train / self._exog_sd
                X_train.index = datetime_index
            else:
                X_train = None
        
        self.model = SARIMAX(
            y_series,
            exog=X_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = self.model.fit(disp=False, maxiter=1000, method='lbfgs')

        if not self.model_fit.mle_retvals['converged']:
            warnings.warn(f"SARIMAX did not converge — results may be unreliable")

        self._is_fitted = True


    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate SARIMAX forecast with future covariates.
        
        Parameters:
        -----------
        horizon : int
            Number of periods to forecast
        X_future : pd.DataFrame, optional
            Future weather covariates (required if use_covariates=True)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        if self.use_covariates and X_future is None:
            raise ValueError("X_future required for SARIMAX prediction")
        
        if X_future is not None:
            if self.exog_cols:
                # Forecast index continues directly after the training index
                future_index = pd.date_range(
                    start=self._train_index[-1] + self._train_index.freq,
                    periods=horizon, freq=self.freq
                )
                X_future = X_future[self.exog_cols].astype(float) / self._exog_sd
                X_future.index = future_index
            else:
                X_future = None
            
        forecast = self.model_fit.forecast(steps=horizon, exog=X_future)
        return np.array(forecast)
    
    def reset(self) -> None:
        """Reset model state"""
        super().reset()
        self.model = None
        self.model_fit = None
        self.exog_cols = None
        self._exog_sd = None
        self._train_index = None
