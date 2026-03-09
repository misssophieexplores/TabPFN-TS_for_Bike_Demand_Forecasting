"""
Statistical forecasting models - FIXED VERSION.
"""
import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.base import BaseForecaster


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
    
    def __init__(self, order: tuple = (2, 1, 2), freq: str = 'h'):
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
        """
        super().__init__("ARIMA", use_covariates=False)
        self.order = order
        self.freq = freq
        self.model = None
        self.model_fit = None
        
    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """Fit ARIMA model"""
        self.model = ARIMA(y_train, order=self.order)
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
    
    def __init__(
        self, 
        order: tuple = (4, 0, 0), 
        seasonal_order: tuple = (1, 0, 1, 24),
        freq: str = 'h'
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
        """
        super().__init__("SARIMAX", use_covariates=True)
        self.order = order
        self.seasonal_order = seasonal_order
        self.freq = freq
        self.model = None
        self.model_fit = None
        
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
        y_series = pd.Series(y_train, index=datetime_index, name='y')
        
        if X_train is not None:
            X_train = X_train.copy()
            X_train.index = datetime_index
        
        self.model = SARIMAX(
            y_series,
            exog=X_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = self.model.fit(disp=False, maxiter=200, method='lbfgs')
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
            future_index = pd.date_range(start='2020-01-01', periods=horizon, freq=self.freq)
            X_future = X_future.copy()
            X_future.index = future_index
            
        forecast = self.model_fit.forecast(steps=horizon, exog=X_future)
        return np.array(forecast)
    
    def reset(self) -> None:
        """Reset model state"""
        super().reset()
        self.model = None
        self.model_fit = None