"""
Base forecaster class that all models inherit from.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    Ensures consistent interface across different model types.
    """
    
    def __init__(self, name: str, use_covariates: bool = False):
        """
        Initialize the forecaster.
        
        Parameters:
        -----------
        name : str
            Name identifier for the model
        use_covariates : bool
            Whether this model uses exogenous covariates
        """
        self.name = name
        self.use_covariates = use_covariates
        self._is_fitted = False
        
    @abstractmethod
    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the forecasting model.
        
        Parameters:
        -----------
        y_train : np.ndarray
            Training target values (historical observations)
        X_train : pd.DataFrame, optional
            Training covariates/features (if model uses them)
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecasts for future time steps.
        
        Parameters:
        -----------
        horizon : int
            Number of steps ahead to forecast
        X_future : pd.DataFrame, optional
            Future covariates (if model uses them)
            
        Returns:
        --------
        np.ndarray
            Forecast values of shape (horizon,)
        """
        pass
    
    def reset(self) -> None:
        """
        Reset model state between CV folds.
        Override if model maintains internal state.
        """
        self._is_fitted = False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', use_covariates={self.use_covariates})"