"""
Forecasting evaluation metrics.
"""
import numpy as np
from typing import Dict


class MetricsCalculator:
    """Calculate forecasting performance metrics"""
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        float
            MAE value
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        float
            RMSE value
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def mase(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_train: np.ndarray, 
        seasonal_period: int = 24
    ) -> float:
        """
        Mean Absolute Scaled Error.
        
        Scales forecast error by naive seasonal forecast error on training set.
        Values < 1 indicate better than naive seasonal forecast.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        y_train : np.ndarray
            Training set values (for scaling)
        seasonal_period : int
            Seasonal period (24 for hourly data with daily seasonality)
            
        Returns:
        --------
        float
            MASE value
        """
        # Calculate naive seasonal forecast error on training set
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
        mae_naive = np.mean(naive_errors)
        
        # Calculate forecast error
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        
        # Return scaled error
        if mae_naive == 0:
            return np.inf
        return float(mae_forecast / mae_naive)
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        
        Handles zero values better than MAPE.
        Returns percentage in range [0, 100].
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        float
            sMAPE value (percentage)
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        smape_values = np.where(
            mask,
            np.abs(y_true - y_pred) / denominator,
            0
        )
        
        return float(100 * np.mean(smape_values))
    
    @classmethod
    def calculate_all(
        cls, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        y_train : np.ndarray
            Training set values (for MASE calculation)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with all metric values
        """
        return {
            'MAE': cls.mae(y_true, y_pred),
            'RMSE': cls.rmse(y_true, y_pred),
            'MASE': cls.mase(y_true, y_pred, y_train),
            'sMAPE': cls.smape(y_true, y_pred)
        }