"""
Machine learning forecasting models.
"""
import numpy as np
import pandas as pd
from typing import Optional
import xgboost as xgb
from  models.base import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost model for time series forecasting.
    
    Uses lagged features and iterative multi-step prediction.
    """
    
    def __init__(self, n_lags: int = 24, **xgb_params):
        """
        Initialize XGBoost forecaster.
        
        Parameters:
        -----------
        n_lags : int
            Number of lagged values to use as features
        **xgb_params : dict
            XGBoost hyperparameters
        """
        super().__init__("XGBoost", use_covariates=True, use_time_features=True)
        self.n_lags = n_lags
        
        # Default XGBoost parameters
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        default_params.update(xgb_params)
        self.xgb_params = default_params
        
        self.model = None
        self.y_train = None
        self.feature_names = None
        
    def _create_lagged_features(
        self, 
        y: np.ndarray, 
        X: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create lagged features from target variable for the full training set.
        
        Parameters:
        -----------
        y : np.ndarray
            Target values
        X : pd.DataFrame, optional
            Additional features (covariates)
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix with lagged values (first n_lags rows removed)
        """
        features = pd.DataFrame()
        
        # Add lagged target values
        for lag in range(1, self.n_lags + 1):
            features[f'lag_{lag}'] = np.roll(y, lag)
        
        # Add covariates if provided
        if X is not None:
            X_reset = X.reset_index(drop=True)
            features = pd.concat([features, X_reset], axis=1)
        
        # Remove initial rows with NaN from lagging
        features = features.iloc[self.n_lags:]
        
        return features

    def _create_predict_row(
        self,
        y_history: np.ndarray,
        X_h: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create a single feature row for one prediction step.

        Parameters:
        -----------
        y_history : np.ndarray
            All observed and predicted values so far
        X_h : pd.DataFrame, optional
            Covariates for this single step (one row)

        Returns:
        --------
        pd.DataFrame
            Single-row feature DataFrame matching training feature names
        """
        # lag_1 = most recent, lag_n = oldest
        lag_vals = y_history[-self.n_lags:][::-1]
        row = {f'lag_{i + 1}': float(lag_vals[i]) for i in range(self.n_lags)}

        if X_h is not None:
            for col in X_h.columns:
                row[col] = float(X_h.iloc[0][col])

        return pd.DataFrame([row])

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        """
        Fit XGBoost model.
        
        Parameters:
        -----------
        y_train : np.ndarray
            Training target values
        X_train : pd.DataFrame, optional
            Training covariates
        """
        self.y_train = y_train.copy()
        
        # Create lagged features
        X_features = self._create_lagged_features(y_train, X_train)
        y_target = y_train[self.n_lags:]
        
        # Store feature names for consistency
        self.feature_names = X_features.columns.tolist()
        
        # Fit model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_features, y_target)
        self._is_fitted = True
        
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecast using iterative multi-step prediction.
        
        Parameters:
        -----------
        horizon : int
            Number of steps to forecast
        X_future : pd.DataFrame, optional
            Future covariates
            
        Returns:
        --------
        np.ndarray
            Forecast values
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        
        forecasts = []
        y_history = self.y_train.copy()
        
        for h in range(horizon):
            # Get covariates for this step if available
            X_h = X_future.iloc[h:h+1].copy() if X_future is not None and h < len(X_future) else None

            # Build single prediction row directly from history
            X_row = self._create_predict_row(y_history, X_h)

            # Ensure correct feature order
            X_row = X_row[self.feature_names]

            # Predict next step
            y_pred = float(self.model.predict(X_row.values)[0])
            forecasts.append(y_pred)
            
            # Update history with prediction
            y_history = np.append(y_history, y_pred)
        
        return np.array(forecasts)
    
    def reset(self) -> None:
        """Reset model state"""
        super().reset()
        self.model = None
        self.y_train = None
        self.feature_names = None