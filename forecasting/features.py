"""
Time feature engineering utilities.

Computes calendar features from a datetime column.
Used by models that need explicit temporal signals (e.g. XGBoost).
TabPFN and SARIMAX do NOT use this — they handle temporal structure internally.
"""

import pandas as pd
from typing import Optional 

TIME_FEATURES = ["hour", "dayofweek", "month", "is_weekend"]


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extract calendar features from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the datetime column. Not modified in place.
    date_col : str
        Name of the datetime column (must be parseable as datetime).

    Returns
    -------
    pd.DataFrame
        New DataFrame with columns: hour, dayofweek, month, is_weekend.
        Index is reset to match positional alignment with y_train / y_test.

    Notes
    -----
    - hour       : 0–23
    - dayofweek  : 0 (Monday) – 6 (Sunday)
    - month      : 1–12
    - is_weekend : 1 if Saturday or Sunday, else 0
    - Holiday is intentionally excluded here; add it via weather_covariates
      in config if needed, since it is already a column in the dataset.
    """
    dt = pd.to_datetime(df[date_col])

    features = pd.DataFrame(index=df.index)
    features["hour"] = dt.dt.hour
    features["dayofweek"] = dt.dt.dayofweek
    features["month"] = dt.dt.month
    features["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    return features.reset_index(drop=True)


def prepare_xgboost_features(
    df: pd.DataFrame,
    date_col: str,
    X_covariates: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build XGBoost feature matrix: covariates + time features.
    Single source of truth used by both tuning and experiment runner.
    """
    time_feats = add_time_features(df, date_col)
    if X_covariates is not None:
        return pd.concat([X_covariates.reset_index(drop=True), time_feats], axis=1)
    return time_feats