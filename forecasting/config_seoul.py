"""
Configuration for forecasting experiments.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ForecastConfig:
    """Configuration for time series forecasting experiments"""

    # --- Data ---
    # Path to the data file, relative to the project root's data/ directory
    data_filename: str = "SeoulBikeData.csv"

    # Name of the datetime column used for sorting and parsing
    date_col: str = "Date"

    # Dataset name for tracking results (defaults to data_filename stem if not specified)
    dataset_name: Optional[str] = None

    # Optional column indicating data quality / imputation (set to None if not present)
    functioning_day_col: Optional[str] = "Functioning Day"

    # Optional holiday column name — appended to weather_covariates at load time.
    # Set to the exact column name in your dataset, or None if not present.
    holiday_col: Optional[str] = "Holiday"

    # Optional season column — normalized to 0–3 int and appended to weather_covariates.
    # Handles: "Winter/Spring/Summer/Autumn" strings, 0.0–3.0, and 1–4 integers.
    # Set to None if not present.
    season_col: Optional[str] = "Seasons"
    # Seoul
    season_mapping: Dict = field(default_factory=lambda: {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3})



    # --- Forecasting ---
    # Forecasting horizons in hours
    horizons: List[int] = field(default_factory=lambda: [6, 24, 48, 168])

    # Maximum number of CV folds (actual may be less based on available data)
    n_folds: int = 20

    # Training window size (defined by TabPFN-TS capacity)
    n_train_samples: int = 4096
    
    # Seasonal period in hours (e.g. 24 for daily seasonality in hourly data)
    seasonal_period: int = 24

    # --- Columns ---
    # Target column name
    target_col: str = "Rented Bike Count"

    # Weather covariate columns
    weather_covariates: List[str] = field(default_factory=lambda: [
        "Temperature",
        "Humidity",
        "Wind speed",
        # "Visibility",
        "Dew point temperature",
        "Solar Radiation",
        "Rainfall",
        "Snowfall"
    ])

    # Map weather columns to degradation variable types
    weather_degradation_mapping: Dict[str, str] = field(default_factory=lambda: {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Wind speed": "wind_speed",
        # "Visibility": "visibility",
        "Solar Radiation": "solar_radiation",
        "Rainfall": "precipitation",
        "Snowfall": "precipitation"
    })

    # --- Weather degradation ---
    degradation_seed: int = 42
    weather_scenarios: List[str] = field(default_factory=lambda: [
        "all_weather",
        "clean_only",
        "degraded"
    ])

    # --- Model parameters ---

    # Path to tuned ARIMA parameters JSON, relative to project root
    arima_params_file: Optional[str] = None #TODO

    # Path to tuned SARIMAX parameters JSON, relative to project root
    # TODO: update once SARIMAX tuning completes
    sarimax_params_file: Optional[str] = None #TODO

    # Path to XGBoost tuned parameters JSON, relative to project root
    xgb_params_file: str = "results/tuning/xgboost_best_params_20260110_042536.json"

    # --- Output ---
    output_dir: str = "results"

    # Version tag appended to result file names (e.g. "v3" -> results_master_v3.csv)
    results_version: str = "v4"

    # --- W&B ---
    wandb_project: str = "bike-forecasting"
    experiment_name: Optional[str] = None


    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.wandb_project}_{self.results_version}"