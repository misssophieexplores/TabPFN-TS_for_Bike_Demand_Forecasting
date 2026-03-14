"""
Shared base configuration for forecasting experiments.

Dataset-specific fields (data_filename, dataset_name, column names,
season_mapping, weather_covariates, model parameter files) are set to None
here and must be overridden in each city config via get_config().

Fields that are truly shared across all datasets (wandb_project,
results_version, horizons, n_folds, etc.) are defined here once.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ForecastConfig:
    """Configuration for time series forecasting experiments"""

    # --- Data (dataset-specific — override in city config) ---
    data_filename: Optional[str] = None
    date_col: Optional[str] = None
    target_col: Optional[str] = None
    dataset_name: Optional[str] = None
    functioning_day_col: Optional[str] = None
    holiday_col: Optional[str] = None
    holiday_mapping: Optional[Dict] = field(default_factory=lambda: {'Yes': 1, 'No': 0})
    season_col: Optional[str] = None
    season_mapping: Optional[Dict] = None
    weather_covariates: Optional[List[str]] = None
    weather_degradation_mapping: Optional[Dict[str, str]] = None
    column_scale_factors: Dict[str, float] = field(default_factory=dict)

    # --- Model parameters (dataset-specific — override in city config) ---
    arima_params_file: Optional[str] = None
    sarimax_params_file: Optional[str] = None
    xgb_params_file: Optional[str] = None

    # # --- Forecasting (shared) ---
    # horizons: List[int] = field(default_factory=lambda: [6, 24, 48, 168])
    # n_folds: int = 20
    # n_train_samples: int = 4096
    # seasonal_period: int = 24
    # --- Forecasting (shared) ---
    horizons: List[int] = field(default_factory=lambda: [6])
    n_folds: int = 2
    n_train_samples: int = 500
    seasonal_period: int = 24


    # --- Weather degradation (shared) ---
    degradation_seed: int = 42
    weather_scenarios: List[str] = field(default_factory=lambda: [
        "all_weather",
        "clean_only",
        "degraded"
    ])

    # --- Output (shared) ---
    output_dir: str = "results"
    results_version: str = "t5-02" 
    verbose: bool = False

    # # --- W&B (shared) ---
    # wandb_project: str = "bike-forecasting"
    # experiment_name: Optional[str] = None

    # --- W&B (shared) ---
    wandb_project: str = "bike-forecasting-testing" #TODO
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.dataset_name}_{self.results_version}"
