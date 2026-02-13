"""
Configuration for forecasting experiments.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ForecastConfig:
    """Configuration for time series forecasting experiments"""
    
    # Dataset name for tracking results (defaults to filename if not specified)
    dataset_name: Optional[str] = None
    
    # Forecasting horizons in hours
    horizons: List[int] = field(default_factory=lambda: [6, 24, 48, 168])
    
    # Maximum number of CV folds (actual may be less based on available data)
    n_folds: int = 20
    
    
    # Minimum training size (for TabPFN-TS requirement)
    max_train_samples: int = 4096
    
    # Target column name
    target_col: str = "Rented Bike Count"
    
    # Weather covariate columns
    weather_covariates: List[str] = field(default_factory=lambda: [
        "Temperature",
        "Humidity", 
        "Wind speed",
        "Visibility",
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
        "Visibility": "visibility",
        "Solar Radiation": "solar_radiation",
        "Rainfall": "precipitation",
        "Snowfall": "precipitation"
    })
    
    # Weather degradation configuration
    degradation_seed: int = 42
    weather_scenarios: List[str] = field(default_factory=lambda: [
        "all_weather",
        "clean_only",
        "degraded"
    ])