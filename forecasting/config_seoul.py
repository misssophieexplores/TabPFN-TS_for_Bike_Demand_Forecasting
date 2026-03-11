"""
Seoul dataset configuration.
Only dataset-specific fields are set here.
All shared settings (wandb_project, results_version, horizons, etc.)
are inherited from config.py and only need changing there.
"""
from config import ForecastConfig


def get_config() -> ForecastConfig:
    config = ForecastConfig()

    # --- Data ---
    config.data_filename = "SeoulBikeData.csv"
    config.dataset_name = "seoul"
    config.date_col = "Date"
    config.target_col = "Rented Bike Count"
    config.functioning_day_col = "Functioning Day"
    config.holiday_col = "Holiday"
    config.season_col = "Seasons"
    config.season_mapping = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}

    # --- Weather ---
    config.weather_covariates = [
        "Temperature",
        "Humidity",
        "Wind speed",
        "Dew point temperature",
        "Solar Radiation",
        "Rainfall",
        "Snowfall",
    ]
    config.weather_degradation_mapping = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Wind speed": "wind_speed",
        "Solar Radiation": "solar_radiation",
        "Rainfall": "precipitation",
        "Snowfall": "precipitation",
    }

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_20260311_190232.json"
    config.sarimax_params_file = "results/tuning/sarimax_best_params_20260311_193129.json"
    config.xgb_params_file = "results/tuning/xgboost_best_params_20260309_172323.json"

    return config
