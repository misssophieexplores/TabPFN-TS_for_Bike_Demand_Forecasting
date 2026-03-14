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
    config.holiday_mapping = {'Holiday': 1, 'No Holiday': 0}
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
        "Visibility",
        "Seasons",
        "Holiday",
    ]
    config.weather_degradation_mapping = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Wind speed": "wind_speed",
        "Solar Radiation": "solar_radiation",
        "Rainfall": "precipitation",
        "Snowfall": "precipitation",
        "Visibility": "visibility"
    }
    config.column_scale_factors = {"visibility": 0.001}

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_20260314_190645.json"
    config.sarimax_params_file = "" 
    config.xgb_params_file = "" #TODO

    return config
