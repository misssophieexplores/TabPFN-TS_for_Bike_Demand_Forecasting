"""
London dataset configuration.
Only dataset-specific fields are set here.
All shared settings (wandb_project, results_version, horizons, etc.)
are inherited from config.py and only need changing there.
"""
from config import ForecastConfig


def get_config() -> ForecastConfig:
    config = ForecastConfig()

    # --- Data ---
    config.data_filename = "LondonBikeData.csv"
    config.dataset_name = "london"
    config.date_col = "timestamp"
    config.target_col = "cnt"
    config.functioning_day_col = None
    config.holiday_col = "is_holiday"
    config.season_col = "season"
    config.season_mapping = {0: 0, 1: 1, 2: 2, 3: 3}  # already 0-based

    # --- Weather ---
    config.weather_covariates = [
        "t1",
        "hum",
        "wind_speed",
        "dew_point_c",
        "solar_radiation_wm2",
        "rainfall_mm",
        "snowfall_cm",
        "visibility_km"
    ]
    config.weather_degradation_mapping = {
        "t1": "temperature",
        "hum": "humidity",
        "wind_speed": "wind_speed",
        "solar_radiation_wm2": "solar_radiation",
        "rainfall_mm": "precipitation",
        "snowfall_cm": "precipitation",
        "visibility_km": "visibility"
    }

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_20260314_190725.json"
    config.sarimax_params_file = "results/tuning/sarimax_best_params_20260315_135209.json" 
    config.xgb_params_file = "results/tuning/xgboost_best_params_20260309_081759.json" 

    return config
