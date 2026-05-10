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
    config.data_filename = "WashingtonBikeData.csv"
    config.dataset_name = "washington"
    config.date_col = "timestamp"
    config.target_col = "cnt"
    config.functioning_day_col = "Functioning Day"
    config.holiday_col = "holiday"
    config.season_col = "season"
    config.season_mapping = {
        1.0: 3, # winter 
        2.0: 0, # spring
        3.0: 1, # summer
        4.0: 2  # autumn
        }
      
#timestamp,season,holiday,casual,registered,cnt,temperature_c,humidity_percent,dew_point_c,rainfall_mm,snowfall_cm,wind_speed_ms,solar_radiation_wm2,solar_radiation_mjm2,Functioning Day
    # --- Weather ---
    config.weather_covariates = [
        "temperature_c",
        "humidity_percent",
        "wind_speed_ms",
        "dew_point_c",
        "solar_radiation_wm2",
        "rainfall_mm",
        "snowfall_cm",
        "visibility_km"
    ]
    config.weather_degradation_mapping = {
        "temperature_c": "temperature",
        "humidity_percent": "humidity",
        "wind_speed_ms": "wind_speed",
        "solar_radiation_wm2": "solar_radiation",
        "rainfall_mm": "precipitation",
        "snowfall_cm": "precipitation",
        "visibility_km": "visibility"
    }

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_20260314_190758.json"
    config.sarimax_params_file = "results/tuning/sarimax_best_params_20260315_120718.json"
    config.xgb_params_file = "results/tuning/xgboost_best_params_20260315_162809.json"
    config.prophet_params_file = "" #TODO
    config.neuralprophet_params_file = "" #TODO
    
    return config
