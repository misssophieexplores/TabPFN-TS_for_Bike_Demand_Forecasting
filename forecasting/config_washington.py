"""
Washington dataset configuration.
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
      
    # CSV columns: timestamp,season,holiday,casual,registered,cnt,temperature_c,humidity_percent,dew_point_c,rainfall_mm,snowfall_cm,precipitation_mm,snow_depth_cm,wind_speed_ms,solar_radiation_wm2,solar_radiation_mjm2,Functioning Day,visibility_km
    # --- Weather ---
    config.weather_covariates = [
        "temperature_c",
        "humidity_percent",
        "wind_speed_ms",
        "dew_point_c",
        "solar_radiation_wm2",
        "precipitation_mm",
        "snow_depth_cm",
        "visibility_km"
    ]
    config.weather_degradation_mapping = {
        "temperature_c": "temperature",
        "humidity_percent": "humidity",
        "wind_speed_ms": "wind_speed",
        "solar_radiation_wm2": "solar_radiation",
        "precipitation_mm": "precipitation",   # rain + snowfall / 0.7 (mm), Open-Meteo/ERA5
        "snow_depth_cm": "snow_depth",         # snow depth on the ground (cm), ERA5-Land
        "visibility_km": "visibility"
    }
    config.timezone = "America/New_York"   # local time of the date column (daylight saving included)
    config.nwp_calibration_file = "weather/nwp/calibration/washington.npz"

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_washington_720_20261003_124552.json"
    config.sarimax_params_file = ""
    config.xgb_params_file = "results/tuning/xgboost_washington_model_params_only.json"
    config.xgb_noweather_params_file = ""
    config.prophet_params_file = "results/tuning/prophet_best_params_washington_720_20261003_125158.json"
    config.neuralprophet_params_file = "results/tuning/neuralprophet_best_params_washington_clean_only_720_20261003_135452.json"
    config.neuralprophet_noweather_params_file = "results/tuning/neuralprophet_best_params_washington_no_weather_720_20261003_145457.json"
    
    return config
