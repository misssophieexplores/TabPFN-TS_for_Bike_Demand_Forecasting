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
      
    # CSV columns: timestamp,season,holiday,casual,registered,cnt,temperature_c,humidity_percent,dew_point_c,rainfall_mm,snowfall_cm,wind_speed_ms,solar_radiation_wm2,solar_radiation_mjm2,Functioning Day,visibility_km
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
    config.rain_col = "rainfall_mm"
    config.snow_col = "snowfall_cm"

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_washington_clean_only_720_20260516_003716.json"
    config.sarimax_params_file = "results/tuning/sarimax_best_params_washington_clean_only_720_20260515_155959.json"
    config.xgb_params_file = "results/tuning/xgboost_best_params_washington_clean_only_720_20260515_233428.json"
    config.xgb_noweather_params_file = None  # set after: tune_xgboost.py --city washington --scenario no_weather
    config.prophet_params_file = "results/tuning/prophet_best_params_washington_720_20260516_000912.json" 
    config.neuralprophet_params_file = "results/tuning/neuralprophet_best_params_washington_clean_only_720_20260516_003612.json"
    config.neuralprophet_noweather_params_file = None  # set after: tune_neuralprophet.py --city washington --scenario no_weather
    
    return config
