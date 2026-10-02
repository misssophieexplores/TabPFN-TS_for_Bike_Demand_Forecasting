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
    config.functioning_day_col = "Functioning Day"
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
    config.rain_col = "rainfall_mm"
    config.timezone = "Europe/London"   # local time of the date column (daylight saving included)
    config.nwp_calibration_file = "weather/nwp/calibration/london.npz"
    config.snow_col = "snowfall_cm"

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_london_720_20261001_170305.json"
    config.sarimax_params_file = "results/tuning/sarimax_best_params_london_clean_only_720_20261001_194055.json"
    config.xgb_params_file = "results/tuning/xgboost_best_params_london_clean_only_720_20261001_225940.json"
    config.xgb_noweather_params_file = "results/tuning/xgboost_best_params_london_no_weather_720_20261002_014031.json"
    config.prophet_params_file = "results/tuning/prophet_best_params_london_720_20261001_170855.json"
    config.neuralprophet_params_file = "results/tuning/neuralprophet_best_params_london_clean_only_720_20261001_175145.json"
    config.neuralprophet_noweather_params_file ="results/tuning/neuralprophet_best_params_london_no_weather_720_20261001_185249.json"

    return config
