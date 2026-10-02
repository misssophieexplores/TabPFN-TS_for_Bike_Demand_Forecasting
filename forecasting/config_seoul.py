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
    config.rain_col = "Rainfall"
    config.timezone = "Asia/Seoul"   # local time of the date column (KST, no daylight saving)
    config.nwp_calibration_file = "weather/nwp/calibration/seoul.npz"
    config.snow_col = "Snowfall"
    config.column_scale_factors = {"Visibility": 0.01}  # raw unit is 10 m -> km (as London/Washington)

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_seoul_720_20261001_170419.json"
    config.sarimax_params_file = "results/tuning/sarimax_best_params_seoul_clean_only_720_20261001_215224.json"
    config.xgb_params_file = "results/tuning/xgboost_best_params_seoul_clean_only_720_20261001_215653.json"
    config.xgb_noweather_params_file = "results/tuning/xgboost_best_params_seoul_no_weather_720_20261002_024216.json"
    config.prophet_params_file = "results/tuning/prophet_best_params_seoul_720_20261001_170843.json"
    config.neuralprophet_params_file = "results/tuning/neuralprophet_best_params_seoul_clean_only_720_20261001_173039.json"
    config.neuralprophet_noweather_params_file ="results/tuning/neuralprophet_best_params_seoul_no_weather_720_20261001_183244.json"
    return config
