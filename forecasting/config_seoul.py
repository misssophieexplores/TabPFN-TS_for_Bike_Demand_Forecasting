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
        "precipitation_mm",
        "snow_depth_cm",
        "Visibility",
        "Seasons",
        "Holiday",
    ]
    config.weather_degradation_mapping = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Wind speed": "wind_speed",
        "Solar Radiation": "solar_radiation",
        "precipitation_mm": "precipitation",   # KMA 강수량 (= Rainfall); Nov-Mar 3-h totals spread over their 3 hours (mm/h)
        "snow_depth_cm": "snow_depth",         # KMA 적설 (= Snowfall): snow depth on the ground (cm)
        "Visibility": "visibility"
    }
    config.timezone = "Asia/Seoul"   # local time of the date column (KST, no daylight saving)
    config.nwp_calibration_file = "weather/nwp/calibration/seoul.npz"
    config.column_scale_factors = {"Visibility": 0.01}  # raw unit is 10 m -> km (as London/Washington)

    # --- Model parameters ---
    config.arima_params_file = "results/tuning/arima_best_params_seoul_720_20261001_170419.json"
    config.sarimax_params_file = None   # re-tune: precipitation_mm (winter 3-h totals spread), 2 Oct 2026
    config.xgb_params_file = None   # re-tune: covariates changed 2 Oct 2026
    config.xgb_noweather_params_file = "results/tuning/xgboost_best_params_seoul_no_weather_720_20261002_024216.json"
    config.prophet_params_file = "results/tuning/prophet_best_params_seoul_720_20261001_170843.json"
    config.neuralprophet_params_file = None   # re-tune: covariates changed 2 Oct 2026
    config.neuralprophet_noweather_params_file ="results/tuning/neuralprophet_best_params_seoul_no_weather_720_20261001_183244.json"
    return config
