"""
Shared base configuration for forecasting experiments.

Dataset-specific fields (data_filename, dataset_name, column names,
season_mapping, weather_covariates, model parameter files) are set to None
here and must be overridden in each city config via get_config().

Fields that are truly shared across all datasets (wandb_project,
results_version, horizons, n_folds, etc.) are defined here once.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ForecastConfig:
    """Configuration for time series forecasting experiments"""

    # --- Data (dataset-specific — override in city config) ---
    data_filename: Optional[str] = None
    date_col: Optional[str] = None
    target_col: Optional[str] = None
    dataset_name: Optional[str] = None
    functioning_day_col: Optional[str] = None
    holiday_col: Optional[str] = None
    holiday_mapping: Optional[Dict] = field(default_factory=lambda: {'Yes': 1, 'No': 0})
    season_col: Optional[str] = None
    season_mapping: Optional[Dict] = None
    weather_covariates: Optional[List[str]] = None
    weather_degradation_mapping: Optional[Dict[str, str]] = None
    # Variable types in weather_degradation_mapping: temperature, humidity,
    # wind_speed, solar_radiation, visibility, precipitation (total
    # precipitation incl. melted snow, mm; one column per city) and snow_depth
    # (snow on the ground, cm; persistence forecast). Since 2 Oct 2026 there is
    # no rain/snow split and no rain_col / snow_col.
    # Local time zone of the date column (IANA name); the measured NWP error
    # model converts the test hours to UTC to find the ECMWF run in use
    timezone: Optional[str] = None
    # Calibration file of the measured NWP error model (path relative to
    # forecasting/), built by weather/nwp/build_nwp_calibration.py
    nwp_calibration_file: Optional[str] = None
    column_scale_factors: Dict[str, float] = field(default_factory=dict)

    # --- Model parameters (dataset-specific — override in city config) ---
    arima_params_file: Optional[str] = None
    sarimax_params_file: Optional[str] = None
    xgb_params_file: Optional[str] = None
    xgb_noweather_params_file: Optional[str] = None
    prophet_params_file: Optional[str] = None
    neuralprophet_params_file: Optional[str] = None
    neuralprophet_noweather_params_file: Optional[str] = None

    # # --- Forecasting (shared) --- 
    horizons: List[int] = field(default_factory=lambda: [6, 24, 48, 168])
    seasonal_period: int = 24
    # # lookback window 4096
    # n_folds: int = 20 # for n_train_samples=4096 (circa 0.74 of the data)
    # n_train_samples: int = 4096
    # lookback window 720
    n_folds: int = 35
    n_train_samples: int = 720 # 30 days (30*24=720)




    # --- Weather degradation (shared) ---
    degradation_seed: int = 42
    # Error model of the degraded scenarios:
    #   "nwp_measured": per-city errors measured from real ECMWF IFS HRES
    #       forecasts (weather/nwp_error_model.py; see weather_methodology.md)
    #   "literature": the earlier model with published error sizes, the same
    #       for every city (weather_degradation.degrade_weather_dataset)
    degradation_model: str = "nwp_measured"
    # Measured model. Default = an operator with an up-to-date, locally
    # corrected weather forecast:
    #   nwp_fresh_forecast=True: the weather forecast starts when the demand
    #       forecast is made, so test hour i gets the error of an (i+1)-hour
    #       forecast in every city. False: the newest ECMWF run available at
    #       that time is used (start hours nwp_run_hours_utc, available
    #       nwp_availability_delay_h hours later), i.e. a 6-17 h old forecast.
    #   nwp_remove_bias=True: the average error (lean) of the forecasts is
    #       removed, e.g. Seoul's forecast being on average 1.5 C too cold.
    # The replayed run comes from the same time of year (+/- nwp_season_window_days).
    #   nwp_seasonal_rain=True: rain miss rate, false-alarm ratio and amount
    #       error of the time of year (monthly bins); False: year-round values.
    nwp_fresh_forecast: bool = True
    nwp_remove_bias: bool = True
    nwp_seasonal_rain: bool = True
    #   nwp_rain_frequency_unbiased=True: the forecast is wet as often as
    #       observed (false alarms = misses), i.e. the rain-frequency bias of
    #       the raw forecast is removed like the temperature/humidity/wind
    #       bias. Seoul's raw forecast is wet about 2.2 times as often as the
    #       station. False: measured false-alarm ratio.
    nwp_rain_frequency_unbiased: bool = True
    nwp_run_hours_utc: List[int] = field(default_factory=lambda: [0, 12])
    nwp_availability_delay_h: int = 6
    nwp_season_window_days: int = 30
    # Degraded scenarios -> factor applied to the calibrated error magnitudes.
    # 'degraded' is the calibrated error model. Noise-magnitude sensitivity
    # scenarios are not run since 2 Oct 2026 (the errors are measured per
    # city); to run them, add e.g. "degraded_x050": 0.5 / "degraded_x150": 1.5
    # here and to weather_scenarios. All degraded scenarios of a (horizon,
    # fold) use the same random numbers (see ARCHITECTURE.md
    # "Noise-magnitude sensitivity" for what is scaled).
    degradation_scales: Dict[str, float] = field(default_factory=lambda: {
        "degraded": 1.0,
    })
    weather_scenarios: List[str] = field(default_factory=lambda: [
        "all_weather",
        "clean_only",
        "degraded",
    ])

    # --- Output (shared) ---
    output_dir: str = "results"
    results_version: str = "v7"
    verbose: bool = False

    # --- W&B (shared) ---
    wandb_project: str = "bike-forecasting"
    experiment_name: Optional[str] = None

    # Tuning parameters:
    tune_folds: Optional[int] = 90
    tune_horizon: int = 24

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.dataset_name}_{self.results_version}"

    def is_degraded(self, scenario: str) -> bool:
        """True if the scenario degrades the test covariates (a key of degradation_scales)."""
        return scenario in self.degradation_scales

    def degradation_label(self) -> str:
        """Error model and its settings, as written to the results
        ('degradation_model' column), e.g.
        'nwp_measured(fresh,no_bias,seasonal_rain,rain_freq_unbiased)'."""
        if self.degradation_model != "nwp_measured":
            return self.degradation_model
        age = "fresh" if self.nwp_fresh_forecast else f"age_delay{self.nwp_availability_delay_h}h"
        bias = "no_bias" if self.nwp_remove_bias else "with_bias"
        rain = "seasonal_rain" if self.nwp_seasonal_rain else "yearround_rain"
        if self.nwp_rain_frequency_unbiased:
            rain += ",rain_freq_unbiased"
        return f"nwp_measured({age},{bias},{rain})"
