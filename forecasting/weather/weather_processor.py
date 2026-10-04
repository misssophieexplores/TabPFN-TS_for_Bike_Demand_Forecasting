"""
Weather data processor for different scenarios.

Orchestrates weather data preparation including:
- Variable selection based on scenario
- On-the-fly degradation with proper seeding
- Integration with weather_degradation module
"""

import zlib

import pandas as pd
import numpy as np
from typing import List, Optional
from config import ForecastConfig
from weather.weather_degradation import (
    prepare_degradation_parameters,
    degrade_weather_dataset,
)
from weather.nwp_error_model import load_error_model, to_utc

DEGRADATION_MODELS = ("nwp_measured", "literature")


def city_seed_term(dataset_name):
    """City part of the degradation seed (measured model): 10,000,000 x
    (CRC32 of the dataset name mod 1000), so the cities draw different random
    numbers for the same horizon and fold (seoul 782, london 181,
    washington 182). 0 if no name."""
    if not dataset_name:
        return 0
    return 10_000_000 * (zlib.crc32(str(dataset_name).encode()) % 1000)


class WeatherProcessor:
    """
    Orchestrates weather data preparation for different scenarios.
    
    Handles these scenarios:
    - all_weather: all config.weather_covariates (8 weather variables +
      holiday + season), no degradation
    - clean_only: 7 degradable variables (no Dew point) + holiday + season,
      no degradation
    - degraded: same columns as clean_only, degradation on the 7 degradable
      variables
    - optional noise-magnitude sensitivity scenarios (e.g. degraded_x150:
      error magnitudes x1.5), not run by default; every key of
      config.degradation_scales is a degraded scenario
    
    Parameters
    ----------
    config : ForecastConfig
        Configuration object with weather settings
        
    Attributes
    ----------
    config : ForecastConfig
        Stored configuration
    degradation_params : dict or None
        Cached degradation parameters (computed from training data)
    """
    
    def __init__(self, config: ForecastConfig):
        """
        Initialize weather processor.
        
        Parameters
        ----------
        config : ForecastConfig
            Configuration with weather_covariates and weather_degradation_mapping
        """
        self.config = config
        self.degradation_params = None
        # Measured NWP model: what the last degraded test window used
        # (forecast start, lead times, replayed run, settings); None for "literature"
        self.last_degradation_info = None
        if config.degradation_model not in DEGRADATION_MODELS:
            raise ValueError(
                f"degradation_model must be one of {DEGRADATION_MODELS}, "
                f"got {config.degradation_model!r}"
            )
        
    def get_weather_columns(self, scenario: str) -> List[str]:
        """
        Get list of weather columns for a given scenario.
        
        Parameters
        ----------
        scenario : str
            'all_weather', 'clean_only' or a degraded scenario (config.degradation_scales)
            
        Returns
        -------
        List[str]
            Weather column names to use for this scenario
            
        Notes
        -----
        - all_weather: all of config.weather_covariates (8 weather
          variables + holiday + season)
        - clean_only: the 7 degradable variables (no Dew point) + holiday + season
        - degraded scenarios: same columns as clean_only
        """
        if scenario == "all_weather":
            # All of weather_covariates (8 weather variables + holiday + season)
            return self.config.weather_covariates.copy()
        
        elif scenario == "clean_only" or self.config.is_degraded(scenario):
            # Degradable variables (7, no Dew point), then holiday and season
            # These are the ones in weather_degradation_mapping
            degradable_vars = list(self.config.weather_degradation_mapping.keys())

            # Ensure order matches original weather_covariates
            ordered_vars = [
                var for var in self.config.weather_covariates
                if var in degradable_vars
            ]

            # Also include non-degradable covariates (e.g. holiday, season)
            for col in [self.config.holiday_col, self.config.season_col]:
                if col and col in self.config.weather_covariates:
                    ordered_vars.append(col)

            return ordered_vars

        else:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Must be one of: {self.config.weather_scenarios}"
            )
    
    def prepare_weather_data(
        self,
        df: pd.DataFrame,
        scenario: str,
        horizon: int,
        fold_idx: int,
        split: str = "train"
    ) -> Optional[pd.DataFrame]:
        """
        Prepare weather data for a specific scenario and fold.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataframe (train or test) with all columns
        scenario : str
            'all_weather', 'clean_only' or a degraded scenario (config.degradation_scales)
        horizon : int
            Forecast horizon in hours (6, 24, 48, 168)
        fold_idx : int
            CV fold index for seed generation
        split : str, default='train'
            'train' or 'test'.  Controls whether and how degradation is applied:

            - 'train': **no degradation even in the 'degraded' scenario**.
              In an operational setting the model is fitted on historical
              *observed* weather, not on NWP forecasts.  Degrading training
              covariates would conflate train/test domain shift with the
              robustness signal we want to measure.

            - 'test': degradation uses **row-varying lead times**: row i
              has lead (i + 1) hours (measured NWP model with
              config.nwp_fresh_forecast=True, the default, and the
              'literature' model). With nwp_fresh_forecast=False the ECMWF
              run available at the forecast issue time (6-17 h old) is used,
              so row i has lead (run age + i + 1) hours. The measured model
              needs config.date_col in df.
            
        Returns
        -------
        pd.DataFrame
            Weather covariates prepared for this scenario
            Contains only the selected weather columns
            
        Notes
        -----
        Degradation is applied on-the-fly during each fold:
        - Prevents data leakage between folds
        - Different degradation per fold (realistic)
        - Reproducible via seed = base_seed + 10000 * horizon + fold_idx
          (+ city_seed_term(dataset_name) in the measured model)
        
        Examples
        --------
        >>> processor = WeatherProcessor(config)
        >>> # Training data — always clean (observed weather)
        >>> X_train = processor.prepare_weather_data(
        ...     train_df, 'degraded', horizon=24, fold_idx=5, split='train'
        ... )
        >>> # Test data — row-varying lead-time noise
        >>> X_test = processor.prepare_weather_data(
        ...     test_df, 'degraded', horizon=24, fold_idx=5, split='test'
        ... )
        """
        # Get appropriate columns for this scenario
        weather_cols = self.get_weather_columns(scenario)
        
        # Extract weather data
        weather_df = df[weather_cols].copy()

        # Degraded scenarios: 'degraded' and the noise-magnitude sensitivity
        # scenarios (keys of config.degradation_scales)
        degraded = self.config.is_degraded(scenario)

        # Degradation parameters (solar cap, wet-hour share) come from the clean training fold,
        # which is always prepared before the test fold.
        if degraded and split == "train":
            self.degradation_params = prepare_degradation_parameters(
                weather_df,
                self.config.weather_degradation_mapping
            )

        # Apply degradation only to test split in the degraded scenarios,
        # scaled by the scenario's factor.
        # Training data always uses clean (observed) weather so that the
        # experiment measures degradation at inference time, not during fitting.
        if degraded and split == "test":
            date_col = self.config.date_col
            timestamps = df[date_col] if date_col and date_col in df.columns else None
            weather_df = self.degrade_dataframe(
                weather_df,
                horizon,
                fold_idx,
                noise_scale=self.config.degradation_scales[scenario],
                timestamps=timestamps,
            )
        
        return weather_df
    
    def degrade_dataframe(
        self,
        df: pd.DataFrame,
        horizon: int,
        fold_idx: int,
        noise_scale: float = 1.0,
        timestamps=None,
    ) -> pd.DataFrame:
        """
        Apply degradation to weather dataframe with row-varying lead times.
        
        Parameters
        ----------
        df : pd.DataFrame
            Weather dataframe with degradable columns (test split only)
        horizon : int
            Forecast horizon in hours (6, 24, 48, 168).  Also the number of
            rows in df for a single test window.
        fold_idx : int
            CV fold index for seed generation
        noise_scale : float, default=1.0
            Factor applied to the error magnitudes (config.degradation_scales
            of the scenario; 1.0 = calibrated error model)
        timestamps : array-like, optional
            Local timestamps of the rows (config.date_col, time zone
            config.timezone). Required for the measured NWP model.
            
        Returns
        -------
        pd.DataFrame
            Degraded weather dataframe
            
        Notes
        -----
        Seed calculation:
        - seed = base_seed + 10000 * horizon + fold_idx; measured NWP model:
          + city_seed_term(config.dataset_name), so the cities draw different
          random numbers (literature model unchanged, reproduces earlier runs)
        - The seed does not depend on noise_scale, so all degraded scenarios
          of a (horizon, fold) use the same random numbers; only the error
          magnitude differs (common random numbers).
        - Unique per (horizon, fold) as long as fold_idx < 10000 (at most
          980 folds, for h=6). The previous base_seed + fold_idx + horizon
          gave duplicates across horizons (e.g. fold 18 at h=6 and fold 0
          at h=24 both got base_seed + 24).

        Lead-time assignment:
        - "nwp_measured", config.nwp_fresh_forecast=True (default): the
          weather forecast starts when the demand forecast is issued (first
          test hour - 1 h); row i gets lead time i + 1. With
          config.nwp_remove_bias=True (default) the average forecast error
          (lean) is removed; with config.nwp_seasonal_rain=True (default)
          the rain error rates are those of the time of year; with
          config.nwp_mean_preserving_caps=True (default) the degraded
          covariates keep the clean value on average after their caps and
          bounds; with config.nwp_rain_amount_unbiased=True (default) false
          alarms add as much rain as misses remove, in expectation. See
          weather/nwp_error_model.py.
        - "nwp_measured", config.nwp_fresh_forecast=False: the newest ECMWF
          run available at the issue time is used (runs at
          config.nwp_run_hours_utc, available config.nwp_availability_delay_h
          hours later). Row i gets that run's lead time: run age (6-17 h) + i + 1.
        - "literature": row i (0-indexed) is the forecast for (i+1) hours
          ahead and receives noise calibrated to lead time (i+1) hours.
        """
        # Seed unique per (horizon, fold): fold_idx < 10000 for all horizons
        if not 0 <= fold_idx < 10000:
            raise ValueError(
                f"fold_idx={fold_idx} out of range [0, 10000): degradation seeds "
                f"would collide across horizons"
            )
        horizon_seed = self.config.degradation_seed + 10000 * horizon + fold_idx

        # Degradation parameters computed from the training fold
        if self.degradation_params is None:
            raise RuntimeError(
                "degradation_params not set: prepare the train split of this fold "
                "(degraded scenario) before the test split"
            )

        # One total-precipitation column per city (mm, rain + melted snow);
        # snow depth has its own type ('snow_depth', persistence forecast)
        precip_cols = [c for c, t in self.config.weather_degradation_mapping.items()
                       if t == "precipitation"]
        if len(precip_cols) > 1:
            raise ValueError(
                f"weather_degradation_mapping has {len(precip_cols)} precipitation columns "
                f"{precip_cols}; expected one total-precipitation column (snow depth: "
                f"type 'snow_depth')"
            )

        if self.config.degradation_model == "nwp_measured":
            if timestamps is None:
                raise ValueError(
                    "The measured NWP error model needs the timestamps of the test "
                    f"rows (column {self.config.date_col!r})"
                )
            if not self.config.nwp_calibration_file:
                raise ValueError("config.nwp_calibration_file is not set for this city")
            model = load_error_model(self.config.nwp_calibration_file)
            times_utc = to_utc(timestamps, self.config.timezone)
            # different random numbers per city (since 2 Oct 2026)
            horizon_seed += city_seed_term(self.config.dataset_name)
            df_degraded, info = model.degrade(
                df,
                times_utc,
                self.config.weather_degradation_mapping,
                self.degradation_params,
                seed=horizon_seed,
                noise_scale=noise_scale,
                run_hours=tuple(self.config.nwp_run_hours_utc),
                delay_h=self.config.nwp_availability_delay_h,
                season_days=self.config.nwp_season_window_days,
                fresh_forecast=self.config.nwp_fresh_forecast,
                remove_bias=self.config.nwp_remove_bias,
                seasonal_rain=self.config.nwp_seasonal_rain,
                rain_intensity_dependent=self.config.nwp_rain_intensity_dependent,
                rain_frequency_unbiased=self.config.nwp_rain_frequency_unbiased,
                rain_amount_unbiased=self.config.nwp_rain_amount_unbiased,
                mean_preserving_caps=self.config.nwp_mean_preserving_caps,
            )
            self.last_degradation_info = info
            return df_degraded

        # "literature": per-row lead times 1..h
        lead_times = np.arange(1, len(df) + 1)
        df_degraded = degrade_weather_dataset(
            df=df,
            horizon_hours=horizon,      # fallback scalar (unused when lead_times given)
            degradation_params=self.degradation_params,
            column_mapping=self.config.weather_degradation_mapping,
            seed=horizon_seed,
            lead_times=lead_times,
            noise_scale=noise_scale
        )
        
        return df_degraded
    
    def get_scenario_summary(self, scenario: str) -> dict:
        """
        Get summary information about a scenario.
        
        Parameters
        ----------
        scenario : str
            Scenario identifier
            
        Returns
        -------
        dict
            Summary with keys: scenario, num_vars, variables, degraded,
            noise_scale (None if the scenario is not degraded)
            
        Examples
        --------
        >>> processor = WeatherProcessor(config)
        >>> summary = processor.get_scenario_summary('degraded')
        >>> print(summary)
        {
            'scenario': 'degraded',
            'num_vars': 7,
            'variables': ['Temperature', 'Humidity', ...],
            'degraded': True
        }
        """
        weather_cols = self.get_weather_columns(scenario)
        
        return {
            'scenario': scenario,
            'num_vars': len(weather_cols),
            'variables': weather_cols,
            'degraded': self.config.is_degraded(scenario),
            'noise_scale': self.config.degradation_scales.get(scenario)
        }