"""
Weather data processor for different scenarios.

Orchestrates weather data preparation including:
- Variable selection based on scenario
- On-the-fly degradation with proper seeding
- Integration with weather_degradation module
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from config import ForecastConfig
from weather.weather_degradation import (
    prepare_degradation_parameters,
    degrade_weather_dataset,
    fix_precipitation_type,
)
from weather.nwp_error_model import load_error_model, to_utc

DEGRADATION_MODELS = ("nwp_measured", "literature")


class WeatherProcessor:
    """
    Orchestrates weather data preparation for different scenarios.
    
    Handles these scenarios:
    - all_weather: All 8 variables, no degradation
    - clean_only: 7 degradable variables (exclude Dew point), no degradation
    - degraded: 7 degradable variables (exclude Dew point), with degradation
    - degraded_x050 / degraded_x150: as degraded, error magnitudes x0.5 / x1.5
      (every key of config.degradation_scales is a degraded scenario)
    
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
        - all_weather: All (8) variables from config.weather_covariates
        - clean_only: Only the 7 degradable variables (excludes Dew point)
        - degraded scenarios: Same 7 degradable variables as clean_only
        """
        if scenario == "all_weather":
            # Return all 8 weather variables
            return self.config.weather_covariates.copy()
        
        elif scenario == "clean_only" or self.config.is_degraded(scenario):
            # Return only degradable variables (7 vars, exclude Dew point)
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
        - seed = base_seed + 10000 * horizon + fold_idx
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
          (lean) is removed. See weather/nwp_error_model.py.
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

        # Columns for the rain/snow phase correction
        temp_cols = [c for c, t in self.config.weather_degradation_mapping.items() if t == "temperature"]
        if len(temp_cols) != 1 or not self.config.rain_col or not self.config.snow_col:
            raise ValueError(
                "Rain/snow correction needs exactly one 'temperature' column in "
                "weather_degradation_mapping and config.rain_col / config.snow_col set"
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
            df_degraded, info = model.degrade(
                df,
                times_utc,
                self.config.weather_degradation_mapping,
                self.degradation_params,
                seed=horizon_seed,
                rain_col=self.config.rain_col,
                noise_scale=noise_scale,
                run_hours=tuple(self.config.nwp_run_hours_utc),
                delay_h=self.config.nwp_availability_delay_h,
                season_days=self.config.nwp_season_window_days,
                fresh_forecast=self.config.nwp_fresh_forecast,
                remove_bias=self.config.nwp_remove_bias,
            )
            self.last_degradation_info = info
            # Rain/snow phase from the degraded temperature (as in the
            # literature model)
            return fix_precipitation_type(
                df_degraded, temp_col=temp_cols[0],
                rain_col=self.config.rain_col, snow_col=self.config.snow_col,
            )

        # "literature": per-row lead times 1..h
        lead_times = np.arange(1, len(df) + 1)
        df_degraded = degrade_weather_dataset(
            df=df,
            horizon_hours=horizon,      # fallback scalar (unused when lead_times given)
            degradation_params=self.degradation_params,
            column_mapping=self.config.weather_degradation_mapping,
            seed=horizon_seed,
            lead_times=lead_times,
            temp_col=temp_cols[0],
            rain_col=self.config.rain_col,
            snow_col=self.config.snow_col,
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