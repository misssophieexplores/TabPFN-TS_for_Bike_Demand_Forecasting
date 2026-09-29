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
    degrade_weather_dataset
)


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

            - 'test': degradation uses **row-varying lead times** (1 h for
              the first prediction step, 2 h for the second, …, horizon h
              for the last step).  This is physically correct because a
              horizon-h forecast covers h consecutive future hours and the
              NWP error grows with each additional hour of lead time.
              The old behaviour (same max-horizon noise on every test row)
              overestimated degradation for near-term steps.
            
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
            weather_df = self.degrade_dataframe(
                weather_df,
                horizon,
                fold_idx,
                noise_scale=self.config.degradation_scales[scenario]
            )
        
        return weather_df
    
    def degrade_dataframe(
        self,
        df: pd.DataFrame,
        horizon: int,
        fold_idx: int,
        noise_scale: float = 1.0
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
        - Row i (0-indexed) is the forecast for (i+1) hours ahead, so it
          receives noise calibrated to lead time (i+1) hours.
        - The first test step gets the smallest noise of the window (1 h
          lead), which is not zero: every error formula has an intercept
          (e.g. temperature 0.79 °C, humidity 13.0 %-points at 1 h). The
          last step gets full-horizon noise.
        """
        # Seed unique per (horizon, fold): fold_idx < 10000 for all horizons
        if not 0 <= fold_idx < 10000:
            raise ValueError(
                f"fold_idx={fold_idx} out of range [0, 10000): degradation seeds "
                f"would collide across horizons"
            )
        horizon_seed = self.config.degradation_seed + 10000 * horizon + fold_idx
        
        # Per-row lead times: step 0 → 1 h, step 1 → 2 h, …, step h-1 → h
        lead_times = np.arange(1, len(df) + 1)
        
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

        # Apply degradation with per-row lead times
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