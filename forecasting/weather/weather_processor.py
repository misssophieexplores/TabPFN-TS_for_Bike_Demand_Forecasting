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
    
    Handles three scenarios:
    - all_weather: All 8 variables, no degradation
    - clean_only: 6 degradable variables (exclude Dew point), no degradation
    - degraded: 6 degradable variables (exclude Dew point), with degradation
    
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
            One of: 'all_weather', 'clean_only', 'degraded'
            
        Returns
        -------
        List[str]
            Weather column names to use for this scenario
            
        Notes
        -----
        - all_weather: All (8) variables from config.weather_covariates
        - clean_only: Only the 6 degradable variables (excludes Dew point)
        - degraded: Same 6 degradable variables as clean_only
        """
        if scenario == "all_weather":
            # Return all 8 weather variables
            return self.config.weather_covariates.copy()
        
        elif scenario in ["clean_only", "degraded"]:
            # Return only degradable variables (6 vars, exclude Dew point and visibility)
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
            One of: 'all_weather', 'clean_only', 'degraded'
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
        - Reproducible via seed = base_seed + fold_idx + horizon
        
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
        
        # Apply degradation only to test split in the 'degraded' scenario.
        # Training data always uses clean (observed) weather so that the
        # experiment measures degradation at inference time, not during fitting.
        if scenario == "degraded" and split == "test":
            weather_df = self.degrade_dataframe(
                weather_df,
                horizon,
                fold_idx
            )
        
        return weather_df
    
    def degrade_dataframe(
        self,
        df: pd.DataFrame,
        horizon: int,
        fold_idx: int
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
            
        Returns
        -------
        pd.DataFrame
            Degraded weather dataframe
            
        Notes
        -----
        Seed calculation:
        - fold_seed = base_seed + fold_idx
        - horizon_seed = fold_seed + horizon
        - This ensures reproducibility and independence

        Lead-time assignment:
        - Row i (0-indexed) represents the forecast for 1 hour ahead at
          step i, so it receives noise calibrated to lead time (i+1) hours.
        - This means the first test step gets near-zero noise (1 h lead) and
          the last step gets full horizon noise — physically correct behaviour.
        """
        # Compute fold-specific seed
        fold_seed = self.config.degradation_seed + fold_idx
        horizon_seed = fold_seed + horizon
        
        # Per-row lead times: step 0 → 1 h, step 1 → 2 h, …, step h-1 → h
        lead_times = np.arange(1, len(df) + 1)
        
        # Compute degradation parameters from data
        # (cheap operation, just percentile calculations)
        degradation_params = prepare_degradation_parameters(
            df, 
            self.config.weather_degradation_mapping
        )
        
        # Apply degradation with per-row lead times
        df_degraded = degrade_weather_dataset(
            df=df,
            horizon_hours=horizon,      # fallback scalar (unused when lead_times given)
            degradation_params=degradation_params,
            column_mapping=self.config.weather_degradation_mapping,
            seed=horizon_seed,
            lead_times=lead_times
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
            Summary with keys: scenario, num_vars, variables, degraded
            
        Examples
        --------
        >>> processor = WeatherProcessor(config)
        >>> summary = processor.get_scenario_summary('degraded')
        >>> print(summary)
        {
            'scenario': 'degraded',
            'num_vars': 6,
            'variables': ['Temperature', 'Humidity', ...],
            'degraded': True
        }
        """
        weather_cols = self.get_weather_columns(scenario)
        
        return {
            'scenario': scenario,
            'num_vars': len(weather_cols),
            'variables': weather_cols,
            'degraded': scenario == "degraded"
        }