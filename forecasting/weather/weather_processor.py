"""
Weather data processor for different scenarios.

Orchestrates weather data preparation including:
- Variable selection based on scenario
- On-the-fly degradation with proper seeding
- Integration with weather_degradation module
"""

# TODO: ADD HOLIDAY AND SEASON!!!! 
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
    - clean_only: 7 degradable variables (exclude Dew point), no degradation
    - degraded: 7 degradable variables (exclude Dew point), with degradation
    
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
        - all_weather: All 8 variables from config.weather_covariates
        - clean_only: Only the 7 degradable variables (excludes Dew point)
        - degraded: Same 7 degradable variables as clean_only
        """
        if scenario == "all_weather":
            # Return all 8 weather variables
            return self.config.weather_covariates.copy()
        
        elif scenario in ["clean_only", "degraded"]:
            # Return only degradable variables (7 vars, exclude Dew point)
            # These are the ones in weather_degradation_mapping
            degradable_vars = list(self.config.weather_degradation_mapping.keys())
            
            # Ensure order matches original weather_covariates
            ordered_vars = [
                var for var in self.config.weather_covariates 
                if var in degradable_vars
            ]
            
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
        fold_idx: int
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
        >>> # Get clean weather (no degradation)
        >>> X_clean = processor.prepare_weather_data(
        ...     test_df, 'clean_only', horizon=24, fold_idx=5
        ... )
        >>> # Get degraded weather
        >>> X_degraded = processor.prepare_weather_data(
        ...     test_df, 'degraded', horizon=24, fold_idx=5
        ... )
        """
        # Get appropriate columns for this scenario
        weather_cols = self.get_weather_columns(scenario)
        
        # Extract weather data
        weather_df = df[weather_cols].copy()
        
        # Apply degradation if scenario is 'degraded'
        if scenario == "degraded":
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
        Apply degradation to weather dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Weather dataframe with degradable columns
        horizon : int
            Forecast horizon in hours (6, 24, 48, 168)
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
        
        Degradation parameters are computed on-the-fly from the data.
        This is cheap (just percentile calculations) and avoids caching issues.
        """
        # Compute fold-specific seed
        fold_seed = self.config.degradation_seed + fold_idx
        horizon_seed = fold_seed + horizon
        
        # Create RNG for this fold and horizon
        rng = np.random.default_rng(seed=horizon_seed)
        
        # Compute degradation parameters from data
        # (cheap operation, just percentile calculations)
        degradation_params = prepare_degradation_parameters(
            df, 
            self.config.weather_degradation_mapping
        )
        
        # Apply degradation
        df_degraded = degrade_weather_dataset(
            df=df,
            horizon_hours=horizon,
            degradation_params=degradation_params,
            column_mapping=self.config.weather_degradation_mapping,
            seed=horizon_seed
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
            'degraded': scenario == "degraded"
        }
