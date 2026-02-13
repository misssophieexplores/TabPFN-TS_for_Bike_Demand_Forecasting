"""
Unit tests for weather degradation functions.

Run with: pytest test_weather_unit.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from config import ForecastConfig
from weather.weather_degradation import (
    degrade_weather_forecast,
    prepare_degradation_parameters,
    degrade_weather_dataset  
)

class TestWeatherDegradation:
    """Unit tests for individual degradation functions"""
    
    def test_temperature_degradation_reasonable(self):
        """Test temperature degradation produces reasonable values"""
        rng = np.random.default_rng(seed=42)
        actual = 15.0
        degraded = degrade_weather_forecast(actual, 'temperature', 24, rng=rng)
        
        # Should be within reasonable range (e.g., ±10°C)
        assert abs(degraded - actual) < 10.0, f"Degraded temp {degraded} too far from actual {actual}"
    
    def test_wind_speed_non_negative(self):
        """Test wind speed is never negative"""
        rng = np.random.default_rng(seed=42)
        
        # Test multiple values
        for actual in [0.5, 1.0, 5.0]:
            degraded = degrade_weather_forecast(actual, 'wind_speed', 168, rng=rng)
            assert degraded >= 0, f"Wind speed {degraded} is negative!"
    
    def test_reproducibility(self):
        """Test same seed produces identical results"""
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        
        actual = 15.0
        deg1 = degrade_weather_forecast(actual, 'temperature', 24, rng=rng1)
        deg2 = degrade_weather_forecast(actual, 'temperature', 24, rng=rng2)
        
        assert deg1 == deg2, "Same seed should produce identical results"
    
    def test_different_seeds_different_results(self):
        """Test different seeds produce different results"""
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=99)
        
        actual = 15.0
        deg1 = degrade_weather_forecast(actual, 'temperature', 24, rng=rng1)
        deg2 = degrade_weather_forecast(actual, 'temperature', 24, rng=rng2)
        
        assert deg1 != deg2, "Different seeds should produce different results"
    
    def test_precipitation_event_detection(self):
        """Test precipitation can have false alarms and misses"""
        rng = np.random.default_rng(seed=42)
        
        # Test actual = 0 (potential false alarm)
        # Run multiple times to see if false alarm occurs
        false_alarms = []
        for i in range(100):
            rng_i = np.random.default_rng(seed=42 + i)
            degraded = degrade_weather_forecast(0.0, 'precipitation', 24, rng=rng_i)
            if degraded > 0:
                false_alarms.append(degraded)
        
        # Should have some false alarms (but not all)
        assert 5 < len(false_alarms) < 95, f"Expected some false alarms, got {len(false_alarms)}/100"
    
    def test_prepare_degradation_parameters(self):
        """Test degradation parameter computation"""
        config = ForecastConfig()
        
        df = pd.DataFrame({
            'Temperature': np.random.randn(100) * 10 + 15,
            'Solar Radiation': np.random.rand(100) * 3
        })
        
        params = prepare_degradation_parameters(df, config.weather_degradation_mapping)
        
        assert 'solar_cap' in params
        assert params['solar_cap'] > 0
    
    def test_degrade_dataframe(self):
        """Test dataframe degradation"""
        config = ForecastConfig()
        
        df = pd.DataFrame({
            'Temperature': [15.0, 16.0, 17.0],
            'Humidity': [60.0, 65.0, 70.0],
            'Wind speed': [2.0, 3.0, 4.0],
            'Solar Radiation': [1.5, 2.0, 2.5]
        })
        
        params = prepare_degradation_parameters(df, config.weather_degradation_mapping)
        
        df_degraded = degrade_weather_dataset(
            df, 24, params, config.weather_degradation_mapping, 
            seed=42  # Changed from rng=
        )
        
        assert df_degraded.shape == df.shape
        assert list(df_degraded.columns) == list(df.columns)
        assert not df_degraded.equals(df)
if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
