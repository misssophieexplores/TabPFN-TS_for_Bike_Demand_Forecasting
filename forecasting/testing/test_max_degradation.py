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
from weather.weather_processor import WeatherProcessor

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
        column_mapping = {
            'Temperature': 'temperature',
            'Solar Radiation': 'solar_radiation',
        }

        df = pd.DataFrame({
            'Temperature': np.random.randn(100) * 10 + 15,
            'Solar Radiation': np.random.rand(100) * 3
        })
        
        params = prepare_degradation_parameters(df, column_mapping)
        
        assert 'solar_cap' in params
        assert params['solar_cap'] > 0
    
    def test_degrade_dataframe(self):
        """Test dataframe degradation"""
        column_mapping = {
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'Wind speed': 'wind_speed',
            'Solar Radiation': 'solar_radiation',
        }

        df = pd.DataFrame({
            'Temperature': [15.0, 16.0, 17.0],
            'Humidity': [60.0, 65.0, 70.0],
            'Wind speed': [2.0, 3.0, 4.0],
            'Solar Radiation': [1.5, 2.0, 2.5]
        })
        
        params = prepare_degradation_parameters(df, column_mapping)
        
        df_degraded = degrade_weather_dataset(
            df, 24, params, column_mapping,
            seed=42
        )
        
        assert df_degraded.shape == df.shape
        assert list(df_degraded.columns) == list(df.columns)
        assert not df_degraded.equals(df)

    def _make_weather_processor(self):
        """Helper: minimal config + WeatherProcessor that doesn't need real data."""
        config = ForecastConfig()
        config.dataset_name = "test"
        config.degradation_seed = 42
        config.weather_covariates = [
            'Temperature', 'Humidity', 'Wind speed', 'Solar Radiation',
            'Rainfall', 'Snowfall',
        ]
        config.weather_degradation_mapping = {
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'Wind speed': 'wind_speed',
            'Solar Radiation': 'solar_radiation',
            'Rainfall': 'precipitation',
            'Snowfall': 'precipitation',
        }
        config.holiday_col = None
        config.season_col = None
        return WeatherProcessor(config)

    def _make_test_df(self, n_rows=24):
        """Helper: synthetic weather DataFrame with n_rows rows."""
        rng = np.random.default_rng(seed=0)
        return pd.DataFrame({
            'Temperature':    rng.normal(15, 5, n_rows),
            'Humidity':       rng.uniform(40, 90, n_rows),
            'Wind speed':     rng.uniform(0.5, 8, n_rows),
            'Solar Radiation': rng.uniform(0, 3, n_rows),
            'Rainfall':       np.zeros(n_rows),
            'Snowfall':       np.zeros(n_rows),
        })

    def test_train_data_never_degraded(self):
        """Training data must be identical for clean_only and degraded scenarios.

        In an operational setting the model is trained on observed weather, not
        NWP forecasts.  Degrading training covariates would conflate fitting-time
        and inference-time uncertainty, so prepare_weather_data(..., split='train')
        must always return clean data regardless of scenario.
        """
        processor = self._make_weather_processor()
        df = self._make_test_df()

        X_train_clean = processor.prepare_weather_data(
            df, 'clean_only', horizon=24, fold_idx=0, split='train'
        )
        X_train_deg = processor.prepare_weather_data(
            df, 'degraded', horizon=24, fold_idx=0, split='train'
        )

        pd.testing.assert_frame_equal(
            X_train_clean.reset_index(drop=True),
            X_train_deg.reset_index(drop=True),
            check_like=True,
            obj="Training data should be identical for clean_only and degraded scenarios"
        )

    def test_noise_grows_with_lead_time(self):
        """Test error magnitude increases from first to last row of the test window.

        Row i is degraded with lead time (i+1) hours, so the second half of the
        test window should have larger absolute errors than the first half on
        average.  Verified across multiple fold seeds to guard against stochastic
        failures on any single seed.
        """
        processor = self._make_weather_processor()
        horizon = 48
        df = self._make_test_df(n_rows=horizon)

        wins = 0
        n_trials = 10
        for fold_idx in range(n_trials):
            X_clean = processor.prepare_weather_data(
                df, 'clean_only', horizon=horizon, fold_idx=fold_idx, split='test'
            )
            X_deg = processor.prepare_weather_data(
                df, 'degraded', horizon=horizon, fold_idx=fold_idx, split='test'
            )

            errors = (X_deg['Temperature'] - X_clean['Temperature']).abs()
            first_half  = errors.iloc[:horizon // 2].mean()
            second_half = errors.iloc[horizon // 2:].mean()
            if second_half > first_half:
                wins += 1

        # Expect the second half to be noisier in the large majority of trials
        assert wins >= 7, (
            f"Expected noise to grow with lead time in ≥7/10 trials, got {wins}/10"
        )
if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])