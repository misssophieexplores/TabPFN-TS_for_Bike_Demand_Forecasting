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
    degrade_weather_dataset,
    precipitation_detection_rates,
    precipitation_event_statistics,
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
        """Dry hours get false alarms with P(false alarm | dry), wet hours are
        missed with the miss rate (single-value function)."""
        wet_fraction = 0.06
        miss_rate, fa_prob = precipitation_detection_rates(24, wet_fraction)
        rng = np.random.default_rng(seed=42)
        n = 20000
        dry = np.array([degrade_weather_forecast(0.0, 'precipitation', 24, rng=rng,
                                                 wet_fraction=wet_fraction) for _ in range(n)])
        wet = np.array([degrade_weather_forecast(2.0, 'precipitation', 24, rng=rng,
                                                 wet_fraction=wet_fraction) for _ in range(n)])
        assert (dry > 0).mean() == pytest.approx(fa_prob, abs=0.005)
        assert (wet == 0).mean() == pytest.approx(miss_rate, abs=0.015)

    def test_precipitation_requires_wet_fraction(self):
        with pytest.raises(ValueError):
            degrade_weather_forecast(0.0, 'precipitation', 24, rng=np.random.default_rng(0))

    def test_event_statistics_hit_cited_values(self):
        """Miss rate and FAR pass through the cited Sukovich et al. (2014)
        values: Day 1 (24 h) 35 %, Day 2 (48 h) 45 %, 50 % cap from 60 h
        (Day 3: 45-55 %)."""
        for h, expected in [(24, 0.35), (48, 0.45), (60, 0.50), (72, 0.50), (168, 0.50)]:
            miss_rate, far = precipitation_event_statistics(h)
            assert miss_rate == pytest.approx(expected)
            assert far == pytest.approx(expected)

    def test_false_alarm_probability_formula(self):
        """P(false alarm | dry) = FAR / (1 - FAR) * POD * p / (1 - p), with FAR
        the false alarm RATIO of Sukovich et al. (2014)."""
        h, p = 24, 0.06
        miss_rate_src, far = precipitation_event_statistics(h)
        pod = 1 - miss_rate_src
        miss_rate, fa_prob = precipitation_detection_rates(h, p)
        assert miss_rate == pytest.approx(1 - pod)
        assert fa_prob == pytest.approx(far / (1 - far) * pod * p / (1 - p))
        assert fa_prob < 0.05                      # a few percent, not ~35%
        assert precipitation_detection_rates(h, 0.0)[1] == 0.0
        assert precipitation_detection_rates(h, 1.0)[1] == 0.0
        with pytest.raises(ValueError):
            precipitation_detection_rates(h, 1.5)

    def test_simulated_forecasts_reproduce_source_far_and_pod(self):
        """The degraded series must reproduce the published statistics
        (Day 1: FAR 0.35, POD 0.65; Day 2: FAR 0.45, POD 0.55):
        FAR = false alarms / forecast events, POD = hits / observed events."""
        rng = np.random.default_rng(1)
        n, p = 100_000, 0.06
        rain = np.where(rng.random(n) < p, rng.gamma(0.7, 1.5, n) + 0.1, 0.0)
        df = pd.DataFrame({'Rainfall': rain})
        mapping = {'Rainfall': 'precipitation'}
        params = {'solar_cap': 1.0, 'wet_fraction': float((rain > 0).mean())}
        obs = rain > 0
        for h, far_source, pod_source in [(24, 0.35, 0.65), (48, 0.45, 0.55)]:
            deg = degrade_weather_dataset(df, h, params, mapping, seed=h)['Rainfall'].to_numpy()
            fc = deg > 0
            hits, false_alarms = (obs & fc).sum(), (~obs & fc).sum()
            assert false_alarms / (false_alarms + hits) == pytest.approx(far_source, abs=0.02)
            assert hits / obs.sum() == pytest.approx(pod_source, abs=0.02)

    def test_wet_fraction_from_training_data(self):
        """Wet hour = rain OR snow > 0 (one precipitation variable)."""
        df = pd.DataFrame({
            'Solar Radiation': np.ones(10),
            'Rainfall': [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
            'Snowfall': [0, 1, 0, 3, 0, 0, 0, 0, 0, 0],
        })
        mapping = {'Solar Radiation': 'solar_radiation',
                   'Rainfall': 'precipitation', 'Snowfall': 'precipitation'}
        assert prepare_degradation_parameters(df, mapping)['wet_fraction'] == pytest.approx(0.3)

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
        config.rain_col = 'Rainfall'
        config.snow_col = 'Snowfall'
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
        # every degraded scenario, including the noise-magnitude sensitivity ones
        for scenario in processor.config.degradation_scales:
            X_train_deg = processor.prepare_weather_data(
                df, scenario, horizon=24, fold_idx=0, split='train'
            )

            pd.testing.assert_frame_equal(
                X_train_clean.reset_index(drop=True),
                X_train_deg.reset_index(drop=True),
                check_like=True,
                obj=f"Training data should be identical for clean_only and {scenario}"
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
            # The train split provides the degradation parameters (solar cap)
            processor.prepare_weather_data(
                df, 'degraded', horizon=horizon, fold_idx=fold_idx, split='train'
            )
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
    # ------------------------------------------------------------------
    # Regression tests for fixes that change results
    # ------------------------------------------------------------------
    def test_degraded_test_requires_train_split(self):
        """The test split cannot be degraded before the train split of the fold."""
        processor = self._make_weather_processor()
        with pytest.raises(RuntimeError):
            processor.prepare_weather_data(
                self._make_test_df(), 'degraded', horizon=24, fold_idx=0, split='test'
            )

    def test_solar_cap_from_training_fold(self):
        """Degraded solar radiation is capped at the 99.5th percentile of the
        clean TRAINING fold, not of the test window."""
        processor = self._make_weather_processor()
        train = self._make_test_df(n_rows=720)
        train['Solar Radiation'] = np.linspace(0, 1.0, 720)
        test = self._make_test_df(n_rows=168)
        test['Solar Radiation'] = 3.0          # far above the training cap
        processor.prepare_weather_data(train, 'degraded', horizon=168, fold_idx=0, split='train')
        cap = np.percentile(train['Solar Radiation'], 99.5)
        assert processor.degradation_params['solar_cap'] == pytest.approx(cap)
        X_deg = processor.prepare_weather_data(test, 'degraded', horizon=168, fold_idx=0, split='test')
        assert X_deg['Solar Radiation'].max() <= cap + 1e-12

    def test_seeds_unique_across_horizons(self):
        """fold 18 at h=6 and fold 0 at h=24 had the same seed with the old
        formula (base_seed + fold_idx + horizon); they must differ now."""
        processor = self._make_weather_processor()
        df = self._make_test_df(n_rows=6)
        out = []
        for horizon, fold_idx in [(6, 18), (24, 0)]:
            processor.prepare_weather_data(df, 'degraded', horizon=horizon, fold_idx=fold_idx, split='train')
            X = processor.prepare_weather_data(df, 'degraded', horizon=horizon, fold_idx=fold_idx, split='test')
            out.append(X['Temperature'].values - df['Temperature'].values)
        assert not np.allclose(out[0], out[1])

    def test_dry_window_with_phase_correction(self):
        """Cold, dry test window: precipitation columns come out as all-zero
        before the phase correction moves false-alarm rain into snow. Must not
        fail on dtype (pandas >= 3 rejects floats in an int column)."""
        processor = self._make_weather_processor()
        df = self._make_test_df(n_rows=6)
        df['Temperature'] = -5.0
        for fold_idx in range(50):
            processor.prepare_weather_data(df, 'degraded', horizon=6, fold_idx=fold_idx, split='train')
            X = processor.prepare_weather_data(df, 'degraded', horizon=6, fold_idx=fold_idx, split='test')
            assert X['Rainfall'].dtype.kind == 'f' and X['Snowfall'].dtype.kind == 'f'

    def test_one_false_alarm_draw_per_hour(self):
        """Dry test window, rain AND snow columns: the share of hours with
        forecast precipitation equals P(false alarm | dry) once, not
        1 - (1 - P)^2 from separate draws per column. The wet-hour share
        comes from the training fold."""
        processor = self._make_weather_processor()
        horizon, n_folds = 24, 400
        train = self._make_test_df(n_rows=720)
        train['Rainfall'] = np.where(np.arange(720) % 20 == 0, 1.0, 0.0)   # 5 % wet hours
        test = self._make_test_df(n_rows=horizon)                         # completely dry
        shares = []
        for fold_idx in range(n_folds):
            processor.prepare_weather_data(train, 'degraded', horizon=horizon, fold_idx=fold_idx, split='train')
            X = processor.prepare_weather_data(test, 'degraded', horizon=horizon, fold_idx=fold_idx, split='test')
            shares.append(((X['Rainfall'] > 0) | (X['Snowfall'] > 0)).mean())
        assert processor.degradation_params['wet_fraction'] == pytest.approx(0.05)
        expected = np.mean([precipitation_detection_rates(lt, 0.05)[1] for lt in range(1, horizon + 1)])
        assert np.mean(shares) == pytest.approx(expected, abs=0.006)

    def test_dry_training_fold_gives_no_false_alarms(self):
        """No wet hour in the training fold: wet-hour share 0, no false alarms."""
        processor = self._make_weather_processor()
        df = self._make_test_df(n_rows=48)
        for fold_idx in range(20):
            processor.prepare_weather_data(df, 'degraded', horizon=48, fold_idx=fold_idx, split='train')
            X = processor.prepare_weather_data(df, 'degraded', horizon=48, fold_idx=fold_idx, split='test')
            assert (X['Rainfall'] == 0).all() and (X['Snowfall'] == 0).all()

    # ------------------------------------------------------------------
    # Noise-magnitude sensitivity (degraded scenarios with noise_scale)
    # ------------------------------------------------------------------
    def test_degraded_scenarios_in_config(self):
        """'degraded' is the calibrated model (1.0); every degraded scenario is
        in weather_scenarios and recognised by is_degraded()."""
        config = ForecastConfig()
        assert config.degradation_scales['degraded'] == 1.0
        for scenario in config.degradation_scales:
            assert config.is_degraded(scenario)
            assert scenario in config.weather_scenarios
        assert not config.is_degraded('clean_only')
        assert not config.is_degraded('all_weather')

    def test_degraded_is_unscaled_model(self):
        """'degraded' passes noise_scale=1.0: its output equals
        degrade_weather_dataset with the default noise_scale (the calibrated
        error model, same seed)."""
        processor = self._make_weather_processor()
        horizon, fold_idx = 24, 5
        df = self._make_test_df(n_rows=horizon)
        df['Rainfall'] = np.tile([0.0, 2.0], horizon // 2)
        processor.prepare_weather_data(df, 'degraded', horizon=horizon, fold_idx=fold_idx, split='train')
        X = processor.prepare_weather_data(df, 'degraded', horizon=horizon, fold_idx=fold_idx, split='test')
        expected = degrade_weather_dataset(
            df[X.columns], horizon, processor.degradation_params,
            processor.config.weather_degradation_mapping,
            seed=processor.config.degradation_seed + 10000 * horizon + fold_idx,
            lead_times=np.arange(1, horizon + 1),
            temp_col='Temperature', rain_col='Rainfall', snow_col='Snowfall',
        )
        pd.testing.assert_frame_equal(X, expected)

    def test_noise_scales_share_random_numbers(self):
        """All degraded scenarios of a (horizon, fold) use the same random
        numbers; only the error magnitude differs. Temperature errors
        (additive Gaussian, no clipping) are proportional to the scale, and
        the wet/dry pattern of precipitation (event detection, not scaled) is
        the same in every scenario."""
        processor = self._make_weather_processor()
        horizon = 48
        df = self._make_test_df(n_rows=horizon)
        df['Temperature'] = 25.0                          # far from the 2 °C rain/snow threshold
        df['Rainfall'] = np.tile([0.0, 2.0], horizon // 2)
        out = {}
        for scenario in processor.config.degradation_scales:
            processor.prepare_weather_data(df, scenario, horizon=horizon, fold_idx=3, split='train')
            out[scenario] = processor.prepare_weather_data(
                df, scenario, horizon=horizon, fold_idx=3, split='test'
            )
        err_1 = out['degraded']['Temperature'] - df['Temperature']
        wet_1 = (out['degraded']['Rainfall'] + out['degraded']['Snowfall']) > 0
        for scenario, scale in processor.config.degradation_scales.items():
            err = out[scenario]['Temperature'] - df['Temperature']
            np.testing.assert_allclose(err, scale * err_1, rtol=1e-9, atol=1e-9)
            wet = (out[scenario]['Rainfall'] + out[scenario]['Snowfall']) > 0
            assert (wet == wet_1).all(), f"{scenario}: precipitation events differ from 'degraded'"

    def test_negative_noise_scale_raises(self):
        with pytest.raises(ValueError):
            degrade_weather_forecast(15.0, 'temperature', 24,
                                     rng=np.random.default_rng(0), noise_scale=-0.5)

    def test_fold_idx_out_of_range(self):
        processor = self._make_weather_processor()
        df = self._make_test_df()
        processor.prepare_weather_data(df, 'degraded', horizon=24, fold_idx=0, split='train')
        with pytest.raises(ValueError):
            processor.prepare_weather_data(df, 'degraded', horizon=24, fold_idx=10000, split='test')

    def test_rain_snow_phase_correction_uses_config_columns(self):
        """Phase correction with non-Seoul column names (London/Washington)."""
        config = ForecastConfig()
        config.dataset_name = "test"
        config.weather_covariates = ['temperature_c', 'solar_radiation_wm2', 'rainfall_mm', 'snowfall_cm']
        config.weather_degradation_mapping = {
            'temperature_c': 'temperature', 'solar_radiation_wm2': 'solar_radiation',
            'rainfall_mm': 'precipitation', 'snowfall_cm': 'precipitation',
        }
        config.holiday_col = None
        config.season_col = None
        config.rain_col = 'rainfall_mm'
        config.snow_col = 'snowfall_cm'
        config.degradation_seed = 42
        processor = WeatherProcessor(config)
        n = 48
        df = pd.DataFrame({'temperature_c': np.full(n, 20.0),      # warm: snow must become rain
                           'solar_radiation_wm2': np.full(n, 100.0),
                           'rainfall_mm': np.zeros(n), 'snowfall_cm': np.full(n, 5.0)})
        processor.prepare_weather_data(df, 'degraded', horizon=n, fold_idx=0, split='train')
        X = processor.prepare_weather_data(df, 'degraded', horizon=n, fold_idx=0, split='test')
        warm = X['temperature_c'] > 2
        assert warm.any()
        assert (X.loc[warm, 'snowfall_cm'] == 0).all()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
