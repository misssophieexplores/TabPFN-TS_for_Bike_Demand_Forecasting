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

# Noise-magnitude sensitivity scenarios: not in the default config since
# 2 Oct 2026, but the mechanism stays available and tested.
SENSITIVITY_SCALES = {"degraded": 1.0, "degraded_x050": 0.5, "degraded_x150": 1.5}


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
        """Helper: minimal config + WeatherProcessor that doesn't need real data.
        These tests cover the "literature" error model (no timestamps needed);
        the measured NWP model is tested in TestMeasuredNWPModel."""
        config = ForecastConfig()
        config.dataset_name = "test"
        config.degradation_model = "literature"
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
        # every degraded scenario, including optional sensitivity ones
        processor.config.degradation_scales = dict(SENSITIVITY_SCALES)
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
        """'degraded' is the calibrated model (1.0) and, since 2 Oct 2026, the
        only degraded scenario by default (no sensitivity runs); every
        degraded scenario is in weather_scenarios and recognised by
        is_degraded()."""
        config = ForecastConfig()
        assert config.degradation_scales == {'degraded': 1.0}
        assert config.weather_scenarios == ['all_weather', 'clean_only', 'degraded']
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
        processor.config.degradation_scales = dict(SENSITIVITY_SCALES)
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
        config.degradation_model = "literature"
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


# ======================================================================
# Measured NWP error model (config.degradation_model = "nwp_measured")
# ======================================================================
import importlib
import json
from weather.nwp_error_model import (
    load_error_model, run_init_and_leads, fresh_start_and_leads, to_utc, ar1, MAX_LEAD,
)

CITY_MODULES = {"seoul": "config_seoul", "london": "config_london", "washington": "config_washington"}


def _city_config(city):
    return importlib.import_module(CITY_MODULES[city]).get_config()


def _synthetic_city_data(config, start, n_hours, seed=0, rain_every=None):
    """Clean weather in the city's own columns, with its date column."""
    rng = np.random.default_rng(seed)
    m = config.weather_degradation_mapping
    t = pd.date_range(start, periods=n_hours, freq="h")
    df = pd.DataFrame({config.date_col: t})
    for col, typ in m.items():
        if typ == "temperature":
            df[col] = 12 + 6 * np.sin(2 * np.pi * (t.hour - 9) / 24) + rng.normal(0, 1, n_hours)
        elif typ == "humidity":
            df[col] = rng.uniform(40, 90, n_hours)
        elif typ == "wind_speed":
            df[col] = rng.uniform(0.5, 6, n_hours)
        elif typ == "solar_radiation":
            df[col] = np.where((t.hour >= 7) & (t.hour <= 17), 300.0, 0.0)
        elif typ == "visibility":
            df[col] = rng.uniform(2, 30, n_hours)
        elif typ == "precipitation":
            df[col] = 0.0
    if rain_every:
        df[config.rain_col] = np.where(np.arange(n_hours) % rain_every == 0, 1.5, 0.0)
    for col in [config.holiday_col, config.season_col]:
        if col:
            df[col] = 0
    return df


def _processor(city, fresh=True, remove_bias=True):
    config = _city_config(city)
    config.nwp_fresh_forecast = fresh
    config.nwp_remove_bias = remove_bias
    config.weather_covariates = list(config.weather_degradation_mapping) + [
        c for c in [config.holiday_col, config.season_col] if c]
    return WeatherProcessor(config)


class TestMeasuredNWPModel:
    """Per-city error model measured from real ECMWF IFS HRES forecasts."""

    def test_default_model_and_city_settings(self):
        cfg = ForecastConfig()
        assert cfg.degradation_model == "nwp_measured"
        assert cfg.nwp_fresh_forecast and cfg.nwp_remove_bias and cfg.nwp_seasonal_rain
        assert cfg.nwp_rain_frequency_unbiased
        assert cfg.degradation_label() == "nwp_measured(fresh,no_bias,seasonal_rain,rain_freq_unbiased)"
        for city in CITY_MODULES:
            config = _city_config(city)
            assert config.timezone
            model = load_error_model(config.nwp_calibration_file)
            assert model.tqw_errors.shape[1:] == (MAX_LEAD + 1, 3)
            assert np.isfinite(model.tqw_errors[:, 1:, :]).all(axis=(1, 2)).sum() > 500

    def test_run_and_lead_times(self):
        """Issue time = first test hour - 1 h; newest 00/12 UTC run that is
        6 h old by then. Lead of the first hour 7..18 h, for every start hour."""
        cases = {  # first test hour (UTC) -> (run start, first lead)
            "2015-06-01 12:00": ("2015-06-01 00:00", 12),
            "2015-06-01 19:00": ("2015-06-01 12:00", 7),
            "2015-06-01 18:00": ("2015-06-01 00:00", 18),
            "2015-06-01 03:00": ("2015-05-31 12:00", 15),
        }
        for first, (init, lead0) in cases.items():
            times = pd.date_range(first, periods=24, freq="h")
            got_init, leads = run_init_and_leads(times)
            assert got_init == pd.Timestamp(init)
            assert leads[0] == lead0 and (np.diff(leads) == 1).all()
        for h in range(24):
            times = pd.date_range(pd.Timestamp("2016-01-10") + pd.Timedelta(hours=h), periods=168, freq="h")
            _, leads = run_init_and_leads(times)
            assert 7 <= leads[0] <= 18 and leads[-1] <= MAX_LEAD

    def test_seasonal_rain_parameters(self):
        """Rain error rates per time of year: 12 monthly bins, linear
        interpolation by day of year (weights sum to 1, a bin centre gets its
        own values); year-round values when switched off. Seoul rain is
        measured against the station, London and Washington against ERA5.
        Washington's forecast misses summer rain more often than winter rain."""
        for city in CITY_MODULES:
            model = load_error_model(_city_config(city).nwp_calibration_file)
            assert model.season_doy is not None and model.miss_rate_seasonal.shape == (12, MAX_LEAD + 1)
            for doy in [1, 15, 100, 196, 300, 365]:
                w = model.season_weights(doy)
                assert w.sum() == pytest.approx(1.0) and (w >= 0).all() and (w > 0).sum() <= 2
            k = 6                                                      # 15 July
            np.testing.assert_allclose(model.rain_params(model.season_doy[k])[0],
                                       model.miss_rate_seasonal[k])
            for got, year_round in zip(model.rain_params(196, seasonal=False),
                                       (model.miss_rate, model.far, model.hit_mean_log, model.hit_sd_log)):
                assert got is year_round
            for arr in (model.miss_rate_seasonal, model.far_seasonal, model.hit_sd_log_seasonal):
                assert np.isfinite(arr[:, 1:169]).all()
            ref = json.loads(model.summary)["precipitation"]["reference"]
            assert ("NOAA ISD" in ref) if city == "seoul" else ("era5" in ref.lower())
        dc = load_error_model(_city_config("washington").nwp_calibration_file)
        assert dc.rain_params(196)[0][24] > dc.rain_params(15)[0][24]

    def test_rain_frequency_option(self):
        """nwp_rain_frequency_unbiased (default on): false alarms equal misses,
        so the degraded data are wet about as often as the clean data;
        off: the measured false-alarm ratio (Seoul's raw forecast is wet
        about twice as often as the station)."""
        assert ForecastConfig().nwp_rain_frequency_unbiased
        ratio = {}
        for unbiased in (False, True):
            proc = _processor("seoul")
            proc.config.nwp_rain_frequency_unbiased = unbiased
            n_obs = n_fc = 0
            for fold in range(120):
                start = pd.Timestamp("2018-06-01") + pd.Timedelta(hours=int(31 * fold))
                df = _synthetic_city_data(proc.config, start, 720 + 48, seed=fold, rain_every=9)
                train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
                proc.prepare_weather_data(train, "degraded", horizon=48, fold_idx=fold, split="train")
                X = proc.prepare_weather_data(test, "degraded", horizon=48, fold_idx=fold, split="test")
                assert proc.last_degradation_info["rain_frequency_unbiased"] == unbiased
                n_obs += int((test["Rainfall"] > 0).sum())
                n_fc += int((X[["Rainfall", "Snowfall"]] > 0).any(axis=1).sum())
            ratio[unbiased] = n_fc / n_obs
        assert ratio[True] == pytest.approx(1.0, abs=0.2)
        assert ratio[False] > 1.4

    def test_rain_capped_at_training_maximum(self):
        """Degraded rain never exceeds the training fold's maximum of the
        column, or the hour's measured amount if that is larger (the
        lognormal amount error has no upper limit)."""
        proc = _processor("seoul")
        cfg = proc.config
        n_below = n_wet = 0
        for fold in range(60):
            df = _synthetic_city_data(cfg, pd.Timestamp("2018-07-01") + pd.Timedelta(hours=37 * fold),
                                      720 + 168, seed=fold, rain_every=3)
            df.loc[df.index < 720, "Rainfall"] = np.where(df.index[df.index < 720] % 3 == 0, 2.0, 0.0)
            df.loc[df.index == 720 + 30, "Rainfall"] = 10.0             # above the training maximum
            train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
            proc.prepare_weather_data(train, "degraded", horizon=168, fold_idx=fold, split="train")
            assert proc.degradation_params["precip_max"]["Rainfall"] == 2.0
            X = proc.prepare_weather_data(test, "degraded", horizon=168, fold_idx=fold, split="test")
            cap = np.maximum(2.0, test["Rainfall"].to_numpy())
            # rain/snow phase correction may move amounts between the columns
            tot = X[["Rainfall", "Snowfall"]].max(axis=1).to_numpy()
            assert (tot <= cap + 1e-9).all()
            wet = (test["Rainfall"] > 0).to_numpy() & (tot > 0)
            n_wet += int(wet.sum()); n_below += int((tot[wet] < cap[wet] - 1e-9).sum())
        assert n_below > 0.3 * n_wet          # the cap does not flatten every hour

    def test_city_seed_term(self):
        """Measured model: the three cities get different random numbers for
        the same horizon and fold; the same city always the same."""
        from weather.weather_processor import city_seed_term
        terms = {c: city_seed_term(c) for c in CITY_MODULES}
        assert len(set(terms.values())) == 3 and city_seed_term(None) == 0
        out = []
        for name in ("seoul", "seoul", "london"):
            proc = _processor("seoul")
            proc.config.dataset_name = name
            df = _synthetic_city_data(proc.config, "2018-05-01", 720 + 24)
            train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
            proc.prepare_weather_data(train, "degraded", horizon=24, fold_idx=3, split="train")
            X = proc.prepare_weather_data(test, "degraded", horizon=24, fold_idx=3, split="test")
            out.append(X["Temperature"].to_numpy())
        np.testing.assert_array_equal(out[0], out[1])
        assert not np.allclose(out[0], out[2])

    def test_fresh_forecast_leads(self):
        """Fresh forecast: lead times 1..n for any start hour; errors replayed
        from the 00/12 UTC run closest to the issue time of day (tie: the
        earlier one)."""
        cases = {  # first test hour (UTC) -> replay start hour
            "2015-06-01 01:00": 0, "2015-06-01 06:00": 0, "2015-06-01 07:00": 0,
            "2015-06-01 08:00": 12, "2015-06-01 13:00": 12, "2015-06-01 19:00": 12,
            "2015-06-01 20:00": 0, "2015-06-02 00:00": 0,
        }
        for first, hour in cases.items():
            times = pd.date_range(first, periods=48, freq="h")
            issue, start_hour, leads = fresh_start_and_leads(times)
            assert issue == times[0] - pd.Timedelta(hours=1)
            assert start_hour == hour, first
            assert list(leads) == list(range(1, 49))

    def test_local_time_to_utc(self):
        t = ["2015-07-01 12:00", "2015-01-15 12:00"]
        assert list(to_utc(t, "Asia/Seoul").hour) == [3, 3]
        assert list(to_utc(t, "Europe/London").hour) == [11, 12]
        assert list(to_utc(t, "America/New_York").hour) == [16, 17]
        # repeated autumn hour (read as daylight-saving time = the first
        # occurrence, kept by load_and_prepare_data) and missing spring hour
        assert to_utc(["2015-10-25 01:00"], "Europe/London")[0] == pd.Timestamp("2015-10-25 00:00")
        assert to_utc(["2012-11-04 01:00"], "America/New_York")[0] == pd.Timestamp("2012-11-04 05:00")
        assert len(to_utc(["2015-03-29 01:00"], "Europe/London")) == 1

    # Test windows across a clock change: naive local hours as in the data
    # after load_and_prepare_data (repeated autumn hour once, non-existent
    # spring hour present). (city, time zone, first test hour, change)
    DST_WINDOWS = [
        ("london", "Europe/London", "2016-03-26 22:00", "spring"),          # 27 Mar 2016 01:00
        ("london", "Europe/London", "2016-10-30 00:00", "autumn"),          # v7 fold
        ("washington", "America/New_York", "2012-03-10 22:00", "spring"),   # 11 Mar 2012 02:00
        ("washington", "America/New_York", "2012-11-04 00:00", "autumn"),   # v7 fold
    ]

    def test_leads_count_rows_across_clock_change(self):
        """Lead time of row i = i + 1 (fresh) or run age + i + 1 (newest run),
        also when the window spans a clock change (in UTC: a 2-h step in
        autumn, a repeated hour in spring). UTC only selects the run: issue
        time = UTC of the first hour - 1 h."""
        for _, tz, first, change in self.DST_WINDOWS:
            for n in (6, 24, 48, 168):
                utc = to_utc(pd.date_range(first, periods=n, freq="h"), tz)
                step = np.diff(utc) / pd.Timedelta(hours=1)
                assert (step == (2 if change == "autumn" else 0)).sum() == 1
                assert (step == 1).sum() == n - 2
                issue, _, leads = fresh_start_and_leads(utc)
                assert issue == utc[0] - pd.Timedelta(hours=1)
                assert list(leads) == list(range(1, n + 1)), (tz, first, n)
                init, leads = run_init_and_leads(utc)
                age = int((issue - init) / pd.Timedelta(hours=1))
                assert init.hour in (0, 12) and 6 <= age <= 17
                assert list(leads) == list(range(age + 1, age + n + 1)), (tz, first, n)

    def test_degrade_across_clock_change(self):
        """End to end (h = 24, fresh and newest run): the replayed temperature
        errors are those of lead times 1..24, or run age + 1..24, also when
        the window spans a clock change."""
        for city, tz, first, _ in self.DST_WINDOWS:
            for fresh in (True, False):
                proc = _processor(city, fresh)
                cfg = proc.config
                df = _synthetic_city_data(cfg, pd.Timestamp(first) - pd.Timedelta(hours=720), 744)
                train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
                proc.prepare_weather_data(train, "degraded", horizon=24, fold_idx=0, split="train")
                X = proc.prepare_weather_data(test, "degraded", horizon=24, fold_idx=0, split="test")
                info = proc.last_degradation_info
                start = pd.Timestamp(info["forecast_start_utc"])
                issue = to_utc(test[cfg.date_col].iloc[:1], tz)[0] - pd.Timedelta(hours=1)
                age = 0 if fresh else int((issue - start) / pd.Timedelta(hours=1))
                leads = age + np.arange(1, 25)
                assert (info["lead_first"], info["lead_last"]) == (leads[0], leads[-1])
                model = load_error_model(cfg.nwp_calibration_file)
                k = np.flatnonzero(model.run_init == pd.Timestamp(info["replayed_run_utc"]))[0]
                cand, _ = model.candidate_runs(info["start_hour_utc"], start.dayofyear, leads,
                                               cfg.nwp_season_window_days)
                expected = model.tqw_errors[k, leads, 0] - model.tqw_errors[cand][:, leads, 0].mean(axis=0)
                tcol = [c for c, t in cfg.weather_degradation_mapping.items() if t == "temperature"][0]
                np.testing.assert_allclose(X[tcol] - test[tcol], expected, atol=1e-9)

    def test_replays_real_errors_of_one_run(self):
        """Temperature, humidity and wind get the actual errors of one ECMWF run
        with the matching start hour and season, minus the average error of
        all such runs (default), or unchanged (nwp_remove_bias=False). The run
        is reported in last_degradation_info."""
        for city in CITY_MODULES:
            for fresh, remove_bias in [(True, True), (False, False), (True, False)]:
                proc = _processor(city, fresh, remove_bias)
                cfg = proc.config
                df = _synthetic_city_data(cfg, "2016-07-10 05:00", 720 + 48)
                train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
                proc.prepare_weather_data(train, "degraded", horizon=48, fold_idx=1, split="train")
                X = proc.prepare_weather_data(test, "degraded", horizon=48, fold_idx=1, split="test")
                info = proc.last_degradation_info
                assert info["fresh_forecast"] == fresh and info["remove_bias"] == remove_bias
                model = load_error_model(cfg.nwp_calibration_file)
                k = np.flatnonzero(model.run_init == pd.Timestamp(info["replayed_run_utc"]))[0]
                times = to_utc(test[cfg.date_col], cfg.timezone)
                if fresh:
                    start, hour, leads = fresh_start_and_leads(times)
                    assert list(leads) == list(range(1, 49))
                else:
                    start, leads = run_init_and_leads(times)
                    hour = start.hour
                    assert 7 <= leads[0] <= 18
                assert str(start) == info["forecast_start_utc"] and hour == info["start_hour_utc"]
                assert model.run_init[k].hour == hour
                d = abs(model.run_init[k].dayofyear - start.dayofyear)
                assert min(d, 366 - d) <= info["season_window_days"]
                expected = model.tqw_errors[k, leads, 0]
                if remove_bias:
                    cand, _ = model.candidate_runs(hour, start.dayofyear, leads, cfg.nwp_season_window_days)
                    assert len(cand) == info["n_candidate_runs"]
                    expected = expected - model.tqw_errors[cand][:, leads, 0].mean(axis=0)
                tcol = [c for c, t in cfg.weather_degradation_mapping.items() if t == "temperature"][0]
                np.testing.assert_allclose(X[tcol] - test[tcol], expected, atol=1e-9)

    def test_needs_timestamps(self):
        proc = _processor("london")
        df = _synthetic_city_data(proc.config, "2016-01-01", 744)
        proc.prepare_weather_data(df.iloc[:720], "degraded", horizon=24, fold_idx=0, split="train")
        with pytest.raises(ValueError):
            proc.degrade_dataframe(df.iloc[720:][list(proc.config.weather_degradation_mapping)], 24, 0)

    def test_noise_scales_share_random_numbers(self):
        """Same run and same random numbers in every degraded scenario:
        temperature errors proportional to the scale, identical rain events
        and identical visibility-below-cap decisions."""
        proc = _processor("seoul")
        cfg = proc.config
        cfg.degradation_scales = dict(SENSITIVITY_SCALES)
        df = _synthetic_city_data(cfg, "2018-03-01", 720 + 168, rain_every=7)
        df.loc[df.index % 3 == 0, "Visibility"] = 20.0            # at the Seoul cap
        train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
        out = {}
        for scen in cfg.degradation_scales:
            proc.prepare_weather_data(train, scen, horizon=168, fold_idx=4, split="train")
            out[scen] = proc.prepare_weather_data(test, scen, horizon=168, fold_idx=4, split="test")
        e1 = out["degraded"]["Temperature"] - test["Temperature"]
        wet1 = (out["degraded"][["Rainfall", "Snowfall"]] > 0).any(axis=1)
        below1 = out["degraded"]["Visibility"] < 20.0 - 1e-9
        for scen, scale in cfg.degradation_scales.items():
            np.testing.assert_allclose(out[scen]["Temperature"] - test["Temperature"], scale * e1, atol=1e-9)
            assert ((out[scen][["Rainfall", "Snowfall"]] > 0).any(axis=1) == wet1).all()
            at = test["Visibility"] >= 20.0
            assert ((out[scen]["Visibility"] < 20.0 - 1e-9)[at] == below1[at]).all() or scale == 0

    def test_bounds(self):
        """Humidity 0..100, wind >= 0, solar 0 at night and <= training cap,
        visibility <= the city's cap (Seoul 20 km, Washington 16 km) or the
        training maximum (London)."""
        for city in CITY_MODULES:
            proc = _processor(city)
            cfg = proc.config
            m = cfg.weather_degradation_mapping
            col = {t: c for c, t in m.items() if t != "precipitation"}
            cfg.degradation_scales = dict(SENSITIVITY_SCALES)
            df = _synthetic_city_data(cfg, "2016-11-01", 720 + 168)
            if city == "seoul":
                df[col["visibility"]] = np.minimum(df[col["visibility"]], 20.0)
            if city == "washington":
                df[col["visibility"]] = np.minimum(df[col["visibility"]], 16.0)
            train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
            for fold in range(5):
                proc.prepare_weather_data(train, "degraded_x150", horizon=168, fold_idx=fold, split="train")
                X = proc.prepare_weather_data(test, "degraded_x150", horizon=168, fold_idx=fold, split="test")
                assert X[col["humidity"]].between(0, 100).all()
                assert (X[col["wind_speed"]] >= 0).all()
                night = test[col["solar_radiation"]] <= 0
                assert (X.loc[night, col["solar_radiation"]] == 0).all()
                assert X[col["solar_radiation"]].max() <= proc.degradation_params["solar_cap"] + 1e-9
                vmax = {"seoul": 20.0, "washington": 16.0}.get(city, proc.degradation_params["visibility_max"])
                assert X[col["visibility"]].max() <= vmax + 1e-9

    def test_reproduces_measured_error_statistics(self):
        """Over many test windows the simulated errors match the calibration:
        precipitation miss rate and false-alarm ratio (of the time of year), visibility falling
        below the Seoul cap, and the temperature error: average ~0 with the
        lean removed (default), the measured Seoul lean (about -1 C or more)
        with nwp_remove_bias=False; spread as in the replayed runs."""
        for remove_bias in (True, False):
            proc = _processor("seoul", fresh=True, remove_bias=remove_bias)
            cfg = proc.config
            cfg.nwp_rain_frequency_unbiased = False      # measured false-alarm ratio
            model = load_error_model(cfg.nwp_calibration_file)
            horizon, n_folds = 24, 300
            t_err, leads_all, exp_var, exp_miss, exp_far = [], [], [], [], []
            H = M = F = 0
            vis_below, vis_at = 0, 0
            for fold in range(n_folds):
                start = pd.Timestamp("2018-01-01") + pd.Timedelta(hours=int(29 * fold))
                df = _synthetic_city_data(cfg, start, 720 + horizon, seed=fold, rain_every=6)
                df["Visibility"] = 20.0
                train, test = df.iloc[:720], df.iloc[720:].reset_index(drop=True)
                proc.prepare_weather_data(train, "degraded", horizon=horizon, fold_idx=fold, split="train")
                X = proc.prepare_weather_data(test, "degraded", horizon=horizon, fold_idx=fold, split="test")
                info = proc.last_degradation_info
                leads = np.arange(info["lead_first"], info["lead_last"] + 1)
                assert info["lead_first"] == 1 and len(leads) == horizon
                t_err.append((X["Temperature"] - test["Temperature"]).to_numpy())
                leads_all.append(leads)
                fs = pd.Timestamp(info["forecast_start_utc"])
                assert info["seasonal_rain"]
                miss_r, far_r, _, _ = model.rain_params(fs.dayofyear)
                exp_miss.append(miss_r[leads].mean()); exp_far.append(far_r[leads].mean())
                cand, _ = model.candidate_runs(info["start_hour_utc"], fs.dayofyear, leads, cfg.nwp_season_window_days)
                exp_var.append(model.tqw_errors[cand][:, leads, 0].var(axis=0).mean())
                ow = test["Rainfall"] > 0
                fw = (X[["Rainfall", "Snowfall"]] > 0).any(axis=1)
                H += int((ow & fw).sum()); M += int((ow & ~fw).sum()); F += int((~ow & fw).sum())
                vis_below += int((X["Visibility"] < 20.0 - 1e-9).sum()); vis_at += len(X)
            leads_all = np.concatenate(leads_all)
            assert M / (H + M) == pytest.approx(np.mean(exp_miss), abs=0.06)
            assert F / (H + F) == pytest.approx(np.mean(exp_far), abs=0.08)
            assert vis_below / vis_at == pytest.approx(model.vis_at_cap_p_below, abs=0.06)
            e = np.concatenate(t_err)
            if remove_bias:
                assert abs(e.mean()) < 0.25
                # spread around the run-pool average, as in the replayed runs
                assert e.std() == pytest.approx(np.sqrt(np.mean(exp_var)), rel=0.2)
            else:
                assert e.mean() < -0.5

    def test_ar1_persistence(self):
        z = ar1(200000, 0.8, np.random.default_rng(1))
        assert np.corrcoef(z[:-1], z[1:])[0, 1] == pytest.approx(0.8, abs=0.01)
        assert z.std() == pytest.approx(1.0, abs=0.01)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
