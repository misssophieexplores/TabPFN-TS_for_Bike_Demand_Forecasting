"""
Weather Forecast Degradation Module

Simulate realistic numerical weather prediction (NWP) forecast errors for
machine learning model evaluation under operational conditions.

Error growth functions are calibrated to published NWP verification statistics.
See weather_methodology.md for detailed documentation and validation evidence.

Version: 1.1.0
"""

import numpy as np


def degrade_weather_forecast(actual_value, variable_type, horizon_hours, 
                            solar_cap=None, rng=None):
    """
    Apply forecast uncertainty to observed weather variables based on lead time.
    
    Simulates realistic forecast errors by adding noise calibrated to operational
    NWP verification statistics. Error magnitudes increase with forecast horizon.
    
    Parameters
    ----------
    actual_value : float
        Observed (true) weather value
    variable_type : str
        Weather variable type. Must be one of:
        - 'temperature' : 2-meter air temperature (°C)
        - 'humidity' : 2-meter relative humidity (%)
        - 'wind_speed' : 10-meter wind speed (m/s)
        - 'solar_radiation' : Solar irradiance (MJ/m² hourly)
        - 'precipitation' : Precipitation amount (mm)
    horizon_hours : int
        Forecast lead time in hours. Typical values: 6, 24, 48, 168
    solar_cap : float, optional
        Maximum physically plausible solar radiation (same units as actual_value).
        Compute as: np.percentile(training_data['solar_radiation'], 99.5)
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, creates new Generator.
        For reproducibility: rng = np.random.default_rng(seed=42)
    
    Returns
    -------
    float
        Weather value with forecast uncertainty applied
    
    Notes
    -----
    **Validation Status** (see weather_methodology.md for details):
    
    - **Temperature**: VALIDATED - ECMWF TM 918 (2024), meteoblue (2017)
      σ(24h) = 1.2°C, σ(168h) = 3.8°C
      
    - **Wind Speed**: VALIDATED - meteoblue (2017), ECMWF TM 918
      σ(24h) = 2.0 m/s, σ(168h) = 3.5 m/s
      
    - **Humidity**: PARTIALLY VALIDATED - Kartsios et al. (2024) GFS/Africa
      σ(24h) = 13.6%, σ(168h) = 16.9%
      
    - **Solar Radiation**: VALIDATED - Kleissl (2013) Table 10.2
      Relative MAE: 18.6% @ 24h, 40.2% @ 168h
      
    - **Precipitation (detection)**: VALIDATED - Sukovich et al. (2014)
      Day 1: POD≈65%, FAR≈35%; Day 2: POD≈55%, FAR≈45%
      
    - **Precipitation (magnitude)**: CONSERVATIVE ESTIMATE - ensemble spread theory
      CV growth: 0.15%/h (half the theoretical bound)
      
    - **Visibility**: ECMWF Forecast User Guide (Owens & Hewson, 2018), Section 9.4
      Constant CV = 25%; horizon-independent (skill non-monotonic with lead time)
    
    **Error Models:**
    
    - **Additive Gaussian** (temperature, humidity, wind): X' = X + ε
    - **Heteroscedastic Gaussian** (solar): σ ∝ X (errors scale with intensity)
    - **Multiplicative lognormal** (precipitation magnitude): X' = X × M
    - **Binary detection** (precipitation events): false alarms + misses
    
    **Limitations:**
    
    - Magnitude errors only; timing/spatial displacement not modeled
    - Independent errors across variables (actual forecasts are correlated)
    - Linear error growth (slight nonlinearity beyond 5-7 days not captured)
    
    For detailed methodology, validation evidence, and references, see:
    weather_methodology.md
    
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Reproducible degradation
    >>> rng = np.random.default_rng(seed=12345)
    >>> temp_degraded = degrade_weather_forecast(
    ...     actual_value=15.3,
    ...     variable_type='temperature',
    ...     horizon_hours=24,
    ...     rng=rng
    ... )
    
    >>> # Solar radiation requires cap parameter
    >>> solar_cap = np.percentile(train_data['solar_radiation'], 99.5)
    >>> rng = np.random.default_rng(seed=12345)
    >>> solar_degraded = degrade_weather_forecast(
    ...     actual_value=2.1,
    ...     variable_type='solar_radiation',
    ...     horizon_hours=48,
    ...     solar_cap=solar_cap,
    ...     rng=rng
    ... )
    """
    
    # Use new Generator with random entropy if rng not provided
    # This ensures consistent API (Generator object) rather than np.random module
    if rng is None:
        rng = np.random.default_rng()
    
    h = horizon_hours
    
    if variable_type == 'temperature':
        # ECMWF TM 918 (2024) + meteoblue (2017)
        # σ(24h) ≈ 1.2°C, σ(168h) ≈ 3.8°C
        sigma = 0.77 + 0.0181 * h  # °C
        noise = rng.normal(0, sigma)
        return actual_value + noise
    
    elif variable_type == 'humidity':
        # Kartsios et al. (2024), Acta Geophysica: GFS 2m RH RMSE over Africa
        # 12h→180h: 13.58%→16.94% (NCEP/GFS, June 2018-May 2020)
        # Formula calibrated to match observed range
        sigma = 13.0 + 0.023 * h  # %-points
        noise = rng.normal(0, sigma)
        degraded = actual_value + noise
        # Enforce physical bounds
        return np.clip(degraded, 0, 100)
    
    elif variable_type == 'wind_speed':
        # meteoblue (2017) + ECMWF TM 918
        # σ(24h) ≈ 2.0 m/s, σ(168h) ≈ 3.5 m/s
        sigma = 1.8 + 0.010 * h  # m/s
        noise = rng.normal(0, sigma)
        degraded = actual_value + noise
        # Truncate at zero (minor negative bias but avoids large positive bias)
        return max(0, degraded)
    
    elif variable_type == 'solar_radiation':
        # No radiation at night
        if actual_value <= 0:
            return 0
        
        # Heteroscedastic additive Gaussian: σ proportional to value
        # Kleissl (2013) Table 10.2: 1-day MAE 18-36%, 7-day 23-46%
        # Hourly growth (0.15%/h = 3.6%/day) interpolates verified endpoints
        relative_mae_pct = 15 + 0.15 * h
        
        # Convert relative MAE to absolute σ for Gaussian noise
        # Normal distribution: σ ≈ 1.253 × MAE
        sigma = 1.253 * (relative_mae_pct / 100) * actual_value
        
        noise = rng.normal(0, sigma)
        degraded = actual_value + noise
        
        # Physical cap prevents unrealistic values
        if solar_cap is None:
            raise ValueError(
                "solar_cap required for solar_radiation. "
                "Compute as: np.percentile(train_data['solar_radiation'], 99.5)"
            )
        return np.clip(degraded, 0, solar_cap)
    
    elif variable_type == 'precipitation':
        # Event detection error rates from Sukovich et al. (2014) CONUS QPF:
        # Day 1: POD≈0.65 (miss≈35%), FAR≈0.35
        # Day 2: POD≈0.55 (miss≈45%), FAR≈0.45
        # Linear interpolation with 50% cap at week-ahead
        miss_rate = min(0.30 + 0.0015 * h, 0.50)      # 30% @ 6h → 50% @ 168h
        false_alarm_rate = min(0.35 + 0.0010 * h, 0.50)  # 35% @ 6h → 50% @ 168h
        
        if actual_value == 0:
            # False alarm: forecast rain when there is none
            if rng.random() < false_alarm_rate:
                # Small false alarm (most are light precipitation)
                return rng.lognormal(mean=np.log(0.5), sigma=0.5)
            else:
                return 0  # Correctly forecast no rain
        
        else:
            # Missed detection: forecast no rain when there is rain
            if rng.random() < miss_rate:
                return 0  # Missed the rain event
            
            # Detected correctly: apply magnitude error
            # Multiplicative lognormal noise (Jolliffe & Stephenson)
            # CV growth rate: 0.15%/h (conservative estimate from ensemble
            # spread theory - half the theoretical bound of 0.30%/h)
            relative_error_pct = 30 + 0.15 * h
            cv = relative_error_pct / 100
            
            sigma_log = np.sqrt(np.log(1 + cv**2))
            mu_log = -0.5 * sigma_log**2  # Mean-preserving
            
            multiplier = rng.lognormal(mean=mu_log, sigma=sigma_log)
            return actual_value * multiplier
    
    elif variable_type == 'visibility':
        # Lognormal multiplicative noise.
        # CV = 25% (constant, horizon-independent)
        # ECMWF characterises visibility as its lowest-skill surface forecast
        # variable. Crucially, Section 9.4.1 explicitly states that shorter
        # lead times are not necessarily more skilful than longer ones — i.e.
        # skill is non-monotonic with horizon. A horizon-dependent growth rate
        # is therefore inconsistent with the cited evidence. A flat CV = 25%
        # is anchored to Bari & Ouagabi (2020): ML-corrected NWP achieves
        # MAE ~1300m, RMSE ~2000m at 24h, implying ~25% relative error for a
        # typical mean visibility of ~7-8km. Raw NWP errors are higher
        # (Gultepe et al., 2006); 25% is therefore a conservative lower bound.
        # Source: Owens & Hewson (2018), ECMWF Forecast User Guide, Sec. 9.4
        cv = 0.25
        sigma_log = np.sqrt(np.log(1 + cv**2))
        mu_log = -0.5 * sigma_log**2  # Mean-preserving: E[multiplier] = 1
        multiplier = rng.lognormal(mean=mu_log, sigma=sigma_log)
        return max(0, actual_value * multiplier)

    else:
        raise ValueError(
            f"Unknown variable type: '{variable_type}'. "
            f"Must be one of: temperature, humidity, wind_speed, "
            f"solar_radiation, precipitation, visibility"
        )


def prepare_degradation_parameters(training_data, column_mapping=None):
    """
    Extract data-dependent parameters needed for degradation.
    
    This function computes statistics from training data that are required
    for physically consistent degradation (e.g., maximum solar radiation).
    
    Parameters
    ----------
    training_data : pd.DataFrame
        Training dataset with solar radiation column
    column_mapping : dict, optional
        Maps actual column names to variable types.
        If None, assumes column is named 'solar_radiation'
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'solar_cap': 99.5th percentile of observed solar radiation
    
    Examples
    --------
    >>> import pandas as pd
    >>> train_df = pd.read_csv('seoul_bike_train.csv')
    >>> 
    >>> # With column mapping
    >>> mapping = {'Solar Radiation': 'solar_radiation'}
    >>> params = prepare_degradation_parameters(train_df, mapping)
    >>> 
    >>> # Without mapping (assumes 'solar_radiation' column exists)
    >>> params = prepare_degradation_parameters(train_df)
    """
    
    # Find solar radiation column name
    solar_col = None
    if column_mapping:
        for col_name, var_type in column_mapping.items():
            if var_type == 'solar_radiation':
                solar_col = col_name
                break
    
    # Fallback to default name if no mapping provided
    if solar_col is None:
        solar_col = 'solar_radiation'
    
    params = {
        'solar_cap': np.percentile(training_data[solar_col], 99.5)
    }
    
    return params


def fix_precipitation_type(df_degraded, temp_col='Temperature', 
                          rain_col='Rainfall', snow_col='Snowfall'):
    """
    Swap rain/snow based on degraded temperature to maintain physical consistency.
    
    After independent degradation, temperature and precipitation type may be
    inconsistent (e.g., snowfall at 15°C or rainfall at -5°C). This function
    corrects precipitation type based on the degraded temperature.
    
    Rule: precipitation type determined by degraded temperature
    - Above 2°C: convert snow to rain
    - Below 2°C: convert rain to snow
    
    Parameters
    ----------
    df_degraded : pd.DataFrame
        Degraded dataframe
    temp_col : str, default='Temperature'
        Temperature column name
    rain_col : str, default='Rainfall'
        Rainfall column name
    snow_col : str, default='Snowfall'
        Snowfall column name
    
    Returns
    -------
    pd.DataFrame
        Dataframe with corrected precipitation types
        
    Notes
    -----
    The 2°C threshold is a simplification. Real precipitation phase transition
    occurs over a range (typically 0-4°C) with mixed precipitation possible.
    This threshold represents typical operational forecast practice.
    
    Examples
    --------
    >>> # After degradation, fix inconsistencies
    >>> df_corrected = fix_precipitation_type(
    ...     df_degraded,
    ...     temp_col='Temperature',
    ...     rain_col='Rainfall',
    ...     snow_col='Snowfall'
    ... )
    """
    df = df_degraded.copy()
    
    # Skip if columns missing
    if temp_col not in df.columns or rain_col not in df.columns or snow_col not in df.columns:
        return df
    
    # Condition 1: Warm temperatures (>2°C) with snow → convert to rain
    warm_with_snow = (df[temp_col] > 2) & (df[snow_col] > 0)
    df.loc[warm_with_snow, rain_col] = df.loc[warm_with_snow, rain_col] + df.loc[warm_with_snow, snow_col]
    df.loc[warm_with_snow, snow_col] = 0
    
    # Condition 2: Cold temperatures (<2°C) with rain → convert to snow
    cold_with_rain = (df[temp_col] < 2) & (df[rain_col] > 0)
    df.loc[cold_with_rain, snow_col] = df.loc[cold_with_rain, snow_col] + df.loc[cold_with_rain, rain_col]
    df.loc[cold_with_rain, rain_col] = 0
    
    return df


def degrade_weather_dataset(df, horizon_hours, degradation_params, 
                            column_mapping=None, seed=42, lead_times=None):
    """
    Apply forecast degradation to all weather variables in a dataset.
    
    Degradation is applied in two passes:
    1. All variables degraded independently
    2. Precipitation types corrected based on degraded temperature
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with weather columns
    horizon_hours : int
        Forecast lead time used when lead_times is None (6, 24, 48, or 168).
        Ignored when lead_times is provided.
    degradation_params : dict
        Parameters from prepare_degradation_parameters()
    column_mapping : dict, optional
        Maps actual column names to variable types.
        Example: {'Temperature': 'temperature', 'Rainfall': 'precipitation'}
        If None, uses exact lowercase matching.
    seed : int or None, default=42
        Random seed for reproducibility. Use seed=None for truly random results.
    lead_times : array-like of int, optional
        Per-row forecast lead times in hours (length must equal len(df)).
        When provided, each row i is degraded using lead_times[i] instead of
        the scalar horizon_hours.  Pass np.arange(1, horizon+1) for a test
        window so that the first predicted hour uses 1-hour noise and the last
        uses full-horizon noise — which is the physically correct behaviour.
    
    Returns
    -------
    pd.DataFrame
        Dataset with degraded weather forecasts
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Load data
    >>> train_df = pd.read_csv('seoul_bike_train.csv')
    >>> test_df = pd.read_csv('seoul_bike_test.csv')
    >>> 
    >>> # Prepare parameters and mapping
    >>> column_mapping = {'Temperature': 'temperature', 'Rainfall': 'precipitation'}
    >>> params = prepare_degradation_parameters(train_df, column_mapping)
    >>> 
    >>> # Degrade test set — each row gets its own lead-time-appropriate noise
    >>> row_lead_times = np.arange(1, len(test_df) + 1)
    >>> test_degraded = degrade_weather_dataset(
    ...     test_df, horizon_hours=24, params, column_mapping,
    ...     seed=42, lead_times=row_lead_times
    ... )
    """
    
    # Create reproducible seed
    rng = np.random.default_rng(seed=seed)
    
    df_degraded = df.copy()

    # Pass 1: Degrade all variables independently
    for col, var_type in column_mapping.items():
        if col not in df.columns:
            continue
        
        if lead_times is not None:
            # Per-row degradation: each step uses its own lead time
            degraded_values = []
            for i, val in enumerate(df[col]):
                lt = int(lead_times[i])
                if var_type == 'solar_radiation':
                    degraded_values.append(
                        degrade_weather_forecast(
                            val, var_type, lt,
                            solar_cap=degradation_params['solar_cap'],
                            rng=rng
                        )
                    )
                else:
                    degraded_values.append(
                        degrade_weather_forecast(val, var_type, lt, rng=rng)
                    )
            df_degraded[col] = degraded_values
        elif var_type == 'solar_radiation':
            df_degraded[col] = df[col].apply(
                lambda x: degrade_weather_forecast(
                    x, var_type, horizon_hours, 
                    solar_cap=degradation_params['solar_cap'],
                    rng=rng
                )
            )
        else:
            df_degraded[col] = df[col].apply(
                lambda x: degrade_weather_forecast(
                    x, var_type, horizon_hours, rng=rng
                )
            )
    
    # Pass 2: Fix precipitation types based on degraded temperature
    # This MUST happen after temperature degradation
    df_degraded = fix_precipitation_type(
        df_degraded,
        temp_col='Temperature',
        rain_col='Rainfall',
        snow_col='Snowfall'
    )

    return df_degraded


def create_forecast_scenarios(df, degradation_params, column_mapping, 
                              horizons=None, seed=42):
    """
    Create multiple independent forecast scenarios with different lead times.
    
    Each horizon receives an independent random stream to ensure scenarios
    represent independent forecast perturbations rather than correlated errors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Observed weather data
    degradation_params : dict
        Parameters from prepare_degradation_parameters()
    column_mapping : dict
        Maps column names to variable types.
        Example: {'Temperature': 'temperature', 'Rainfall': 'precipitation'}
    horizons : list of int, optional
        Forecast lead times in hours. Default: [6, 24, 48, 168]
    seed : int or None, default=42
        Base random seed for reproducibility. Each horizon gets seed+h
        to ensure independence. Use seed=None for truly random results.
    
    Returns
    -------
    dict
        Dictionary mapping horizon string (e.g., '24h') to degraded DataFrame
    
    Notes
    -----
    Horizon scenarios are generated with independent random streams. If seed
    is provided, horizon h uses seed (seed+h), ensuring reproducibility while
    maintaining independence across horizons.
    
    Examples
    --------
    >>> # Define column mapping
    >>> column_mapping = {
    ...     'Temperature': 'temperature',
    ...     'Rainfall': 'precipitation',
    ...     'Snowfall': 'precipitation'
    ... }
    >>> 
    >>> # REPRODUCIBLE (recommended, default seed=42)
    >>> scenarios = create_forecast_scenarios(test_df, params, column_mapping)
    >>> 
    >>> # REPRODUCIBLE with custom seed
    >>> scenarios = create_forecast_scenarios(test_df, params, column_mapping, seed=99)
    >>> 
    >>> # NOT REPRODUCIBLE (different results each run)
    >>> scenarios = create_forecast_scenarios(test_df, params, column_mapping, seed=None)
    >>> 
    >>> # Use in modeling
    >>> print(scenarios.keys())  # dict_keys(['6h', '24h', '48h', '168h'])
    >>> predictions_24h = model.predict(scenarios['24h'])
    >>> predictions_168h = model.predict(scenarios['168h'])
    >>> 
    >>> # Scenarios are independent even with same base seed
    >>> assert not np.allclose(
    ...     scenarios['24h']['Temperature'].values,
    ...     scenarios['48h']['Temperature'].values
    ... )
    """
    
    if horizons is None:
        horizons = [6, 24, 48, 168]
    
    scenarios = {}
    
    for h in horizons:
        # Create independent seed per horizon
        # This ensures each horizon scenario is an independent perturbation
        horizon_seed = None if seed is None else seed + h
        
        scenarios[f'{h}h'] = degrade_weather_dataset(
            df, h, degradation_params, column_mapping, seed=horizon_seed
        )
    
    return scenarios


def validate_degradation_statistics(df_actual, df_forecast, variable, 
                                    horizon_hours, verbose=False):
    """
    Verify that degradation produces expected error statistics.
    
    Computes MAE, RMSE, and bias from degraded forecasts and compares
    to theoretically expected values. Useful for validating implementation.
    
    Parameters
    ----------
    df_actual : pd.DataFrame
        Observed values
    df_forecast : pd.DataFrame
        Degraded forecast values
    variable : str
        Column name to validate
    horizon_hours : int
        Forecast horizon used
    verbose : bool, optional
        If True, prints validation results to stdout. Default: False.
        Use verbose=True only for interactive development/testing.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'mae': Observed mean absolute error
        - 'rmse': Observed root mean square error
        - 'bias': Observed bias (should be near zero)
        - 'expected_sigma': Theoretical standard deviation (or None)
        - 'expected_mae': Theoretical MAE (or None)
        - 'ratio': RMSE / expected_sigma (or None)
    
    Notes
    -----
    Only validates additive Gaussian variables (temperature, humidity, wind_speed).
    Multiplicative lognormal (precipitation) and heteroscedastic Gaussian 
    (solar_radiation) variables are not currently validated due to value-dependent 
    error statistics.
    
    Examples
    --------
    >>> # Validate temperature degradation
    >>> stats = validate_degradation_statistics(
    ...     test_df, test_24h, 'temperature', 24
    ... )
    >>> assert abs(stats['bias']) < 0.1  # Check for bias
    >>> assert 0.95 < stats['ratio'] < 1.05  # Check calibration
    """
    
    errors = df_forecast[variable] - df_actual[variable]
    mae_observed = np.abs(errors).mean()
    rmse_observed = np.sqrt((errors**2).mean())
    bias = errors.mean()
    
    # Compute expected sigma based on variable type
    # Only implemented for additive Gaussian variables
    h = horizon_hours
    
    if variable in ['temperature', 'temp', 'Temperature']:
        expected_sigma = 0.77 + 0.0181 * h
        expected_mae = expected_sigma * 0.798  # Theoretical MAE for normal
    elif variable in ['humidity', 'relative_humidity', 'Humidity']:
        expected_sigma = 13.0 + 0.023 * h
        expected_mae = expected_sigma * 0.798
    elif variable in ['wind_speed', 'wind', 'Wind speed']:
        expected_sigma = 1.8 + 0.010 * h
        # Note: truncation at zero introduces small negative bias
        expected_mae = expected_sigma * 0.798
    else:
        # Multiplicative lognormal and heteroscedastic variables not validated
        expected_sigma = None
        expected_mae = None
    
    results = {
        'mae': mae_observed,
        'rmse': rmse_observed,
        'bias': bias,
        'expected_sigma': expected_sigma,
        'expected_mae': expected_mae,
        'ratio': rmse_observed / expected_sigma if expected_sigma else None
    }
    
    if verbose:
        print(f"\n{variable.upper()} Validation (horizon = {horizon_hours}h)")
        print("=" * 60)
        print(f"  Observed MAE:      {mae_observed:.3f}")
        print(f"  Observed RMSE:     {rmse_observed:.3f}")
        print(f"  Observed Bias:     {bias:.3f} (should be ≈ 0)")
        
        if expected_sigma is not None:
            print(f"\n  Expected σ:        {expected_sigma:.3f}")
            print(f"  Expected MAE:      {expected_mae:.3f}")
            print(f"  RMSE/Expected σ:   {results['ratio']:.3f} (should be ≈ 1.0)")
            
            if abs(bias) > 0.1 * expected_sigma:
                print(f"\n  WARNING: Bias exceeds 10% of expected σ")
            if not (0.9 < results['ratio'] < 1.1):
                print(f"  WARNING: RMSE deviates >10% from expected σ")
        else:
            print(f"\n  Note: Expected statistics not available for {variable}")
            print(f"        (validation only supports additive Gaussian variables)")
    
    return results