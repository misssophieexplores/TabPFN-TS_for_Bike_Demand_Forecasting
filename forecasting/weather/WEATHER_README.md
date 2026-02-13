# Weather Forecast Degradation

Implementation of NWP forecast error simulation for machine learning model evaluation.


## Installation
```bash
pip install numpy pandas
```

Requires Python 3.8+ and NumPy 1.17+.

## Quick Start
```python
import numpy as np
import pandas as pd
from weather_degradation import (
    degrade_weather_forecast,
    prepare_degradation_parameters,
    create_forecast_scenarios
)

# Load training data
train_df = pd.read_csv('data/train.csv')

# Compute data-dependent parameters
params = prepare_degradation_parameters(train_df)

# Create forecast scenarios at multiple horizons
test_df = pd.read_csv('data/test.csv')
scenarios = create_forecast_scenarios(test_df, params, seed=42)

# Use degraded forecasts in modeling
model.fit(train_df)
predictions_24h = model.predict(scenarios['24h'])
predictions_168h = model.predict(scenarios['168h'])
```

## Reproducibility

### Ensuring Reproducible Results

For reproducible experiments, **always provide a seed**:
```python
# REPRODUCIBLE - Recommended for research/production
scenarios = create_forecast_scenarios(test_df, params, seed=42)

# NOT REPRODUCIBLE - Uses random entropy
scenarios = create_forecast_scenarios(test_df, params)  # Different results each run
```

### Single-value degradation with reproducibility:
```python
# REPRODUCIBLE
rng = np.random.default_rng(seed=42)
temp_forecast = degrade_weather_forecast(15.3, 'temperature', 24, rng=rng)

# NOT REPRODUCIBLE
temp_forecast = degrade_weather_forecast(15.3, 'temperature', 24)  # No rng provided
```

### Best Practice for Research
```python
import numpy as np

# Set global random seed for complete reproducibility
RANDOM_SEED = 42

# All degradation will be reproducible
scenarios = create_forecast_scenarios(
    test_df, 
    degradation_params, 
    seed=RANDOM_SEED
)

# Results are identical across runs with same seed
```

## Features

- **Source-calibrated error growth**: Parameters derived from ECMWF operational verification where available
- **Multiple noise models**: Gaussian (homoscedastic/heteroscedastic) and lognormal multiplicative
- **Fully reproducible**: Controlled random number generation with independent horizon streams
- **Production-ready**: Input validation, physical bounds, comprehensive testing

## Supported Variables

| Variable | Error Model | Calibration Source |
|----------|-------------|-------------------|
| Temperature | Additive Gaussian | ECMWF TM 918 (2024), Figs 7, 13, 14, 24, 32 |
| Humidity | Additive Gaussian | Assumed (no verification data available) |
| Wind Speed | Additive Gaussian (truncated) | ECMWF TM 918 (2024), Figs 13, 14, 27, 32 |
| Solar Radiation | Heteroscedastic Gaussian | Kleissl (2013) |
| Precipitation | Binary detection + Lognormal | Jolliffe & Stephenson (2003) |
| Visibility | Multiplicative lognormal | Assumption-based |

## Error Growth by Horizon

### Temperature (σ in °C)
- 6h: 0.88°C
- 24h: 1.20°C
- 48h: 1.64°C
- 72h: 2.07°C
- 120h: 2.94°C
- 168h: 3.81°C

### Humidity (σ in %-points)
- 6h: 4.91%
- 24h: 6.13%
- 48h: 7.76%
- 72h: 9.40%
- 120h: 12.66%
- 168h: 15.92%

### Wind Speed (σ in m/s)
- 6h: 1.45 m/s
- 24h: 1.59 m/s
- 48h: 1.78 m/s
- 72h: 1.98 m/s
- 120h: 2.36 m/s
- 168h: 2.74 m/s

### Solar Radiation (relative MAE in %)
- 6h: 15.9%
- 24h: 18.6%
- 48h: 22.2%
- 72h: 25.8%
- 120h: 33.0%
- 168h: 40.2%

### Precipitation (CV in %)
- 6h: 30.9%
- 24h: 33.6%
- 48h: 37.2%
- 72h: 40.8%
- 120h: 48.0%
- 168h: 55.2%

### Visibility (CV in %)
- 6h: 10.6%
- 24h: 12.4%
- 48h: 14.8%
- 72h: 17.2%
- 120h: 22.0%
- 168h: 26.8%

## Documentation

- Full methodology: See `weather_methodology.md`
- API reference: See docstrings in `weather_degradation.py`
- Usage examples: Run `python weather_degradation.py`

## Citation

If using this methodology, cite the relevant verification sources:
```bibtex
@techreport{ecmwf2024tm918,
  title={Evaluation of ECMWF forecasts, including 2023-2024 upgrades},
  author={Haiden, T. and Janousek, M. and Vitart, F. and Ben-Bouallegue, Z. and Ferranti, L. and Prates, F. and Richardson, D.S.},
  year={2024},
  number={918},
  institution={European Centre for Medium-Range Weather Forecasts},
  type={Technical Memorandum}
}

@book{jolliffe2003,
  title={Forecast Verification: A Practitioner's Guide in Atmospheric Science},
  editor={Jolliffe, Ian T. and Stephenson, David B.},
  year={2003},
  publisher={John Wiley \& Sons}
}

@book{kleissl2013,
  title={Solar Energy Forecasting and Resource Assessment},
  editor={Kleissl, Jan},
  year={2013},
  publisher={Academic Press}
}
```

