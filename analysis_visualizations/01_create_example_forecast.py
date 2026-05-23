import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'forecasting'))

import pandas as pd
import config_seoul
from models.tabpfn_pipeline_model import TabPFNPipelineForecaster

FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures' / 'seoul'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Config & data ─────────────────────────────────────────────────────────────
config   = config_seoul.get_config()
filepath = PROJECT_ROOT / 'data' / config.data_filename

df_raw = pd.read_csv(filepath, parse_dates=[config.date_col])
df_raw = df_raw[df_raw[config.functioning_day_col].str.strip() == 'Yes']
df_raw[config.season_col]  = df_raw[config.season_col].map(config.season_mapping)
df_raw[config.holiday_col] = df_raw[config.holiday_col].map(config.holiday_mapping)
df_raw = df_raw.set_index(config.date_col).sort_index()

HORIZON = max(config.horizons)
CUTOFF  = df_raw.index.max() - pd.Timedelta(hours=config.n_folds * HORIZON)

# ── Train / future ────────────────────────────────────────────────────────────
train  = df_raw.loc[CUTOFF - pd.Timedelta(hours=config.n_train_samples) : CUTOFF]
future = df_raw.loc[CUTOFF + pd.Timedelta(hours=1) : CUTOFF + pd.Timedelta(hours=HORIZON)]

y_train  = train[config.target_col].values
X_train  = train[config.weather_covariates]
X_future = future[config.weather_covariates]

# ── Fit & predict ─────────────────────────────────────────────────────────────
model = TabPFNPipelineForecaster()
model.fit(y_train, X_train)
yhat  = model.predict(HORIZON, X_future)

# ── Save ──────────────────────────────────────────────────────────────────────
EXAMPLE_DATA_DIR = PROJECT_ROOT / 'data' / 'example_data'
EXAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

out = pd.DataFrame({'ds': future.index, 'yhat': yhat, 'y_actual': future[config.target_col].values})
out['cutoff'] = CUTOFF
out.to_csv(EXAMPLE_DATA_DIR / 'forecast_seoul_predictions.csv', index=False)
print(f"Saved → {EXAMPLE_DATA_DIR / 'forecast_seoul_predictions.csv'}")