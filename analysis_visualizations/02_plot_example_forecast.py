import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'forecasting'))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config_seoul

FIGURES_DIR     = PROJECT_ROOT / 'results' / 'figures' / 'seoul'
EXAMPLE_DATA_DIR = PROJECT_ROOT / 'data' / 'example_data'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load predictions ──────────────────────────────────────────────────────────
pred = pd.read_csv(EXAMPLE_DATA_DIR / 'forecast_seoul_predictions.csv', parse_dates=['ds', 'cutoff'])
CUTOFF = pred['cutoff'].iloc[0]

# ── Load actuals (1 month before cutoff) ──────────────────────────────────────
config   = config_seoul.get_config()
df_raw   = pd.read_csv(PROJECT_ROOT / 'data' / config.data_filename, parse_dates=[config.date_col])
df_raw   = df_raw[df_raw[config.functioning_day_col].str.strip() == 'Yes']
df_raw   = df_raw.set_index(config.date_col).sort_index()
actuals  = df_raw.loc[CUTOFF - pd.Timedelta(days=30) : CUTOFF, config.target_col]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(actuals.index,  actuals.values,    color="#4C9BE8", linewidth=1.2, label="Actual")
ax.plot(pred['ds'],     pred['y_actual'],  color="#4C9BE8", linewidth=1.2)
ax.plot(pred['ds'],     pred['yhat'],      color="#FF6B6B", linewidth=2, linestyle="--", label="Forecast")
ax.axvline(CUTOFF, color="#0F3A65", linewidth=1.5, linestyle="--", label="Cutoff")
ax.set_xlim(actuals.index[0], pred['ds'].iloc[-1])

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

ax.set_ylabel("Seoul rented bikes (hourly)")
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'forecast_seoul.png', dpi=150, bbox_inches='tight')
print(f"Saved → {FIGURES_DIR / 'forecast_seoul.png'}")