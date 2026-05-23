# analysis_visualizations/plot_data_usage.py
"""
Visualise training / tuning / test data usage for each city dataset.

Regions (based on h=168, the longest horizon):
  - Never used        : data_start → earliest_tune_train_start
  - Burn-in           : earliest_tune_train_start → tune_cutoff
                        (training lookback for tuning fold 0 only; never a test row)
  - Tuning test       : tune_cutoff → cv_cutoff
                        (also serves as training lookback for early CV folds)
  - CV train & test   : cv_cutoff → latest_cv_train_end
                        (rolling window overlap within CV eval zone)
  - CV test only      : latest_cv_train_end → data_end
"""

import sys
from pathlib import Path

# analysis_visualizations/ → project root → forecasting/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'forecasting'))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import config_london
import config_seoul
import config_washington

FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['DejaVu Sans'],
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#e0e0e0',
    'grid.linewidth':    0.6,
    'grid.linestyle':    '--',
})

SERIES_COLORS = {
    'London':     '#1f77b4',
    'Seoul':      '#d62728',
    'Washington': '#2ca02c',
}

# region colour, alpha, label
REGIONS = [
    ('#aaaaaa', 0.35, 'Never used'),
    ("#545353ff", 0.35, 'Burn-in (training)'),
    ('#2ca02c', 0.30, 'Tuning test'),
    ("#f080c5", 0.40, 'CV test'),
]

datasets = [
    ('Seoul',      config_seoul,      PROJECT_ROOT / 'data' / 'SeoulBikeData.csv'),
    ('London',     config_london,     PROJECT_ROOT / 'data' / 'LondonBikeData.csv'),
    ('Washington', config_washington, PROJECT_ROOT / 'data' / 'WashingtonBikeData.csv'),
]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
fig.subplots_adjust(hspace=0.45)

for ax, (city, cfg_mod, filepath) in zip(axes, datasets):
    config = cfg_mod.get_config()
    config.verbose = False

    df = pd.read_csv(filepath, parse_dates=[config.date_col])

    # Filter non-functioning days where applicable
    fday = config.functioning_day_col
    if fday and fday in df.columns:
        df = df[df[fday].str.strip() == 'Yes']

    # # Weekly aggregation (drop first/last partial week)
    # Test daily aggregation 
    series = (
        df.set_index(config.date_col)[config.target_col]
        .resample('d').sum()
        .iloc[1:-1]
    )

    # ── Key dates ──────────────────────────────────────────────────────────────
    data_start = df[config.date_col].min()
    data_end   = df[config.date_col].max()
    max_h      = max(config.horizons)

    # CV cutoff: mirrors TimeSeriesCV.get_cutoff_date()
    cv_cutoff           = data_end - pd.Timedelta(hours=config.n_folds * max_h)
    # Tuning cutoff: same formula applied to data before cv_cutoff
    tune_cutoff         = cv_cutoff - pd.Timedelta(hours=config.tune_folds * config.tune_horizon)
    # Earliest point ever used as training data (tuning fold 0 lookback)
    earliest_tune_train = tune_cutoff - pd.Timedelta(hours=config.n_train_samples)

    boundaries = [
        data_start,
        earliest_tune_train,   # never used → burn-in
        tune_cutoff,           # burn-in → tuning test
        cv_cutoff,             # tuning test → CV test
        data_end,
    ]

    # ── Print key dates ────────────────────────────────────────────────────────
    print(f"\n{city}:")
    print(f"CUTOFF DATE: {cv_cutoff}\n\n")
    print(f"  data_start          : {data_start}")
    print(f"  earliest_tune_train : {earliest_tune_train}  "
          f"(never used: {max(earliest_tune_train - data_start, pd.Timedelta(0))})")
    print(f"  tune_cutoff         : {tune_cutoff}  "
          f"(burn-in: {tune_cutoff - max(earliest_tune_train, data_start)})")
    print(f"  cv_cutoff           : {cv_cutoff}  "
          f"(tuning test: {cv_cutoff - tune_cutoff})")
    print(f"  data_end            : {data_end}  "
          f"(CV test: {data_end - cv_cutoff})")

    # ── Shade regions ──────────────────────────────────────────────────────────
    for (color, alpha, _), start, end in zip(REGIONS, boundaries[:-1], boundaries[1:]):
        if end > start:  # skip zero-width spans (e.g. if never-used is absent)
            ax.axvspan(start, end, alpha=alpha, color=color, zorder=0, linewidth=0)

    # ── Boundary lines ─────────────────────────────────────────────────────────
    for date in boundaries[1:-1]:
        if data_start < date < data_end:
            ax.axvline(date, color='#444444', linewidth=0.9, linestyle=':', zorder=3)

    # ── Series ─────────────────────────────────────────────────────────────────
    ax.plot(series.index, series.values,
            color=SERIES_COLORS[city], linewidth=1.5, zorder=2)

    # ── Annotations ────────────────────────────────────────────────────────────
    ymax = series.max()
    ax.set_ylim(bottom=0, top=ymax * 1.18)
    label_y = ymax * 1.08

    # midpoints = [
    #     (data_start,          earliest_tune_train,  'unused'),
    #     (earliest_tune_train, tune_cutoff,           f'burn-in'),
    #     (tune_cutoff,         cv_cutoff,             f'tuning test'),
    #     (cv_cutoff,           data_end,              f'CV test'),
    # ]
    # for s, e, lbl in midpoints:
    #     s_clipped = max(s, data_start)
    #     if e > s_clipped:
    #         mid = s_clipped + (e - s_clipped) / 2
    #         ax.text(mid, label_y, lbl, ha='center', va='bottom',
    #                 fontsize=7.5, color='#333333', linespacing=1.3)

    # ── Axes formatting ────────────────────────────────────────────────────────
    ax.set_title(f'{city}  '
                 f'({df[config.date_col].min().strftime("%b %Y")}–'
                 f'{df[config.date_col].max().strftime("%b %Y")})',
                 fontweight='bold', loc='left', fontsize=11)
    ax.set_ylabel('Daily rentals', labelpad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.tick_params(axis='x', rotation=30)

# ── Shared legend ──────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=color, alpha=alpha + 0.15, label=label)
    for color, alpha, label in REGIONS
]
axes[0].legend(
    handles=patches,
    frameon=True, framealpha=0.9, edgecolor='#cccccc',
    loc='upper right', fontsize=9, ncol=2
)

fig.savefig(FIGURES_DIR / 'timeseries_data_usage.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved to {FIGURES_DIR / 'timeseries_data_usage.png'}")