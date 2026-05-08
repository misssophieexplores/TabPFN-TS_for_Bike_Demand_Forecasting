"""
Visualise training / test data usage for each city dataset.

Regions (based on h=168, the longest horizon):
  - Never used      : data_start → earliest_train_start
  - Training only   : earliest_train_start → first_fold_date
  - Train & test    : first_fold_date → latest_train_end   (rolling window overlap)
  - Test only       : latest_train_end → data_end
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
    ('#4a90d9', 0.30, 'Training only'),
    ('#9b59b6', 0.30, 'Training & test (overlap)'),
    ('#e67e22', 0.35, 'Test only'),
]

datasets = [
    ('London',     config_london,     PROJECT_ROOT / 'data' / 'LondonBikeData.csv'),
    ('Seoul',      config_seoul,      PROJECT_ROOT / 'data' / 'SeoulBikeData.csv'),
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

    # Weekly aggregation (drop first/last partial week)
    series = (
        df.set_index(config.date_col)[config.target_col]
        .resample('W').sum()
        .iloc[1:-1]
    )

    # ── Key dates ──────────────────────────────────────────────────────────────
    data_start = df[config.date_col].min()
    data_end   = df[config.date_col].max()
    max_h      = max(config.horizons)

    first_fold_date      = data_end  - pd.Timedelta(hours=config.n_folds * max_h)
    earliest_train_start = first_fold_date - pd.Timedelta(hours=config.n_train_samples)
    latest_train_end     = data_end  - pd.Timedelta(hours=max_h)

    boundaries = [
        data_start,
        earliest_train_start,
        first_fold_date,
        latest_train_end,
        data_end,
    ]
    # ── Print key dates
    # ──────────────────────────────────────────────────────────────
    print(f"\n{city}:")
    print(f"  Data range:           {data_start} → {data_end}")
    print(f"  Earliest train start: {earliest_train_start} (unused: {earliest_train_start - data_start})")
    print(f"  First fold date:     {first_fold_date} (train only: {first_fold_date - earliest_train_start})")   

    # ── Shade regions ──────────────────────────────────────────────────────────
    for (color, alpha, _), start, end in zip(REGIONS, boundaries[:-1], boundaries[1:]):
        ax.axvspan(start, end, alpha=alpha, color=color, zorder=0, linewidth=0)

    # ── Boundary lines ─────────────────────────────────────────────────────────
    for date in boundaries[1:-1]:
        ax.axvline(date, color='#444444', linewidth=0.9, linestyle=':', zorder=3)

    # ── Series ─────────────────────────────────────────────────────────────────
    ax.plot(series.index, series.values,
            color=SERIES_COLORS[city], linewidth=1.5, zorder=2)

    # ── Annotations ────────────────────────────────────────────────────────────
    ymax = series.max()
    ax.set_ylim(bottom=0, top=ymax * 1.18)

    label_y = ymax * 1.08
    midpoints = [
        (data_start,            earliest_train_start, 'unused'),
        (earliest_train_start,  first_fold_date,      f'train\n(n={config.n_train_samples:,})'),
        (first_fold_date,       latest_train_end,     'train\n& test'),
        (latest_train_end,      data_end,             f'test\nh={max_h}h'),
    ]
    for s, e, lbl in midpoints:
        mid = s + (e - s) / 2
        ax.text(mid, label_y, lbl, ha='center', va='bottom',
                fontsize=7.5, color='#333333', linespacing=1.3)

    # ── Axes formatting ────────────────────────────────────────────────────────
    ax.set_title(f'{city}  '
                 f'({df[config.date_col].min().strftime("%b %Y")}–'
                 f'{df[config.date_col].max().strftime("%b %Y")})',
                 fontweight='bold', loc='left', fontsize=11)
    ax.set_ylabel('Weekly rentals', labelpad=6)
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