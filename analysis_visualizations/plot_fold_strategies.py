"""
analysis_visualizations/plot_fold_strategies.py

One figure per PARAM_SETS entry. Layout per figure:
  Row 0        : time series, one per city (shared across all strategies)
  Rows 1–3     : one Gantt per strategy, 3 cities as columns, with spacing between

Strategies:
  1. Consistent   — n_eval & n_tuning same for all cities (n_tuning from Seoul)
  2. Fixed tuning / Max eval — n_tuning from Seoul, n_eval maximised per city
  3. Fixed eval   / Max tuning — n_eval fixed, n_tuning maximised per city

Seoul is the reference (shortest) dataset; all 3 strategies are identical for Seoul.

PARAM_SETS = [(n_train_samples_hours, n_eval_folds), ...]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'forecasting'))

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config_london
import config_seoul
import config_washington

FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures' /'fold_strategies'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── USER CONFIGURATION ─────────────────────────────────────────────────────────
PARAM_SETS = [
    (720, 35),
    (4096, 20),
]

DATASETS = [
    ('Seoul',      config_seoul,      'SeoulBikeData.csv'),
    ('London',     config_london,     'LondonBikeData.csv'),
    ('Washington', config_washington, 'WashingtonBikeData.csv'),
]

REFERENCE_CITY = 'Seoul'

STRATEGIES = [
    ('consistent', 'Consistent Folds'),
    ('max_eval',   'Max Evaluation'),
    ('max_tuning', 'Max Tuning'),
]

# ── COLORS ─────────────────────────────────────────────────────────────────────
CITY_COLORS = {
    'Seoul':      '#d62728',
    'London':     '#1f77b4',
    'Washington': '#2ca02c',
}

C_LEFTOVER = '#c8c8c8'
C_BURNIN   = '#4a90d9'
C_TUNING   = '#9b59b6'
C_EVAL_BI  = '#85c1e9'
C_EVAL     = '#e67e22'
# C_CUTOFF   = '#c0392b'
# C_TUNSTART = '#7f8c8d'

GANTT_ROWS = [
    (0, 'Eval folds',    C_EVAL),
    (1, 'Eval burn-in',  C_EVAL_BI),
    (2, 'Tuning folds',  C_TUNING),
    (3, 'Tun. burn-in',  C_BURNIN),
    (4, 'Leftover',      C_LEFTOVER),
]

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size':       8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_city(cfg_mod, filename, n_train):
    config = cfg_mod.get_config()
    config.verbose = False
    config.n_train_samples = n_train
    df = pd.read_csv(PROJECT_ROOT / 'data' / filename, parse_dates=[config.date_col])
    df = df.sort_values(config.date_col).reset_index(drop=True)
    fday = config.functioning_day_col
    if fday and fday in df.columns:
        df = df[df[fday].str.strip() == 'Yes']
    df = df.drop_duplicates(subset=config.date_col, keep='first').reset_index(drop=True)
    return df, config


# ── FOLD LOGIC ─────────────────────────────────────────────────────────────────

def seoul_reference(df, config, n_eval_folds):
    avail = len(df) - config.n_train_samples
    max_folds = int(avail // max(config.horizons))
    return max_folds, max_folds - n_eval_folds


def fold_counts(df, config, strategy, n_eval_folds, n_tuning_fixed):
    city_max = int((len(df) - config.n_train_samples) // max(config.horizons))
    if strategy == 'consistent':
        return n_tuning_fixed, n_eval_folds
    elif strategy == 'max_eval':
        return n_tuning_fixed, city_max - n_tuning_fixed
    else:
        return city_max - n_eval_folds, n_eval_folds


def compute_boundaries(df, config, n_tuning, n_eval):
    n_train   = config.n_train_samples
    max_h     = max(config.horizons)
    total     = len(df)
    max_folds = int((total - n_train) // max_h)
    ts        = df[config.date_col]

    def d(i):
        return ts.iloc[int(np.clip(i, 0, len(ts) - 1))]

    ev_start_i = total - n_eval * max_h
    bi_end_i   = ev_start_i - n_tuning * max_h
    lo_end_i   = bi_end_i - n_train
    ebi_start_i = ev_start_i - n_train   # exactly n_train long

    return {
        'n_train': n_train, 'max_h': max_h,
        'max_folds': max_folds, 'tuning': n_tuning, 'eval': n_eval,
        'd_start':  d(0),
        'd_lo_end': d(lo_end_i),
        'd_bi_end': d(bi_end_i),
        'd_tu_end': d(ev_start_i),
        'd_ebi_st': d(ebi_start_i),
        'd_end':    d(total - 1),
    }


# ── DRAWING ────────────────────────────────────────────────────────────────────

def gantt_bar(ax, y, start, end, color, alpha=0.85):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        return
    s, e = mdates.date2num(start), mdates.date2num(end)
    ax.broken_barh([(s, e - s)], (y - 0.38, 0.76),
                   facecolors=color, alpha=alpha, edgecolors='none')


def draw_series(ax, df, config, city):
    series = (df.set_index(config.date_col)[config.target_col]
              .resample('W').sum().iloc[1:-1])
    ax.plot(series.index, series.values, color=CITY_COLORS[city], linewidth=1.4, zorder=3)
    ax.set_ylim(0, series.max() * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, color='#ebebeb', linewidth=0.5, linestyle='--')
    ax.tick_params(axis='x', rotation=30, labelsize=7, pad=4)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.tick_params(axis='y', labelsize=7)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.set_ylabel('weekly rentals', fontsize=7)


def draw_gantt(ax, bnd, show_x_labels=True, show_leftover=True):
    b = bnd
    if show_leftover:
        gantt_bar(ax, 4, b['d_start'],  b['d_lo_end'],  C_LEFTOVER)
    gantt_bar(ax, 3, b['d_lo_end'], b['d_bi_end'],  C_BURNIN)
    gantt_bar(ax, 2, b['d_bi_end'], b['d_tu_end'],  C_TUNING)
    gantt_bar(ax, 1, b['d_ebi_st'], b['d_tu_end'],  C_EVAL_BI)
    gantt_bar(ax, 0, b['d_tu_end'], b['d_end'],     C_EVAL)

    ax.set_yticks([r[0] for r in GANTT_ROWS])
    ax.set_yticklabels([r[1] for r in GANTT_ROWS], fontsize=6.5)
    ax.set_ylim(-0.5, 5.0)
    ax.grid(True, axis='x', color='#ebebeb', linewidth=0.5, linestyle='--')
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.tick_params(axis='x', rotation=30, labelsize=6.5,
                   labelbottom=show_x_labels)
    ax.spines['left'].set_visible(False)

    # Stats
    overlap = min(b['tuning'], int(np.ceil(b['n_train'] / b['max_h'])))
    ax.set_xlabel(
        f"max={b['max_folds']}  tune={b['tuning']}  eval={b['eval']}"
        f"  ·  eval burn-in ≈{overlap} tuning fold(s)",
        fontsize=6, color='#666666', labelpad=3,
    )


# ── FIGURE GENERATION ─────────────────────────────────────────────────────────

legend_handles = [
    mpatches.Patch(color=C_LEFTOVER, alpha=0.85, label='Leftover'),
    mpatches.Patch(color=C_BURNIN,   alpha=0.85, label='Tuning burn-in'),
    mpatches.Patch(color=C_TUNING,   alpha=0.85, label='Tuning folds'),
    mpatches.Patch(color=C_EVAL_BI,  alpha=0.85, label='Eval burn-in'),
    mpatches.Patch(color=C_EVAL,     alpha=0.85, label='Evaluation folds'),
    # mlines.Line2D([0], [0], color=C_TUNSTART, ls='none', marker='|',
    #               markersize=9, markeredgewidth=1.5, label='Tuning start date'),
    # mlines.Line2D([0], [0], color=C_CUTOFF, ls='none', marker='|',
                #   markersize=9, markeredgewidth=1.5, label='Cutoff date (tuning end / eval start)'),
]

for n_train, n_eval_folds in PARAM_SETS:

    city_data = {}
    for city, cfg_mod, filename in DATASETS:
        city_data[city] = load_city(cfg_mod, filename, n_train)

    df_ref, config_ref = city_data[REFERENCE_CITY]
    seoul_max, n_tuning_fixed = seoul_reference(df_ref, config_ref, n_eval_folds)

    if n_tuning_fixed < 0:
        print(f'SKIP: n_eval_folds={n_eval_folds} > Seoul max={seoul_max} '
              f'for n_train={n_train}h')
        continue

    n_s = len(STRATEGIES)
    n_c = len(DATASETS)

    # Layout: row 0 = series (tall), rows 1..n_s = gantts
    # Extra 0.3 spacer rows between gantts achieved via height_ratios padding
    height_ratios = [2.5] + [0.55, 1.1] * n_s   # spacer + gantt per strategy
    n_rows = 1 + 2 * n_s

    fig = plt.figure(figsize=(7.0 * n_c, 8.0 + 1.8 * n_s))
    gs  = gridspec.GridSpec(
        n_rows, n_c,
        height_ratios=height_ratios,
        hspace=0.0,   # spacing controlled by spacer rows
        wspace=0.15,
        top=0.93, bottom=0.10, left=0.07, right=0.98,
    )

    # — Series row (row 0) ——————————————————————————————————————————————————
    ax_series = {}
    for ci, (city, cfg_mod, filename) in enumerate(DATASETS):
        df, config = city_data[city]
        ax = fig.add_subplot(gs[0, ci])
        draw_series(ax, df, config, city)
        ref = '  ★ reference' if city == REFERENCE_CITY else ''
        ax.set_title(f'{city}{ref}', fontsize=11, fontweight='bold',
                     color=CITY_COLORS[city], pad=6)
        ax_series[ci] = ax

    # — Gantt rows (rows 2, 4, 6 = skip spacer rows 1, 3, 5) ——————————————
    for si, (strat_id, strat_label) in enumerate(STRATEGIES):
        gantt_row = 1 + si * 2 + 1   # rows: 2, 4, 6
        is_last = (si == n_s - 1)

        for ci, (city, cfg_mod, filename) in enumerate(DATASETS):
            df, config = city_data[city]
            n_tuning, n_eval = fold_counts(
                df, config, strat_id, n_eval_folds, n_tuning_fixed)
            bnd = compute_boundaries(df, config, n_tuning, n_eval)

            ax = fig.add_subplot(gs[gantt_row, ci], sharex=ax_series[ci])
            draw_gantt(ax, bnd, show_x_labels=is_last,
                       show_leftover=(strat_id == 'consistent'))

            # Strategy label on left column only
            if ci == 0:
                ax.set_ylabel(strat_label, fontsize=7.5, fontweight='bold', labelpad=6)

    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=4,
        fontsize=7, frameon=True, framealpha=0.95, edgecolor='#cccccc',
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        f'CV Fold Strategy Comparison  |  n_train = {n_train:,} h  ·  '
        f'n_eval_folds = {n_eval_folds}  ·  '
        f'n_tuning_folds (Seoul ★) = {n_tuning_fixed}',
        fontsize=11, fontweight='bold',
    )

    outfile = FIGURES_DIR / f'fold_strategies_ntrain{n_train}_neval{n_eval_folds}.png'
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {outfile}')

print('Done.')