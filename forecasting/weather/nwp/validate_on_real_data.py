"""
End-to-end check of the measured NWP error model on the real bike data:
the same CV folds as the experiments (TimeSeriesCV.split, partial_last_fold),
every horizon, scenario 'degraded', with the settings of the city configs
(default: fresh forecast, average lean removed). Writes the simulated
covariate errors per test hour (sim_errors_<seed>.csv) for comparison with
the calibration (ECMWF errors at the same lead times), and prints per city
the mean change of each covariate (degraded - clean): temperature, humidity
and wind in their units; solar radiation (daylight hours), visibility (hours
below the cap) and precipitation (total amount) as degraded/clean ratios.
Precipitation is also split into hits (degraded/clean amount over the hours
wet in both), missed amount (clean amount in hours wet in clean and dry in
degraded) and false-alarm amount (degraded amount in hours dry in clean and
wet in degraded), the last two as shares of the clean total.

Every number comes with (+/-): half-width of the 95 % range when the test
windows are resampled (bootstrap within each horizon, 1000 draws), i.e. how
much the number varies by chance with one seed. With the default settings
(nwp_mean_preserving_caps, nwp_rain_amount_unbiased) the expected changes
are 0 / x1, except humidity at exactly 100 % and calm wind (at their bounds)
and visibility at its cap; a number within its (+/-) of 0 / x1 is
consistent with no bias (ARCHITECTURE.md, Known Limitation 24).

Run from the folder that contains data/ (as the experiments). Seed: SEED=42 (default).
"""
import sys, time, importlib
import os
from pathlib import Path
# forecasting/ folder (this file: forecasting/weather/nwp/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from evaluation.cv import TimeSeriesCV
from weather.weather_processor import WeatherProcessor
from weather.nwp_error_model import load_error_model

rows = []
for city, mod in [("seoul", "config_seoul"), ("london", "config_london"), ("washington", "config_washington")]:
    cfg = importlib.import_module(mod).get_config()
    df = pd.read_csv(f"data/{cfg.data_filename}")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col, kind="stable").drop_duplicates(cfg.date_col).reset_index(drop=True)
    for col, f in cfg.column_scale_factors.items():
        df[col] = df[col] * f
    cfg.weather_covariates = [c for c in cfg.weather_covariates if c in df.columns and c not in (cfg.holiday_col, cfg.season_col)]
    cfg.holiday_col = cfg.season_col = None
    import os; cfg.degradation_seed = int(os.environ.get("SEED", "42"))
    proc = WeatherProcessor(cfg)
    model = load_error_model(cfg.nwp_calibration_file)
    cv = TimeSeriesCV(cfg)
    m = cfg.weather_degradation_mapping
    col = {t: c for c, t in m.items() if t != "precipitation"}
    for h in cfg.horizons:
        splits = cv.split(df, h, partial_last_fold=True)
        t0 = time.time()
        for fold, (tr, te) in enumerate(splits):
            proc.prepare_weather_data(tr, "degraded", h, fold, split="train")
            X = proc.prepare_weather_data(te, "degraded", h, fold, split="test")
            te = te.reset_index(drop=True); X = X.reset_index(drop=True)
            info = proc.last_degradation_info
            leads = np.arange(info["lead_first"], info["lead_last"] + 1)
            pcols = [c for c, t in m.items() if t == "precipitation"]
            precip_ob = te[pcols].clip(lower=0).sum(axis=1)
            ow = precip_ob > 0
            fw = (X[pcols] > 0).any(axis=1)
            for i in range(len(te)):
                rows.append(dict(city=city, horizon=h, fold=fold, step=i + 1, lead=int(leads[i]),
                                 dT=X[col["temperature"]][i] - te[col["temperature"]][i],
                                 dRH=X[col["humidity"]][i] - te[col["humidity"]][i],
                                 dWS=X[col["wind_speed"]][i] - te[col["wind_speed"]][i],
                                 sol_c=te[col["solar_radiation"]][i], sol_d=X[col["solar_radiation"]][i],
                                 vis_c=te[col["visibility"]][i], vis_d=X[col["visibility"]][i],
                                 precip_ob_mm=float(precip_ob[i]),
                                 precip_fc_mm=float(X[pcols].clip(lower=0).sum(axis=1)[i]),
                                 vis_below_cap=bool(model.vis_cap is None
                                                    or te[col["visibility"]][i] < model.vis_cap - 1e-6),
                                 precip_class=("light_lt1" if 0 < precip_ob[i] < 1 else
                                               "strong_ge1" if precip_ob[i] >= 1 else "dry"),
                                 ow=bool(ow[i]), fw=bool(fw[i]), n_cand=info["n_candidate_runs"],
                                 start=info["forecast_start_utc"],
                                 season_days=info["season_window_days"]))
        print(f"{city} h={h}: {len(splits)} folds, {time.time() - t0:.1f} s", flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"sim_errors_{os.environ.get('SEED','42')}.csv", index=False)  # written to the current folder
print("rows", len(rows))


def window_sums(g):
    """Per test window (horizon, fold): the sums each statistic is a ratio of."""
    day, below = g["sol_c"] > 0, g["vis_below_cap"]
    hit, miss, fa = g["ow"] & g["fw"], g["ow"] & ~g["fw"], ~g["ow"] & g["fw"]
    fc, ob = g["precip_fc_mm"], g["precip_ob_mm"]
    cols = dict(n=np.ones(len(g)), dT=g["dT"], dRH=g["dRH"], dWS=g["dWS"],
                sol_d=g["sol_d"].where(day, 0.0), sol_c=g["sol_c"].where(day, 0.0),
                vis_d=g["vis_d"].where(below, 0.0), vis_c=g["vis_c"].where(below, 0.0),
                fc=fc, ob=ob, hit_fc=fc.where(hit, 0.0), hit_ob=ob.where(hit, 0.0),
                miss_ob=ob.where(miss, 0.0), fa_fc=fc.where(fa, 0.0))
    return pd.DataFrame(cols, index=g.index).groupby([g["horizon"], g["fold"]]).sum()


STATS = {
    "dT": lambda S: S["dT"] / S["n"], "dRH": lambda S: S["dRH"] / S["n"],
    "dWS": lambda S: S["dWS"] / S["n"], "solar": lambda S: S["sol_d"] / S["sol_c"],
    "vis": lambda S: S["vis_d"] / S["vis_c"], "total": lambda S: S["fc"] / S["ob"],
    "hit": lambda S: S["hit_fc"] / S["hit_ob"], "missed": lambda S: S["miss_ob"] / S["ob"],
    "fa": lambda S: S["fa_fc"] / S["ob"],
}


def bootstrap_halfwidth(w, n_draws=1000, seed=0):
    """95 % half-width (1.96 SD) of each statistic when the test windows are
    resampled with replacement within each horizon."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _, wh in w.groupby(level="horizon"):
        a = wh.to_numpy()
        counts = rng.multinomial(len(a), np.full(len(a), 1.0 / len(a)), size=n_draws)
        tot = tot + counts @ a
    S = pd.DataFrame(tot, columns=w.columns)
    return {k: 1.96 * float(np.nanstd(f(S))) for k, f in STATS.items()}


print("\nMean change degraded vs clean (all horizons; per horizon in the CSV);")
print("(+/-): 95 % range from resampling the test windows (chance variation of one seed):")
for city, g in out.groupby("city", sort=False):
    w = window_sums(g)
    v = {k: float(f(w.sum())) for k, f in STATS.items()}
    e = bootstrap_halfwidth(w)
    print(f"  {city:10s} temperature {v['dT']:+.3f} (+/-{e['dT']:.3f}) C, humidity {v['dRH']:+.2f} (+/-{e['dRH']:.2f}) %-pts, "
          f"wind {v['dWS']:+.3f} (+/-{e['dWS']:.3f}) m/s, solar (daylight) x{v['solar']:.3f} (+/-{e['solar']:.3f})")
    print(f"  {'':10s} visibility below cap x{v['vis']:.2f} (+/-{e['vis']:.2f}), "
          f"precipitation total x{v['total']:.2f} (+/-{e['total']:.2f})")
    # precipitation total = hits + false alarms; clean total = hits + misses
    print(f"  {'':10s} precipitation hit amount x{v['hit']:.2f} (+/-{e['hit']:.2f}) (hours wet in both), "
          f"missed amount {v['missed']:.1%} (+/-{e['missed']:.1%}), "
          f"false-alarm amount {v['fa']:.1%} (+/-{e['fa']:.1%}) of the clean total")
