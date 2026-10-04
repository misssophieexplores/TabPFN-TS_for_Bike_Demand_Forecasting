"""
End-to-end check of the measured NWP error model on the real bike data:
the same CV folds as the experiments (TimeSeriesCV.split, partial_last_fold),
every horizon, scenario 'degraded', with the settings of the city configs
(default: fresh forecast, average lean removed). Writes the simulated
covariate errors per test hour (sim_errors_<seed>.csv) for comparison with
the calibration (ECMWF errors at the same lead times), and prints per city
the mean change of each covariate (degraded - clean): temperature, humidity
and wind in their units; visibility (hours below the cap) and precipitation
(total amount) as degraded/clean ratios. Precipitation is also split into
hits (degraded/clean amount over the hours wet in both), missed amount (clean
amount in hours wet in clean and dry in degraded) and false-alarm amount
(degraded amount in hours dry in clean and wet in degraded), the last two as
shares of the clean total. Visibility below the cap and hit amounts are
mean-preserving after their caps (config.nwp_mean_preserving_caps, default),
so these ratios are ~1 up to sampling variation (about +/-0.04 per seed for
hits); humidity (clipped at 0-100) and the precipitation total (false-alarm
amounts smaller than missed amounts in Seoul) are not exactly 0 / 1
(ARCHITECTURE.md, Known Limitation 24).

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
print("\nMean change degraded vs clean (all horizons; per horizon in the CSV):")
for city, g in out.groupby("city", sort=False):
    b = g[g["vis_below_cap"]]
    print(f"  {city:10s} temperature {g['dT'].mean():+.3f} C, humidity {g['dRH'].mean():+.2f} %-pts, "
          f"wind {g['dWS'].mean():+.3f} m/s | visibility below cap x{b['vis_d'].mean() / b['vis_c'].mean():.2f}, "
          f"precipitation total x{g['precip_fc_mm'].sum() / g['precip_ob_mm'].sum():.2f}")
    # precipitation total = hits + false alarms; clean total = hits + misses
    hit, miss, fa = g["ow"] & g["fw"], g["ow"] & ~g["fw"], ~g["ow"] & g["fw"]
    clean_total = g["precip_ob_mm"].sum()
    print(f"  {'':10s} precipitation hit amount x{g.loc[hit, 'precip_fc_mm'].sum() / g.loc[hit, 'precip_ob_mm'].sum():.2f} "
          f"(hours wet in both), missed amount {g.loc[miss, 'precip_ob_mm'].sum() / clean_total:.1%}, "
          f"false-alarm amount {g.loc[fa, 'precip_fc_mm'].sum() / clean_total:.1%} of the clean total")
