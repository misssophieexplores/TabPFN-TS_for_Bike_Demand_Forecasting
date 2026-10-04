"""
Pre-flight check before the paper runs (writes nothing to results/).

Run on a compute node, from the repository root, in the environment of the
experiments:
    python forecasting/testing/preflight.py                  # all cities
    python forecasting/testing/preflight.py --cities london  # one city
    python forecasting/testing/preflight.py --skip-models    # config/data checks only (seconds)

Checks
  Environment: code version (git_dirty), library versions, W&B mode.
  Per city:
    1. Params files named in the city config: exist, have provenance, and were
       tuned for this city, with this n_train_samples, with the expected
       scenario (clean_only / no_weather), and on data ending at the current
       evaluation cutoff.
    2. build_models() succeeds (same call as run_weather_baseline.py).
    3. Data: holiday column not all zero after loading; NWP calibration file loads.
    4. CV: expected number of folds and 5,880 evaluation hours for every horizon.
    5. One fold per model (last fold, h = 24): clean_only for every model, and
       degraded for models with covariates; the forecast has 24 finite values.
       This also starts TimesFM (server + weights) and loads the TabPFN
       checkpoint, so missing weights on the node show up here.
Exit code 1 if any check fails.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # forecasting/

import numpy as np
import pandas as pd

import run_experiments as rx
from evaluation.cv import TimeSeriesCV
from provenance import get_code_version, get_library_versions
from weather.nwp_error_model import load_error_model
from weather.weather_processor import WeatherProcessor

# params-file field -> (model key, expected 'scenario' in the JSON, or None = no check)
PARAMS = {
    "arima_params_file": ("arima", None),
    "sarimax_params_file": ("sarimax", "clean_only"),
    "xgb_params_file": ("xgboost", "clean_only"),
    "xgb_noweather_params_file": ("xgboost_noweather", "no_weather"),
    "prophet_params_file": ("prophet", None),
    "neuralprophet_params_file": ("neuralprophet", "clean_only"),
    "neuralprophet_noweather_params_file": ("neuralprophet_noweather", "no_weather"),
}

FAILURES = []


def fail(city, what, msg):
    FAILURES.append((city, what, msg))
    print(f"  [FAIL] {what}: {msg}")


def ok(what, msg=""):
    print(f"  [OK]   {what}{': ' + msg if msg else ''}")


# Keys build_models() reads from each params file (a file without them fails there)
REQUIRED_KEYS = {
    "arima": ["order", "with_intercept"],
    "sarimax": ["order", "seasonal_order", "with_intercept"],
    "xgboost": ["n_lags", "xgb_params"],
    "xgboost_noweather": ["n_lags", "xgb_params"],
    "prophet": ["prophet_params"],
    "neuralprophet": ["n_lags", "neuralprophet_params"],
    "neuralprophet_noweather": ["n_lags", "neuralprophet_params"],
}


def check_params(config, cutoff):
    """Check every params file named in the city config. Never raises: each
    problem is reported and the remaining files and checks still run."""
    import json
    from provenance import code_id
    seen = {}
    code_ids = {}   # params field -> code ID / commit the file was tuned with
    lib_versions = {}
    # Covariates the experiments give the clean_only models (after
    # load_and_prepare_data, i.e. with holiday and season, in experiment order)
    exp_covs = WeatherProcessor(config).get_weather_columns("clean_only")
    for field, (key, scenario) in PARAMS.items():
        path = getattr(config, field)
        if not path:  # None or "" (an empty string would open the current folder)
            fail(config.dataset_name, field, f"not set ({path!r})")
            continue
        if path in seen:
            fail(config.dataset_name, field, f"same file as {seen[path]} ({path})")
        seen[path] = field
        if not Path(path).is_file():
            fail(config.dataset_name, field, f"file not found: {path}")
            continue
        try:
            with open(path) as f:
                p = json.load(f)
            if not isinstance(p, dict):
                raise ValueError("not a JSON object")
        except Exception as e:
            fail(config.dataset_name, field, f"{path} is not a params JSON file "
                 f"({type(e).__name__}: {e})")
            continue
        problems = []
        if "provenance" not in p:
            problems.append("no provenance")
        else:
            code_ids[field] = p["provenance"].get("git_commit")
            lib_versions[field] = p["provenance"].get("library_versions")
        if p.get("city") != config.dataset_name:
            problems.append(f"city={p.get('city')!r}")
        if p.get("n_train_samples") != config.n_train_samples:
            problems.append(f"n_train_samples={p.get('n_train_samples')}")
        if scenario is not None and p.get("scenario") != scenario:
            problems.append(f"scenario={p.get('scenario')!r}, expected {scenario!r}")
        last = (p.get("tuning_period") or {}).get("last_timestamp")
        if last is None or pd.Timestamp(last) != cutoff:
            problems.append(f"tuning data end {last} != evaluation cutoff {cutoff}")
        missing = [k for k in REQUIRED_KEYS[key] if k not in p]
        if missing:
            problems.append(f"missing {missing}" + (" (old tuning code)" if "with_intercept" in missing else ""))
        if key.startswith("xgboost") and "n_estimators" not in (p.get("xgb_params") or {}):
            problems.append("xgb_params has no n_estimators")
        # Tuned on the covariates (and, for XGBoost, the column order) the
        # experiments use: clean_only columns, or none for no_weather
        if scenario is not None:  # SARIMAX, XGBoost(_NoWeather), NeuralProphet(_NoWeather)
            expected = [] if scenario == "no_weather" else exp_covs
            if "covariates_used" not in p:
                problems.append("no covariates_used")
            elif list(p["covariates_used"]) != list(expected):
                problems.append(f"covariates_used {p['covariates_used']} != experiment columns {expected}")
        if problems:
            fail(config.dataset_name, field, f"{path}: " + "; ".join(problems))
        else:
            ok(field, f"{Path(path).name} (tuned with {code_ids.get(field)})")

    # Provenance overview (information, not a failure): the code ID covers all
    # .py/.npz files in forecasting/ incl. the city configs, so files tuned
    # before and after a params-path edit already differ. Whether a change
    # between tuning and now affects a model is answered by `git diff`.
    if code_ids:
        distinct = sorted(set(map(str, code_ids.values())))
        print(f"  [INFO] params files tuned with {len(distinct)} code version(s): {distinct}; "
              f"current code ID {code_id()}")
        if len(distinct) > 1:
            for cid in distinct:
                print(f"         {cid}: {[f for f, c in code_ids.items() if str(c) == cid]}")
    libs = {f: json.dumps(v, sort_keys=True) for f, v in lib_versions.items() if v}
    if len(set(libs.values())) > 1:
        print("  [WARN] params files were tuned with different library versions:")
        for f, v in libs.items():
            print(f"         {f}: {v}")


def run_one_fold(config, model, df, scenario, horizon=24):
    cv = TimeSeriesCV(config)
    splits = cv.split(df, horizon, partial_last_fold=True)
    fold_idx = len(splits) - 1
    train_df, test_df = splits[fold_idx]
    wp = WeatherProcessor(config)
    y_train, X_train, y_test, X_test = rx.prepare_fold_inputs(
        config, model, wp, train_df, test_df, scenario, horizon, fold_idx)
    model.reset()
    t0 = time.perf_counter()
    model.fit(y_train, X_train)
    y_pred = np.asarray(model.predict(len(test_df), X_test), dtype=float).ravel()
    secs = time.perf_counter() - t0
    if len(y_pred) != len(test_df) or not np.all(np.isfinite(y_pred)):
        raise RuntimeError(f"forecast has {len(y_pred)} values, "
                           f"{int((~np.isfinite(y_pred)).sum())} non-finite")
    mae = float(np.mean(np.abs(y_pred - y_test)))
    return secs, mae, int((y_pred < 0).sum())


def check_city(get_config, run_models):
    config = get_config()
    city = config.dataset_name
    print(f"\n===== {city} | degradation: {config.degradation_label()} | "
          f"scenarios run by main.py: {['clean_only'] + list(config.degradation_scales)}")
    df, _ = rx.load_and_prepare_data(config)
    cv = TimeSeriesCV(config)
    cutoff = cv.get_cutoff_date(df)

    # 1. params files
    check_params(config, cutoff)

    # 3. data
    if config.holiday_col:
        n_hol = int(df[config.holiday_col].sum())
        if n_hol == 0:
            fail(city, "holiday column", f"'{config.holiday_col}' is all 0 after loading "
                 f"(pandas {pd.__version__}: check the holiday mapping)")
        else:
            ok("holiday column", f"{n_hol} holiday hours")
    try:
        load_error_model(config.nwp_calibration_file)
        ok("NWP calibration file", config.nwp_calibration_file)
    except Exception as e:
        fail(city, "NWP calibration file", str(e))

    # 4. CV coverage
    for h in config.horizons:
        splits = cv.split(df, h, partial_last_fold=True)
        hours = sum(len(t) for _, t in splits)
        if len(splits) != cv.expected_n_folds(h) or hours != cv.get_eval_hours():
            fail(city, f"CV h={h}", f"{len(splits)} folds / {hours} h, expected "
                 f"{cv.expected_n_folds(h)} / {cv.get_eval_hours()}")
    ok("CV", f"cutoff {cutoff}, folds per horizon "
       f"{[cv.expected_n_folds(h) for h in config.horizons]}, {cv.get_eval_hours()} h each")

    # 2. build_models
    try:
        models = rx.build_models(config)
        ok("build_models", ", ".join(m.name for m in models))
    except Exception as e:
        fail(city, "build_models", f"{type(e).__name__}: {e}")
        return

    # 5. one fold per model
    if not run_models:
        return
    for model in models:
        scenarios = ["clean_only"] + ([s for s in config.degradation_scales] if model.use_covariates else [])
        for scen in scenarios:
            what = f"{model.name} | {scen} | h=24, last fold"
            try:
                secs, mae, n_neg = run_one_fold(config, model, df, scen)
                ok(what, f"{secs:.1f} s, MAE {mae:.1f}" + (f", {n_neg} negative forecasts" if n_neg else ""))
            except Exception as e:
                fail(city, what, f"{type(e).__name__}: {e}")
                traceback.print_exc(limit=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cities", nargs="+", default=["seoul", "washington", "london"],
                        choices=["seoul", "washington", "london"])
    parser.add_argument("--skip-models", action="store_true",
                        help="only config/data/params checks (no model fits)")
    args = parser.parse_args()

    print("===== environment")
    cv_ = get_code_version()
    print(f"  code: {cv_}")
    if cv_.get("git_dirty"):
        FAILURES.append(("env", "git", "uncommitted changes (results would be tagged git_dirty=True)"))
        print("  [FAIL] git: uncommitted changes")
    print(f"  libraries: {get_library_versions()}")
    mode = os.getenv("WANDB_MODE", "online (default)")
    print(f"  WANDB_MODE={mode}, WANDB_ENTITY={'set' if os.getenv('WANDB_ENTITY') else 'NOT SET'}")
    if mode not in ("offline", "disabled") and not os.getenv("HTTPS_PROXY"):
        import socket
        try:
            socket.create_connection(("api.wandb.ai", 443), timeout=5).close()
            print("  [OK]   api.wandb.ai reachable")
        except OSError:
            # warning only: W&B may still work through a proxy
            print("  [WARN] api.wandb.ai not reachable from this node and WANDB_MODE is online: "
                  "wandb.init() would fail and main.py would skip the city. "
                  "Set WANDB_MODE=offline (sync later with `wandb sync`).")

    import config_seoul, config_washington, config_london
    getters = {"seoul": config_seoul.get_config, "washington": config_washington.get_config,
               "london": config_london.get_config}
    for city in args.cities:
        try:
            check_city(getters[city], run_models=not args.skip_models)
        except Exception as e:
            fail(city, "city check", f"{type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n===== summary")
    if FAILURES:
        for city, what, msg in FAILURES:
            print(f"  [FAIL] {city} | {what}: {msg}")
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
