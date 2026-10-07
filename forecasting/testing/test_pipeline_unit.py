"""
Unit tests for the evaluation pipeline: CV folds, metrics, and
tuning-vs-experiment consistency. Synthetic data only (no CSVs needed).

Run with: pytest forecasting/testing/test_pipeline_unit.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))                      # forecasting/
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "tuning"))  # tuning scripts

import numpy as np
import pandas as pd
import pytest

from config_seoul import get_config
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator as M

N_HOURS = 8760  # one year, as Seoul (shortest dataset)
COVS = ["Temperature", "Humidity", "Wind speed", "Dew point temperature",
        "Solar Radiation", "precipitation_mm", "snow_depth_cm", "Visibility"]


def make_df(n=N_HOURS, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2017-12-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Date": ts,
        "Rented Bike Count": (500 + 400 * np.sin(2 * np.pi * ts.hour.values / 24)
                              + rng.normal(0, 50, n)).clip(0),
        "Functioning Day": "Yes",
        "Holiday": (rng.random(n) < 0.03).astype(int),
        "Seasons": (ts.month.values % 12) // 3,
    })
    for c in COVS:
        df[c] = rng.normal(size=n)
    df["Solar Radiation"] = df["Solar Radiation"].abs()
    df["precipitation_mm"] = df["precipitation_mm"].clip(0)
    df["snow_depth_cm"] = df["snow_depth_cm"].clip(0)
    return df


@pytest.fixture
def config():
    return get_config()


# ----------------------------------------------------------------------
# CV
# ----------------------------------------------------------------------
class TestCV:
    def test_train_test_disjoint_and_train_size(self, config):
        df = make_df()
        for train_df, test_df in TimeSeriesCV(config).split(df, 24, partial_last_fold=True):
            assert len(train_df) == config.n_train_samples
            assert train_df["Date"].max() < test_df["Date"].min()
            assert train_df["Date"].max() + pd.Timedelta(hours=1) == test_df["Date"].min()

    def test_every_horizon_covers_the_evaluation_period(self, config):
        df = make_df()
        cv = TimeSeriesCV(config)
        cutoff = cv.get_cutoff_date(df)
        for h in config.horizons:
            splits = cv.split(df, h, partial_last_fold=True)
            test_hours = pd.concat([t["Date"] for _, t in splits])
            assert len(splits) == cv.expected_n_folds(h)
            assert test_hours.is_unique
            assert len(test_hours) == cv.get_eval_hours()
            assert test_hours.min() == cutoff + pd.Timedelta(hours=1)
            assert test_hours.max() == df["Date"].max()

    def test_tuning_never_touches_the_evaluation_period(self, config):
        df = make_df()
        cv = TimeSeriesCV(config)
        cutoff = cv.get_cutoff_date(df)
        tune_df = df[df["Date"] <= cutoff]
        splits = cv.split(tune_df, config.tune_horizon)
        assert len(splits) == config.tune_folds  # Seoul length: exactly 90 tune folds
        for train_df, test_df in splits:
            assert test_df["Date"].max() <= cutoff
            assert train_df["Date"].max() <= cutoff

    def test_gap_is_reported(self, config, capsys):
        df = make_df().drop(index=range(2000, 2003))
        cv = TimeSeriesCV(config)
        tune_df = df[df["Date"] <= cv.get_cutoff_date(df)]
        cv.split(tune_df, 24)
        assert "WARNING: fold with test window" in capsys.readouterr().out

    def test_tune_and_search_folds(self, config):
        df = make_df()
        cv = TimeSeriesCV(config)
        tune_df = df[df["Date"] <= cv.get_cutoff_date(df)].copy()
        splits = cv.split(tune_df, 24)
        assert cv.tune_fold_indices(splits) == list(range(90))
        # a fully imputed test day is not a tune fold
        day = splits[40][1].index
        tune_df.loc[day, "Functioning Day"] = "No"
        splits = cv.split(tune_df, 24)
        assert 40 not in cv.tune_fold_indices(splits)
        # same spread folds in every script (NeuralProphet, ARIMA, SARIMAX)
        assert TimeSeriesCV.spread_fold_indices(list(range(90)), 6) == [0, 18, 36, 53, 71, 89]

    def test_split_info_aligned(self, config):
        df = make_df()
        df.loc[2000:2010, "Functioning Day"] = "No"
        cv = TimeSeriesCV(config)
        splits = cv.split(df[df["Date"] <= cv.get_cutoff_date(df)], 24)
        info = cv.get_split_info(splits)
        expected = [int((t["Functioning Day"] == "No").sum()) for _, t in splits]
        assert info["test_imputed"].tolist() == expected


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
class TestMetrics:
    def test_mase_uses_seasonal_naive_on_training_data(self):
        y_train = np.tile(np.arange(24.0), 10) + np.repeat(np.arange(10.0), 24)  # naive error = 1
        assert M.mase(np.array([5.0]), np.array([7.0]), y_train) == pytest.approx(2.0)

    def test_mase_ignores_imputed_training_pairs(self):
        y_train = np.tile(np.arange(24.0), 10) + np.repeat(np.arange(10.0), 24)
        y_train[100] = 1e6
        mask = np.ones_like(y_train, dtype=bool)
        mask[100] = False
        assert M.mase(np.array([5.0]), np.array([7.0]), y_train, train_mask=mask) == pytest.approx(2.0)

    def test_smape_zero_counts(self):
        assert M.smape(np.array([0.0]), np.array([4.0])) == pytest.approx(200.0)
        assert M.smape(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)

    def test_imputed_test_hours_not_scored(self):
        m = M.calculate_all(np.array([10.0, 1000.0]), np.array([10.0, 0.0]), np.arange(100.0),
                            test_mask=np.array([True, False]))
        assert m["MAE"] == 0.0

    def test_broken_forecast_raises(self):
        with pytest.raises(ValueError):
            M.calculate_all(np.array([1.0, 2.0]), np.array([1.0, np.nan]), np.arange(100.0))
        with pytest.raises(ValueError):
            M.calculate_all(np.array([1.0, 2.0]), np.array([1.0]), np.arange(100.0))

    def test_win_rate_and_skill_score(self):
        e_j = np.array([0.5, 1.0, 2.0, 1.0])
        e_b = np.array([1.0, 1.0, 1.0, 2.0])
        assert M.win_rate(e_j, e_b) == pytest.approx((1 + 0.5 + 0 + 1) / 4)
        expected = 1 - np.exp(np.mean(np.log([0.5, 1.0, 2.0, 0.5])))
        assert M.skill_score(e_j, e_b) == pytest.approx(expected)

    def _results(self):
        rows = []
        for d in ["a", "b", "c"]:
            for h in [6, 24, 48, 168]:
                rows.append(dict(dataset=d, horizon=h, weather_scenario="clean_only",
                                 model="Seasonal_Naive", MASE_mean=1.0))
                for sc in ["clean_only", "degraded"]:
                    rows.append(dict(dataset=d, horizon=h, weather_scenario=sc,
                                     model="X", MASE_mean=0.8))
        return pd.DataFrame(rows)

    def test_comparative_metrics(self):
        out = M.compute_comparative_metrics(self._results())
        assert set(out["weather_scenario"]) == {"clean_only", "degraded"}
        assert (out["n_tasks"] == 12).all()
        assert (out["win_rate"] == 1.0).all()

    def test_comparative_metrics_rejects_duplicates(self):
        r = self._results()
        with pytest.raises(ValueError):
            M.compute_comparative_metrics(pd.concat([r, r.iloc[[1]]]))


# ----------------------------------------------------------------------
# Tuning vs experiment consistency (XGBoost)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["clean_only", "no_weather"])
def test_xgboost_tuning_equals_experiment(config, scenario):
    """tune_xgboost (no early stopping) and the experiment path
    (build inputs with prepare_fold_inputs, XGBoostForecaster) must give
    identical forecasts for the same parameters. no_weather: tuning with
    --scenario no_weather and XGBoostForecaster_NoWeather (calendar features only)."""
    pytest.importorskip("xgboost")
    import tune_xgboost as tx
    from models.ml_models import XGBoostForecaster, XGBoostForecaster_NoWeather
    from run_experiments import prepare_fold_inputs
    from weather.weather_processor import WeatherProcessor

    df = make_df()
    cfg_tune = get_config()
    cfg_tune.weather_covariates = tx.select_covariates(cfg_tune, df, scenario)
    cv = TimeSeriesCV(cfg_tune)
    tune_df = df[df["Date"] <= cv.get_cutoff_date(df)]
    train_df, test_df = cv.split(tune_df, 24)[-1]

    params = {"objective": "reg:squarederror", "random_state": 42, "tree_method": "hist",
              "n_jobs": 1, "n_estimators": 30, "max_depth": 4, "learning_rate": 0.1}

    captured = {}
    orig = tx.iterative_forecast
    tx.iterative_forecast = lambda **kw: captured.setdefault("pred", orig(**kw))
    try:
        tx.evaluate_params_on_fold(train_df, test_df, cfg_tune, params, 24, 8000,
                                   use_early_stopping=False)
    finally:
        tx.iterative_forecast = orig

    cfg_exp = get_config()
    cfg_exp.weather_covariates = COVS + ["Holiday", "Seasons"]  # as after load_and_prepare_data
    model_cls = XGBoostForecaster_NoWeather if scenario == "no_weather" else XGBoostForecaster
    model = model_cls(n_lags=24, **params)
    y_train, X_train, _, X_test = prepare_fold_inputs(
        cfg_exp, model, WeatherProcessor(cfg_exp), train_df, test_df, "clean_only", 24, 0
    )
    if scenario == "no_weather":
        assert list(X_train.columns) == ["hour", "dayofweek", "month", "is_weekend"]
    model.fit(y_train, X_train)
    pred = model.predict(len(test_df), X_test)
    np.testing.assert_allclose(pred, captured["pred"], rtol=0, atol=1e-9)


# ----------------------------------------------------------------------
# Tuning output: checked before tuning, restorable from the log
# (3 Oct 2026: read-only folder on the cluster, params files lost)
# ----------------------------------------------------------------------
def test_tuning_output_dir_checked_before_tuning(tmp_path):
    from provenance import check_output_dir
    assert check_output_dir(tmp_path / "a" / "b").is_dir()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(PermissionError):
        check_output_dir(blocker / "tuning")      # no folder can be made inside a file


def test_params_json_restorable_from_log(tmp_path, capsys):
    import json
    from provenance import save_params_json
    out = save_params_json({"city": "seoul", "n_lags": 24}, tmp_path / "p.json")
    log = capsys.readouterr().out
    body = log.split("----- BEGIN PARAMS JSON p.json -----\n")[1].split("\n----- END PARAMS JSON p.json -----")[0]
    assert json.loads(body) == json.loads(out.read_text())
    assert "provenance" in json.loads(body)


def test_empty_params_path_is_not_set(config):
    import run_experiments as rx
    config.xgb_noweather_params_file = ""
    with pytest.raises(ValueError, match="not set"):
        rx.build_models(config, keys=["xgboost_noweather"])


# ----------------------------------------------------------------------
# ARIMA / SARIMAX intercept (pmdarima with_intercept -> statsmodels trend)
# ----------------------------------------------------------------------
def test_intercept_with_one_difference_is_a_drift():
    """pmdarima's intercept with d = 1 is a constant drift. Both experiment
    models must forecast a straight line (constant step), not a curve."""
    pytest.importorskip("statsmodels")
    from models.statistical import ARIMAForecaster, SARIMAXForecaster, trend_from_intercept

    rng = np.random.default_rng(0)
    y = np.cumsum(2.0 + rng.normal(0, 1, 720))  # random walk with drift 2
    order = (0, 1, 0)

    arima = ARIMAForecaster(order=order, trend=trend_from_intercept(True, order))
    sarimax = SARIMAXForecaster(order=order, seasonal_order=(0, 0, 0, 0),
                                trend=trend_from_intercept(True, order, (0, 0, 0, 0), sarimax=True))
    arima.fit(y)
    sarimax.fit(y, None)
    for pred in (arima.predict(168), sarimax.predict(168, pd.DataFrame(index=range(168)))):
        steps = np.diff(pred)
        assert np.ptp(steps) < 1e-6                      # constant drift, not a growing slope
        assert steps.mean() == pytest.approx(2.0, abs=0.2)


def test_trend_from_intercept_mapping():
    pytest.importorskip("statsmodels")
    from models.statistical import trend_from_intercept
    assert trend_from_intercept(False, (1, 1, 1), (1, 1, 1, 24), sarimax=True) == "n"
    assert trend_from_intercept(True, (2, 0, 1), sarimax=False) == "c"
    assert trend_from_intercept(True, (2, 1, 1), sarimax=False) == "t"
    assert trend_from_intercept(True, (2, 0, 1), (1, 0, 1, 24), sarimax=True) == "c"
    assert trend_from_intercept(True, (2, 1, 1), (1, 0, 1, 24), sarimax=True) == "c"
    assert trend_from_intercept(True, (2, 0, 1), (1, 1, 1, 24), sarimax=True) == "c"
    with pytest.raises(ValueError):
        trend_from_intercept(True, (2, 2, 1), sarimax=False)


# ----------------------------------------------------------------------
# SARIMAX selection: converged candidates only (7 Oct 2026)
# ----------------------------------------------------------------------
def _candidates():
    # Seoul, 7 Oct 2026 tuning file: lowest MAE did not converge on 48 of 90 folds
    def c(order, sorder, mae, nc):
        return {"order": order, "seasonal_order": sorder, "with_intercept": True, "trend": "c",
                "status": "ok", "mae_mean": mae, "rmse_mean": mae * 1.3, "non_converged_folds": nc}
    return [c([2, 0, 2], [1, 0, 1, 24], 92.15, 48), c([2, 0, 0], [1, 0, 0, 24], 94.05, 0),
            c([3, 0, 0], [2, 0, 0, 24], 94.30, 4), c([2, 0, 0], [2, 0, 0, 24], 94.22, 0),
            {"order": [5, 0, 0], "seasonal_order": [1, 0, 0, 24], "with_intercept": True,
             "status": "not selectable: failed on fold 3"}]


def test_sarimax_selection_requires_convergence():
    pytest.importorskip("statsmodels")
    from arima_search import select_candidate, RULE_CONVERGED, RULE_LOWEST_MAE
    best, rule = select_candidate(_candidates(), require_converged=True)
    assert (best["order"], best["seasonal_order"], rule) == ([2, 0, 0], [1, 0, 0, 24], RULE_CONVERGED)
    best, rule = select_candidate(_candidates(), require_converged=False)
    assert (best["order"], rule) == ([2, 0, 2], RULE_LOWEST_MAE)
    # no candidate converged on every fold -> lowest MAE, and the rule says so
    none = [dict(c, non_converged_folds=1) for c in _candidates() if c["status"] == "ok"]
    best, rule = select_candidate(none, require_converged=True)
    assert best["order"] == [2, 0, 2] and "no candidate converged" in rule


def test_reselect_sarimax_keeps_tuning_fields():
    pytest.importorskip("statsmodels")
    from reselect_sarimax import reselect
    source = {"city": "seoul", "scenario": "clean_only", "n_train_samples": 720,
              "tuning_period": {"first_timestamp": "a", "last_timestamp": "b"},
              "order": [2, 0, 2], "seasonal_order": [1, 0, 1, 24], "with_intercept": True,
              "trend": "c", "covariates_used": ["Temperature", "Holiday"], "m": 24,
              "tuning": {"best_tune_mae_mean": 92.15, "candidates": _candidates()},
              "provenance": {"git_commit": "code:x", "git_dirty": None}}
    p = reselect(source, "src.json")
    assert (p["order"], p["seasonal_order"], p["trend"]) == ([2, 0, 0], [1, 0, 0, 24], "c")
    assert p["tuning"]["best_tune_mae_mean"] == 94.05
    for k in ("city", "scenario", "n_train_samples", "tuning_period", "covariates_used", "m"):
        assert p[k] == source[k]
    assert "provenance" not in p                      # save_params_json adds the current one
    assert p["reselection"]["source_provenance"] == source["provenance"]
    assert p["reselection"]["source_selection"]["order"] == [2, 0, 2]
    with pytest.raises(ValueError, match="already a re-selected"):
        reselect(p, "p.json")


# ----------------------------------------------------------------------
# Parallel folds (SARIMAX) give the same results as one after another
# ----------------------------------------------------------------------
_TIMING_KEYS = {"timestamp", "fit_time_s", "predict_time_s", "runtime_s"}


@pytest.mark.parametrize("scenario", ["clean_only", "degraded"])
def test_parallel_folds_equal_sequential(scenario):
    pytest.importorskip("statsmodels")
    pytest.importorskip("joblib")
    pytest.importorskip("threadpoolctl")
    from fold_runner import run_fold, run_folds_parallel
    from models.statistical import SARIMAXForecaster
    from weather.weather_processor import WeatherProcessor

    cfg = get_config()
    cfg.weather_covariates = COVS + ["Holiday", "Seasons"]  # as after load_and_prepare_data
    splits = TimeSeriesCV(cfg).split(make_df(), 24, partial_last_fold=True)[:6]
    model = SARIMAXForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 24), trend="c")
    args = (cfg, model, WeatherProcessor(cfg))
    code = {"git_commit": "code:test", "git_dirty": None}

    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):   # one math thread, as in the workers
        seq = [run_fold(*args, tr, te, scenario, 24, i, "run", code) for i, (tr, te) in enumerate(splits)]
    par = run_folds_parallel(*args, splits, scenario, 24, "run", code, n_workers=3)
    assert [o[0] for o in par] == ["ok"] * len(splits)
    for (m_seq, f_seq), (_, m_par, f_par) in zip(seq, par):
        assert {k: v for k, v in m_seq.items() if k not in _TIMING_KEYS} == \
               {k: v for k, v in m_par.items() if k not in _TIMING_KEYS}
        pd.testing.assert_frame_equal(f_seq, f_par)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
