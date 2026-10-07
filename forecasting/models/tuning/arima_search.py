"""
Order search shared by tune_arima.py and tune_sarimax.py.

Procedure
---------
1. Candidates: pmdarima auto_arima (stepwise, AIC; d and D by its unit-root
   tests; intercept chosen by pmdarima) on the training window of each of the
   --search-folds candidate folds (default 6, spread evenly over the tune
   folds). Each distinct (order, seasonal_order, with_intercept) is a candidate.
2. Selection: every candidate is fitted on ALL tune folds (the same 90 folds
   as XGBoost and Prophet) with the model used in the experiments
   (ARIMAForecaster / SARIMAXForecaster, statsmodels, trend from
   with_intercept via trend_from_intercept), with exactly the experiment
   inputs (fold_runner.prepare_fold_inputs), and scored on the next
   tune_horizon hours: MAE, imputed hours excluded. The candidate with the
   lowest mean MAE is selected - the same criterion and folds as XGBoost and
   Prophet. With require_converged=True (SARIMAX, since 7 Oct 2026) only
   candidates whose optimizer converged on every tune fold are selectable,
   as long as at least one candidate did (select_candidate()).

A candidate that fails on a tune fold, or has no statsmodels trend
equivalent, is not selectable (reason saved in the JSON).
"""
import traceback
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from config import ForecastConfig
from evaluation.cv import TimeSeriesCV
from evaluation.metrics import MetricsCalculator
from models.statistical import ARIMAForecaster, SARIMAXForecaster, trend_from_intercept
from fold_runner import prepare_fold_inputs
from weather.weather_processor import WeatherProcessor


def _drop_constant(X: pd.DataFrame):
    """Same rule as SARIMAXForecaster: drop covariates constant in the training window."""
    if X is None:
        return None
    keep = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
    return X[keep] if keep else None


def _make_model(seasonal: bool, order, seasonal_order, trend):
    if seasonal:
        return SARIMAXForecaster(order=order, seasonal_order=seasonal_order, trend=trend)
    return ARIMAForecaster(order=order, trend=trend)


RULE_LOWEST_MAE = "lowest mean MAE on all tune folds"
RULE_CONVERGED = ("lowest mean MAE among the candidates whose optimizer converged on "
                  "every tune fold")


def select_candidate(candidates: List[Dict], require_converged: bool):
    """
    (selected candidate, rule applied) from the scored candidates (dicts of the
    tuning JSON). Only status "ok" candidates are selectable. With
    require_converged, candidates with non_converged_folds > 0 are left out
    if at least one candidate converged on every fold; otherwise the rule
    falls back to the lowest mean MAE (the rule string says so).
    """
    ok = [c for c in candidates if c.get("status") == "ok"]
    if not ok:
        raise RuntimeError(f"No selectable candidate: {candidates}")
    if require_converged:
        converged = [c for c in ok if c.get("non_converged_folds", 0) == 0]
        if converged:
            return min(converged, key=lambda c: c["mae_mean"]), RULE_CONVERGED
        return (min(ok, key=lambda c: c["mae_mean"]),
                RULE_LOWEST_MAE + " (no candidate converged on every tune fold)")
    return min(ok, key=lambda c: c["mae_mean"]), RULE_LOWEST_MAE


def search_orders(
    df: pd.DataFrame,
    config: ForecastConfig,
    seasonal: bool,
    scenario: str,
    auto_arima_kwargs: Dict,
    search_folds: int = 6,
    verbose: bool = True,
    require_converged: bool = False,
) -> Dict:
    from pmdarima import auto_arima

    cv = TimeSeriesCV(config)
    calc = MetricsCalculator()
    weather_proc = WeatherProcessor(config)
    h = config.tune_horizon
    fday = config.functioning_day_col

    # Only tune on pre-cutoff data — never touch the held-out test period
    cutoff_date = cv.get_cutoff_date(df)
    tune_df = df[df[config.date_col] <= cutoff_date].copy()
    splits = cv.split(tune_df, h)
    tune_idx = cv.tune_fold_indices(splits)
    search_idx = cv.spread_fold_indices(tune_idx, search_folds)
    if not search_idx:
        raise RuntimeError("No scored tune folds available.")

    # Prototype only decides which inputs prepare_fold_inputs builds
    proto = _make_model(seasonal, (1, 0, 0), (0, 0, 0, 0), "n")
    covariates = weather_proc.get_weather_columns(scenario) if proto.use_covariates else []

    if verbose:
        print(f"Cutoff date (held-out test start): {cutoff_date}")
        print(f"Candidate folds (auto_arima): {search_idx} | scoring folds: all {len(tune_idx)} tune folds")
        print(f"Covariates ({len(covariates)}): {covariates}")

    # --- 1. Candidates from auto_arima on every search fold
    candidates: Dict[tuple, Dict] = {}
    for i in search_idx:
        train_df, test_df = splits[i]
        y_tr, X_tr, _, _ = prepare_fold_inputs(config, proto, weather_proc, train_df, test_df, scenario, h, i)
        X_fit = _drop_constant(X_tr)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fitted = auto_arima(y_tr, X=None if X_fit is None else X_fit.values, **auto_arima_kwargs)
        order = tuple(int(v) for v in fitted.order)
        seasonal_order = tuple(int(v) for v in fitted.seasonal_order) if seasonal else (0, 0, 0, 0)
        key = (order, seasonal_order, bool(fitted.with_intercept))
        cand = candidates.setdefault(key, {
            "order": list(order), "seasonal_order": list(seasonal_order),
            "with_intercept": key[2], "found_on_folds": [], "aic_on_found_folds": [],
        })
        cand["found_on_folds"].append(int(i))
        cand["aic_on_found_folds"].append(float(fitted.aic()))
        if verbose:
            print(f"  fold {i}: auto_arima -> order={order} seasonal={seasonal_order} "
                  f"intercept={key[2]} AIC={fitted.aic():.1f}")

    # --- 2. Score every candidate with the experiment model on ALL tune folds
    for key, cand in candidates.items():
        order, seasonal_order, with_intercept = key
        try:
            trend = trend_from_intercept(with_intercept, order, seasonal_order, sarimax=seasonal)
        except ValueError as e:
            cand["status"] = f"not selectable: {e}"
            continue
        cand["trend"] = trend
        maes, rmses, non_converged = [], [], 0
        for i in tune_idx:
            train_df, test_df = splits[i]
            model = _make_model(seasonal, order, seasonal_order, trend)
            y_tr, X_tr, y_te, X_te = prepare_fold_inputs(
                config, model, weather_proc, train_df, test_df, scenario, h, i
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(y_tr, X_tr)
                    y_pred = model.predict(len(test_df), X_te)
                # "converge" matches both SARIMAXForecaster's own warning and
                # statsmodels' ConvergenceWarning (ARIMA), as in run_experiments.py
                non_converged += any("converge" in str(w.message).lower() for w in caught)
                m = calc.calculate_all(
                    y_te, y_pred, y_tr,
                    test_mask=calc.observed_mask(test_df, fday),
                    train_mask=calc.observed_mask(train_df, fday),
                )
            except Exception as e:
                if verbose:
                    traceback.print_exc()
                cand["status"] = f"not selectable: failed on fold {i}: {str(e)[:200]}"
                break
            maes.append(float(m["MAE"]))
            rmses.append(float(m["RMSE"]))
        else:
            cand.update(status="ok", fold_mae=maes, mae_mean=float(np.mean(maes)),
                        rmse_mean=float(np.mean(rmses)), non_converged_folds=int(non_converged))
        if verbose:
            print(f"  candidate {key}: {cand.get('status')} "
                  f"{'MAE=%.2f' % cand['mae_mean'] if 'mae_mean' in cand else ''}")

    best, rule = select_candidate(list(candidates.values()), require_converged)

    return {
        "tuning_period": cv.get_tuning_period(tune_df),
        "order": best["order"],
        "seasonal_order": best["seasonal_order"],
        "with_intercept": best["with_intercept"],
        "trend": best["trend"],
        "covariates_used": list(covariates),
        "tuning": {
            "search_type": "auto_arima candidates (stepwise AIC per candidate fold), "
                           "selected by mean MAE of the experiment model on all tune folds",
            "tune_folds": len(tune_idx),
            "scoring_folds": len(tune_idx),
            "candidate_folds": len(search_idx),
            "candidate_fold_indices": [int(i) for i in search_idx],
            "metric_optimized": "MAE",
            "selection_rule": rule,
            "best_tune_mae_mean": best["mae_mean"],
            "best_tune_rmse_mean": best["rmse_mean"],
            "auto_arima_kwargs": {k: v for k, v in auto_arima_kwargs.items() if k != "trace"},
            "candidates": list(candidates.values()),
        },
    }
