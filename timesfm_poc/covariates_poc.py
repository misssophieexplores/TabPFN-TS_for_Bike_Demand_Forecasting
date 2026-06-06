#!/usr/bin/env python3
"""
TimesFM 2.5 Covariates (XReg) POC

Minimal end-to-end test of forecast_with_covariates() on the TimesFM 2.5 model.
Builds tiny synthetic 3-series retail data, runs covariate + baseline forecasts,
and prints shapes + MAE so you can confirm covariate integration works and helps.

Requires:
    pip install "timesfm[xreg]"
    # plus a torch backend installed for your OS/accelerator

Notes:
    - TimesFM 2.5 uses TimesFM_2p5_200M_torch.from_pretrained(...) + .compile(ForecastConfig(...)).
      The old TimesFm(hparams=..., checkpoint=...) constructor is V1/2.0 and will NOT work here.
    - 2.5 dropped the `freq` indicator; forecast_with_covariates takes `horizon=` instead.
    - Dynamic covariates must span BOTH context AND horizon (length = CONTEXT_LEN + HORIZON_LEN).
    - Static covariates are ONE value per series.
"""

from __future__ import annotations

import numpy as np
import timesfm

N_STORES = 3
CONTEXT_LEN = 24
HORIZON_LEN = 12
TOTAL_LEN = CONTEXT_LEN + HORIZON_LEN  # 36


def generate_sales_data() -> dict:
    """Synthetic 3-store weekly sales with known covariate effects.

    Returns dict with per-store full-length (36-week) sales arrays and
    covariate arrays. The last HORIZON_LEN weeks of `sales` are the held-out
    ground truth used only for scoring.
    """
    rng = np.random.default_rng(42)

    stores = {
        "store_A": {"type": "premium", "region": "urban", "base_sales": 1000},
        "store_B": {"type": "standard", "region": "suburban", "base_sales": 750},
        "store_C": {"type": "discount", "region": "rural", "base_sales": 500},
    }
    base_prices = {"store_A": 12.0, "store_B": 10.0, "store_C": 7.5}

    data: dict = {"stores": {}, "covariates": {}}
    price_by, promo_by, holiday_by, dow_by = {}, {}, {}, {}

    for store_id, config in stores.items():
        bp = base_prices[store_id]
        weeks = np.arange(TOTAL_LEN)

        trend = config["base_sales"] * (1 + 0.005 * weeks)
        seasonality = 80 * np.sin(2 * np.pi * weeks / 52)
        noise = rng.normal(0, 40, TOTAL_LEN)
        base = (trend + seasonality + noise).astype(np.float32)

        price = (bp + rng.uniform(-0.5, 0.5, TOTAL_LEN)).astype(np.float32)
        price_effect = (-20 * (price - bp)).astype(np.float32)

        holidays = np.zeros(TOTAL_LEN, dtype=np.float32)
        for hw in [0, 11, 23, 35]:
            if hw < TOTAL_LEN:
                holidays[hw] = 1.0
        holiday_effect = (200 * holidays).astype(np.float32)

        promotion = rng.choice([0.0, 1.0], TOTAL_LEN, p=[0.8, 0.2]).astype(np.float32)
        promo_effect = (150 * promotion).astype(np.float32)

        day_of_week = np.tile(np.arange(7), TOTAL_LEN // 7 + 1)[:TOTAL_LEN].astype(np.int32)

        sales = np.maximum(base + price_effect + holiday_effect + promo_effect, 50.0)

        data["stores"][store_id] = {"sales": sales, "config": config}
        price_by[store_id] = price
        promo_by[store_id] = promotion
        holiday_by[store_id] = holidays
        dow_by[store_id] = day_of_week

    data["covariates"] = {
        "price": price_by,
        "promotion": promo_by,
        "holiday": holiday_by,
        "day_of_week": dow_by,
        "store_type": {sid: stores[sid]["type"] for sid in stores},
        "region": {sid: stores[sid]["region"] for sid in stores},
    }
    return data


def main() -> None:
    data = generate_sales_data()
    store_ids = list(data["stores"].keys())

    # inputs = CONTEXT portion only
    inputs = [data["stores"][s]["sales"][:CONTEXT_LEN].tolist() for s in store_ids]

    # dynamic covariates: list-of-lists, each spanning context + horizon (all 36 weeks)
    dyn_num = {"price": [data["covariates"]["price"][s].tolist() for s in store_ids]}
    dyn_cat = {
        "promotion":   [data["covariates"]["promotion"][s].astype(int).tolist()  for s in store_ids],
        "holiday":     [data["covariates"]["holiday"][s].astype(int).tolist()     for s in store_ids],
        "day_of_week": [data["covariates"]["day_of_week"][s].tolist()             for s in store_ids],
    }
    # static covariates: ONE value per series
    stat_cat = {
        "store_type": [data["covariates"]["store_type"][s] for s in store_ids],
        "region":     [data["covariates"]["region"][s]     for s in store_ids],
    }

    # held-out ground truth (last HORIZON_LEN weeks)
    truth = np.array([data["stores"][s]["sales"][CONTEXT_LEN:] for s in store_ids])

    # --- load TimesFM 2.5 ---
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=168,
            return_backcast=True,  # required for forecast_with_covariates (XReg)
        )
    )

    # --- forecast WITH covariates ---
    # NOTE: this installed build infers the horizon from covariate length
    # (covariate length - context length), so do NOT pass horizon= here.
    cov_forecast, xreg_forecast = model.forecast_with_covariates(
        inputs=inputs,
        dynamic_numerical_covariates=dyn_num,
        dynamic_categorical_covariates=dyn_cat,
        static_numerical_covariates={},
        static_categorical_covariates=stat_cat,
        xreg_mode="xreg + timesfm",  # default; try "timesfm + xreg" too
    )
    cov_forecast = np.asarray(cov_forecast)

    # --- baseline forecast (no covariates) for comparison ---
    point, _ = model.forecast(horizon=HORIZON_LEN, inputs=inputs)
    point = np.asarray(point)

    # --- report ---
    print("inputs:", len(inputs), "series | context", CONTEXT_LEN, "| horizon", HORIZON_LEN)
    print("cov_forecast shape:   ", cov_forecast.shape)
    print("xreg (2nd return):    ", np.asarray(xreg_forecast).shape)
    print("baseline shape:       ", point.shape)
    print()
    print(f"covariate MAE: {np.mean(np.abs(cov_forecast - truth)):.3f}")
    print(f"baseline  MAE: {np.mean(np.abs(point - truth)):.3f}")


if __name__ == "__main__":
    main()