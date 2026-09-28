"""
forecasting/run_timesfm_server.py — Persistent TimesFM server.

Loads the model once, then processes folds on demand via stdin/stdout.
Run with .timesfm_venv/bin/python. Never call directly — managed by
TimesFMForecaster in timesfm_model.py.

Protocol (newline-delimited):
    startup: READY|<timesfm version>
    stdin:   <in_path>|<out_path>
    stdout:  OK|<out_path>   or   ERROR|<message>

Input parquet: column "y" (context values, then NaN for the horizon) and, for
the covariate model, one column per covariate covering context + horizon.

Forecast config: the TimesFM 2.5 configuration recommended by the authors
(timesfm README), with max_horizon = 168 (longest experiment horizon) and
return_backcast = True (required by forecast_with_covariates).
Point forecast = median (quantile index 5).
"""
import sys
from importlib import metadata

import numpy as np
import pandas as pd
import timesfm

MAX_CONTEXT = 1024
MAX_HORIZON = 168


def load_model():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )
    return model


def run_inference(model, in_path, out_path):
    df = pd.read_parquet(in_path)

    is_train = df["y"].notna()
    y_train  = df.loc[is_train, "y"].values.astype(np.float32)
    horizon  = int((~is_train).sum())

    cov_cols = [c for c in df.columns if c != "y"]
    dynamic_covariates = (
        {col: [df[col].values.astype(np.float32)] for col in cov_cols}
        if cov_cols else None
    )

    if dynamic_covariates:
        # In-context linear regression on the covariates ("xreg + timesfm"),
        # TimesFM forecasts the residual. Returns the horizon only.
        timesfm_out, _ = model.forecast_with_covariates(
            inputs=[y_train],
            dynamic_numerical_covariates=dynamic_covariates,
        )
        y_pred = np.asarray(timesfm_out[0])
    else:
        # With return_backcast=True, forecast() returns the in-sample backcast
        # followed by the forecast: the forecast is the LAST `horizon` values.
        point_forecast, _ = model.forecast(
            inputs=[y_train],
            horizon=horizon,
        )
        y_pred = np.asarray(point_forecast[0])[-horizon:]

    if len(y_pred) != horizon:
        raise RuntimeError(f"TimesFM returned {len(y_pred)} values, expected {horizon}")

    pd.DataFrame({"y_pred": y_pred}).to_parquet(out_path, index=False)


def main():
    model = load_model()
    print(f"READY|{metadata.version('timesfm')}", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        in_path, out_path = line.split("|", 1)
        try:
            run_inference(model, in_path, out_path)
            print(f"OK|{out_path}", flush=True)
        except Exception as e:
            print(f"ERROR|{e}", flush=True)


if __name__ == "__main__":
    main()
