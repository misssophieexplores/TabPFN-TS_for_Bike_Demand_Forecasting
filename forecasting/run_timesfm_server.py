"""
forecasting/run_timesfm_server.py — Persistent TimesFM server.

Loads the model once, then processes folds on demand via stdin/stdout.
Run with .timesfm_venv/bin/python. Never call directly — managed by
TimesFMForecaster in timesfm_model.py.

Protocol (newline-delimited):
    stdin:  <in_path>|<out_path>
    stdout: OK|<out_path>   or   ERROR|<message>
"""
import sys

import numpy as np
import pandas as pd
import timesfm


def load_model():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=168,
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
        timesfm_out, _ = model.forecast_with_covariates(
            inputs=[y_train],
            dynamic_numerical_covariates=dynamic_covariates,
        )
        y_pred = np.array(timesfm_out[0])[:horizon]
    else:
        point_forecast, _ = model.forecast(
            inputs=[y_train],
            horizon=horizon,
        )
        y_pred = point_forecast[0, :horizon]

    pd.DataFrame({"y_pred": y_pred}).to_parquet(out_path, index=False)


def main():
    model = load_model()
    print("READY", flush=True)

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