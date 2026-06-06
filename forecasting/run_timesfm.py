"""
forecasting/run_timesfm.py — TimesFM with covariates runner.

Runs in .timesfm_venv (numpy >=2.0, jax). Called via subprocess from
TimesFMForecaster in the main pipeline.

Input parquet columns:
    y       : float, NaN for forecast horizon rows
    <covar> : one column per numerical covariate (full train+test sequence)

Output parquet:
    y_pred  : float, predicted values for the horizon

Usage:
    .timesfm_venv/bin/python forecasting/run_timesfm.py \
        --in /tmp/tfm_input.parquet \
        --out /tmp/tfm_output.parquet
"""
import argparse

import numpy as np
import pandas as pd
import timesfm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="input_path",  required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)

    # Split on NaN in y
    is_train = df["y"].notna()
    y_train  = df.loc[is_train, "y"].values.astype(np.float32)
    horizon  = int((~is_train).sum())

    cov_cols = [c for c in df.columns if c != "y"]

    # Full train+test covariate sequences (required by forecast_with_covariates)
    dynamic_covariates = (
        {col: [df[col].values.astype(np.float32)] for col in cov_cols}
        if cov_cols else None
    )

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=168,
            return_backcast=True,  # required for forecast_with_covariates
        )
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

    pd.DataFrame({"y_pred": y_pred}).to_parquet(args.output_path, index=False)



if __name__ == "__main__":
    main()
