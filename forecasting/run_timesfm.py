"""
forecasting/run_timesfm.py — one-off TimesFM run from the command line.

Not used by the pipeline: the experiments call the persistent server
run_timesfm_server.py (via models/timesfm_model.py). This script uses the
server's load_model() and run_inference(), so a manual run gives exactly the
same forecasts as the pipeline (same TimesFM config, same forecast slicing).

Runs in .timesfm_venv.

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

from run_timesfm_server import load_model, run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="input_path",  required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    model = load_model()
    run_inference(model, args.input_path, args.output_path)


if __name__ == "__main__":
    main()
