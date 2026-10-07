"""
Re-select the SARIMAX order from an existing tuning file with the current
selection rule (arima_search.select_candidate, require_converged=True): the
lowest mean MAE among the candidates whose optimizer converged on every tune
fold. Nothing is refitted: the candidates, their 90-fold MAEs and their
convergence counts come from the tuning file, which tune_sarimax.py wrote
before the rule existed (7 Oct 2026).

The new params file keeps every field of the source file (city, scenario,
tuning period, covariates, all candidates), with the selected order and
tuning.selection_rule replaced, and records the source file and its
provenance under "reselection". Its own provenance is the code that ran the
re-selection.

Usage (from the repo root; one SLURM job on the cluster):
    python forecasting/models/tuning/reselect_sarimax.py \
        --source results/tuning/sarimax_best_params_seoul_clean_only_720_20261007_063501.json \
        --output results/tuning/sarimax_best_params_seoul_clean_only_720_converged.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # forecasting/
sys.path.insert(0, str(Path(__file__).resolve().parent))      # models/tuning/

import argparse
import copy
import json
from datetime import datetime

from provenance import check_output_dir, save_params_json
from arima_search import select_candidate


def reselect(source: dict, source_name: str) -> dict:
    """New params dict from a SARIMAX tuning dict (see module docstring)."""
    tuning = source.get("tuning", {})
    candidates = tuning.get("candidates")
    if not candidates:
        raise ValueError(f"{source_name}: no tuning.candidates")
    if source.get("m") is None or "seasonal_order" not in source:
        raise ValueError(f"{source_name}: not a SARIMAX tuning file")
    if "reselection" in source:
        raise ValueError(f"{source_name} is already a re-selected file; use the original tuning file")
    best, rule = select_candidate(candidates, require_converged=True)

    params = copy.deepcopy(source)
    params.pop("provenance", None)   # the new file gets its own (save_params_json)
    params["order"] = best["order"]
    params["seasonal_order"] = best["seasonal_order"]
    params["with_intercept"] = best["with_intercept"]
    params["trend"] = best["trend"]
    params["tuning"]["selection_rule"] = rule
    params["tuning"]["best_tune_mae_mean"] = best["mae_mean"]
    params["tuning"]["best_tune_rmse_mean"] = best["rmse_mean"]
    params["reselection"] = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "source_file": source_name,
        "source_provenance": source.get("provenance"),
        "source_selection": {
            "order": source["order"],
            "seasonal_order": source["seasonal_order"],
            "with_intercept": source["with_intercept"],
            "trend": source.get("trend"),
            "best_tune_mae_mean": tuning.get("best_tune_mae_mean"),
        },
        "rule": rule,
        "note": "candidates and fold scores taken from source_file, nothing refitted",
    }
    return params


def main():
    parser = argparse.ArgumentParser(description="Re-select the SARIMAX order (converged candidates only)")
    parser.add_argument("--source", required=True, help="Tuning file written by tune_sarimax.py")
    parser.add_argument("--output", required=True, help="New params file")
    args = parser.parse_args()

    source_path, output = Path(args.source), Path(args.output)
    if output.resolve() == source_path.resolve():
        raise SystemExit("--output must differ from --source")
    check_output_dir(output.parent)
    with open(source_path) as f:
        source = json.load(f)
    params = reselect(source, str(args.source))

    old, new = params["reselection"]["source_selection"], params
    print(f"{params['city']}: {old['order']}{old['seasonal_order']} intercept={old['with_intercept']} "
          f"(MAE {old['best_tune_mae_mean']:.2f}) -> {new['order']}{new['seasonal_order']} "
          f"intercept={new['with_intercept']} (MAE {new['tuning']['best_tune_mae_mean']:.2f})")
    print(f"Rule: {params['tuning']['selection_rule']}")
    save_params_json(params, output)
    print(f"  --> config.sarimax_params_file = '{output}'")


if __name__ == "__main__":
    main()
