"""
main.py — Multi-city forecasting orchestrator.

Runs run_weather_baseline.py sequentially for each city, passing in the
appropriate city config. No manual file swapping needed.

Usage:
    # Run all cities
    python forecasting/main.py

    # Run a subset
    python forecasting/main.py --cities seoul washington
"""
import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from run_weather_baseline import main as run_baseline


# ---------------------------------------------------------------------------
# City registry — add new cities here, no other file needs to change
# ---------------------------------------------------------------------------

def get_configs():
    """Import and return all city configs as a dict."""
    from config_seoul import get_config as seoul_config
    from config_washington import get_config as washington_config
    from config_london import get_config as london_config

    return {
        "seoul":      seoul_config(),
        "washington": washington_config(),
        "london":     london_config(),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run forecasting experiments for all cities sequentially."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        choices=["seoul", "washington", "london"],
        default=["seoul", "washington", "london"],
        help="Which cities to run (default: all).",
    )
    args = parser.parse_args()

    configs = get_configs()
    selected = {city: configs[city] for city in args.cities}

    # Set up error log file
    first_config = next(iter(selected.values()))
    error_log_path = Path(first_config.output_dir) / f"errors_{first_config.results_version}.log"
    error_log_path.parent.mkdir(exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MULTI-CITY FORECASTING | Cities: {', '.join(selected.keys())}")

    results = {}
    overall_start = time.time()

    for city, config in selected.items():
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING: {city.upper()}")

        start = time.time()
        try:
            run_baseline(config=config, no_confirm=True)
            elapsed = time.time() - start
            print(f"\n[OK] {city} completed in {elapsed / 60:.1f} min")
            results[city] = "OK"
        except Exception:
            elapsed = time.time() - start
            print(f"\n[FAILED] {city} failed after {elapsed / 60:.1f} min")
            print(f"  See {error_log_path} for details")
            with open(error_log_path, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FAILED: {city}\n")
                f.write(f"{'='*80}\n")
                traceback.print_exc(file=f)
            results[city] = "FAILED"

    # ------------------------------------------------------------------
    # Comparative metrics — once, pooled across all cities
    # ------------------------------------------------------------------
    from run_experiments import compute_and_log_comparative_metrics
    if any(s == "OK" for s in results.values()):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing comparative metrics (pooled across cities)")
        try:
            comp = compute_and_log_comparative_metrics(first_config, log_wandb=False)
            print(comp[['model', 'n_tasks', 'win_rate', 'skill_score']].to_string(index=False))
        except ValueError as e:
            print(f"[FAILED] comparative metrics: {e}")


    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL CITIES COMPLETE")
    for city, status in results.items():
        print(f"  [{status:6s}]  {city}")
    print(f"\n  Total wall time: {total_elapsed / 60:.1f} min")

    if any(s == "FAILED" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()