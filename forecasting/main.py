"""
main.py — Multi-city forecasting orchestrator.

Runs run_weather_baseline.py sequentially for each city, passing in the
appropriate city config. No manual file swapping needed.

Usage:
    # Run all cities
    python main.py

    # Run a subset
    python main.py --cities seoul washington
"""
import argparse
import sys
import time
import traceback
from datetime import datetime

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

    print("\n" + "=" * 80)
    print(f"MULTI-CITY FORECASTING  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cities: {', '.join(selected.keys())}")
    print("=" * 80)

    results = {}
    overall_start = time.time()

    for city, config in selected.items():
        print(f"\n{'=' * 80}")
        print(f"STARTING: {city.upper()}")
        print(f"{'=' * 80}")

        start = time.time()
        try:
            run_baseline(config=config, no_confirm=True)
            elapsed = time.time() - start
            print(f"\n[OK] {city} completed in {elapsed / 60:.1f} min")
            results[city] = "OK"
        except Exception:
            elapsed = time.time() - start
            print(f"\n[FAILED] {city} failed after {elapsed / 60:.1f} min")
            traceback.print_exc()
            results[city] = "FAILED"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 80)
    print("ALL CITIES COMPLETE")
    print("=" * 80)
    for city, status in results.items():
        print(f"  [{status:6s}]  {city}")
    print(f"\n  Total wall time: {total_elapsed / 60:.1f} min")

    if any(s == "FAILED" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
