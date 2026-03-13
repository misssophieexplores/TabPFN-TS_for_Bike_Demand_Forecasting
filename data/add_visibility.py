import requests
import pandas as pd
import time
from datetime import date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import os

load_dotenv()
VISUALCROSSING_KEY = os.getenv('VISUALCROSSING_KEY')

# ── Config ────────────────────────────────────────────────────────────────────
DATASETS = {
    "London": {
        "path": "data/LondonBikeData.csv",
        "location": "London,UK",
    },
    "Washington": {
        "path": "data/WashingtonBikeData.csv",
        "location": "Washington,DC",
    },
}

VC_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
SLEEP_BETWEEN_CALLS = 0.3

# Set to True to fetch only the first 5 hourly records per dataset for testing.
# A test file (<original_name>_visibility_test.csv) will be written so you can
# verify the merge looks correct before committing to the full download.
# Flip to False for the real run.
TEST_MODE = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def month_chunks(start: date, end: date):
    cursor = start.replace(day=1)
    while cursor <= end:
        chunk_start = max(cursor, start)
        chunk_end   = min((cursor + relativedelta(months=1) - relativedelta(days=1)), end)
        yield chunk_start, chunk_end
        cursor += relativedelta(months=1)


def fetch_chunk(location: str, start: date, end: date) -> pd.DataFrame:
    url = f"{VC_BASE}/{location}/{start}/{end}"
    params = {
        "key":       VISUALCROSSING_KEY,
        "unitGroup": "metric",
        "include":   "hours",
        "elements":  "datetime,visibility",
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"  WARNING HTTP {response.status_code}: {response.text[:120]}")
        return pd.DataFrame(columns=["timestamp", "visibility_km"])

    records = []
    for day in response.json().get("days", []):
        for hour in day.get("hours", []):
            records.append({
                "timestamp":     pd.to_datetime(f"{day['datetime']} {hour['datetime']}"),
                "visibility_km": hour.get("visibility"),
            })
            if TEST_MODE and len(records) >= 5:
                return pd.DataFrame(records)
    return pd.DataFrame(records)


def add_visibility(name: str, path: str, location: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({path})")
    print(f"{'='*60}")
    if TEST_MODE:
        print(f"  TEST MODE: fetching 5 records only, saving to test file.")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Guard: already done? (skipped in test mode so you can re-run freely)
    if not TEST_MODE and "visibility_km" in df.columns and df["visibility_km"].notna().any():
        filled = df["visibility_km"].notna().sum()
        print(f"  'visibility_km' already present ({filled} non-null values). Skipping.")
        return

    start = df["timestamp"].min().date()
    end   = df["timestamp"].max().date()
    chunks = list(month_chunks(start, end))
    print(f"  Date range: {start} -> {end}  ({len(chunks)} monthly chunks)")

    all_visibility = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"    [{i:>2}/{len(chunks)}] {chunk_start} -> {chunk_end}", end=" ... ")
        chunk_df = fetch_chunk(location, chunk_start, chunk_end)
        all_visibility.append(chunk_df)
        print(f"{len(chunk_df)} records")
        time.sleep(SLEEP_BETWEEN_CALLS)
        if TEST_MODE:
            break  # one chunk is enough to verify end-to-end

    visibility_df = pd.concat(all_visibility, ignore_index=True)
    merged = pd.merge(df, visibility_df, on="timestamp", how="left")

    if TEST_MODE:
        test_path = path.replace(".csv", "_visibility_test.csv")
        merged.to_csv(test_path, index=False)
        print(f"\n  Sample visibility values fetched:")
        print(merged[["timestamp", "visibility_km"]].dropna(subset=["visibility_km"]).head().to_string(index=False))
        print(f"\n  Full test merge saved to: {test_path}")
        print(f"  Check that file looks correct, then set TEST_MODE = False and re-run.")
        return

    # Report coverage before imputation
    missing_before = merged["visibility_km"].isna().sum()
    total          = len(merged)
    print(f"\n  Coverage before imputation: {total - missing_before}/{total} rows filled "
          f"({missing_before} missing = {missing_before/total*100:.1f}%)")

    # Fill gaps with linear interpolation (= average between surrounding values).
    # limit=5 ensures we never interpolate across a gap larger than 5 consecutive
    # hours -- anything longer is left as NaN and flagged below.
    if missing_before > 0:
        print(f"  Filling gaps with linear interpolation...")
        merged["visibility_km"] = merged["visibility_km"].interpolate(
            method="linear", limit=5, limit_direction="both"
        )

    missing_after = merged["visibility_km"].isna().sum()
    if missing_after > 0:
        print(f"  WARNING: {missing_after} values still missing after interpolation "
              f"(gaps > 5 hours). Timestamps:")
        print(merged[merged["visibility_km"].isna()]["timestamp"].to_string(index=False))
    else:
        print(f"  All gaps filled.")

    merged.to_csv(path, index=False)
    print(f"  Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not VISUALCROSSING_KEY:
        raise EnvironmentError("VISUALCROSSING_KEY not found in environment / .env file.")

    for name, cfg in DATASETS.items():
        add_visibility(name, cfg["path"], cfg["location"])

    print(f"\n{'='*60}")
    if TEST_MODE:
        print("  Test run complete. Check the _visibility_test.csv files,")
        print("  then set TEST_MODE = False to run for real.")
    else:
        print("  All done!")
    print(f"{'='*60}\n")