"""Cleaning pipeline for the NYC TLC Case-B trip-duration prototype.

This module never mutates raw files. It writes cleaned copies to data/interim
and one CSV log recording the exact impact of each approved rule.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PICKUP_COL = "tpep_pickup_datetime"
DROPOFF_COL = "tpep_dropoff_datetime"
MAX_DURATION_SECONDS = 4 * 60 * 60
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def expected_month_from_filename(path: Path) -> pd.Period:
    """Return the YYYY-MM period encoded in a TLC trip filename."""
    match = re.search(r"(\d{4}-\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot infer YYYY-MM from filename: {path.name}")
    return pd.Period(match.group(1), freq="M")


def clean_trips(raw_trips: pd.DataFrame, expected_month: pd.Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the approved, minimal cleaning policy to one TLC monthly file.

    Drop rules remove rows that cannot supply a valid Case-B target. Distance
    and passenger-count anomalies remain in the data with explicit flags so
    EDA can determine their eventual feature policy.
    """
    required = {PICKUP_COL, DROPOFF_COL, "PULocationID", "DOLocationID", "trip_distance", "passenger_count"}
    missing = required.difference(raw_trips.columns)
    if missing:
        raise ValueError(f"Missing required TLC columns: {sorted(missing)}")

    pickup = pd.to_datetime(raw_trips[PICKUP_COL], errors="coerce")
    dropoff = pd.to_datetime(raw_trips[DROPOFF_COL], errors="coerce")
    duration_seconds = (dropoff - pickup).dt.total_seconds()

    missing_timestamp = pickup.isna() | dropoff.isna()
    non_positive_duration = duration_seconds.isna() | duration_seconds.le(0)
    pickup_outside_file_month = pickup.dt.to_period("M").ne(expected_month)
    duration_over_limit = duration_seconds.gt(MAX_DURATION_SECONDS)
    missing_zone = raw_trips["PULocationID"].isna() | raw_trips["DOLocationID"].isna()

    drop_mask = (
        missing_timestamp
        | non_positive_duration
        | pickup_outside_file_month
        | duration_over_limit
        | missing_zone
    )

    log = pd.DataFrame(
        {
            "rule": [
                "missing_timestamp",
                "non_positive_duration",
                "pickup_outside_file_month",
                "duration_over_4_hours",
                "missing_pickup_or_dropoff_zone",
                "total_rows_removed_union",
            ],
            "action": ["DROP", "DROP", "DROP", "DROP", "DROP", "DROP"],
            "affected_rows": [
                int(missing_timestamp.sum()),
                int(non_positive_duration.sum()),
                int(pickup_outside_file_month.sum()),
                int(duration_over_limit.sum()),
                int(missing_zone.sum()),
                int(drop_mask.sum()),
            ],
        }
    )
    log["affected_percent"] = (100 * log["affected_rows"] / len(raw_trips)).round(4)

    cleaned = raw_trips.loc[~drop_mask].copy()
    cleaned["trip_duration_seconds"] = duration_seconds.loc[~drop_mask]
    cleaned["flag_zero_distance"] = cleaned["trip_distance"].eq(0)
    cleaned["flag_extreme_distance_over_100_miles"] = cleaned["trip_distance"].gt(100)
    cleaned["flag_missing_passenger_count"] = cleaned["passenger_count"].isna()
    cleaned["flag_zero_passenger_count"] = cleaned["passenger_count"].eq(0)
    cleaned["flag_passenger_count_over_six"] = cleaned["passenger_count"].gt(6)

    assert cleaned[PICKUP_COL].notna().all()
    assert cleaned[DROPOFF_COL].notna().all()
    assert cleaned["trip_duration_seconds"].between(1, MAX_DURATION_SECONDS).all()
    assert cleaned[PICKUP_COL].dt.to_period("M").eq(expected_month).all()
    assert cleaned[["PULocationID", "DOLocationID"]].notna().all().all()

    return cleaned, log


def clean_directory(raw_dir: Path, interim_dir: Path) -> pd.DataFrame:
    """Clean all monthly TLC Parquet files and write a combined rule log."""
    files = sorted(raw_dir.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No TLC Parquet files found in {raw_dir}")

    interim_dir.mkdir(parents=True, exist_ok=True)
    monthly_logs = []
    for path in files:
        cleaned, log = clean_trips(pd.read_parquet(path), expected_month_from_filename(path))
        output_path = interim_dir / f"{path.stem}_cleaned.parquet"
        cleaned.to_parquet(output_path, index=False)
        log.insert(0, "source_file", path.name)
        log.insert(1, "rows_before", len(cleaned) + int(log.loc[log["rule"] == "total_rows_removed_union", "affected_rows"].iloc[0]))
        log.insert(2, "rows_after", len(cleaned))
        monthly_logs.append(log)

    combined_log = pd.concat(monthly_logs, ignore_index=True)
    combined_log.to_csv(interim_dir / "nyc_tlc_cleaning_log.csv", index=False)
    return combined_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean NYC TLC files into data/interim without modifying raw files.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "nyc_tlc")
    parser.add_argument("--interim-dir", type=Path, default=PROJECT_ROOT / "data" / "interim" / "nyc_tlc")
    args = parser.parse_args()
    log = clean_directory(args.raw_dir, args.interim_dir)
    removed = log.loc[log["rule"] == "total_rows_removed_union", "affected_rows"].sum()
    before = log.loc[log["rule"] == "total_rows_removed_union", "rows_before"].sum()
    print(f"Cleaning complete: removed {removed:,} of {before:,} rows. Log: {args.interim_dir / 'nyc_tlc_cleaning_log.csv'}")


if __name__ == "__main__":
    main()
