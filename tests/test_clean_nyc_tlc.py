from pathlib import Path

import pandas as pd

from src.data.clean_nyc_tlc import clean_trips, expected_month_from_filename


def test_expected_month_is_read_from_tlc_filename():
    assert expected_month_from_filename(Path("yellow_tripdata_2025-07.parquet")) == pd.Period("2025-07", freq="M")


def test_clean_trips_drops_only_invalid_target_or_context_rows():
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-01 08:00", "2025-01-02 09:00", "2025-01-03 10:00", "2009-01-01 10:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-01-01 08:10", "2025-01-02 09:00", "2025-01-03 15:01", "2009-01-01 10:20"]),
            "PULocationID": [1, 1, 1, 1],
            "DOLocationID": [2, 2, 2, 2],
            "trip_distance": [1.0, 0.0, 150.0, 1.0],
            "passenger_count": [1.0, None, 7.0, 1.0],
        }
    )

    cleaned, log = clean_trips(raw, pd.Period("2025-01", freq="M"))

    assert len(cleaned) == 1
    assert cleaned["trip_duration_seconds"].iloc[0] == 600
    assert not cleaned["flag_zero_distance"].iloc[0]
    assert log.loc[log["rule"] == "total_rows_removed_union", "affected_rows"].iloc[0] == 3
