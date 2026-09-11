"""Load interim data and produce chronological train/val/test splits.

Implements chronological time-based split strictly preventing temporal data leakage.
Attaches taxi zone coordinates to derive haversine and Manhattan distances.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.features.build_features import (
    FEATURE_NAMES,
    add_datetime_features,
    haversine_distance_km,
    manhattan_distance_km,
)
from src.utils.io import load_config, resolve_path


def load_interim_cleaned_sample(
    sample_per_month: int = 50000,
    random_state: int = 42,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Load and combine a balanced, chronological sample of cleaned TLC data across all months."""
    config = config or load_config()
    interim_dir = resolve_path(config["paths"]["interim_dir"]) / "nyc_tlc"
    centroids_file = interim_dir / "taxi_zone_centroids.csv"

    if not centroids_file.exists():
        raise FileNotFoundError(f"Taxi zone centroids file not found: {centroids_file}")

    centroids = pd.read_csv(centroids_file)

    cleaned_files = sorted(interim_dir.glob("*_cleaned.parquet"))
    if not cleaned_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {interim_dir}")

    dfs = []
    for f in cleaned_files:
        month_df = pd.read_parquet(
            f,
            columns=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "passenger_count",
                "trip_duration_seconds",
            ],
        )
        if sample_per_month and len(month_df) > sample_per_month:
            month_df = month_df.sample(n=sample_per_month, random_state=random_state)
        dfs.append(month_df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["tpep_pickup_datetime"] = pd.to_datetime(combined["tpep_pickup_datetime"])
    combined["tpep_dropoff_datetime"] = pd.to_datetime(combined["tpep_dropoff_datetime"])

    # Merge pickup coordinates
    combined = combined.merge(
        centroids[["LocationID", "latitude", "longitude"]].rename(
            columns={
                "LocationID": "PULocationID",
                "latitude": "pickup_latitude",
                "longitude": "pickup_longitude",
            }
        ),
        on="PULocationID",
        how="left",
    )

    # Merge dropoff coordinates
    combined = combined.merge(
        centroids[["LocationID", "latitude", "longitude"]].rename(
            columns={
                "LocationID": "DOLocationID",
                "latitude": "dropoff_latitude",
                "longitude": "dropoff_longitude",
            }
        ),
        on="DOLocationID",
        how="left",
    )

    # Fill any missing coords with NYC center
    combined["pickup_latitude"] = combined["pickup_latitude"].fillna(40.7580)
    combined["pickup_longitude"] = combined["pickup_longitude"].fillna(-73.9855)
    combined["dropoff_latitude"] = combined["dropoff_latitude"].fillna(40.7829)
    combined["dropoff_longitude"] = combined["dropoff_longitude"].fillna(-73.9654)

    # Add datetime features
    combined = add_datetime_features(combined, "tpep_pickup_datetime")

    # Add distance features
    hav = haversine_distance_km(
        combined["pickup_latitude"],
        combined["pickup_longitude"],
        combined["dropoff_latitude"],
        combined["dropoff_longitude"],
    )
    man = manhattan_distance_km(
        combined["pickup_latitude"],
        combined["pickup_longitude"],
        combined["dropoff_latitude"],
        combined["dropoff_longitude"],
    )

    # If intra-zone (distance ~ 0) but trip_distance exists, estimate from trip_distance
    route_km = combined["trip_distance"] * 1.60934
    intra_zone = hav < 0.1
    combined["distance_haversine_km"] = np.where(intra_zone, np.maximum(0.5, route_km * 0.7), hav)
    combined["distance_manhattan_km"] = np.where(intra_zone, np.maximum(0.7, route_km * 0.9), man)

    # Clean passenger count
    combined["passenger_count"] = combined["passenger_count"].fillna(1).clip(1, 6)

    # Target in minutes
    combined["trip_duration_minutes"] = combined["trip_duration_seconds"] / 60.0

    # Basic business bounds
    valid = (
        (combined["trip_duration_minutes"] >= 0.5)
        & (combined["trip_duration_minutes"] <= 180.0)
        & (combined["distance_haversine_km"] >= 0.1)
        & (combined["distance_haversine_km"] <= 100.0)
    )
    return combined[valid].copy()


def time_based_split(
    df: pd.DataFrame,
    datetime_col: str = "tpep_pickup_datetime",
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test chronologically (no shuffling) to eliminate leakage."""
    config = config or load_config()
    split_cfg = config.get("split", {"train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15})
    df = df.sort_values(datetime_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * split_cfg["train_frac"])
    val_end = train_end + int(n * split_cfg["val_frac"])

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def build_and_save_processed_datasets(
    sample_per_month: int = 50000,
    config: Optional[dict] = None,
) -> Tuple[Path, Path, Path]:
    """Build the train, val, and test splits and save them to data/processed."""
    config = config or load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_interim_cleaned_sample(sample_per_month=sample_per_month, config=config)
    train_df, val_df, test_df = time_based_split(df, datetime_col="tpep_pickup_datetime", config=config)

    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Save feature metadata
    dict_path = processed_dir / "feature_dictionary.json"
    meta = {
        "features": FEATURE_NAMES,
        "target_primary": "trip_duration_minutes",
        "target_secondary": "trip_duration_seconds",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_start": str(train_df["tpep_pickup_datetime"].min()),
        "train_end": str(train_df["tpep_pickup_datetime"].max()),
        "val_start": str(val_df["tpep_pickup_datetime"].min()),
        "val_end": str(val_df["tpep_pickup_datetime"].max()),
        "test_start": str(test_df["tpep_pickup_datetime"].min()),
        "test_end": str(test_df["tpep_pickup_datetime"].max()),
    }
    with open(dict_path, "w") as f:
        json.dump(meta, f, indent=2)

    return train_path, val_path, test_path


if __name__ == "__main__":
    t_path, v_path, ts_path = build_and_save_processed_datasets(sample_per_month=50000)
    print(f"Processed splits saved to {t_path}, {v_path}, {ts_path}")

