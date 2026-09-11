"""Feature engineering for Case B (predict at trip start).

Only includes features derivable from pickup location, destination, and
start time — i.e. what's actually available at inference time. Do not add
features that require info only known after the trip ends; that's a
Case-A/Case-B leakage bug, not a modeling choice.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
FEATURE_NAMES = [
    "distance_haversine_km",
    "distance_manhattan_km",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "sin_hour",
    "cos_hour",
    "passenger_count",
]


def haversine_distance_km(
    lat1: Union[pd.Series, np.ndarray, float],
    lon1: Union[pd.Series, np.ndarray, float],
    lat2: Union[pd.Series, np.ndarray, float],
    lon2: Union[pd.Series, np.ndarray, float],
) -> Union[pd.Series, np.ndarray, float]:
    """Great-circle distance in km between two lat/lon points (vectorized or scalar)."""
    is_scalar = np.isscalar(lat1)
    l1, o1, l2, o2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [l1, o1, l2, o2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    res = EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    if isinstance(lat1, pd.Series):
        return pd.Series(res, index=lat1.index)
    return float(res) if is_scalar else res


def manhattan_distance_km(
    lat1: Union[pd.Series, np.ndarray, float],
    lon1: Union[pd.Series, np.ndarray, float],
    lat2: Union[pd.Series, np.ndarray, float],
    lon2: Union[pd.Series, np.ndarray, float],
) -> Union[pd.Series, np.ndarray, float]:
    """L1-style Manhattan distance (lat-diff + lon-diff in km)."""
    lat_dist = haversine_distance_km(lat1, lon1, lat2, lon1)
    lon_dist = haversine_distance_km(lat2, lon1, lat2, lon2)
    return lat_dist + lon_dist


def add_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Add hour/day-of-week/month/weekend/rush-hour and cyclical features derived from trip start."""
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    # Rush hour: weekdays 07-09h and 16-19h
    is_rush_hour_time = dt.dt.hour.isin([7, 8, 9, 16, 17, 18, 19])
    df["is_rush_hour"] = (is_rush_hour_time & (df["is_weekend"] == 0)).astype(int)

    # Cyclical hour encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)
    return df


def add_distance_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """Add haversine and Manhattan distance features using dataframe coordinates."""
    df = df.copy()
    p_lat = df["pickup_latitude"] if "pickup_latitude" in df.columns else df.get("PULocationID")
    p_lon = df["pickup_longitude"] if "pickup_longitude" in df.columns else df.get("PULocationID")
    d_lat = df["dropoff_latitude"] if "dropoff_latitude" in df.columns else df.get("DOLocationID")
    d_lon = df["dropoff_longitude"] if "dropoff_longitude" in df.columns else df.get("DOLocationID")

    df["distance_haversine_km"] = haversine_distance_km(p_lat, p_lon, d_lat, d_lon)
    df["distance_manhattan_km"] = manhattan_distance_km(p_lat, p_lon, d_lat, d_lon)
    return df


def extract_features_for_inference(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    pickup_datetime: Union[str, datetime],
    passenger_count: int = 1,
) -> pd.DataFrame:
    """Prepare a single sample DataFrame adhering strictly to the training feature schema."""
    if isinstance(pickup_datetime, str):
        dt = datetime.fromisoformat(pickup_datetime.replace("Z", "+00:00"))
    else:
        dt = pickup_datetime

    hour = dt.hour
    dayofweek = dt.weekday()
    is_weekend = 1 if dayofweek in (5, 6) else 0
    is_rush_hour = 1 if (hour in (7, 8, 9, 16, 17, 18, 19) and is_weekend == 0) else 0

    hav_km = float(haversine_distance_km(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude))
    man_km = float(manhattan_distance_km(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude))

    features = {
        "distance_haversine_km": [hav_km],
        "distance_manhattan_km": [man_km],
        "pickup_hour": [hour],
        "pickup_dayofweek": [dayofweek],
        "is_weekend": [is_weekend],
        "is_rush_hour": [is_rush_hour],
        "sin_hour": [float(np.sin(2 * np.pi * hour / 24.0))],
        "cos_hour": [float(np.cos(2 * np.pi * hour / 24.0))],
        "passenger_count": [passenger_count if passenger_count is not None else 1],
    }
    return pd.DataFrame(features)[FEATURE_NAMES]

