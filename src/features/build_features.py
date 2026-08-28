"""Feature engineering for Case B (predict at trip start).

Only includes features derivable from pickup location, destination, and
start time — i.e. what's actually available at inference time. Do not add
features that require info only known after the trip ends; that's a
Case-A/Case-B leakage bug, not a modeling choice.

Functions here are standard geometry/calendar utilities, not modeling
decisions, so they're implemented directly rather than left as stubs.
"""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1: pd.Series, lon1: pd.Series,
                           lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Great-circle distance in km between two lat/lon points (vectorized).

    This is a straight-line proxy for route distance, not actual road
    distance — useful as a baseline feature, but expect gradient-boosting
    and DL models to benefit from a real routing-engine distance later if
    one becomes available.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def manhattan_distance_km(lat1: pd.Series, lon1: pd.Series,
                           lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """L1-style distance (lat-diff + lon-diff, each haversine'd separately).

    Often a better proxy than straight-line haversine for grid-like street
    networks (e.g. much of Manhattan).
    """
    lat_dist = haversine_distance_km(lat1, lon1, lat2, lon1)
    lon_dist = haversine_distance_km(lat2, lon1, lat2, lon2)
    return lat_dist + lon_dist


def add_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Add hour/day-of-week/month/weekend/rush-hour features derived from
    the trip start time. All available at inference time for Case B.
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_rush_hour"] = dt.dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    return df


def add_distance_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add haversine and Manhattan distance features using column names
    from the data contract in config.yaml.
    """
    df = df.copy()
    cols = config["dataset"]["required_columns"]
    df["distance_haversine_km"] = haversine_distance_km(
        df[cols["pickup_latitude"]], df[cols["pickup_longitude"]],
        df[cols["dropoff_latitude"]], df[cols["dropoff_longitude"]],
    )
    df["distance_manhattan_km"] = manhattan_distance_km(
        df[cols["pickup_latitude"]], df[cols["pickup_longitude"]],
        df[cols["dropoff_latitude"]], df[cols["dropoff_longitude"]],
    )
    return df
