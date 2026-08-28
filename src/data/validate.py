"""Schema and sanity validation against the data contract in config.yaml.

Run this right after loading raw data and again after any dataset swap
(e.g. real company data) to catch schema drift early.
"""
import pandas as pd


def validate_schema(df: pd.DataFrame, config: dict) -> list[str]:
    """Check that all required columns from the data contract are present.

    Args:
        df: dataframe to check.
        config: parsed config dict.

    Returns:
        List of missing column names (empty list = valid).
    """
    required = config["dataset"]["required_columns"].values()
    target = config["dataset"]["target_column"]
    expected = list(required) + [target]
    return [col for col in expected if col not in df.columns]


def flag_out_of_bounds(df: pd.DataFrame, config: dict) -> pd.Series:
    """Return a boolean mask of rows violating coordinate or trip-duration
    sanity bounds defined in config.yaml. Does not drop rows — leaves that
    decision to the caller (Phase 4/5 territory).

    Args:
        df: dataframe with pickup/dropoff coordinates and target column.
        config: parsed config dict.

    Returns:
        Boolean Series, True where a row is out of bounds.
    """
    cols = config["dataset"]["required_columns"]
    bounds = config["dataset"]["coordinate_bounds"]
    dur_bounds = config["dataset"]["trip_duration_bounds_seconds"]
    target = config["dataset"]["target_column"]

    bad_coords = (
        ~df[cols["pickup_longitude"]].between(bounds["lon_min"], bounds["lon_max"])
        | ~df[cols["pickup_latitude"]].between(bounds["lat_min"], bounds["lat_max"])
        | ~df[cols["dropoff_longitude"]].between(bounds["lon_min"], bounds["lon_max"])
        | ~df[cols["dropoff_latitude"]].between(bounds["lat_min"], bounds["lat_max"])
    )
    bad_duration = ~df[target].between(dur_bounds["min"], dur_bounds["max"])

    return bad_coords | bad_duration
