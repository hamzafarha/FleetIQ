"""Load raw data and produce train/val/test splits.

Phase 3 (Dataset design) territory. Implementations here should read
column names and paths from config, never hardcode them, so this module
keeps working once real company data replaces the NYC Taxi prototype.
"""
import pandas as pd

from src.utils.io import load_config, resolve_path


def load_raw_data(config: dict = None) -> pd.DataFrame:
    """Load the raw training data referenced in config.yaml.

    Args:
        config: parsed config dict (loads default config.yaml if None).

    Returns:
        Raw dataframe, unfiltered and unvalidated.
    """
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    train_file = config["dataset"]["train_file"]
    return pd.read_csv(raw_dir / train_file)


def time_based_split(df: pd.DataFrame, datetime_col: str, config: dict = None):
    """Split into train/val/test chronologically (no shuffling).

    Taxi demand is time-dependent, so a random split would leak future
    traffic patterns into training. Split fractions come from
    config.yaml -> split.

    Args:
        df: dataframe sorted or sortable by datetime_col.
        datetime_col: name of the pickup datetime column.
        config: parsed config dict (loads default config.yaml if None).

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    config = config or load_config()
    split_cfg = config["split"]
    df = df.sort_values(datetime_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * split_cfg["train_frac"])
    val_end = train_end + int(n * split_cfg["val_frac"])

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
