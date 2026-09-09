"""Config loading and path helpers shared across the pipeline.

Keeping this centralized means notebooks and src/ modules never hardcode
paths or column names — everything traces back to config/config.yaml.
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "config.yaml"
    if (PROJECT_ROOT / "configs" / "config.yaml").exists()
    else PROJECT_ROOT / "config" / "config.yaml"
)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the project config.yaml as a dict.

    Args:
        config_path: path to the YAML config file.

    Returns:
        Parsed config as a nested dict.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml (relative to project root) to an
    absolute Path, so notebooks and scripts behave the same regardless of
    the current working directory they're launched from.
    """
    return PROJECT_ROOT / relative_path
