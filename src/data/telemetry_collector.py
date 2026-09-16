"""Tunisian Market Passive Telemetry Collector.

Enables 'Shadow Logging' of completed rides on the Tunisian road network.
Collects real (or simulated pilot) trips to build the first genuine
Tunisian taxi dataset and bootstrap local model fine-tuning.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from src.utils.io import load_config

DEFAULT_LOG_PATH = Path("data/interim/tunisia_telemetry_log.csv")

HEADERS = [
    "ride_id",
    "timestamp",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "actual_duration_seconds",
    "predicted_duration_seconds",
    "error_seconds",
    "road_distance_km",
    "traffic_density",
    "weather_condition",
    "routing_engine",
]


def _get_log_path() -> Path:
    try:
        cfg = load_config()
        configured_path = cfg.get("tunisia_market", {}).get("telemetry", {}).get("log_file")
        if configured_path:
            return Path(configured_path)
    except Exception:
        pass
    return DEFAULT_LOG_PATH


def init_telemetry_file(log_path: Optional[Path] = None) -> Path:
    path = log_path or _get_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
    return path


def log_completed_ride(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    actual_duration_seconds: int,
    predicted_duration_seconds: Optional[int] = None,
    road_distance_km: Optional[float] = None,
    traffic_density: Optional[float] = None,
    weather_condition: Optional[str] = None,
    routing_engine: str = "calibrated_tunisia",
    ride_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a completed ride record to the Tunisian telemetry log."""
    path = init_telemetry_file(log_path)
    rid = ride_id or f"tn_ride_{uuid.uuid4().hex[:10]}"
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    error_sec = (
        (predicted_duration_seconds - actual_duration_seconds)
        if predicted_duration_seconds is not None
        else None
    )

    row = [
        rid,
        ts,
        round(pickup_latitude, 6),
        round(pickup_longitude, 6),
        round(dropoff_latitude, 6),
        round(dropoff_longitude, 6),
        actual_duration_seconds,
        predicted_duration_seconds if predicted_duration_seconds is not None else "",
        error_sec if error_sec is not None else "",
        round(road_distance_km, 3) if road_distance_km is not None else "",
        traffic_density if traffic_density is not None else "",
        weather_condition if weather_condition is not None else "",
        routing_engine,
    ]

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return {
        "status": "logged",
        "ride_id": rid,
        "timestamp": ts,
        "actual_duration_seconds": actual_duration_seconds,
        "predicted_duration_seconds": predicted_duration_seconds,
        "error_seconds": error_sec,
    }


def get_telemetry_stats(log_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compute summary metrics over the collected Tunisian telemetry log."""
    path = log_path or _get_log_path()
    if not path.exists():
        return {
            "total_rides_logged": 0,
            "mean_error_seconds": 0.0,
            "mean_mae_minutes": 0.0,
            "status": "COLD_START_INSUFFICIENT_DATA",
            "threshold_for_retraining": 500,
            "ready_for_fine_tuning": False,
        }

    total = 0
    errors: List[float] = []

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            err_str = row.get("error_seconds")
            if err_str:
                try:
                    errors.append(abs(float(err_str)))
                except ValueError:
                    pass

    mae_min = round((sum(errors) / len(errors) / 60.0), 2) if errors else 0.0

    return {
        "total_rides_logged": total,
        "logged_with_predictions": len(errors),
        "mean_mae_minutes": mae_min,
        "threshold_for_retraining": 500,
        "ready_for_fine_tuning": total >= 500,
        "status": "READY_FOR_LOCAL_TRAINING" if total >= 500 else "COLLECTING_TELEMETRY",
    }
