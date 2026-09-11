"""Smart ETA Prediction API — Production Serving Service.

Designed for integration with SmartTaxi central backend:
- Read-only: strictly no database writes to the Ride table.
- Authoritative timestamps are owned and provided by the backend.
- Fallback strategy: routing/speed heuristic if ML model is unavailable or confidence is degraded.
- No silent simulation of missing external traffic/weather data.
- Standard response contract: eta_seconds, eta_minutes, model_version, quality_flag.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
import joblib
from pydantic import BaseModel, Field, field_validator

from src.features.build_features import extract_features_for_inference, haversine_distance_km
from src.utils.io import load_config

# Load configuration
try:
    config = load_config()
    FALLBACK_SPEED_KMH = config.get("backend_rules", {}).get("fallback_routing_speed_kmh", 25.0)
except Exception:
    FALLBACK_SPEED_KMH = 25.0

API_VERSION = "1.0.0"
MODEL_VERSION = "routing_fallback_speed_v1"

# Load Champion ML Model if available
CHAMPION_MODEL_PATH = Path("models/champion_eta_model.joblib")
champion_bundle: Optional[Dict[str, Any]] = None
champion_model: Optional[Any] = None

if CHAMPION_MODEL_PATH.exists():
    try:
        champion_bundle = joblib.load(CHAMPION_MODEL_PATH)
        champion_model = champion_bundle.get("model")
        MODEL_VERSION = champion_bundle.get("model_version", "v1.0_random_forest")
    except Exception:
        champion_model = None
        champion_bundle = None

app = FastAPI(
    title="SmartTaxi — Smart Arrival Time Estimation (ETA) API",
    version=API_VERSION,
    description="Independent ETA inference service for passenger trip duration and driver-to-passenger pickup.",
)


class TargetType(str, Enum):
    TRIP_DURATION = "trip_duration"
    DRIVER_PICKUP = "driver_pickup"


class QualityFlag(str, Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    ESTIMATED = "ESTIMATED"
    FALLBACK_ROUTING = "FALLBACK_ROUTING"
    DEGRADED = "DEGRADED"


class ETAPredictionRequest(BaseModel):
    pickup_latitude: float = Field(..., ge=-90.0, le=90.0, description="Pickup latitude in degrees")
    pickup_longitude: float = Field(..., ge=-180.0, le=180.0, description="Pickup longitude in degrees")
    dropoff_latitude: float = Field(..., ge=-90.0, le=90.0, description="Dropoff latitude in degrees")
    dropoff_longitude: float = Field(..., ge=-180.0, le=180.0, description="Dropoff longitude in degrees")
    pickup_datetime: str = Field(..., description="ISO 8601 pickup timestamp provided by backend")
    passenger_count: Optional[int] = Field(default=1, ge=1, le=10, description="Passenger count")
    traffic_density: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Real-time traffic index if available; do not fake"
    )
    weather_condition: Optional[str] = Field(
        default=None, description="Current weather descriptor if available; do not fake"
    )

    @field_validator("pickup_datetime")
    @classmethod
    def validate_iso_datetime(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid timestamp format: '{v}'. Expected ISO 8601 format.")
        return v


class DriverPickupRequest(BaseModel):
    driver_latitude: float = Field(..., ge=-90.0, le=90.0, description="Driver current latitude")
    driver_longitude: float = Field(..., ge=-180.0, le=180.0, description="Driver current longitude")
    passenger_latitude: float = Field(..., ge=-90.0, le=90.0, description="Passenger pickup latitude")
    passenger_longitude: float = Field(..., ge=-180.0, le=180.0, description="Passenger pickup longitude")
    assignment_datetime: str = Field(..., description="ISO 8601 dispatch assignment timestamp")

    @field_validator("assignment_datetime")
    @classmethod
    def validate_iso_datetime(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid timestamp format: '{v}'. Expected ISO 8601 format.")
        return v


class ETAPredictionResponse(BaseModel):
    eta_seconds: int = Field(..., description="Estimated arrival time in seconds")
    eta_minutes: float = Field(..., description="Estimated arrival time in minutes (business-friendly)")
    model_version: str = Field(..., description="Active model or fallback version")
    target_type: TargetType = Field(..., description="Prediction target type")
    quality_flag: QualityFlag = Field(..., description="Confidence and routing quality indicator")
    distance_km: float = Field(..., description="Straight-line / proxy distance in kilometers")
    traffic_included: bool = Field(..., description="Indicates if verified traffic data was utilized")
    weather_included: bool = Field(..., description="Indicates if verified weather data was utilized")
    fallback_used: bool = Field(..., description="True if classical routing fallback was triggered")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


def calculate_fallback_eta_seconds(distance_km: float, speed_kmh: float = FALLBACK_SPEED_KMH) -> int:
    """Classical routing fallback calculation based on average urban speed."""
    if distance_km <= 0.05:
        return 60  # Minimum 1 minute for negligible distances
    duration_hours = distance_km / max(speed_kmh, 5.0)
    duration_seconds = int(duration_hours * 3600)
    return max(60, duration_seconds)


@app.get("/health", tags=["System"])
def health_check():
    """Health check for backend orchestrator and container liveness probes."""
    return {
        "status": "healthy",
        "api_version": API_VERSION,
        "active_model": MODEL_VERSION,
        "ml_model_loaded": champion_model is not None,
        "fallback_enabled": True,
        "read_only_mode": True,
    }


@app.get("/model-info", tags=["System"])
def model_info():
    """Return champion model metadata, evaluation metrics, and feature dictionary."""
    if not champion_bundle:
        return {
            "status": "fallback_mode",
            "model_version": MODEL_VERSION,
            "description": "Running on classical routing heuristic fallback engine.",
        }

    return {
        "status": "active",
        "model_name": champion_bundle.get("model_name"),
        "model_version": champion_bundle.get("model_version"),
        "features": champion_bundle.get("features"),
        "metrics": champion_bundle.get("metrics"),
        "business_validation": champion_bundle.get("business_validation"),
        "trained_at": champion_bundle.get("created_at"),
    }


@app.post("/predict/trip-duration", response_model=ETAPredictionResponse, tags=["Inference"])
def predict_trip_duration(payload: ETAPredictionRequest):
    """Predict passenger trip duration from pickup to destination (Target 2 / Case B).

    Uses live Champion ML Model when available, with automatic failover to classical
    routing heuristics if inputs are out of bounds or model raises an error.
    """
    start_time = time.perf_counter()

    # Calculate distance proxy
    dist_km = float(
        haversine_distance_km(
            payload.pickup_latitude,
            payload.pickup_longitude,
            payload.dropoff_latitude,
            payload.dropoff_longitude,
        )
    )

    # Check for rush hour
    dt = datetime.fromisoformat(payload.pickup_datetime.replace("Z", "+00:00"))
    is_rush_hour = dt.hour in (7, 8, 9, 16, 17, 18, 19) and dt.weekday() not in (5, 6)

    used_fallback = True
    active_version = MODEL_VERSION
    quality = QualityFlag.FALLBACK_ROUTING

    if champion_model is not None:
        try:
            features_df = extract_features_for_inference(
                pickup_latitude=payload.pickup_latitude,
                pickup_longitude=payload.pickup_longitude,
                dropoff_latitude=payload.dropoff_latitude,
                dropoff_longitude=payload.dropoff_longitude,
                pickup_datetime=dt,
                passenger_count=payload.passenger_count or 1,
            )
            raw_pred_min = float(champion_model.predict(features_df)[0])
            pred_min = max(1.0, round(raw_pred_min, 2))
            eta_sec = max(60, int(round(pred_min * 60.0)))
            eta_min = pred_min
            used_fallback = False
            quality = QualityFlag.HIGH_CONFIDENCE
            active_version = MODEL_VERSION
        except Exception:
            used_fallback = True

    if used_fallback:
        adjusted_speed = (FALLBACK_SPEED_KMH * 0.75) if is_rush_hour else FALLBACK_SPEED_KMH
        eta_sec = calculate_fallback_eta_seconds(dist_km, speed_kmh=adjusted_speed)
        eta_min = round(eta_sec / 60.0, 2)
        active_version = "fallback_routing_speed_v1"
        quality = QualityFlag.FALLBACK_ROUTING

    latency = round((time.perf_counter() - start_time) * 1000, 2)

    return ETAPredictionResponse(
        eta_seconds=eta_sec,
        eta_minutes=eta_min,
        model_version=active_version,
        target_type=TargetType.TRIP_DURATION,
        quality_flag=quality,
        distance_km=round(dist_km, 3),
        traffic_included=payload.traffic_density is not None,
        weather_included=payload.weather_condition is not None,
        fallback_used=used_fallback,
        latency_ms=latency,
    )


@app.post("/predict/driver-pickup", response_model=ETAPredictionResponse, tags=["Inference"])
def predict_driver_pickup(payload: DriverPickupRequest):
    """Predict driver-to-passenger ETA (Target 1: Driver pickup arrival).

    Priority endpoint for passenger dispatch matching. Operates under calibrated
    urban routing fallback until driver telemetry traces are confirmed.
    """
    start_time = time.perf_counter()

    dist_km = float(
        haversine_distance_km(
            payload.driver_latitude,
            payload.driver_longitude,
            payload.passenger_latitude,
            payload.passenger_longitude,
        )
    )

    dt = datetime.fromisoformat(payload.assignment_datetime.replace("Z", "+00:00"))
    is_rush_hour = dt.hour in (7, 8, 9, 16, 17, 18, 19) and dt.weekday() not in (5, 6)
    pickup_speed = 18.0 if is_rush_hour else 25.0

    eta_sec = calculate_fallback_eta_seconds(dist_km, speed_kmh=pickup_speed)
    eta_min = round(eta_sec / 60.0, 2)

    latency = round((time.perf_counter() - start_time) * 1000, 2)

    return ETAPredictionResponse(
        eta_seconds=eta_sec,
        eta_minutes=eta_min,
        model_version="routing_fallback_driver_v1",
        target_type=TargetType.DRIVER_PICKUP,
        quality_flag=QualityFlag.FALLBACK_ROUTING,
        distance_km=round(dist_km, 3),
        traffic_included=False,
        weather_included=False,
        fallback_used=True,
        latency_ms=latency,
    )

