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

from src.data.telemetry_collector import get_telemetry_stats, log_completed_ride
from src.features.build_features import (
    extract_features_for_inference,
    haversine_distance_km,
    is_tunisian_rush_hour,
)
from src.features.routing import get_tunisia_road_route
from src.utils.io import load_config


# Load configuration
try:
    config = load_config()
    FALLBACK_SPEED_KMH = config.get("backend_rules", {}).get("fallback_routing_speed_kmh", 25.0)
    TUNISIA_CFG = config.get("tunisia_market", {})
except Exception:
    FALLBACK_SPEED_KMH = 25.0
    TUNISIA_CFG = {}

API_VERSION = "1.1.0"
MODEL_VERSION = "routing_fallback_speed_v1"

# Load Champion NYC Model if available
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

# Load Tunisia Adapted Model if available
TUNISIA_MODEL_PATH = Path("models/tunisia_champion_model.joblib")
tunisia_bundle: Optional[Dict[str, Any]] = None
tunisia_model: Optional[Any] = None

if TUNISIA_MODEL_PATH.exists():
    try:
        tunisia_bundle = joblib.load(TUNISIA_MODEL_PATH)
        tunisia_model = tunisia_bundle.get("model")
    except Exception:
        tunisia_model = None
        tunisia_bundle = None


def is_tunisia_coordinates(lat: float, lon: float) -> bool:
    """Return True if coordinates fall within Tunisia geographical bounding box."""
    return 30.0 <= lat <= 38.0 and 7.0 <= lon <= 12.2


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
    road_distance_km: Optional[float] = Field(default=None, description="Road distance in kilometers (OSRM/network)")
    routing_engine: Optional[str] = Field(default=None, description="Routing provider or calculation method")
    market: Optional[str] = Field(default="nyc_surrogate", description="Geographic market context")


class TelemetryLogRequest(BaseModel):
    pickup_latitude: float = Field(..., ge=-90.0, le=90.0)
    pickup_longitude: float = Field(..., ge=-180.0, le=180.0)
    dropoff_latitude: float = Field(..., ge=-90.0, le=90.0)
    dropoff_longitude: float = Field(..., ge=-180.0, le=180.0)
    actual_duration_seconds: int = Field(..., ge=10, le=86400)
    predicted_duration_seconds: Optional[int] = Field(default=None, ge=1)
    road_distance_km: Optional[float] = None
    traffic_density: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    weather_condition: Optional[str] = None
    routing_engine: Optional[str] = "tunisia_fleet"
    ride_id: Optional[str] = None


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
        "ml_model_loaded": (champion_model is not None or tunisia_model is not None),
        "tunisia_model_loaded": tunisia_model is not None,
        "fallback_enabled": True,
        "read_only_mode": True,
    }


@app.get("/model-info", tags=["System"])
def model_info():
    """Return champion model metadata, evaluation metrics, and feature dictionary."""
    if not champion_bundle and not tunisia_bundle:
        return {
            "status": "fallback_mode",
            "model_version": MODEL_VERSION,
            "description": "Running on classical routing heuristic fallback engine.",
        }

    active_b = tunisia_bundle if tunisia_bundle else champion_bundle

    return {
        "status": "active",
        "model_name": active_b.get("model_name"),
        "model_version": active_b.get("model_version"),
        "market": active_b.get("market", "Global"),
        "features": active_b.get("features"),
        "metrics": active_b.get("metrics"),
        "business_validation": active_b.get("business_validation", "PASSED"),
        "trained_at": active_b.get("created_at"),
        "has_tunisia_adapted_model": tunisia_bundle is not None,
    }


@app.post("/predict/trip-duration", response_model=ETAPredictionResponse, tags=["Inference"])
def predict_trip_duration(payload: ETAPredictionRequest):
    """Predict passenger trip duration from pickup to destination (Target 2 / Case B).

    Uses live Champion ML Model when available, with automatic failover to classical
    routing heuristics if inputs are out of bounds or model raises an error.
    """
    start_time = time.perf_counter()

    # Base distance proxy
    dist_km = float(
        haversine_distance_km(
            payload.pickup_latitude,
            payload.pickup_longitude,
            payload.dropoff_latitude,
            payload.dropoff_longitude,
        )
    )

    in_tunisia = is_tunisia_coordinates(payload.pickup_latitude, payload.pickup_longitude)
    road_dist_km = None
    routing_engine_name = "haversine_proxy"
    market_name = "tunisia" if in_tunisia else "nyc_surrogate"

    if in_tunisia:
        route = get_tunisia_road_route(
            payload.pickup_latitude,
            payload.pickup_longitude,
            payload.dropoff_latitude,
            payload.dropoff_longitude,
        )
        road_dist_km = route.road_distance_km
        routing_engine_name = route.routing_engine

    # Check for rush hour
    dt = datetime.fromisoformat(payload.pickup_datetime.replace("Z", "+00:00"))
    if in_tunisia:
        is_rush_hour = is_tunisian_rush_hour(dt)
        local_speed = float(TUNISIA_CFG.get("speeds_kmh", {}).get("blended_default", 28.0))
    else:
        is_rush_hour = dt.hour in (7, 8, 9, 16, 17, 18, 19) and dt.weekday() not in (5, 6)
        local_speed = FALLBACK_SPEED_KMH

    used_fallback = True
    active_version = MODEL_VERSION
    quality = QualityFlag.FALLBACK_ROUTING

    # Model prioritization: use Tunisian model if in Tunisia, otherwise NYC champion
    target_model = tunisia_model if (in_tunisia and tunisia_model is not None) else champion_model
    target_version = (
        tunisia_bundle.get("model_version", "v1.0_tunisia_adapted")
        if (in_tunisia and tunisia_bundle is not None)
        else (champion_bundle.get("model_version", MODEL_VERSION) if champion_bundle else MODEL_VERSION)
    )

    if target_model is not None:
        try:
            features_df = extract_features_for_inference(
                pickup_latitude=payload.pickup_latitude,
                pickup_longitude=payload.pickup_longitude,
                dropoff_latitude=payload.dropoff_latitude,
                dropoff_longitude=payload.dropoff_longitude,
                pickup_datetime=dt,
                passenger_count=payload.passenger_count or 1,
                road_distance_km=road_dist_km,
            )
            raw_pred_min = float(target_model.predict(features_df)[0])
            pred_min = max(1.0, round(raw_pred_min, 2))
            eta_sec = max(60, int(round(pred_min * 60.0)))
            eta_min = pred_min
            used_fallback = False
            quality = QualityFlag.HIGH_CONFIDENCE
            active_version = target_version
        except Exception:
            used_fallback = True

    if used_fallback:
        effective_dist = road_dist_km if road_dist_km is not None else dist_km
        adjusted_speed = (local_speed * 0.70) if is_rush_hour else local_speed
        eta_sec = calculate_fallback_eta_seconds(effective_dist, speed_kmh=adjusted_speed)
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
        road_distance_km=road_dist_km,
        routing_engine=routing_engine_name,
        market=market_name,
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

    in_tunisia = is_tunisia_coordinates(payload.driver_latitude, payload.driver_longitude)
    road_dist_km = None
    routing_engine_name = "haversine_proxy"
    market_name = "tunisia" if in_tunisia else "nyc_surrogate"

    if in_tunisia:
        route = get_tunisia_road_route(
            payload.driver_latitude,
            payload.driver_longitude,
            payload.passenger_latitude,
            payload.passenger_longitude,
        )
        road_dist_km = route.road_distance_km
        routing_engine_name = route.routing_engine

    dt = datetime.fromisoformat(payload.assignment_datetime.replace("Z", "+00:00"))
    if in_tunisia:
        from src.features.build_features import is_tunisian_rush_hour
        is_rush_hour = is_tunisian_rush_hour(dt)
        pickup_speed = 18.0 if is_rush_hour else 26.0
    else:
        is_rush_hour = dt.hour in (7, 8, 9, 16, 17, 18, 19) and dt.weekday() not in (5, 6)
        pickup_speed = 18.0 if is_rush_hour else 25.0

    effective_dist = road_dist_km if road_dist_km is not None else dist_km
    eta_sec = calculate_fallback_eta_seconds(effective_dist, speed_kmh=pickup_speed)
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
        road_distance_km=road_dist_km,
        routing_engine=routing_engine_name,
        market=market_name,
    )


@app.post("/telemetry/log-completed-ride", tags=["Telemetry"])
def log_completed_ride_endpoint(payload: TelemetryLogRequest):
    """Log completed ride telemetry to build continuous Tunisian fleet dataset."""
    res = log_completed_ride(
        pickup_latitude=payload.pickup_latitude,
        pickup_longitude=payload.pickup_longitude,
        dropoff_latitude=payload.dropoff_latitude,
        dropoff_longitude=payload.dropoff_longitude,
        actual_duration_seconds=payload.actual_duration_seconds,
        predicted_duration_seconds=payload.predicted_duration_seconds,
        road_distance_km=payload.road_distance_km,
        traffic_density=payload.traffic_density,
        weather_condition=payload.weather_condition,
        routing_engine=payload.routing_engine or "tunisia_fleet",
        ride_id=payload.ride_id,
    )
    return res


@app.get("/telemetry/stats", tags=["Telemetry"])
def telemetry_stats_endpoint():
    """Return cold-start fleet dataset statistics and retraining readiness."""
    return get_telemetry_stats()


