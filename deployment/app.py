"""Phase 9 (Deployment) service for SmartTaxi ETA.

Wired to the champion model artifact from models/ with fallback routing heuristics.
Run with: uvicorn deployment.app:app --reload
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Optional

from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field

from src.features.build_features import extract_features_for_inference, haversine_distance_km

app = FastAPI(
    title="SmartTaxi — Smart Arrival Time Estimation (ETA) API",
    version="1.0.0",
    description="Production deployment serving service for SmartTaxi central backend.",
)

# Load Champion ML Model if available
CHAMPION_MODEL_PATH = Path("models/champion_eta_model.joblib")
champion_bundle = None
champion_model = None

if CHAMPION_MODEL_PATH.exists():
    try:
        champion_bundle = joblib.load(CHAMPION_MODEL_PATH)
        champion_model = champion_bundle.get("model")
    except Exception:
        champion_model = None


class TripRequest(BaseModel):
    pickup_latitude: float = Field(..., ge=-90.0, le=90.0)
    pickup_longitude: float = Field(..., ge=-180.0, le=180.0)
    dropoff_latitude: float = Field(..., ge=-90.0, le=90.0)
    dropoff_longitude: float = Field(..., ge=-180.0, le=180.0)
    pickup_datetime: str
    passenger_count: Optional[int] = Field(default=1, ge=1, le=10)


class TripResponse(BaseModel):
    predicted_duration_minutes: float
    eta_seconds: int
    model_version: str
    fallback_used: bool
    distance_km: float
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": champion_model is not None,
        "model_version": champion_bundle.get("model_version", "fallback_routing_speed_v1") if champion_bundle else "fallback_routing_speed_v1",
    }


@app.post("/predict", response_model=TripResponse)
def predict(trip: TripRequest):
    start = time.perf_counter()
    dist_km = float(
        haversine_distance_km(
            trip.pickup_latitude,
            trip.pickup_longitude,
            trip.dropoff_latitude,
            trip.dropoff_longitude,
        )
    )

    fallback_used = True
    model_ver = "fallback_routing_speed_v1"

    if champion_model is not None:
        try:
            dt = datetime.fromisoformat(trip.pickup_datetime.replace("Z", "+00:00"))
            features_df = extract_features_for_inference(
                pickup_latitude=trip.pickup_latitude,
                pickup_longitude=trip.pickup_longitude,
                dropoff_latitude=trip.dropoff_latitude,
                dropoff_longitude=trip.dropoff_longitude,
                pickup_datetime=dt,
                passenger_count=trip.passenger_count or 1,
            )
            raw_min = float(champion_model.predict(features_df)[0])
            pred_min = max(1.0, round(raw_min, 2))
            eta_sec = max(60, int(round(pred_min * 60.0)))
            fallback_used = False
            model_ver = champion_bundle.get("model_version", "v1.0_champion")
        except Exception:
            fallback_used = True

    if fallback_used:
        hours = dist_km / 25.0
        eta_sec = max(60, int(hours * 3600))
        pred_min = round(eta_sec / 60.0, 2)

    latency = round((time.perf_counter() - start) * 1000, 2)

    return TripResponse(
        predicted_duration_minutes=pred_min,
        eta_seconds=eta_sec,
        model_version=model_ver,
        fallback_used=fallback_used,
        distance_km=round(dist_km, 3),
        latency_ms=latency,
    )

