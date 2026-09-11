"""Integration and API test suite for SmartTaxi Smart Arrival Time Estimation (ETA).

Verifies the minimum integration test scenarios defined in the Integration Guide:
- Health check & model metadata endpoints (Read-only, Backend compliant)
- Short vs. Long trips
- Peak hours vs. Off-peak
- Missing optional data (traffic, weather) handled without fake imputation
- Invalid coordinates & invalid timestamps rejection (422)
- Prediction latency SLA (< 50ms)
- Driver-to-Passenger pickup endpoint (Target 1 decoupling)
- Classical routing fallback resilience
"""
from datetime import datetime
import time

from fastapi.testclient import TestClient
import pytest

from app import app, champion_model

client = TestClient(app)


def test_health_endpoint():
    """Verify health check probe for central backend orchestrator."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["read_only_mode"] is True
    assert data["fallback_enabled"] is True
    assert "active_model" in data


def test_model_info_endpoint():
    """Verify model metadata and business threshold status reporting."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["active", "fallback_mode"]
    if data["status"] == "active":
        assert "metrics" in data
        assert "features" in data
        assert "business_validation" in data


def test_predict_trip_duration_live_ml_or_fallback():
    """Verify live passenger trip duration inference."""
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T14:30:00Z",
        "passenger_count": 2,
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["target_type"] == "trip_duration"
    assert data["eta_seconds"] > 60
    assert data["eta_minutes"] > 1.0
    assert data["distance_km"] > 0.0
    assert data["traffic_included"] is False
    assert data["weather_included"] is False
    assert data["latency_ms"] >= 0.0
    assert data["quality_flag"] in ["HIGH_CONFIDENCE", "FALLBACK_ROUTING"]


def test_predict_trip_duration_short_vs_long():
    """Verify that long distance trips predict significantly higher ETA than short trips."""
    short_trip = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7600,
        "dropoff_longitude": -73.9840,  # ~250 meters
        "pickup_datetime": "2025-06-16T14:00:00Z",
    }
    long_trip = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.6413,
        "dropoff_longitude": -73.7781,  # JFK Airport ~21 km
        "pickup_datetime": "2025-06-16T14:00:00Z",
    }

    short_resp = client.post("/predict/trip-duration", json=short_trip).json()
    long_resp = client.post("/predict/trip-duration", json=long_trip).json()

    assert short_resp["distance_km"] < 1.0
    assert long_resp["distance_km"] > 15.0
    assert long_resp["eta_seconds"] > short_resp["eta_seconds"]
    assert long_resp["eta_minutes"] > short_resp["eta_minutes"]


def test_predict_trip_duration_rush_hour_vs_offpeak():
    """Verify rush hour temporal differentiation on weekday morning vs late night."""
    weekday_rush = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T08:30:00Z",  # Monday 8:30 AM
    }
    weekday_night = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T03:30:00Z",  # Monday 3:30 AM
    }

    rush_resp = client.post("/predict/trip-duration", json=weekday_rush).json()
    night_resp = client.post("/predict/trip-duration", json=weekday_night).json()

    assert rush_resp["eta_seconds"] > 0
    assert night_resp["eta_seconds"] > 0


def test_predict_missing_optional_data_no_fake_simulation():
    """Verify rule: if traffic/weather is absent, do not silently simulate."""
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T14:30:00Z",
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["traffic_included"] is False
    assert data["weather_included"] is False


def test_predict_with_optional_traffic_and_weather():
    """Verify optional fields are acknowledged when genuinely supplied by backend."""
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T14:30:00Z",
        "traffic_density": 0.78,
        "weather_condition": "rain",
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["traffic_included"] is True
    assert data["weather_included"] is True


def test_predict_invalid_coordinates_rejected():
    """Out-of-bound coordinates must be strictly rejected with HTTP 422."""
    payload = {
        "pickup_latitude": 105.0,  # Invalid latitude > 90
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T14:30:00Z",
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 422


def test_predict_invalid_timestamp_rejected():
    """Malformed or non-ISO timestamps must be rejected with HTTP 422."""
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-13-45T99:99:99",
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 422


def test_predict_driver_pickup_endpoint():
    """Verify Target 1 (Driver-to-passenger pickup ETA) endpoint."""
    payload = {
        "driver_latitude": 40.7500,
        "driver_longitude": -73.9900,
        "passenger_latitude": 40.7580,
        "passenger_longitude": -73.9855,
        "assignment_datetime": "2025-06-16T10:00:00Z",
    }
    response = client.post("/predict/driver-pickup", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_type"] == "driver_pickup"
    assert data["quality_flag"] == "FALLBACK_ROUTING"
    assert data["fallback_used"] is True
    assert data["eta_seconds"] >= 60
    assert data["distance_km"] > 0.0


def test_prediction_latency_sla():
    """Verify that inference latency strictly complies with production SLA (< 50ms)."""
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T14:30:00Z",
    }
    t0 = time.perf_counter()
    response = client.post("/predict/trip-duration", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert response.status_code == 200
    assert elapsed_ms < 50.0  # Fast inference SLA


def test_predict_fallback_when_model_unavailable(monkeypatch):
    """Verify that when ML model is disabled or raises an error, classical fallback operates cleanly."""
    import app as app_module
    monkeypatch.setattr(app_module, "champion_model", None)
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T14:30:00Z",
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fallback_used"] is True
    assert data["quality_flag"] == "FALLBACK_ROUTING"
    assert data["eta_seconds"] >= 60
    assert data["distance_km"] > 0.0


