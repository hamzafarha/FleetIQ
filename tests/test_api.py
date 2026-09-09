from fastapi.testclient import TestClient
import pytest

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["read_only_mode"] is True
    assert data["fallback_enabled"] is True


def test_predict_trip_duration_normal_trip():
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-15T14:30:00Z",
        "passenger_count": 2,
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "eta_seconds" in data
    assert "eta_minutes" in data
    assert data["target_type"] == "trip_duration"
    assert data["eta_seconds"] > 0
    assert data["eta_minutes"] > 0.0
    assert data["distance_km"] > 0.0
    assert data["traffic_included"] is False
    assert data["weather_included"] is False
    assert data["fallback_used"] is True
    assert data["quality_flag"] in ["ESTIMATED", "FALLBACK_ROUTING"]


def test_predict_trip_duration_rush_hour():
    # Peak rush hour at 8:30 AM
    rush_payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-15T08:30:00Z",
    }
    off_peak_payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-15T14:30:00Z",
    }

    rush_resp = client.post("/predict/trip-duration", json=rush_payload).json()
    off_peak_resp = client.post("/predict/trip-duration", json=off_peak_payload).json()

    # Rush hour ETA should be longer due to congestion speed adjustment
    assert rush_resp["eta_seconds"] >= off_peak_resp["eta_seconds"]


def test_predict_invalid_coordinates():
    # Latitude > 90 is invalid
    payload = {
        "pickup_latitude": 120.0,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-15T14:30:00Z",
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_invalid_timestamp():
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "not-a-valid-datetime",
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 422


def test_predict_driver_pickup_endpoint():
    payload = {
        "driver_latitude": 40.7500,
        "driver_longitude": -73.9900,
        "passenger_latitude": 40.7580,
        "passenger_longitude": -73.9855,
        "assignment_datetime": "2025-06-15T10:00:00Z",
    }
    response = client.post("/predict/driver-pickup", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_type"] == "driver_pickup"
    assert data["quality_flag"] == "FALLBACK_ROUTING"
    assert data["eta_seconds"] >= 60
