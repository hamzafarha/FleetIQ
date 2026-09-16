"""Unit & Integration tests for Tunisian Market Adaptation.

Verifies:
1. Grand Tunis OSRM routing & tortuosity fallback (road_distance >= haversine)
2. Tunisian congestion and rush-hour schedule detection
3. Passive telemetry logging and aggregation
4. FastAPI inference handling with Tunisian market metadata
"""
from datetime import datetime
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from app import app
from src.data.telemetry_collector import get_telemetry_stats, log_completed_ride
from src.features.build_features import is_tunisian_rush_hour
from src.features.routing import get_tunisia_road_route

client = TestClient(app)

# Grand Tunis Landmarks
AEROPORT_TUNIS = (36.8510, 10.2272)
AV_BOURGUIBA = (36.7992, 10.1802)
LA_MARSA = (36.8782, 10.3247)


def test_tunisia_road_route_landmarks():
    """Verify routing between Aéroport Carthage and Centre-Ville."""
    route = get_tunisia_road_route(
        pickup_latitude=AEROPORT_TUNIS[0],
        pickup_longitude=AEROPORT_TUNIS[1],
        dropoff_latitude=AV_BOURGUIBA[0],
        dropoff_longitude=AV_BOURGUIBA[1],
    )
    assert route.haversine_distance_km > 5.0
    assert route.road_distance_km >= route.haversine_distance_km
    assert route.tortuosity_ratio >= 1.0
    assert route.freeflow_duration_seconds > 180.0
    assert route.routing_engine in ["osrm_openstreetmap", "calibrated_tunisia_tortuosity"]


def test_tunisia_rush_hour_detection():
    """Verify detection of Tunisian specific congestion windows."""
    # Tuesday 08:15 (Morning rush) -> True
    dt_morning = datetime(2026, 9, 15, 8, 15)
    assert is_tunisian_rush_hour(dt_morning) is True

    # Thursday 13:10 (Midday school/office rush) -> True
    dt_midday = datetime(2026, 9, 17, 13, 10)
    assert is_tunisian_rush_hour(dt_midday) is True

    # Wednesday 17:45 (Evening rush) -> True
    dt_evening = datetime(2026, 9, 16, 17, 45)
    assert is_tunisian_rush_hour(dt_evening) is True

    # Wednesday 22:30 (Night off-peak) -> False
    dt_night = datetime(2026, 9, 16, 22, 30)
    assert is_tunisian_rush_hour(dt_night) is False

    # Sunday 13:15 (Sunday is not a workday) -> False
    dt_sunday = datetime(2026, 9, 20, 13, 15)
    assert is_tunisian_rush_hour(dt_sunday) is False


def test_telemetry_logging(tmp_path: Path):
    """Verify that completed rides are persisted and stats aggregated."""
    test_csv = tmp_path / "test_telemetry.csv"

    res = log_completed_ride(
        pickup_latitude=AEROPORT_TUNIS[0],
        pickup_longitude=AEROPORT_TUNIS[1],
        dropoff_latitude=LA_MARSA[0],
        dropoff_longitude=LA_MARSA[1],
        actual_duration_seconds=900,
        predicted_duration_seconds=840,
        road_distance_km=12.5,
        routing_engine="osrm_openstreetmap",
        log_path=test_csv,
    )
    assert res["status"] == "logged"
    assert res["error_seconds"] == -60

    stats = get_telemetry_stats(log_path=test_csv)
    assert stats["total_rides_logged"] == 1
    assert stats["mean_mae_minutes"] == 1.0


def test_api_predict_tunisia_market():
    """Verify that FastAPI identifies Tunisian coordinates and enriches the response."""
    payload = {
        "pickup_latitude": AEROPORT_TUNIS[0],
        "pickup_longitude": AEROPORT_TUNIS[1],
        "dropoff_latitude": LA_MARSA[0],
        "dropoff_longitude": LA_MARSA[1],
        "pickup_datetime": "2026-09-16T12:45:00Z",
        "passenger_count": 2,
    }
    response = client.post("/predict/trip-duration", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["market"] == "tunisia"
    assert data["road_distance_km"] is not None
    assert data["road_distance_km"] >= data["distance_km"]
    assert data["routing_engine"] is not None
    assert data["eta_seconds"] >= 60
    assert data["eta_minutes"] > 0


def test_api_telemetry_endpoints(tmp_path: Path, monkeypatch):
    """Verify logging endpoint /telemetry/log-completed-ride and /telemetry/stats."""
    test_csv = tmp_path / "api_telemetry.csv"
    monkeypatch.setattr("src.data.telemetry_collector._get_log_path", lambda: test_csv)

    log_payload = {
        "pickup_latitude": AV_BOURGUIBA[0],
        "pickup_longitude": AV_BOURGUIBA[1],
        "dropoff_latitude": LA_MARSA[0],
        "dropoff_longitude": LA_MARSA[1],
        "actual_duration_seconds": 1200,
        "predicted_duration_seconds": 1150,
        "road_distance_km": 18.2,
        "routing_engine": "osrm_openstreetmap",
    }
    response = client.post("/telemetry/log-completed-ride", json=log_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "logged"

    stats_resp = client.get("/telemetry/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_rides_logged"] >= 1
