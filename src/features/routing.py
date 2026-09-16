"""Tunisian Market Road Network Routing Module.

Provides real road-network distance and free-flow duration computation
for Grand Tunis and Tunisian urban zones using:
1. Open Source Routing Machine (OSRM) OpenStreetMap API
2. Calibrated Grand Tunis tortuosity fallback (road_dist ≈ 1.35 * haversine)
3. In-memory LRU caching to maintain sub-15ms inference SLA
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
from typing import Any, Dict, List, Optional
import requests

from src.features.build_features import haversine_distance_km
from src.utils.io import load_config

logger = logging.getLogger(__name__)

# Default Grand Tunis calibration defaults
DEFAULT_TORTUOSITY_FACTOR = 1.35
DEFAULT_BLENDED_SPEED_KMH = 28.0
DEFAULT_OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
DEFAULT_TIMEOUT_SEC = 1.2


@dataclass(frozen=True)
class RouteEstimate:
    road_distance_km: float
    freeflow_duration_seconds: float
    haversine_distance_km: float
    routing_engine: str
    tortuosity_ratio: float
    geometry: Optional[List[List[float]]] = None


def _get_market_config() -> Dict[str, Any]:
    try:
        cfg = load_config()
        return cfg.get("tunisia_market", {})
    except Exception:
        return {}


@functools.lru_cache(maxsize=2048)
def _query_osrm_cached(
    lat1_r: float,
    lon1_r: float,
    lat2_r: float,
    lon2_r: float,
    osrm_url: str,
    timeout_sec: float,
) -> Optional[Dict[str, float]]:
    """Query OSRM API with rounded coordinates and cache results."""
    # OSRM expects: {longitude},{latitude};{longitude},{latitude}
    query_url = f"{osrm_url}/{lon1_r},{lat1_r};{lon2_r},{lat2_r}?overview=full&geometries=geojson"
    try:
        resp = requests.get(query_url, timeout=timeout_sec)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                best_route = data["routes"][0]
                dist_m = float(best_route.get("distance", 0.0))
                dur_s = float(best_route.get("duration", 0.0))
                return {
                    "distance_km": round(dist_m / 1000.0, 3),
                    "duration_seconds": round(dur_s, 1),
                    "geometry": best_route.get("geometry", {}).get("coordinates"),
                }
    except Exception as e:
        logger.debug(f"OSRM request failed (will use calibrated fallback): {e}")
    return None


def get_tunisia_road_route(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    use_live_osrm: Optional[bool] = None,
) -> RouteEstimate:
    """Calculate true road distance and baseline duration in Tunisia.

    Falls back smoothly to calibrated Grand Tunis tortuosity and speed
    profile if OSRM is unreachable or disabled.
    """
    market_cfg = _get_market_config()
    routing_cfg = market_cfg.get("routing", {})
    speeds_cfg = market_cfg.get("speeds_kmh", {})

    tortuosity = float(routing_cfg.get("fallback_tortuosity_factor", DEFAULT_TORTUOSITY_FACTOR))
    speed_kmh = float(speeds_cfg.get("blended_default", DEFAULT_BLENDED_SPEED_KMH))
    osrm_url = str(routing_cfg.get("osrm_url", DEFAULT_OSRM_URL))
    timeout_sec = float(routing_cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SEC))

    if use_live_osrm is None:
        use_live_osrm = bool(routing_cfg.get("use_osrm_live", True))

    # Base haversine distance
    hav_km = float(
        haversine_distance_km(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
        )
    )

    if hav_km < 0.03:
        return RouteEstimate(
            road_distance_km=0.05,
            freeflow_duration_seconds=60.0,
            haversine_distance_km=round(hav_km, 3),
            routing_engine="negligible_distance",
            tortuosity_ratio=1.0,
        )

    # Try live OSRM routing if enabled
    if use_live_osrm:
        # Round coordinates to 4 decimal places (~11 meters) for effective caching
        lat1_r = round(pickup_latitude, 4)
        lon1_r = round(pickup_longitude, 4)
        lat2_r = round(dropoff_latitude, 4)
        lon2_r = round(dropoff_longitude, 4)

        osrm_res = _query_osrm_cached(lat1_r, lon1_r, lat2_r, lon2_r, osrm_url, timeout_sec)
        if osrm_res and osrm_res["distance_km"] > 0:
            road_dist = osrm_res["distance_km"]
            duration_s = max(60.0, osrm_res["duration_seconds"])
            ratio = round(road_dist / max(hav_km, 0.05), 2)
            return RouteEstimate(
                road_distance_km=road_dist,
                freeflow_duration_seconds=duration_s,
                haversine_distance_km=round(hav_km, 3),
                routing_engine="osrm_openstreetmap",
                tortuosity_ratio=ratio,
                geometry=osrm_res.get("geometry"),
            )

    # Graceful fallback: Calibrated Grand Tunis tortuosity
    road_dist = round(hav_km * tortuosity, 3)
    duration_s = max(60.0, round((road_dist / speed_kmh) * 3600.0, 1))
    return RouteEstimate(
        road_distance_km=road_dist,
        freeflow_duration_seconds=duration_s,
        haversine_distance_km=round(hav_km, 3),
        routing_engine="calibrated_tunisia_tortuosity",
        tortuosity_ratio=tortuosity,
    )
