"""Streaming simulation for real-time ride requests and ETA prediction.

Generates sequential ride requests mimicking driver-passenger dispatch events
and evaluates API throughput, latency, and fallback behavior.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import random
import sys
import time
import requests

NYC_BOUNDS = {
    "lat_min": 40.60,
    "lat_max": 40.85,
    "lon_min": -74.05,
    "lon_max": -73.85,
}


def generate_random_ride(ride_id: int) -> dict:
    """Generate a realistic synthetic ride request inside the NYC metro area."""
    p_lat = round(random.uniform(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]), 6)
    p_lon = round(random.uniform(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]), 6)
    d_lat = round(random.uniform(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]), 6)
    d_lon = round(random.uniform(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]), 6)

    return {
        "ride_id": f"ride_{ride_id:04d}",
        "pickup_latitude": p_lat,
        "pickup_longitude": p_lon,
        "dropoff_latitude": d_lat,
        "dropoff_longitude": d_lon,
        "pickup_datetime": datetime.now(timezone.utc).isoformat(),
        "passenger_count": random.randint(1, 4),
        "traffic_density": round(random.uniform(0.1, 0.9), 2) if random.random() > 0.5 else None,
        "weather_condition": "rain" if random.random() > 0.8 else None,
    }


def stream_simulation(api_url: str, num_trips: int = 10, interval_seconds: float = 0.5):
    """Stream simulated ride events to the prediction API."""
    print(f"🚀 Starting ETA streaming simulation to: {api_url}")
    print(f"📦 Total events: {num_trips} | Interval: {interval_seconds}s\n")
    print(f"{'Ride ID':<10} | {'Status':<8} | {'Dist (km)':<10} | {'ETA (min)':<10} | {'Quality':<16} | {'Latency (ms)':<12}")
    print("-" * 75)

    success_count = 0
    total_latency_ms = 0.0

    for i in range(1, num_trips + 1):
        ride = generate_random_ride(i)
        start = time.perf_counter()
        try:
            resp = requests.post(f"{api_url}/predict/trip-duration", json=ride, timeout=2.0)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            total_latency_ms += elapsed_ms

            if resp.status_code == 200:
                data = resp.json()
                success_count += 1
                print(
                    f"{ride['ride_id']:<10} | "
                    f"OK (200) | "
                    f"{data['distance_km']:<10.2f} | "
                    f"{data['eta_minutes']:<10.2f} | "
                    f"{data['quality_flag']:<16} | "
                    f"{elapsed_ms:<12.1f}"
                )
            else:
                print(f"{ride['ride_id']:<10} | ERR ({resp.status_code}) | {'-':<10} | {'-':<10} | {'-':<16} | {elapsed_ms:<12.1f}")
        except Exception as err:
            print(f"{ride['ride_id']:<10} | FAILED   | Error: {err}")

        if i < num_trips:
            time.sleep(interval_seconds)

    print("\n" + "=" * 75)
    print(f"✅ Simulation finished: {success_count}/{num_trips} requests succeeded.")
    if success_count > 0:
        avg_lat = round(total_latency_ms / success_count, 2)
        print(f"⏱️ Average client latency: {avg_lat} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time ride event stream")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of ETA API")
    parser.add_argument("--num-trips", type=int, default=10, help="Number of trips to simulate")
    parser.add_argument("--interval", type=float, default=0.2, help="Interval between requests in seconds")
    args = parser.parse_args()

    stream_simulation(args.api_url, args.num_trips, args.interval)
