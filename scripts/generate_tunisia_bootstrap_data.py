"""Generate Synthetic Bootstrap Dataset for Grand Tunis Taxi Trips.

Simulates realistic taxi trips across key Grand Tunis origin-destination pairs
(Aéroport Tunis-Carthage, Les Berges du Lac, Centre-Ville, Marsa, Sidi Bou Saïd,
Ennasr, ESPRIT / Ghazela, etc.) reflecting local road tortuosity, rush-hour
choke points, and speed profiles.

Output: data/processed/tunisia_bootstrap_trips.parquet
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features.build_features import (
    FEATURE_NAMES,
    haversine_distance_km,
    is_tunisian_rush_hour,
    manhattan_distance_km,
)

from src.features.routing import get_tunisia_road_route

GRAND_TUNIS_HUBS = {
    "aeroport_carthage": (36.8510, 10.2272, "Aéroport Tunis-Carthage"),
    "centre_ville": (36.7992, 10.1802, "Centre-Ville Bourguiba"),
    "lac_1": (36.8335, 10.2341, "Les Berges du Lac 1"),
    "lac_2": (36.8436, 10.2743, "Les Berges du Lac 2"),
    "la_marsa": (36.8782, 10.3247, "La Marsa Corniche"),
    "sidi_bou_said": (36.8703, 10.3418, "Sidi Bou Saïd"),
    "carthage": (36.8529, 10.3243, "Carthage Byrsa"),
    "ennasr_2": (36.8480, 10.1565, "Ennasr 2 Hédi Nouira"),
    "esprit_ghazela": (36.8973, 10.1895, "Technopôle El Ghazela / ESPRIT"),
    "menzah_9": (36.8439, 10.1417, "El Menzah 9"),
    "place_barcelone": (36.7950, 10.1805, "Gare Place Barcelone"),
    "la_goulette": (36.8183, 10.3050, "La Goulette Port"),
    "le_bardo": (36.8092, 10.1343, "Le Bardo Musée"),
}


def generate_tunisian_trips(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    random.seed(random_seed)
    np.random.seed(random_seed)

    hub_keys = list(GRAND_TUNIS_HUBS.keys())
    records = []

    base_start_time = datetime(2026, 8, 1, 6, 0, 0, tzinfo=timezone.utc)

    for i in range(num_samples):
        # Pick distinct origin and destination
        o_key, d_key = random.sample(hub_keys, 2)
        o_lat, o_lon, o_name = GRAND_TUNIS_HUBS[o_key]
        d_lat, d_lon, d_name = GRAND_TUNIS_HUBS[d_key]

        # Add small random jitter (+/- ~300 meters) to avoid discrete point clumping
        p_lat = o_lat + np.random.normal(0, 0.0025)
        p_lon = o_lon + np.random.normal(0, 0.0025)
        d_lat = d_lat + np.random.normal(0, 0.0025)
        d_lon = d_lon + np.random.normal(0, 0.0025)

        # Random timestamp across a 30-day window with realistic diurnal curve
        day_offset = random.randint(0, 30)
        # Bimodal hour distribution (morning rush, lunch, evening rush)
        hour_choice = random.choices(
            population=list(range(24)),
            weights=[
                1, 1, 1, 1, 2, 4, 8, 12, 11, 7, 7, 8, 12, 10, 6, 7, 10, 14, 12, 9, 7, 5, 3, 2
            ],
            k=1,
        )[0]
        minute_choice = random.randint(0, 59)
        trip_dt = base_start_time + timedelta(days=day_offset, hours=hour_choice, minutes=minute_choice)

        hav_km = float(haversine_distance_km(p_lat, p_lon, d_lat, d_lon))
        man_km = float(manhattan_distance_km(p_lat, p_lon, d_lat, d_lon))

        # Grand Tunis tortuosity ratio (~1.30 to 1.45)
        tortuosity = random.uniform(1.30, 1.42)
        road_km = max(0.5, hav_km * tortuosity)

        is_rush = is_tunisian_rush_hour(trip_dt)
        is_weekend = trip_dt.weekday() in (5, 6)

        # Speed model based on route type and time
        # Express routes (e.g. Marsa, Lac, Ghazela) faster than dense central (Centre-Ville, Bardo, Barcelone)
        dense_zones = {"centre_ville", "place_barcelone", "le_bardo"}
        is_dense = (o_key in dense_zones) or (d_key in dense_zones)

        if is_dense:
            base_speed = random.uniform(18.0, 26.0)
        else:
            base_speed = random.uniform(35.0, 55.0)

        if is_rush:
            speed_kmh = max(10.0, base_speed * random.uniform(0.50, 0.75))
        elif is_weekend:
            speed_kmh = base_speed * random.uniform(1.05, 1.20)
        else:
            speed_kmh = base_speed * random.uniform(0.85, 1.05)

        duration_sec = int(max(90, (road_km / speed_kmh) * 3600.0 + random.normalvariate(0, 45)))
        duration_min = round(duration_sec / 60.0, 2)
        passengers = random.choices([1, 2, 3, 4], weights=[0.65, 0.20, 0.10, 0.05])[0]

        records.append({
            "trip_id": f"tn_boot_{i+1:05d}",
            "pickup_datetime": trip_dt.isoformat(),
            "pickup_latitude": p_lat,
            "pickup_longitude": p_lon,
            "dropoff_latitude": d_lat,
            "dropoff_longitude": d_lon,
            "origin_hub": o_name,
            "destination_hub": d_name,
            "distance_haversine_km": hav_km,
            "distance_manhattan_km": man_km,
            "road_distance_km": round(road_km, 3),
            "trip_duration_seconds": duration_sec,
            "trip_duration_minutes": duration_min,
            "pickup_hour": trip_dt.hour,
            "pickup_dayofweek": trip_dt.weekday(),
            "is_weekend": int(is_weekend),
            "is_rush_hour": int(is_rush),
            "sin_hour": float(np.sin(2 * np.pi * trip_dt.hour / 24.0)),
            "cos_hour": float(np.cos(2 * np.pi * trip_dt.hour / 24.0)),
            "passenger_count": passengers,
            "average_speed_kmh": round(road_km / (duration_sec / 3600.0), 1),
        })

    df = pd.DataFrame(records)
    return df


def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "tunisia_bootstrap_trips.parquet"

    print(f"Generating Grand Tunis bootstrap taxi trips dataset...")
    df = generate_tunisian_trips(num_samples=3000, random_seed=42)
    df.to_parquet(out_file, index=False)
    print(f"Generated {len(df)} realistic Tunisian trips saved to {out_file}")
    print("Trip duration statistics (minutes):")
    print(df["trip_duration_minutes"].describe())
    print("\nSpeed statistics (km/h):")
    print(df["average_speed_kmh"].describe())


if __name__ == "__main__":
    main()
