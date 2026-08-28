import pandas as pd

from src.features.build_features import haversine_distance_km


def test_haversine_zero_distance_for_identical_points():
    lat = pd.Series([40.75])
    lon = pd.Series([-73.98])
    d = haversine_distance_km(lat, lon, lat, lon)
    assert d.iloc[0] == 0.0


def test_haversine_known_distance_nyc_landmarks():
    # Times Square to Central Park, roughly ~3.2 km apart
    lat1, lon1 = pd.Series([40.7580]), pd.Series([-73.9855])
    lat2, lon2 = pd.Series([40.7829]), pd.Series([-73.9654])
    d = haversine_distance_km(lat1, lon1, lat2, lon2)
    assert 2.5 < d.iloc[0] < 4.0
