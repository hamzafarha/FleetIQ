import numpy as np
import pandas as pd
import pytest

from src.models.baseline import HistoricalAvgSpeedBaseline


def test_baseline_fit_and_predict_global():
    # 3 trips: 5km in 15min (20 km/h), 10km in 30min (20 km/h), 20km in 60min (20 km/h)
    distances = pd.Series([5.0, 10.0, 20.0])
    durations = pd.Series([15.0, 30.0, 60.0])

    baseline = HistoricalAvgSpeedBaseline()
    baseline.fit(distances, durations)

    assert baseline.is_fitted is True
    assert pytest.approx(baseline.global_speed_kmh, rel=1e-2) == 20.0

    # Predict for 10km -> should be ~30 minutes
    preds = baseline.predict(pd.Series([10.0]))
    assert pytest.approx(preds[0], rel=1e-2) == 30.0


def test_baseline_hourly_segmentation():
    # Rush hour (hour 8): 10km in 40min (15 km/h)
    # Off-peak (hour 14): 10km in 20min (30 km/h)
    distances = pd.Series([10.0, 10.0])
    durations = pd.Series([40.0, 20.0])
    hours = pd.Series([8, 14])

    baseline = HistoricalAvgSpeedBaseline()
    baseline.fit(distances, durations, hours=hours)

    assert pytest.approx(baseline.hourly_speeds[8], rel=1e-2) == 15.0
    assert pytest.approx(baseline.hourly_speeds[14], rel=1e-2) == 30.0

    # Predict rush hour vs off peak for 10km
    rush_pred = baseline.predict(pd.Series([10.0]), hours=pd.Series([8]))
    off_peak_pred = baseline.predict(pd.Series([10.0]), hours=pd.Series([14]))

    assert rush_pred[0] > off_peak_pred[0]
    assert pytest.approx(rush_pred[0], rel=1e-2) == 40.0
    assert pytest.approx(off_peak_pred[0], rel=1e-2) == 20.0


def test_baseline_minimum_duration():
    # Very short distance should have minimum floor of 1.0 minute
    baseline = HistoricalAvgSpeedBaseline(default_speed_kmh=30.0)
    baseline.is_fitted = True
    preds = baseline.predict(0.01)
    assert preds >= 1.0
