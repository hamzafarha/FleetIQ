"""Historical average speed baseline model for ETA prediction.

Predicts trip duration as: distance / historical_average_speed.
Supports global averaging as well as hourly segmentation to capture
peak and off-peak urban traffic dynamics.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class HistoricalAvgSpeedBaseline:
    """Predicts trip duration in minutes from route distance and optional hour.

    Estimates average urban speed from training data and applies:
        predicted_duration_minutes = (distance_km / speed_kmh) * 60.0
    """

    def __init__(self, default_speed_kmh: float = 25.0):
        self.default_speed_kmh = default_speed_kmh
        self.global_speed_kmh: float = default_speed_kmh
        self.hourly_speeds: dict[int, float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        distance_km: Union[pd.Series, np.ndarray],
        duration_min: Union[pd.Series, np.ndarray],
        hours: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "HistoricalAvgSpeedBaseline":
        """Fit the baseline model by calculating average speed.

        Args:
            distance_km: trip distance in kilometers.
            duration_min: actual trip duration in minutes.
            hours: optional integer hour of day (0-23) for hourly segmentation.
        """
        dist = np.asarray(distance_km, dtype=float)
        dur = np.asarray(duration_min, dtype=float)

        # Filter zero/negative distances or durations for speed computation
        valid_mask = (dist > 0.05) & (dur > 0.5)
        if not np.any(valid_mask):
            self.global_speed_kmh = self.default_speed_kmh
            self.is_fitted = True
            return self

        valid_dist = dist[valid_mask]
        valid_dur = dur[valid_mask]

        # Calculate implied speed (km/h)
        implied_speeds = valid_dist / (valid_dur / 60.0)
        # Clip speeds to plausible urban ranges (3 to 90 km/h)
        clipped_speeds = np.clip(implied_speeds, 3.0, 90.0)

        self.global_speed_kmh = float(np.median(clipped_speeds))

        if hours is not None:
            h_arr = np.asarray(hours)[valid_mask]
            for h in range(24):
                mask_h = h_arr == h
                if np.any(mask_h):
                    self.hourly_speeds[h] = float(np.median(clipped_speeds[mask_h]))
                else:
                    self.hourly_speeds[h] = self.global_speed_kmh

        self.is_fitted = True
        return self

    def predict(
        self,
        distance_km: Union[pd.Series, np.ndarray, float],
        hours: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Predict trip duration in minutes.

        Args:
            distance_km: trip distance in kilometers.
            hours: optional hour of day (0-23) to apply hourly speed adjustments.

        Returns:
            Numpy array of predicted durations in minutes.
        """
        is_scalar = np.isscalar(distance_km)
        dist = np.atleast_1d(np.asarray(distance_km, dtype=float))

        if hours is not None and len(self.hourly_speeds) > 0:
            h_arr = np.atleast_1d(np.asarray(hours, dtype=int))
            speeds = np.array([self.hourly_speeds.get(h, self.global_speed_kmh) for h in h_arr])
        else:
            speeds = np.full_like(dist, fill_value=self.global_speed_kmh)

        duration_hours = dist / np.maximum(speeds, 1.0)
        duration_minutes = duration_hours * 60.0
        # Ensure minimum 1.0 minute duration
        preds = np.maximum(duration_minutes, 1.0)

        return preds[0] if is_scalar else preds

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted baseline model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HistoricalAvgSpeedBaseline":
        """Load fitted baseline model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
