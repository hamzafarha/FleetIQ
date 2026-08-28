"""Phase 6 baseline: historical average speed x route distance.

Deliberately left as a stub. The actual averaging strategy (global avg?
segmented by hour/zone? which distance feature?) is a Phase 6 modeling
decision, not something to lock in while still scaffolding the repo.
"""
import pandas as pd


class HistoricalAvgSpeedBaseline:
    """Predicts trip duration as distance / historical_avg_speed.

    TODO (Phase 6): decide segmentation granularity for the average speed
    — e.g. global, by pickup_hour, by borough/zone — before implementing
    `fit`. Keep the interface fit/predict so it's swappable with the
    gradient-boosting and DL models later.
    """

    def __init__(self):
        self.avg_speed_kmh: float | None = None

    def fit(self, distance_km: pd.Series, duration_min: pd.Series) -> "HistoricalAvgSpeedBaseline":
        raise NotImplementedError("Implement in Phase 6 once segmentation strategy is decided.")

    def predict(self, distance_km: pd.Series) -> pd.Series:
        raise NotImplementedError("Implement in Phase 6 once segmentation strategy is decided.")
