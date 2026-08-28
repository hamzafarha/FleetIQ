"""Standard regression metrics for trip-duration prediction.

These are implementation-agnostic and don't constitute a "modeling
decision," so they're implemented directly rather than stubbed.
"""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, in the same units as the target (minutes)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error, in the same units as the target (minutes)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error. epsilon avoids divide-by-zero on
    very short trips.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))) * 100


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Convenience wrapper returning all metrics at once — use this in
    notebooks so every model reports the same set, comparably.
    """
    return {
        "mae_minutes": mae(y_true, y_pred),
        "rmse_minutes": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }
