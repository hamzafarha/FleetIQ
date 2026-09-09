"""Standard regression metrics for trip-duration prediction.

Supports technical and business evaluation:
- MAE (in minutes and seconds for business stakeholders)
- RMSE (in minutes)
- MAPE (percentage error)
- R² (coefficient of determination)
- Business acceptance threshold validation
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, in the same units as target (minutes)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error, in the same units as target (minutes)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error. epsilon avoids divide-by-zero on short trips."""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))) * 100


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R² score)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Convenience wrapper returning standard regression metrics.

    Expresses MAE both in minutes and seconds to meet business reporting requirements.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae_min = mae(y_t, y_p)
    return {
        "mae_minutes": round(mae_min, 3),
        "mae_seconds": round(mae_min * 60.0, 1),
        "rmse_minutes": round(rmse(y_t, y_p), 3),
        "mape_pct": round(mape(y_t, y_p), 2),
        "r2": round(r2(y_t, y_p), 4),
    }


def evaluate_business_thresholds(
    metrics: dict,
    max_acceptable_mae_min: float = 3.5,
    min_acceptable_r2: float = 0.60,
) -> dict:
    """Evaluate whether model performance meets production business gates."""
    mae_pass = metrics.get("mae_minutes", float("inf")) <= max_acceptable_mae_min
    r2_pass = metrics.get("r2", -float("inf")) >= min_acceptable_r2
    overall_pass = mae_pass and r2_pass

    return {
        "passed_all": overall_pass,
        "mae_threshold_met": mae_pass,
        "r2_threshold_met": r2_pass,
        "max_acceptable_mae_min": max_acceptable_mae_min,
        "min_acceptable_r2": min_acceptable_r2,
        "status": "APPROVED_FOR_PRODUCTION" if overall_pass else "BELOW_BUSINESS_THRESHOLD",
    }
