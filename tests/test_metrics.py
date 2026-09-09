import numpy as np

from src.evaluation.metrics import (
    mae,
    rmse,
    mape,
    r2,
    evaluate_all,
    evaluate_business_thresholds,
)


def test_perfect_prediction_gives_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_known_mae():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    assert mae(y_true, y_pred) == 2.0


def test_evaluate_all_returns_expected_keys():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    result = evaluate_all(y_true, y_pred)
    assert set(result.keys()) == {
        "mae_minutes",
        "mae_seconds",
        "rmse_minutes",
        "mape_pct",
        "r2",
    }
    assert result["mae_minutes"] == 2.0
    assert result["mae_seconds"] == 120.0


def test_business_thresholds_passing():
    metrics = {"mae_minutes": 2.8, "r2": 0.72}
    gate = evaluate_business_thresholds(metrics, max_acceptable_mae_min=3.5, min_acceptable_r2=0.60)
    assert gate["passed_all"] is True
    assert gate["status"] == "APPROVED_FOR_PRODUCTION"


def test_business_thresholds_failing():
    metrics = {"mae_minutes": 4.2, "r2": 0.50}
    gate = evaluate_business_thresholds(metrics, max_acceptable_mae_min=3.5, min_acceptable_r2=0.60)
    assert gate["passed_all"] is False
    assert gate["status"] == "BELOW_BUSINESS_THRESHOLD"
