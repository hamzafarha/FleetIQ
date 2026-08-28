import numpy as np

from src.evaluation.metrics import mae, rmse, mape, evaluate_all


def test_perfect_prediction_gives_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_known_mae():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    assert mae(y_true, y_pred) == 2.0


def test_evaluate_all_returns_expected_keys():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    result = evaluate_all(y_true, y_pred)
    assert set(result.keys()) == {"mae_minutes", "rmse_minutes", "mape_pct"}
