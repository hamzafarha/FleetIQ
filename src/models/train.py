"""Model training, benchmarking, and selection pipeline for SmartTaxi ETA.

Compares:
1. Historical Average Speed Baseline (hourly segmented)
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. CatBoost Regressor

Evaluates on strictly out-of-time test split:
- MAE (minutes & seconds)
- RMSE (minutes)
- MAPE (%)
- R² score
- Slice evaluation (Rush hour vs. Off-peak, Short trips < 3km vs. Long trips >= 10km)
- Inference latency per sample (ms)
- Business acceptance threshold validation

Saves champion model to `models/champion_eta_model.joblib`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.evaluation.metrics import evaluate_all, evaluate_business_thresholds
from src.features.build_features import FEATURE_NAMES
from src.models.baseline import HistoricalAvgSpeedBaseline
from src.utils.io import load_config, resolve_path


def load_datasets(
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    """Load train, val, and test splits from processed directory."""
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    target_col = "trip_duration_minutes"
    features = [f for f in FEATURE_NAMES if f in train_df.columns]
    return train_df, val_df, test_df, features, target_col


def measure_inference_latency(model: Any, X_sample: pd.DataFrame, runs: int = 100) -> float:
    """Measure average single-sample inference latency in milliseconds."""
    single_row = X_sample.iloc[[0]]
    # Warmup
    for _ in range(10):
        _ = model.predict(single_row)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model.predict(single_row)
    total_time = time.perf_counter() - t0
    return round((total_time / runs) * 1000.0, 3)


def train_and_evaluate_models(config: dict = None) -> Dict[str, Any]:
    """Train all tabular models, evaluate on test set, and save the champion."""
    config = config or load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed datasets...")
    train_df, val_df, test_df, features, target_col = load_datasets(processed_dir)

    X_train = train_df[features]
    y_train = train_df[target_col].values
    X_val = val_df[features]
    y_val = val_df[target_col].values
    X_test = test_df[features]
    y_test = test_df[target_col].values

    print(f"Dataset shape: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Features utilized: {features}")

    results = {}
    fitted_models = {}

    # 1. Baseline Model (Historical average speed segmented by hour)
    print("\n--- [1/5] Fitting Historical Average Speed Baseline ---")
    baseline = HistoricalAvgSpeedBaseline(default_speed_kmh=25.0)
    baseline.fit(
        distance_km=train_df["distance_haversine_km"],
        duration_min=train_df[target_col],
        hours=train_df["pickup_hour"],
    )
    baseline_preds = baseline.predict(
        distance_km=test_df["distance_haversine_km"],
        hours=test_df["pickup_hour"],
    )
    baseline_metrics = evaluate_all(y_test, baseline_preds)

    # Latency for baseline
    t0 = time.perf_counter()
    for _ in range(100):
        _ = baseline.predict(distance_km=X_test["distance_haversine_km"].iloc[0], hours=X_test["pickup_hour"].iloc[0])
    baseline_latency = round(((time.perf_counter() - t0) / 100.0) * 1000.0, 3)
    baseline_metrics["latency_ms"] = baseline_latency

    results["Baseline (Avg Speed)"] = baseline_metrics
    fitted_models["Baseline (Avg Speed)"] = baseline
    baseline.save(models_dir / "baseline_model.pkl")
    print(f"Baseline Test MAE: {baseline_metrics['mae_minutes']} min, R2: {baseline_metrics['r2']}")

    # 2. Random Forest Regressor
    print("\n--- [2/5] Training Random Forest Regressor ---")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = evaluate_all(y_test, rf_preds)
    rf_metrics["latency_ms"] = measure_inference_latency(rf, X_test)
    results["Random Forest"] = rf_metrics
    fitted_models["Random Forest"] = rf
    print(f"Random Forest Test MAE: {rf_metrics['mae_minutes']} min, R2: {rf_metrics['r2']}")

    # 3. XGBoost Regressor
    print("\n--- [3/5] Training XGBoost Regressor ---")
    xgb = XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb.predict(X_test)
    xgb_metrics = evaluate_all(y_test, xgb_preds)
    xgb_metrics["latency_ms"] = measure_inference_latency(xgb, X_test)
    results["XGBoost"] = xgb_metrics
    fitted_models["XGBoost"] = xgb
    print(f"XGBoost Test MAE: {xgb_metrics['mae_minutes']} min, R2: {xgb_metrics['r2']}")

    # 4. LightGBM Regressor
    print("\n--- [4/5] Training LightGBM Regressor ---")
    lgbm = LGBMRegressor(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgbm_preds = lgbm.predict(X_test)
    lgbm_metrics = evaluate_all(y_test, lgbm_preds)
    lgbm_metrics["latency_ms"] = measure_inference_latency(lgbm, X_test)
    results["LightGBM"] = lgbm_metrics
    fitted_models["LightGBM"] = lgbm
    print(f"LightGBM Test MAE: {lgbm_metrics['mae_minutes']} min, R2: {lgbm_metrics['r2']}")

    # 5. CatBoost Regressor
    print("\n--- [5/5] Training CatBoost Regressor ---")
    cat = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.07,
        thread_count=-1,
        random_seed=42,
        verbose=0,
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    cat_preds = cat.predict(X_test)
    cat_metrics = evaluate_all(y_test, cat_preds)
    cat_metrics["latency_ms"] = measure_inference_latency(cat, X_test)
    results["CatBoost"] = cat_metrics
    fitted_models["CatBoost"] = cat
    print(f"CatBoost Test MAE: {cat_metrics['mae_minutes']} min, R2: {cat_metrics['r2']}")

    # --- Sliced Subgroup Analysis (Short vs Long trips, Rush Hour vs Off-peak) ---
    print("\n--- Performing Subgroup Slice Analysis on Test Set ---")
    short_mask = test_df["distance_haversine_km"] < 3.0
    long_mask = test_df["distance_haversine_km"] >= 10.0
    rush_mask = test_df["is_rush_hour"] == 1
    offpeak_mask = test_df["is_rush_hour"] == 0

    slice_evaluations = {}
    preds_dict = {
        "Baseline (Avg Speed)": baseline_preds,
        "Random Forest": rf_preds,
        "XGBoost": xgb_preds,
        "LightGBM": lgbm_preds,
        "CatBoost": cat_preds,
    }

    for name, p in preds_dict.items():
        slice_evaluations[name] = {
            "overall_mae_min": round(float(np.mean(np.abs(y_test - p))), 3),
            "short_trips_mae_min": round(float(np.mean(np.abs(y_test[short_mask] - p[short_mask]))), 3),
            "long_trips_mae_min": round(float(np.mean(np.abs(y_test[long_mask] - p[long_mask]))), 3),
            "rush_hour_mae_min": round(float(np.mean(np.abs(y_test[rush_mask] - p[rush_mask]))), 3),
            "offpeak_mae_min": round(float(np.mean(np.abs(y_test[offpeak_mask] - p[offpeak_mask]))), 3),
        }

    # Identify Champion Model (Lowest MAE and High R²)
    model_candidates = [m for m in results.keys() if m != "Baseline (Avg Speed)"]
    champion_name = min(model_candidates, key=lambda m: results[m]["mae_minutes"])
    champion_model = fitted_models[champion_name]

    print(f"\n=======================================================")
    print(f" CHAMPION MODEL SELECTED: {champion_name}")
    print(f" Test MAE: {results[champion_name]['mae_minutes']} min ({results[champion_name]['mae_seconds']} sec)")
    print(f" Test RMSE: {results[champion_name]['rmse_minutes']} min")
    print(f" Test R2: {results[champion_name]['r2']}")
    print(f" Single-row Latency: {results[champion_name]['latency_ms']} ms")
    print(f"=======================================================")

    # Check Business Gates
    business_check = evaluate_business_thresholds(
        results[champion_name],
        max_acceptable_mae_min=3.5,
        min_acceptable_r2=0.60,
    )

    # Feature Importance for Champion
    feature_importances = {}
    if hasattr(champion_model, "feature_importances_"):
        fi = champion_model.feature_importances_
        if len(fi) == len(features):
            norm_fi = (fi / np.sum(fi)) * 100.0
            feature_importances = {feat: round(float(score), 2) for feat, score in zip(features, norm_fi)}
            feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # Save Champion Model Artifacts
    champion_save_path = models_dir / "champion_eta_model.joblib"
    champion_bundle = {
        "model": champion_model,
        "model_name": champion_name,
        "model_version": f"v1.0_{champion_name.lower().replace(' ', '_')}",
        "features": features,
        "metrics": results[champion_name],
        "business_validation": business_check,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(champion_bundle, champion_save_path)
    print(f"Saved Champion model bundle to {champion_save_path}")

    # Export Full Comparison Report JSON
    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_model": champion_name,
        "business_status": business_check["status"],
        "models_summary": results,
        "slice_analysis": slice_evaluations,
        "champion_feature_importance": feature_importances,
    }
    with open(models_dir / "model_comparison_results.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)

    return comparison_summary


if __name__ == "__main__":
    train_and_evaluate_models()

