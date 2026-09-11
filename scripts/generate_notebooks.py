"""Generate complete, runnable Jupyter notebooks (03 to 08) conforming to the project architecture."""
import json
from pathlib import Path


def make_cell(cell_type: str, source: list[str]):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" if not s.endswith("\n") else s for s in source],
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def write_notebook(path: Path, cells: list[dict]):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {path.name}")


def generate_all():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(exist_ok=True)

    # 03_dataset_design.ipynb
    nb03_cells = [
        make_cell("markdown", [
            "# FleetIQ / SmartTaxi — 03. Dataset Design & Chronological Split",
            "",
            "## Objective",
            "Build model-ready partitions (`train.parquet`, `val.parquet`, `test.parquet`) from interim cleaned data.",
            "Enforces **chronological time-based splitting** to strictly eliminate future data leakage.",
        ]),
        make_cell("code", [
            "import sys",
            "from pathlib import Path",
            "sys.path.append('..')",
            "",
            "import pandas as pd",
            "from src.utils.io import load_config",
            "from src.data.make_dataset import build_and_save_processed_datasets",
            "",
            "config = load_config()",
            "print('Config loaded successfully for project:', config['project']['name'])",
        ]),
        make_cell("markdown", [
            "## 1. Load Interim Data & Build Chronological Splits",
            "Merges NYC TLC zone centroids and splits temporally into Train (70%), Val (15%), and Test (15%).",
        ]),
        make_cell("code", [
            "train_path, val_path, test_path = build_and_save_processed_datasets(sample_per_month=50000, config=config)",
            "print(f'Train split: {train_path}')",
            "print(f'Validation split: {val_path}')",
            "print(f'Test split: {test_path}')",
        ]),
        make_cell("markdown", [
            "## 2. Inspect Split Boundaries & Non-Overlapping Timestamps",
        ]),
        make_cell("code", [
            "train_df = pd.read_parquet(train_path)",
            "val_df = pd.read_parquet(val_path)",
            "test_df = pd.read_parquet(test_path)",
            "",
            "print(f'Train: {len(train_df):,} rows | {train_df[\"tpep_pickup_datetime\"].min()} -> {train_df[\"tpep_pickup_datetime\"].max()}')",
            "print(f'Val:   {len(val_df):,} rows | {val_df[\"tpep_pickup_datetime\"].min()} -> {val_df[\"tpep_pickup_datetime\"].max()}')",
            "print(f'Test:  {len(test_df):,} rows | {test_df[\"tpep_pickup_datetime\"].min()} -> {test_df[\"tpep_pickup_datetime\"].max()}')",
            "",
            "assert train_df['tpep_pickup_datetime'].max() < val_df['tpep_pickup_datetime'].min()",
            "assert val_df['tpep_pickup_datetime'].max() < test_df['tpep_pickup_datetime'].min()",
            "print('✅ Zero temporal data leakage verified!')",
        ]),
    ]
    write_notebook(nb_dir / "03_dataset_design.ipynb", nb03_cells)

    # 04_feature_engineering.ipynb
    nb04_cells = [
        make_cell("markdown", [
            "# FleetIQ / SmartTaxi — 04. Feature Engineering",
            "",
            "## Objective",
            "Validate spatial and temporal feature transformations implemented in `src/features/build_features.py`.",
            "Includes Haversine, Manhattan distance proxies, rush-hour indicators, and cyclical hour encodings.",
        ]),
        make_cell("code", [
            "import sys",
            "sys.path.append('..')",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from src.features.build_features import (",
            "    haversine_distance_km,",
            "    manhattan_distance_km,",
            "    add_datetime_features,",
            "    extract_features_for_inference,",
            "    FEATURE_NAMES,",
            ")",
            "",
            "print('Target feature schema:', FEATURE_NAMES)",
        ]),
        make_cell("markdown", [
            "## 1. Verify Spatial Distance Functions",
        ]),
        make_cell("code", [
            "# Times Square (40.7580, -73.9855) to Central Park (40.7829, -73.9654)",
            "p_lat, p_lon = 40.7580, -73.9855",
            "d_lat, d_lon = 40.7829, -73.9654",
            "",
            "hav_dist = haversine_distance_km(p_lat, p_lon, d_lat, d_lon)",
            "man_dist = manhattan_distance_km(p_lat, p_lon, d_lat, d_lon)",
            "print(f'Haversine Distance: {hav_dist:.3f} km')",
            "print(f'Manhattan Distance: {man_dist:.3f} km')",
            "assert man_dist >= hav_dist, 'Manhattan L1 distance must be >= Haversine straight line'",
        ]),
        make_cell("markdown", [
            "## 2. Test Real-time Inference Feature Extraction",
        ]),
        make_cell("code", [
            "features_df = extract_features_for_inference(",
            "    pickup_latitude=p_lat,",
            "    pickup_longitude=p_lon,",
            "    dropoff_latitude=d_lat,",
            "    dropoff_longitude=d_lon,",
            "    pickup_datetime='2025-06-16T08:30:00Z',",
            "    passenger_count=2,",
            ")",
            "features_df",
        ]),
    ]
    write_notebook(nb_dir / "04_feature_engineering.ipynb", nb04_cells)

    # 05_eda.ipynb
    nb05_cells = [
        make_cell("markdown", [
            "# FleetIQ / SmartTaxi — 05. Exploratory Data Analysis (EDA)",
            "",
            "## Objective",
            "Explore distributions of trip durations, distance proxies, and rush hour variations across the training set.",
        ]),
        make_cell("code", [
            "import sys",
            "sys.path.append('..')",
            "",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "train_df = pd.read_parquet('../data/processed/train.parquet')",
            "print(f'Loaded {len(train_df):,} training records')",
            "train_df[['trip_duration_minutes', 'distance_haversine_km', 'pickup_hour', 'is_rush_hour']].describe()",
        ]),
        make_cell("markdown", [
            "## 1. Trip Duration & Distance Distributions",
        ]),
        make_cell("code", [
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))",
            "sns.histplot(train_df['trip_duration_minutes'], bins=50, kde=True, ax=ax1, color='#2b5c8f')",
            "ax1.set_title('Trip Duration Distribution (Minutes)')",
            "ax1.set_xlim(0, 60)",
            "",
            "sns.histplot(train_df['distance_haversine_km'], bins=50, kde=True, ax=ax2, color='#2ca02c')",
            "ax2.set_title('Haversine Distance Distribution (km)')",
            "ax2.set_xlim(0, 25)",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_cell("markdown", [
            "## 2. Average Trip Duration by Hour of Day",
        ]),
        make_cell("code", [
            "hourly_avg = train_df.groupby('pickup_hour')['trip_duration_minutes'].mean()",
            "plt.figure(figsize=(10, 4))",
            "hourly_avg.plot(kind='bar', color='#d62728', edgecolor='black', alpha=0.85)",
            "plt.title('Average Trip Duration by Hour of Day')",
            "plt.xlabel('Hour (0-23)')",
            "plt.ylabel('Mean Duration (minutes)')",
            "plt.grid(True, alpha=0.3)",
            "plt.show()",
        ]),
    ]
    write_notebook(nb_dir / "05_eda.ipynb", nb05_cells)

    # 06_baseline_modeling.ipynb
    nb06_cells = [
        make_cell("markdown", [
            "# FleetIQ / SmartTaxi — 06. Baseline Modeling",
            "",
            "## Objective",
            "Fit and evaluate the Historical Average Speed heuristic baseline model (`src.models.baseline`).",
            "Measures global urban speed and hourly segmented speeds to establish the lower bound benchmark.",
        ]),
        make_cell("code", [
            "import sys",
            "sys.path.append('..')",
            "",
            "import pandas as pd",
            "from src.models.baseline import HistoricalAvgSpeedBaseline",
            "from src.evaluation.metrics import evaluate_all",
            "",
            "train_df = pd.read_parquet('../data/processed/train.parquet')",
            "test_df = pd.read_parquet('../data/processed/test.parquet')",
            "",
            "baseline = HistoricalAvgSpeedBaseline(default_speed_kmh=25.0)",
            "baseline.fit(",
            "    distance_km=train_df['distance_haversine_km'],",
            "    duration_min=train_df['trip_duration_minutes'],",
            "    hours=train_df['pickup_hour'],",
            ")",
            "",
            "print(f'Fitted Global Speed: {baseline.global_speed_kmh:.2f} km/h')",
            "print('Sample Hourly Speeds (km/h):', {h: round(baseline.hourly_speeds[h], 1) for h in range(0, 24, 4)})",
        ]),
        make_cell("markdown", [
            "## 1. Evaluate Baseline on Held-out Out-of-Time Test Set",
        ]),
        make_cell("code", [
            "y_test = test_df['trip_duration_minutes'].values",
            "preds = baseline.predict(test_df['distance_haversine_km'], hours=test_df['pickup_hour'])",
            "",
            "metrics = evaluate_all(y_test, preds)",
            "for k, v in metrics.items():",
            "    print(f'{k:<16}: {v}')",
        ]),
    ]
    write_notebook(nb_dir / "06_baseline_modeling.ipynb", nb06_cells)

    # 07_advanced_modeling.ipynb
    nb07_cells = [
        make_cell("markdown", [
            "# FleetIQ / SmartTaxi — 07. Advanced Tabular Modeling",
            "",
            "## Objective",
            "Train and compare tabular machine learning models (Random Forest, XGBoost, LightGBM, CatBoost).",
            "Uses `src.models.train.train_and_evaluate_models` and serializes the champion model bundle.",
        ]),
        make_cell("code", [
            "import sys",
            "sys.path.append('..')",
            "",
            "import json",
            "import pandas as pd",
            "from src.models.train import train_and_evaluate_models",
            "",
            "summary = train_and_evaluate_models()",
            "print('Champion selected:', summary['champion_model'])",
        ]),
        make_cell("markdown", [
            "## 1. Tabular Model Comparison Summary",
        ]),
        make_cell("code", [
            "models_summary = summary['models_summary']",
            "df_res = pd.DataFrame(models_summary).T",
            "df_res[['mae_minutes', 'mae_seconds', 'rmse_minutes', 'r2', 'latency_ms']]",
        ]),
    ]
    write_notebook(nb_dir / "07_advanced_modeling.ipynb", nb07_cells)

    # 08_evaluation.ipynb
    nb08_cells = [
        make_cell("markdown", [
            "# FleetIQ / SmartTaxi — 08. Evaluation & Business Acceptance",
            "",
            "## Objective",
            "Validate model performance against business acceptance gates, analyze subgroup slices, and inspect feature importances.",
        ]),
        make_cell("code", [
            "import sys",
            "sys.path.append('..')",
            "",
            "import json",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "with open('../models/model_comparison_results.json') as f:",
            "    results = json.load(f)",
            "",
            "print('Champion Model:', results['champion_model'])",
            "print('Business Status:', results['business_status'])",
        ]),
        make_cell("markdown", [
            "## 1. Subgroup Slice Evaluation (Short/Long, Rush/Off-peak)",
        ]),
        make_cell("code", [
            "slices_df = pd.DataFrame(results['slice_analysis']).T",
            "slices_df",
        ]),
        make_cell("markdown", [
            "## 2. Feature Importance",
        ]),
        make_cell("code", [
            "fi = results['champion_feature_importance']",
            "pd.Series(fi).plot(kind='barh', figsize=(8, 4), color='#2b5c8f', edgecolor='black')",
            "plt.title('Champion Model Feature Importance (%)')",
            "plt.xlabel('Importance (%)')",
            "plt.show()",
        ]),
    ]
    write_notebook(nb_dir / "08_evaluation.ipynb", nb08_cells)


if __name__ == "__main__":
    generate_all()
