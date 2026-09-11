"""Complete end-to-end pipeline execution script for SmartTaxi Smart ETA Prediction.

Executes all architecture stages:
1. Data splitting & validation (Chronological time-based split)
2. Feature engineering & extraction
3. Tabular model training & benchmarking (Baseline, RF, XGBoost, LightGBM, CatBoost)
4. Champion model selection & serialization
5. Evaluation figures generation
6. Test suite execution (pytest)
7. Health and inference endpoint verification
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def log_step(step_name: str):

    print("\n" + "=" * 70)
    print(f"[*] [STEP] {step_name}")
    print("=" * 70)


def run_cmd(cmd: list[str], description: str):
    print(f"Running: {' '.join(cmd)} ({description})")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error during {description}:\n{res.stderr}\n{res.stdout}")
        sys.exit(res.returncode)
    print(f"[OK] {description} completed successfully.")
    if res.stdout.strip():
        print(res.stdout.strip().split("\n")[-1])



def main():
    start_time = time.perf_counter()
    python_bin = sys.executable

    log_step("1. Build and Validate Chronological Dataset Splits")
    run_cmd(
        [python_bin, "-m", "src.data.make_dataset"],
        "Time-based Train/Val/Test Dataset Generation",
    )

    log_step("2. Train and Benchmark Tabular Models (Baseline, RF, XGB, LightGBM, CatBoost)")
    run_cmd(
        [python_bin, "-m", "src.models.train"],
        "Tabular Model Training and Champion Selection",
    )

    log_step("3. Generate Evaluation Figures and Benchmark Plots")
    run_cmd(
        [
            python_bin,
            "-c",
            """
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

with open('models/model_comparison_results.json') as f:
    res = json.load(f)
summary = res['models_summary']
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(summary.keys(), [summary[m]['mae_minutes'] for m in summary.keys()], color='#2b5c8f', edgecolor='black')
ax.set_ylabel('MAE (min)')
ax.set_title('Test MAE Comparison (Minutes)')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
Path('reports/figures').mkdir(parents=True, exist_ok=True)
plt.savefig('reports/figures/model_comparison_benchmark.png', dpi=150)
print('Updated reports/figures/model_comparison_benchmark.png')
""",
        ],
        "Benchmark Figure Generation",
    )

    log_step("4. Run Complete Pytest Integration Suite")
    run_cmd(
        [python_bin, "-m", "pytest", "tests/", "-v"],
        "Pytest Automated Test Suite",
    )

    log_step("5. Validate API Serving & Health Check")
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    health = client.get("/health").json()
    model_info = client.get("/model-info").json()

    print("Health Check Response :", health)
    print("Active Model Name     :", model_info.get("model_name"))
    print("Active Model Version  :", model_info.get("model_version"))
    print("Champion Test MAE     :", model_info.get("metrics", {}).get("mae_minutes"), "min")

    # Sample Trip prediction
    trip_sample = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7829,
        "dropoff_longitude": -73.9654,
        "pickup_datetime": "2025-06-16T14:30:00Z",
    }
    trip_res = client.post("/predict/trip-duration", json=trip_sample).json()
    print("Trip Sample Prediction:", trip_res)

    # Sample Driver Pickup prediction
    driver_sample = {
        "driver_latitude": 40.7500,
        "driver_longitude": -73.9900,
        "passenger_latitude": 40.7580,
        "passenger_longitude": -73.9855,
        "assignment_datetime": "2025-06-16T10:00:00Z",
    }
    driver_res = client.post("/predict/driver-pickup", json=driver_sample).json()
    print("Driver Pickup Prediction:", driver_res)

    total_duration = round(time.perf_counter() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"SUCCESS: FULL PIPELINE COMPLETED IN {total_duration} SECONDS!")
    print("=" * 70)



if __name__ == "__main__":
    main()
