# SmartTaxi — Smart Arrival Time Estimation (ETA)
## Complete Architecture, Pipeline, Technology Stack & Operational Handbook

**Author:** Hamza — Machine Learning & ETA Lead  
**Project:** SmartTaxi Ecosystem / TSE Consultant INT / ESPRIT  
**Date:** September 2026  
**Document Version:** 2.0 (Production & Integration Guide)  

---

## 1. Executive Summary & Project Mandate

The **Smart Arrival Time Estimation (ETA)** project is an independent, production-grade Machine Learning microservice built for the **SmartTaxi** platform. In an urban mobility ecosystem, precise arrival estimations are critical for:
- Dispatch algorithm optimization (assigning the most suitable taxi).
- Passenger experience (reducing perceived waiting time and building trust).
- Driver efficiency (minimizing empty cruising and optimizing pickup routes).

### Role in the SmartTaxi Ecosystem
Within the central backend architecture (ASP.NET Core .NET 10, coordinated by Houda Ghenmi), the ETA module acts as a **purely consultative, stateless prediction service**:
1. **Zero Database Writes:** The ETA service never writes directly to PostgreSQL or the `Ride` table.
2. **Authoritative Backend Timestamps:** Official timestamps, ride lifecycle transitions, and billing remain the exclusive responsibility of the central backend.
3. **No Silent Data Simulation:** If external traffic or weather feeds are absent, the service never fabricates synthetic values; it explicitly returns flags (`traffic_included: false`, `weather_included: false`).
4. **Resilience & Fallback Assurance:** If the ML model encounters an anomaly, high latency, or reload downtime, an integrated classical urban routing heuristic takes over automatically, guaranteeing 100% service uptime.

---

## 2. Project Layout & Architectural Structure

The repository follows a clean, modular Machine Learning Systems engineering structure:

```
smart-eta-prediction/
├── configs/
│   └── config.yaml                     # Central config: data paths, features, bounds, business SLAs
├── data/
│   ├── raw/nyc_tlc/                    # Raw TLC Parquet files + taxi_zone_lookup.csv (read-only)
│   ├── interim/nyc_tlc/                # Cleaned Parquets, cleaning log, taxi_zone_centroids.csv
│   ├── processed/                      # Model-ready train.parquet, val.parquet, test.parquet, schema
│   └── external/                       # Third-party feeds (weather, traffic)
├── docs/
│   └── project_charter.md              # Business problem formulation, decisions log, target rules
├── notebooks/
│   ├── 01_dataset_audit.ipynb          # Raw data quality audit & feasibility assessment
│   ├── 02_data_cleaning.ipynb          # Cleaning rules verification & interim parquet generation
│   ├── 03_dataset_design.ipynb         # Time-based chronological split (Train/Val/Test)
│   ├── 04_feature_engineering.ipynb    # Spatial & temporal feature transformations
│   ├── 05_eda.ipynb                    # Exploratory data analysis (distributions, peak hours)
│   ├── 06_baseline_modeling.ipynb      # Average speed heuristic baseline with hourly splits
│   ├── 07_advanced_modeling.ipynb      # Tabular models benchmark (RF, XGB, LightGBM, CatBoost)
│   └── 08_evaluation.ipynb             # Out-of-time evaluation, subgroup slices & business gates
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── clean_nyc_tlc.py            # Automated raw data cleaning & audit logger
│   │   ├── make_dataset.py             # Chronological splitting & processed dataset builder
│   │   └── validate.py                 # Schema validation utilities
│   ├── features/
│   │   └── build_features.py           # Haversine, Manhattan distance, cyclical encodings, inference extractor
│   ├── models/
│   │   ├── baseline.py                 # Historical average speed baseline class
│   │   └── train.py                    # Complete training, benchmarking & champion serialization
│   ├── evaluation/
│   │   └── metrics.py                  # MAE (min & sec), RMSE, R², MAPE, business threshold checks
│   └── utils/
│       └── io.py                       # YAML config loader & robust path resolution
├── models/
│   ├── champion_eta_model.joblib       # Serialized champion model bundle (Random Forest v1.0)
│   ├── baseline_model.pkl              # Serialized heuristic baseline model
│   └── model_comparison_results.json   # Full benchmark metrics, slices, and feature importances
├── reports/
│   ├── figures/                        # Benchmark charts, feature importance, slice analysis PNGs
│   ├── rapport_integration_eta_smarttaxi.md  # Official integration session report
│   └── SmartTaxi_ETA_Project_Comprehensive_Handbook.md # This comprehensive report
├── scripts/
│   ├── run_pipeline.py                 # Automated end-to-end pipeline runner (Data -> ML -> Tests -> Serving)
│   └── generate_notebooks.py           # Programmatic notebook generator adhering to src/ imports
├── tests/
│   ├── test_api.py                     # Integration tests (short/long, rush hour, invalid coords, fallback)
│   ├── test_baseline.py                # Unit tests for baseline fitting and predict
│   ├── test_clean_nyc_tlc.py           # Unit tests for cleaning bounds and month filtering
│   ├── test_features.py                # Unit tests for Haversine & Manhattan geometry
│   └── test_metrics.py                 # Unit tests for regression metrics & business gates
├── deployment/
│   ├── Dockerfile                      # Deployment container definition
│   └── app.py                          # Phase 9 serving service wired to champion model
├── app.py                              # Primary production FastAPI serving microservice
├── app_streamlit.py                    # Interactive Streamlit dashboard & live map visualizer
├── simulate_stream.py                  # Real-time event streaming simulator & latency load tester
├── Dockerfile                          # Production FastAPI Dockerfile with healthcheck
├── Dockerfile.streamlit                # Production Streamlit Dockerfile
├── smart-eta-api-task-def.json         # AWS ECS Fargate Task Definition (API microservice)
├── streamlit-task-def.json             # AWS ECS Fargate Task Definition (Streamlit UI)
├── Makefile                            # Standard automation commands (setup, test, api, streamlit, etc.)
├── pyproject.toml                      # Project metadata
├── requirements.txt                    # Frozen dependency specifications
└── README.md                           # Quickstart guide & repository introduction
```

---

## 3. Technology Stack

| Layer | Technologies | Role & Key Features |
| :--- | :--- | :--- |
| **Language & Environment** | Python 3.12 / 3.11, Pip, Virtualenv | Core execution runtime. |
| **Data Processing** | Pandas 2.2, NumPy 1.26, PyArrow 25.0 | High-performance Parquet reading, vectorized array math. |
| **Spatial Geometry** | PyProj 3.8, PyShp 3.1 | Reprojection of NYC taxi zone shapefiles (`EPSG:2263` $\rightarrow$ `EPSG:4326`) to compute accurate centroid coordinates. |
| **Machine Learning** | Scikit-Learn 1.4, LightGBM 4.3, XGBoost 2.0, CatBoost 1.2, Joblib 1.4 | Tabular regression modeling, gradient boosting, multi-threaded training, model artifact serialization. |
| **Serving & API** | FastAPI 0.141, Uvicorn 0.52, Pydantic 2.13, Starlette 1.6 | Asynchronous ASGI HTTP microservice, strict schema validation, OpenAPI / Swagger documentation. |
| **Visualization & UI** | Streamlit 1.63, Altair 6.2, PyDeck, Matplotlib 3.8, Seaborn 0.13 | Interactive web application, real-time map plotting, benchmark visual analytics. |
| **Testing & Quality** | Pytest 8.2, TestClient | Automated unit and integration test harness (24 tests). |
| **DevOps & Containers** | Docker, AWS ECS Fargate (Task Definitions) | Multi-stage container builds, automated healthcheck probes, cloud-ready deployment specs. |

---

## 4. End-to-End Pipeline Steps (How Everything Works)

### Step 1: Raw Data Audit & Feasibility Check
* **File:** `notebooks/01_dataset_audit.ipynb`, `docs/project_charter.md`
* **Purpose:** Audit 15.7M raw records across 4 quarters of 2025 (`yellow_tripdata_2025-01.parquet` to `10.parquet`).
* **Key Finding:** TLC records contain trip start/end timestamps and taxi zone IDs (`PULocationID`, `DOLocationID`). They support **Trip Duration prediction** (Target 2), but lack pre-pickup driver dispatch tracking, proving why **Driver Pickup ETA** (Target 1) must be decoupled into an independent endpoint running in high-reliability routing fallback mode.

### Step 2: Data Cleaning & Spatial Centroids Computation
* **Files:** `src/data/clean_nyc_tlc.py`, `data/interim/nyc_tlc/nyc_tlc_cleaning_log.csv`
* **Rules Applied:**
  - Drop records where duration $\le 0$ or $> 4$ hours.
  - Drop records with missing pickup/dropoff zone IDs.
  - Drop records outside the file's designated month.
  - Reproject NYC shapefile polygons to compute WGS84 centroids (`latitude`, `longitude`) for all 265 taxi zones.

### Step 3: Dataset Design & Leakage-Free Splitting
* **File:** `src/data/make_dataset.py`, `notebooks/03_dataset_design.ipynb`
* **Method:** Strict chronological time-based split without random shuffling to prevent future information from leaking into training:
  - **Train (70%):** 138,718 trips (Jan 1, 2025 to Jul 26, 2025)
  - **Validation (15%):** 29,725 trips (Jul 26, 2025 to Oct 13, 2025)
  - **Test Out-of-Time (15%):** 29,726 trips (Oct 13, 2025 to Oct 31, 2025)

### Step 4: Spatial & Temporal Feature Engineering
* **File:** `src/features/build_features.py`, `notebooks/04_feature_engineering.ipynb`
* **Features Built:**
  - `distance_haversine_km`: Great-circle distance proxy between coordinates.
  - `distance_manhattan_km`: L1 Manhattan distance proxy modeling city street grids.
  - `pickup_hour`, `pickup_dayofweek`, `is_weekend`.
  - `is_rush_hour`: Binary indicator (weekdays 7–9h and 16–19h).
  - `sin_hour`, `cos_hour`: Trigonometric cyclical hour encoding avoiding the 23:59 $\rightarrow$ 00:00 discontinuity.
  - `passenger_count`: Bounded between 1 and 6.

### Step 5: Exploratory Data Analysis (EDA)
* **File:** `notebooks/05_eda.ipynb`
* **Insights:** Non-linear relationship between distance and duration during morning and evening rush hours. Identifies extreme urban congestion slowdowns (implied speeds dropping from 28 km/h off-peak to under 14 km/h during peak Manhattan traffic).

### Step 6: Baseline Heuristic Modeling
* **Files:** `src/models/baseline.py`, `notebooks/06_baseline_modeling.ipynb`
* **Algorithm:** Historical average speed segmented by 24 hourly buckets.
* **Test Performance:** MAE = 7.91 min, $R^2 = 0.258$.
* **Failure Mode:** Fails severely on trips $\ge 10$ km (MAE = 29.65 min) because linear speed cannot capture compounding intersection delays.

### Step 7: Advanced Tabular ML Benchmarking & Model Selection
* **Files:** `src/models/train.py`, `notebooks/07_advanced_modeling.ipynb`
* **Models Compared:** Random Forest, XGBoost, LightGBM, CatBoost.

#### Benchmark Results on Out-of-Time Test Set (29,726 trips):
| Model | MAE (min) | MAE (sec) | RMSE (min) | $R^2$ Score | MAPE (%) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Avg Speed)** | 7.91 min | 474.4 s | 13.14 min | 0.258 | 41.9% | **0.02 ms** |
| **Random Forest (Champion)** | **5.20 min** | **312.1 s** | **8.54 min** | **0.687** | **32.1%** | 21.37 ms |
| **LightGBM Regressor** | 5.28 min | 316.8 s | 8.63 min | 0.680 | 33.0% | **1.10 ms** |
| **XGBoost Regressor** | 5.30 min | 317.7 s | 8.65 min | 0.678 | 33.2% | 1.88 ms |
| **CatBoost Regressor** | 5.40 min | 324.1 s | 8.77 min | 0.670 | 34.2% | 1.25 ms |

* **Champion Selection:** Random Forest achieves the lowest absolute error (5.20 min MAE) with $R^2 = 0.687$ (+165% improvement over baseline). LightGBM serves as an ultra-low latency alternative (1.10 ms).
* **Artifact Saved:** `models/champion_eta_model.joblib`.

### Step 8: Evaluation, Subgroup Slices & Business Gates
* **Files:** `src/evaluation/metrics.py`, `notebooks/08_evaluation.ipynb`
* **Subgroup Slice Performance:**
  - *Short Trips (< 3 km):* Random Forest MAE = **3.64 min** (meets the $\le 4.0$ min urban expectation).
  - *Long Trips ($\ge 10$ km):* Random Forest MAE = **11.31 min** (a **62% error reduction** compared to baseline's 29.65 min).
  - *Rush Hour Trips:* Random Forest MAE = **5.33 min**.
* **Feature Importance Hierarchy:**
  1. `distance_haversine_km`: **84.6%**
  2. `distance_manhattan_km`: **5.1%**
  3. Cyclical hour (`cos_hour`, `sin_hour`, `pickup_hour`): **7.2%**
  4. Calendar features (`pickup_dayofweek`, `is_weekend`): **2.6%**
  5. `passenger_count`, `is_rush_hour`: **0.5%**

### Step 9: Production API & Fallback Routing Engine
* **Files:** `app.py`, `deployment/app.py`
* **Endpoints:**
  - `GET /health`: Liveness probe for backend orchestrator.
  - `GET /model-info`: Returns active model version, test metrics, and feature list.
  - `POST /predict/trip-duration`: Live ML inference with automated failover.
  - `POST /predict/driver-pickup`: Calibrated urban speed heuristic (25 km/h normal, 18 km/h rush hour).
* **Automated Fallback Mechanism:** Any model exception or out-of-distribution feature payload triggers automatic fallback to calibrated routing speed without returning HTTP 500. The response explicitly sets `fallback_used: true` and `quality_flag: "FALLBACK_ROUTING"`.

### Step 10: Interactive Streamlit Dashboard
* **File:** `app_streamlit.py`
* **Capabilities:**
  - Interactive map with pickup and destination pins.
  - Real-time ETA calculation (seconds, minutes, distance, latency).
  - Driver pickup estimation module.
  - Model benchmark visualization tabs.
  - Real-time event streaming simulator with live latency metrics.
  - SmartTaxi backend integration governance rules display.

### Step 11: Real-Time Event Streaming Simulator
* **File:** `simulate_stream.py`
* **Purpose:** Generates synthetic ride requests across NYC coordinates at customizable intervals (e.g. 0.2s) to load-test the API, verify latency SLAs (< 50ms), and inspect fallback flags under concurrency.

### Step 12: Automated Pipeline Runner
* **File:** `scripts/run_pipeline.py`
* **Execution:** Orchestrates Data splitting $\rightarrow$ Model training $\rightarrow$ Figure generation $\rightarrow$ Pytest suite (24 tests) $\rightarrow$ Live API health validation in ~33 seconds.

---

## 5. Complete Command Reference Handbook

### 5.1. Environment Setup & Activation

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install all dependencies
.venv\Scripts\pip.exe install -r requirements.txt
```

### 5.2. Running the Automated Pipeline

```powershell
# Run entire pipeline (data, training, figures, tests, api check)
.venv\Scripts\python.exe scripts/run_pipeline.py

# Or via Makefile
make run-pipeline
```

### 5.3. Running Automated Tests (Pytest)

```powershell
# Run complete test suite (all 24 tests)
.venv\Scripts\pytest.exe -v

# Run only API integration tests
.venv\Scripts\pytest.exe tests/test_api.py -v

# Run only feature engineering tests
.venv\Scripts\pytest.exe tests/test_features.py -v
```

### 5.4. Running the Production FastAPI Server

```powershell
# Launch FastAPI server with hot-reload on port 8000
.venv\Scripts\uvicorn.exe app:app --host 0.0.0.0 --port 8000 --reload

# Test health check in terminal
curl http://127.0.0.1:8000/health

# Test model metadata
curl http://127.0.0.1:8000/model-info
```
*Interactive Swagger Documentation:* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5.5. Running the Streamlit Dashboard UI

```powershell
# Launch Streamlit interface on port 8501
.venv\Scripts\streamlit.exe run app_streamlit.py --server.port 8501
```
*Browser URL:* [http://localhost:8501](http://localhost:8501)

### 5.6. Running the Real-Time Streaming Simulator

```powershell
# Stream 15 consecutive ride events at 0.2s intervals
.venv\Scripts\python.exe simulate_stream.py --num-trips 15 --interval 0.2 --api-url http://127.0.0.1:8000
```

### 5.7. Docker & Containerization Commands

```powershell
# Build API container image
docker build -t smart-eta-api:latest -f Dockerfile .

# Run API container on port 8000
docker run -d -p 8000:8000 --name eta-api smart-eta-api:latest

# Build Streamlit UI container image
docker build -t smart-eta-streamlit:latest -f Dockerfile.streamlit .

# Run Streamlit container on port 8501
docker run -d -p 8501:8501 -e API_URL="http://host.docker.internal:8000" --name eta-ui smart-eta-streamlit:latest
```

### 5.8. Git Workflow & Version Control

```powershell
# Check status of changed files
git status

# Add all project files
git add .

# Recommended conventional commit message
git commit -m "feat(eta): implement end-to-end ML pipeline, serving API, fallback, and integration suite"

# Push to remote repository
git push origin main
```

---

## 6. Summary of Compliance with Integration Guide

| Requirement from Integration Guide | Implementation Status | Evidence / Verification |
| :--- | :---: | :--- |
| **Verrouiller la cible exacte avant entraînement** | **Conforme** | Cibles découplées : Driver Pickup (`/predict/driver-pickup`) et Trip Duration (`/predict/trip-duration`). |
| **Dictionnaire des features réellement disponibles** | **Conforme** | Spécifié dans `data/processed/feature_dictionary.json` ; pas de simulation silencieuse. |
| **Baseline + comparaison modèles tabulaires** | **Conforme** | 5 modèles comparés (Baseline, RF, XGBoost, LightGBM, CatBoost) dans `models/model_comparison_results.json`. |
| **Évaluer MAE/RMSE et exprimer MAE en minutes** | **Conforme** | MAE exprimée en minutes et secondes dans les métriques et dans le payload API. |
| **Exposer une API de prédiction indépendante** | **Conforme** | FastAPI avec `eta_seconds`, `eta_minutes`, `model_version`, `quality_flag`, `latency_ms`. |
| **Stratégie de fallback documentée** | **Conforme** | Fallback routier heuristique urbain avec activation automatique en cas d'erreur. |
| **Aucune écriture directe dans Ride** | **Conforme** | Service 100% stateless et read-only sans connexion d'écriture à PostgreSQL. |
| **Tests d'intégration minimum** | **Conforme** | 24 tests `pytest` couvrant trajets courts/longs, rush hour, données manquantes, coordonnées invalides, latence et indisponibilité. |
| **Conteneurisation préparée** | **Conforme** | `Dockerfile`, `Dockerfile.streamlit`, et task definitions AWS ECS configurées. |
