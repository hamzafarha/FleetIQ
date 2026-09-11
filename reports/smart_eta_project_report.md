# Smart ETA Prediction Project Report

## 1. Project Scope and Objective

Smart ETA Prediction is a production-track Smart Arrival Time Estimation system for taxi dispatch. It is prototyped on the public NYC Taxi Trip Duration dataset while keeping the architecture compatible with future company data.

The repository is organized around an end-to-end, config-driven machine learning workflow that produces processed datasets, trains several tabular regression models, selects a champion model, and exposes prediction endpoints with FastAPI and a Streamlit demo/dashboard.

## 2. Project Structure

The top-level repository contains:

- `app.py`: production FastAPI API service.
- `app_streamlit.py`: Streamlit dashboard UI.
- `simulate_stream.py`: synthetic stream simulator for API stress/load testing.
- `src/`: reusable source-code modules for data, feature engineering, training, evaluation, and shared utilities.
- `data/`: raw, interim, processed, and external data folders.
- `configs/` and `config/`: YAML project configuration.
- `docs/`: business and architecture notes.
- `notebooks/`: numbered exploratory and pipeline notebooks from 01 through 08.
- `models/`: serialized model bundle and model comparison artifacts.
- `reports/`: generated reports and figures.
- `tests/`: pytest test suite.

## 3. Core Pipeline Overview

The repository pipeline is designed in a staged sequence:

1. Data preparation and validation from `src.data.make_dataset`.
2. Model training and evaluation from `src.models.train`.
3. Champion model selection and artifact export to `models/champion_eta_model.joblib`.
4. Metric export and figures generation.
5. API health and inference validation.
6. Integration test execution.

The canonical orchestration command is implemented in `scripts/run_pipeline.py` and performs all stages via `subprocess`.

## 4. Data Pipeline Steps

### Step 1: Load and Clean Data

The data source is the NYC TLC relational tabular dataset. Cleaning and validation rules live in `src/data/clean_nyc_tlc.py` and in the notebook series. The source data and interim data are organized under `data/raw/nyc_tlc/` and `data/interim/nyc_tlc/`.

### Step 2: Create Chronological Splits

The dataset is loaded by `src.data.make_dataset`, which merges cleaned data with taxi zone centroids, adds datetime features, computes haversine and Manhattan distances, and saves chronological train/validation/test Parquet files to `data/processed/`.

### Step 3: Feature Engineering

`src/features/build_features.py` adds the model-ready feature list:

- `distance_haversine_km`
- `distance_manhattan_km`
- `pickup_hour`
- `pickup_dayofweek`
- `is_weekend`
- `is_rush_hour`
- `sin_hour`
- `cos_hour`
- `passenger_count`

These features are designed to be available during inference and avoid leakage.

### Step 4: Train and Benchmark Models

`src/models/train.py` trains and benchmarks the following models:

- Historical average speed baseline
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

It computes MAE, RMSE, MAPE, R², business acceptance gates, and slice evaluation for short trips, long trips, rush-hour, and off-peak periods.

### Step 5: Save Champion Model Bundle

The selected champion model is saved as a joblib bundle containing:

- the fitted model object,
- the model name,
- the model version,
- feature list,
- evaluation metrics,
- business gate pass/fail metadata,
- creation timestamp.

### Step 6: Operational API and Streamlit

The FastAPI app in `app.py` exposes endpoints for:

- `/health`
- `/model-info`
- `/predict/trip-duration`
- `/predict/driver-pickup`

The Streamlit UI in `app_streamlit.py` provides demo and monitoring screens for ETA inference, benchmark reporting, and API simulation.

## 5. Tech Stack

The stack documented by the project files is:

### Python and Data

- Python >= 3.10
- pandas
- numpy
- pyarrow
- pyyaml
- scikit-learn
- joblib
- matplotlib
- seaborn
- jupyter

### ML and Model Stack

- xgboost
- lightgbm
- catboost
- sklearn ensemble models
- a historical average speed heuristic baseline

### Deployment and API

- FastAPI
- Uvicorn
- Pydantic
- Streamlit
- requests
- httpx

### Tests and Quality

- pytest
- py_compile
- ruff and black in optional dev dependencies

## 6. Command Set the User Should Know

Project commands commonly used in this repository:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
make setup
make run-pipeline
make api
make streamlit
make simulate
make test
make lint
make clean
```

Direct module commands:

```bash
python scripts/run_pipeline.py
python -m src.data.make_dataset
python -m src.models.train
python simulate_stream.py --num-trips 15
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
streamlit run app_streamlit.py --server.port=8501
pytest tests/ -v
```

## 7. Configuration Model

The project is intentionally config-driven. YAML files define paths, train/val/test splits, and model metadata in `configs/config.yaml`. This keeps column names, paths, and model settings centralized rather than hardcoded across the source tree.

## 8. Production and Governance Notes

The repository emphasizes:

- read-only API contract,
- no silent simulation of missing data,
- fallback routing behavior when the model is unavailable,
- clean model-bundle serialization,
- business threshold checks for MAE and R²,
- chronological dataset splitting to avoid leakage.
