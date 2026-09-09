# Smart Arrival Time Estimation

Production-track ETA prediction system for a taxi dispatch context, built
at TSE Consultant INT. Prototyped end-to-end on the public NYC Taxi Trip
Duration dataset (Kaggle) while real company data access is pending;
architecture is kept structurally compatible so real data can be dropped
in later without a redesign.

See [`docs/project_charter.md`](docs/project_charter.md) for the current
problem definition, open questions, and decisions log — read that before
touching code.

## Project stages & notebooks

- [x] `docs/project_charter.md`: Business understanding & problem formulation
- [x] `notebooks/01_dataset_audit.ipynb`: NYC TLC raw data audit & feasibility check
- [x] `notebooks/02_data_cleaning.ipynb`: Minimal cleaning rules & interim validation
- [ ] `notebooks/03_dataset_design.ipynb`: Time-based train/val/test split
- [ ] `notebooks/04_feature_engineering.ipynb`: Temporal, distance & rush-hour features
- [ ] `notebooks/05_eda.ipynb`: Exploratory data analysis & feature distributions
- [ ] `notebooks/06_baseline_modeling.ipynb`: Average speed heuristic baseline
- [ ] `notebooks/07_advanced_modeling.ipynb`: Tabular models (RF, XGBoost, LightGBM, CatBoost)
- [ ] `notebooks/08_evaluation.ipynb`: Final test evaluation, MAE/RMSE/R² & business gates
- [x] `app.py`: Production FastAPI inference service with fallback routing
- [x] `simulate_stream.py`: Real-time ride stream simulation & latency load test

## Repo layout

```
configs/            Data contracts, paths, hyperparameters (config.yaml)
data/
  raw/              Immutable original data — never edited in place
  interim/          Intermediate cleaned data
  processed/        Final, model-ready datasets
  external/         Third-party data (traffic, weather, etc.)
docs/               Project charter, decisions log, architecture notes
notebooks/          Sequential notebooks (01 to 08)
src/
  data/             Loading, validation, train/val/test splitting
  features/         Feature engineering (distance, time features, etc.)
  models/           Baseline + trained model code
  evaluation/       Metrics and evaluation harness
  utils/            Config loading, shared helpers
models/             Serialized trained model artifacts
reports/figures/    Generated plots for write-ups
tests/              Unit tests for src/ and app.py
app.py              Production FastAPI serving endpoint
simulate_stream.py  Streaming simulator
Dockerfile          Inference API container definition
Dockerfile.streamlit Streamlit dashboard container definition
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

This repo does not ship data. Download the NYC Taxi Trip Duration dataset
from Kaggle and place the raw CSVs in `data/raw/`:

```bash
kaggle competitions download -c nyc-taxi-trip-duration -p data/raw/
```

## Config-driven, not hardcoded

`config/config.yaml` defines the data contract (required columns, target
column, coordinate bounds) so that swapping in real company data later is
a config change, not a rewrite. Do not hardcode column names or paths
inside `src/` — read them from config.

## Workflow convention

Each numbered notebook in `notebooks/` corresponds to one pipeline stage
and imports its logic from `src/` rather than reimplementing it inline —
notebooks are for exploration and narrative, `src/` is the source of
truth for anything reused downstream (including eventual deployment).
