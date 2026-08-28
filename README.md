# Smart Arrival Time Estimation

Production-track ETA prediction system for a taxi dispatch context, built
at TSE Consultant INT. Prototyped end-to-end on the public NYC Taxi Trip
Duration dataset (Kaggle) while real company data access is pending;
architecture is kept structurally compatible so real data can be dropped
in later without a redesign.

See [`docs/project_charter.md`](docs/project_charter.md) for the current
problem definition, open questions, and decisions log — read that before
touching code.

## Project stages

- [x] 1. Business understanding
- [x] 2. Problem formulation *(target = trip duration, scenario = Case B)*
- [ ] 3. Dataset design
- [ ] 4. Feature engineering
- [ ] 5. EDA
- [ ] 6. Baseline modeling
- [ ] 7. Advanced modeling
- [ ] 8. Evaluation
- [ ] 9. Deployment
- [ ] 10. Monitoring

## Repo layout

```
config/             Data contracts, paths, hyperparameters (config.yaml)
data/
  raw/              Immutable original data — never edited in place
  interim/          Intermediate cleaned data
  processed/        Final, model-ready datasets
  external/         Third-party data (traffic, weather, etc.)
docs/               Project charter, decisions log, architecture notes
notebooks/          One notebook per pipeline stage, numbered in order
src/
  data/             Loading, validation, train/val/test splitting
  features/         Feature engineering (distance, time features, etc.)
  models/           Baseline + trained model code
  evaluation/        Metrics and evaluation harness
  utils/            Config loading, shared helpers
models/             Serialized trained model artifacts (gitignored)
reports/figures/    Generated plots for write-ups
tests/              Unit tests for src/
deployment/         Serving code (FastAPI stub, Dockerfile)
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
