# Project Charter — Smart Arrival Time Estimation

This file is the single source of truth for decisions made during Phase 1
(Problem Definition). Update it as decisions are confirmed or revised —
do not let scope decisions live only in chat history or notebooks.

## Status: Phase 1 (Problem Definition) — in progress

## 1. Business Context
- **Company**: TSE Consultant INT
- **Domain**: Taxi dispatch / passenger ETA
- **Objective**: Build a production-quality ETA prediction system that
  generalizes from a public prototyping dataset to real company data once
  it becomes available.

## 2. Problem Formulation
- **Primary target**: Trip duration (pickup → dropoff), in minutes.
  Chosen for clean ground-truth computation, direct business value, and
  alignment with public benchmarks (easy to validate the pipeline).
- **Prediction scenario**: **Case B** — prediction at trip start.
  Inputs available at prediction time: pickup location, destination,
  start time.
  - Case A (prediction at ride request, before pickup is guaranteed) is
    explicitly a *different* ML problem — different feature availability,
    different error tolerance, different downstream use. Not in scope
    until Case B is validated end-to-end.
- **Working problem statement**: *Given pickup location, destination, and
  start time, predict trip duration in minutes to support passenger ETAs
  and dispatcher planning.*

## 3. Open Questions (blocking Phase 3+ decisions on real data)
Track answers here as soon as they come back from the TSE supervisor.

| Question | Why it matters | Status |
|---|---|---|
| Pickup ETA vs. trip ETA — which is the actual priority? | Determines whether Case A work is ever in scope | Open |
| Are raw GPS traces available, or only origin-destination records? | Architectural fork: sequence models (GRU/LSTM) are only worth building if trace data exists | Open |
| Trip volume / data size | Affects model complexity budget and infra choices | Open |
| Can external traffic/weather data be integrated? | Affects feature engineering scope | Open |

## 4. Prototyping Strategy
Real company data is not yet available. To avoid blocking progress:
- **Prototyping dataset**: [NYC Taxi Trip Duration (Kaggle)](https://www.kaggle.com/c/nyc-taxi-trip-duration)
  — chosen because its schema (pickup/dropoff lat-lon, timestamps,
  passenger count, vendor id) is structurally close to what a real taxi
  company would provide for Case B.
- The pipeline (`src/`) is built against this dataset's schema but kept
  structurally swappable — see `config/config.yaml` for the data contract
  that a real dataset must satisfy to drop in without code changes.
- Sequence-based branches (GRU/LSTM on GPS traces) are **not** built
  against this dataset by default, since NYC Taxi Trip Duration does not
  include raw GPS traces — only endpoints. That branch is deferred until
  the GPS-trace question above is resolved. If it stays unresolved, the
  Porto Taxi dataset (which does include trace polylines) is the fallback
  for prototyping that branch specifically.

## 5. Modeling Progression (planned, not yet started)
1. **Baseline**: historical average speed × route distance
2. **Gradient boosting**: XGBoost / LightGBM on tabular + engineered features
3. **Hybrid deep learning**: GRU/LSTM (sequence) + MLP (static features),
   in the spirit of DeepTTE / WDR-style architectures, scoped down for
   internship constraints. Transformer variant as a recurrence-free
   alternative if time allows.

## 6. Explicit Non-Goals (for now)
- Case A (request-time prediction) is not being designed in parallel.
- No algorithm or code decisions are being finalized until data
  availability (Section 3) is confirmed.
