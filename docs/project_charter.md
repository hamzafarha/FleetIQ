# Project Charter — Smart Arrival Time Estimation (ETA)

**Reference Document for SmartTaxi Central Backend Integration & Modeling Pipeline**

## 1. Business Context & Mandate
- **Company**: TSE Consultant INT / SmartTaxi Ecosystem
- **Domain**: Taxi dispatch, passenger ride-hailing, dynamic fleet routing.
- **Objective**: Build an end-to-end, production-ready ETA prediction system comparing tabular models (Baseline, Random Forest, XGBoost, LightGBM, CatBoost), exposed through a dedicated microservice with fallback capabilities.

---

## 2. Target Formulation & Roadmap

| Target Name | Business Scenario | Prototyping Feasibility | Production Endpoint | Fallback Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Passenger Trip Duration** *(Primary Model)* | Pickup $\rightarrow$ Dropoff duration (Case B) | **Ready**: NYC TLC 2025 Parquet provides pickup & dropoff timestamps. | `/predict/trip-duration` | Historical avg speed $\times$ distance |
| **2. Driver-to-Passenger ETA** *(Secondary Model)* | Driver assignment $\rightarrow$ Arrival at passenger | **Pending Data**: TLC data lacks driver dispatch telemetry & pre-pickup GPS. | `/predict/driver-pickup` | Classical routing heuristic (25 km/h urban speed) |

> [!IMPORTANT]
> **Data Reality Check**: NYC TLC yellow taxi records provide passenger trip timestamps and zones, allowing immediate training of the **Trip Duration** model. Driver-to-passenger ETA is explicitly decoupled as a dedicated endpoint running in robust routing fallback mode until telemetry/dispatch logs are made available.

---

## 3. SmartTaxi Backend Integration Rules
To ensure seamless integration with the central backend:

1. **Zero Direct Writes**: The ML service never writes directly to the `Ride` table or any production state database.
2. **Authoritative Timestamp Source**: All official ride state transitions and timestamps originate from the central backend.
3. **No Silent Simulation**: If external traffic or weather feeds are absent, the service never silently fabricates values; it reports `traffic_included: false` and `weather_included: false`.
4. **Standard API Contract**:
   - `eta_seconds` (integer)
   - `eta_minutes` (float for dispatch and passenger UI)
   - `model_version`
   - `quality_flag` (`HIGH_CONFIDENCE`, `ESTIMATED`, `FALLBACK_ROUTING`, `DEGRADED`)
   - `target_type`
5. **Fallback Mandate**: If the model fails, degrades, or inputs are out-of-distribution, classical routing speed heuristics return an actionable fallback ETA rather than throwing an unhandled service outage.

---

## 4. Modeling Progression & Comparison
1. **Baseline**: Heuristic average speed $\times$ route distance (segmented by global and hour-of-day).
2. **Tabular Models**:
   - Random Forest Regressor
   - XGBoost Regressor
   - LightGBM Regressor
   - CatBoost Regressor
3. **Evaluation Metrics**:
   - **MAE** (in minutes and seconds for business stakeholders)
   - **RMSE** (penalizes large unexpected estimation delays)
   - **R²** (variance explained)
   - **MAPE** (relative percentage accuracy)
4. **Business Acceptance Thresholds**:
   - Overall MAE: $\le 3.5$ minutes.
   - Rush Hour MAE Tolerance: $\le 4.5$ minutes.
   - Target $R^2$: $\ge 0.60$.
5. **Split Strategy**: Strictly time-based (e.g., 70% train / 15% val / 15% test) to prevent temporal data leakage.

---

## 5. Deployment Market & Cold-Start Strategy (Grand Tunis, Tunisia)
For the full technical breakdown, see [`docs/tunisia_adaptation_strategy.md`](tunisia_adaptation_strategy.md).

1. **Surrogate Prototype Justification**: NYC TLC dataset acts as an industrial benchmark to validate the entire MLOps workflow end-to-end.
2. **Hybrid Two-Tier Engine**: 
   - **Tier 1 (Road Routing)**: OpenStreetMap OSRM calculates real route distance and free-flow duration on Grand Tunis axes, with graceful fallback to calibrated local tortuosity ($\times 1.35$).
   - **Tier 2 (ML Congestion Correction)**: Dedicated LightGBM model adapted to Tunisian rush hours (morning 07:30–09:00, midday 12:30–14:00, evening 17:00–19:00).
3. **Passive Shadow Telemetry**: Real fleet rides are recorded via `POST /telemetry/log-completed-ride` to trigger continuous re-training once 500+ rides are logged.

