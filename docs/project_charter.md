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

## 5. Marché Cible & Stratégie de Transition (Cold-Start & Domain Adaptation)

### 5.1 Contexte Marché Tunisien vs. Dataset de Prototypage (NYC TLC)
- **Marché Cible Réel** : Tunisie (déploiement initial sur le Grand Tunis : Tunis, Ariana, Ben Arous, La Manouba, puis extension Sousse et Sfax).
- **Problématique de Données Locales** : Il n'existe aucun jeu de données public tunisien de mobilité à grande échelle (plusieurs millions de courses horodatées avec traces GPS) permettant d'entraîner et de stress-tester un pipeline IA de niveau industriel.
- **Rôle du Dataset NYC TLC Yellow Taxi** :
  - Servir de **banc d'essai mondial et reproductible** pour concevoir l'ingestion, valider la séparation stricte train/val/test sans fuite temporelle, auditer la latence d'inférence en streaming (< 20 ms), et prouver la supériorité de modèles tabulaires avancés (Random Forest, CatBoost, LightGBM) par rapport aux heuristiques de vitesse moyenne.

### 5.2 Conception Agnostique de la Géographie (Zero Lock-in NYC)
Pour éviter tout biais structurel lié à New York :
1. **Absence de Dépendance Spatiale Rigide** : Le modèle n'apprend pas d'identifiants de zones fixes new-yorkaises (`PULocationID` bruts en dur).
2. **Variables d'Entrée Relatives** :
   - Distance géodésique Haversine ($km$) et proxy de détour Manhattan ($km$).
   - Composantes temporelles cycliques universelles ($\sin/\cos$ de l'heure, jour de la semaine).
   - Indicateurs d'heures de pointe adaptables et nombre de passagers.
3. **Contrat d'API Standardisé** : Les entrées (`pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`) acceptent n'importe quel point du globe au format standard WGS84, en particulier la Tunisie ($[36.7^\circ - 37.0^\circ \text{N}, 10.0^\circ - 10.4^\circ \text{E}]$).

### 5.3 Feuille de Route de Déploiement Local (Cold-Start Loop)
La transition vers une précision optimale sur le terrain tunisien s'articule en trois phases :

| Phase | Statut | Mécanisme d'Estimation | Justification & Objectif Métier |
| :--- | :--- | :--- | :--- |
| **Phase 1 : Lancement & Fallback Calibré** | **Actuelle (Prêt)** | Modèle Champion ML + Fallback Heuristique Urbain (25 km/h en ville, 55 km/h voies rapides) | Assure une haute disponibilité (100% de réponses en < 20 ms) et élimine les prédictions aberrantes grâce aux bornes de sécurité. |
| **Phase 2 : Collecte Passive de Télémétrie** | **En cours (Intégration Backend)** | Logging passif en base analytique des trajets réels SmartTaxi (départ, arrivée, horodatages réels, durée effective) | Constitution du premier dataset haute fidélité de mobilité urbaine en Tunisie sans perturber le backend transactionnel. |
| **Phase 3 : Fine-Tuning & Réentraînement Local** | **Post-Lancement (Seuil : 1 000 courses)** | Réentraînement direct des modèles tabulaires (CatBoost / LightGBM) sur les courses locales | Le pipeline existant (`src/data/`, `src/features/`, `src/models/`) ingère le nouveau fichier Parquet tunisien et régénère l'artefact champion sans modifier une seule ligne du service API. |

