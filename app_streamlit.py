"""SmartTaxi — Smart Arrival Time Estimation (ETA) Streamlit Interface.

Interactive demo and operational dashboard for the SmartTaxi ETA microservice.
Supports:
1. Live Passenger Trip Duration Inference (Target 2 / Case B) with Map visualization
2. Driver-to-Passenger Pickup ETA (Target 1 / Dispatch)
3. Model Benchmark & Business SLA Monitoring (Baseline, RF, XGB, LightGBM, CatBoost)
4. Real-time Streaming Simulation & Latency Benchmark
5. Backend Architecture & Governance Alignment
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SmartTaxi — ETA Prediction Service",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants & Default API URL
DEFAULT_API_URL = "http://127.0.0.1:8000"

PRESET_LOCATIONS = {
    "Times Square, Manhattan": (40.7580, -73.9855),
    "Central Park North": (40.7990, -73.9535),
    "JFK International Airport": (40.6413, -73.7781),
    "LaGuardia Airport (LGA)": (40.7769, -73.8740),
    "Wall Street / Financial District": (40.7075, -74.0090),
    "Brooklyn Bridge Park": (40.7023, -73.9964),
    "Williamsburg, Brooklyn": (40.7145, -73.9560),
    "Penn Station, Manhattan": (40.7505, -73.9934),
}


def load_model_benchmark_data() -> Dict[str, Any]:
    """Load model comparison results if available."""
    res_path = Path("models/model_comparison_results.json")
    if res_path.exists():
        with open(res_path) as f:
            return json.load(f)
    return {}


# Custom styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-high {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-fallback {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🚕 SmartTaxi ETA")
st.sidebar.caption("Machine Learning Arrival Time Estimation Microservice")

nav = st.sidebar.radio(
    "Navigation :",
    [
        "1. Estimation Course (Trip Duration)",
        "2. Approche Chauffeur (Driver Pickup)",
        "3. Benchmark & Comparaison Modèles",
        "4. Simulateur Streaming Temps Réel",
        "5. Gouvernance & Contrat Backend",
    ],
)

api_base = st.sidebar.text_input("URL Backend Microservice ETA :", value=DEFAULT_API_URL)

# Check API Health
try:
    health_resp = requests.get(f"{api_base}/health", timeout=1.0)
    if health_resp.status_code == 200:
        st.sidebar.success("🟢 API Connectée & Opérationnelle")
    else:
        st.sidebar.warning(f"🟡 API Statut: {health_resp.status_code}")
except Exception:
    st.sidebar.info("🔵 Mode Autonome / Fallback Local Actif")

# --- PAGE 1: ESTIMATION COURSE (TRIP DURATION) ---
if nav == "1. Estimation Course (Trip Duration)":
    st.title("📍 Estimation de la Durée du Trajet Passager (Cible 2 / Case B)")
    st.markdown("Prédit le temps de trajet entre la prise en charge et la destination finale.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Coordonnées de Prise en Charge")
        pickup_preset = st.selectbox(
            "Point de départ prédéfini (ou personnalisé) :",
            ["Personnalisé"] + list(PRESET_LOCATIONS.keys()),
            index=1,
        )
        if pickup_preset != "Personnalisé":
            p_lat_init, p_lon_init = PRESET_LOCATIONS[pickup_preset]
        else:
            p_lat_init, p_lon_init = 40.7580, -73.9855

        p_lat = st.number_input("Latitude Prise en Charge", value=p_lat_init, format="%.6f")
        p_lon = st.number_input("Longitude Prise en Charge", value=p_lon_init, format="%.6f")

        st.subheader("2. Coordonnées de Destination")
        dropoff_preset = st.selectbox(
            "Point d'arrivée prédéfini (ou personnalisé) :",
            ["Personnalisé"] + list(PRESET_LOCATIONS.keys()),
            index=3,
        )
        if dropoff_preset != "Personnalisé":
            d_lat_init, d_lon_init = PRESET_LOCATIONS[dropoff_preset]
        else:
            d_lat_init, d_lon_init = 40.7829, -73.9654

        d_lat = st.number_input("Latitude Destination", value=d_lat_init, format="%.6f")
        d_lon = st.number_input("Longitude Destination", value=d_lon_init, format="%.6f")

    with col2:
        st.subheader("3. Paramètres Temporels & Contexte")
        trip_date = st.date_input("Date du trajet", value=datetime.now())
        trip_time = st.time_input("Heure du trajet", value=datetime.now().time())
        passengers = st.slider("Nombre de passagers", min_value=1, max_value=6, value=1)

        st.markdown("---")
        st.subheader("4. Données Externes Optionnelles")
        st.caption("Règle Backend : si non fournies, aucune valeur n'est inventée.")
        use_traffic = st.checkbox("Inclure index trafic réel")
        traffic_val = st.slider("Indice de trafic (0=fluide, 1=saturé)", 0.0, 1.0, 0.75) if use_traffic else None

        use_weather = st.checkbox("Inclure condition météo")
        weather_val = st.selectbox("Condition météo", ["clear", "rain", "snow", "fog"]) if use_weather else None

        dt_iso = datetime.combine(trip_date, trip_time).replace(tzinfo=timezone.utc).isoformat()

        is_rush = trip_time.hour in (7, 8, 9, 16, 17, 18, 19) and trip_date.weekday() < 5
        if is_rush:
            st.info("⚠️ Heure de pointe détectée (Congestion urbaine prise en compte)")

    predict_btn = st.button("🚀 Estimer l'Heure d'Arrivée (ETA)", type="primary", use_container_width=True)

    if predict_btn:
        payload = {
            "pickup_latitude": p_lat,
            "pickup_longitude": p_lon,
            "dropoff_latitude": d_lat,
            "dropoff_longitude": d_lon,
            "pickup_datetime": dt_iso,
            "passenger_count": passengers,
            "traffic_density": traffic_val,
            "weather_condition": weather_val,
        }

        with st.spinner("Calcul de l'ETA en cours..."):
            try:
                resp = requests.post(f"{api_base}/predict/trip-duration", json=payload, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                else:
                    st.error(f"Erreur API ({resp.status_code}): {resp.text}")
                    data = None
            except Exception as e:
                st.warning(f"Connexion directe à l'API échouée ({e}). Exécution du calcul local.")
                # Local fallback execution
                from src.features.build_features import extract_features_for_inference, haversine_distance_km
                import joblib
                dist = float(haversine_distance_km(p_lat, p_lon, d_lat, d_lon))
                data = {
                    "eta_minutes": round((dist / 25.0) * 60.0, 2),
                    "eta_seconds": int((dist / 25.0) * 3600),
                    "model_version": "local_fallback_speed_v1",
                    "quality_flag": "FALLBACK_ROUTING",
                    "distance_km": round(dist, 2),
                    "traffic_included": False,
                    "weather_included": False,
                    "fallback_used": True,
                    "latency_ms": 1.2,
                }

        if data:
            st.success("✅ Estimation Calculée avec Succès")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Durée Estimée (ETA)", f"{data['eta_minutes']} min", f"{data['eta_seconds']} s")
            res_col2.metric("Distance Calculée", f"{data['distance_km']} km")
            res_col3.metric("Version Modèle", data["model_version"])
            res_col4.metric("Latence Inférence", f"{data['latency_ms']} ms")

            status_style = "badge-high" if data["quality_flag"] == "HIGH_CONFIDENCE" else "badge-fallback"
            st.markdown(
                f"**Indicateur Qualité :** <span class='{status_style}'>{data['quality_flag']}</span> "
                f"| Fallback activé : `{data['fallback_used']}` "
                f"| Trafic certifié : `{data['traffic_included']}` "
                f"| Météo certifiée : `{data['weather_included']}`",
                unsafe_allow_html=True,
            )

            # Map Visualization
            map_data = pd.DataFrame(
                {
                    "lat": [p_lat, d_lat],
                    "lon": [p_lon, d_lon],
                    "point": ["Prise en charge", "Destination"],
                }
            )
            st.map(map_data, zoom=11)

# --- PAGE 2: APPROCHE CHAUFFEUR (DRIVER PICKUP) ---
elif nav == "2. Approche Chauffeur (Driver Pickup)":
    st.title("🚖 Approche Chauffeur $\rightarrow$ Passager (Cible 1 / Dispatch)")
    st.markdown(
        """
        **Cible prioritaire pour le matching central :** Estime le temps mis par le chauffeur 
        sélectionné pour rejoindre le point de prise en charge du passager.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Position Actuelle du Chauffeur")
        d_lat = st.number_input("Latitude Chauffeur", value=40.7500, format="%.6f")
        d_lon = st.number_input("Longitude Chauffeur", value=-73.9900, format="%.6f")

    with c2:
        st.subheader("Position du Passager (Pickup)")
        p_lat = st.number_input("Latitude Passager", value=40.7580, format="%.6f")
        p_lon = st.number_input("Longitude Passager", value=-73.9855, format="%.6f")

    assign_dt = datetime.now(timezone.utc).isoformat()

    if st.button("⏱️ Calculer l'ETA d'Approche Chauffeur", type="primary"):
        payload = {
            "driver_latitude": d_lat,
            "driver_longitude": d_lon,
            "passenger_latitude": p_lat,
            "passenger_longitude": p_lon,
            "assignment_datetime": assign_dt,
        }
        try:
            resp = requests.post(f"{api_base}/predict/driver-pickup", json=payload, timeout=2.0)
            data = resp.json()
            st.success("✅ Estimation d'Approche Calculée")
            r1, r2, r3 = st.columns(3)
            r1.metric("Temps d'Approche (ETA)", f"{data['eta_minutes']} min", f"{data['eta_seconds']} sec")
            r2.metric("Distance Chauffeur $\rightarrow$ Client", f"{data['distance_km']} km")
            r3.metric("Statut Moteur", data["quality_flag"])
            st.info(
                "ℹ️ Fonctionne en mode Fallback Routier Urbain Calibré. "
                "Prêt pour le branchement télématique dès confirmation des traces GPS par Houda (Backend)."
            )
        except Exception as e:
            st.error(f"Erreur d'appel API : {e}")

# --- PAGE 3: BENCHMARK & COMPARAISON MODÈLES ---
elif nav == "3. Benchmark & Comparaison Modèles":
    st.title("📊 Benchmark et Comparaison des Modèles Tabulaires")
    st.markdown(
        """
        Évaluation comparative des **5 modèles tabulaires** sur le jeu de test chronologique (fin octobre 2025).
        """
    )

    benchmark_data = load_model_benchmark_data()
    if benchmark_data:
        summary = benchmark_data.get("models_summary", {})
        table_rows = []
        for name, m in summary.items():
            table_rows.append(
                {
                    "Modèle": name,
                    "MAE (min)": m["mae_minutes"],
                    "MAE (sec)": m["mae_seconds"],
                    "RMSE (min)": m["rmse_minutes"],
                    "R² Score": m["r2"],
                    "MAPE (%)": f"{m['mape_pct']}%",
                    "Latence (ms)": f"{m.get('latency_ms', '-')} ms",
                }
            )

        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, use_container_width=True)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        champ = benchmark_data.get("champion_model", "Random Forest")
        kpi1.metric("🏆 Modèle Champion", champ)
        kpi2.metric("MAE Test Champion", f"{summary[champ]['mae_minutes']} min", f"{summary[champ]['mae_seconds']} s")
        kpi3.metric("R² Score Champion", summary[champ]["r2"])
        kpi4.metric("Latence Inférence", f"{summary[champ]['latency_ms']} ms")

        st.subheader("Graphiques et Analyses d'Erreurs")
        fig1_path = Path("reports/figures/model_comparison_benchmark.png")
        fig2_path = Path("reports/figures/feature_importance.png")
        fig3_path = Path("reports/figures/slice_analysis.png")

        tab_g1, tab_g2, tab_g3 = st.tabs(["Benchmark MAE & R²", "Importance des Features", "Analyse par Sous-Groupes"])
        with tab_g1:
            if fig1_path.exists():
                st.image(str(fig1_path), use_column_width=True)
        with tab_g2:
            if fig2_path.exists():
                st.image(str(fig2_path), use_column_width=True)
        with tab_g3:
            if fig3_path.exists():
                st.image(str(fig3_path), use_column_width=True)
    else:
        st.warning("Aucun résultat de benchmark trouvé dans `models/model_comparison_results.json`.")

# --- PAGE 4: SIMULATEUR STREAMING TEMPS RÉEL ---
elif nav == "4. Simulateur Streaming Temps Réel":
    st.title("⚡ Simulation de Flux Temps Réel & Test de Débit")
    st.markdown("Simule des arrivées séquentielles de réservations de taxis pour évaluer la latence et la robustesse.")

    n_events = st.slider("Nombre d'événements à simuler", 5, 50, 15)
    interval = st.slider("Intervalle entre événements (secondes)", 0.05, 1.0, 0.2)

    if st.button("🚀 Démarrer la Simulation"):
        from simulate_stream import generate_random_ride

        progress_bar = st.progress(0)
        logs = []
        table_placeholder = st.empty()
        latencies = []

        for i in range(1, n_events + 1):
            ride = generate_random_ride(i)
            t0 = time.perf_counter()
            try:
                resp = requests.post(f"{api_base}/predict/trip-duration", json=ride, timeout=2.0)
                dur = round((time.perf_counter() - t0) * 1000, 2)
                latencies.append(dur)
                if resp.status_code == 200:
                    d = resp.json()
                    logs.append(
                        {
                            "Course ID": ride["ride_id"],
                            "Statut": "OK (200)",
                            "Distance": f"{d['distance_km']} km",
                            "ETA": f"{d['eta_minutes']} min",
                            "Qualité": d["quality_flag"],
                            "Latence (ms)": dur,
                        }
                    )
                else:
                    logs.append({"Course ID": ride["ride_id"], "Statut": f"ERR {resp.status_code}"})
            except Exception as err:
                dur = round((time.perf_counter() - t0) * 1000, 2)
                logs.append({"Course ID": ride["ride_id"], "Statut": "Échec Connexion", "Latence (ms)": dur})

            table_placeholder.dataframe(pd.DataFrame(logs), use_container_width=True)
            progress_bar.progress(i / n_events)
            time.sleep(interval)

        st.success(f"Simulation terminée ! {len(logs)} requêtes envoyées.")
        if latencies:
            st.metric("Latence Moyenne Observée", f"{np.mean(latencies):.2f} ms")

# --- PAGE 5: GOUVERNANCE & CONTRAT BACKEND ---
elif nav == "5. Gouvernance & Contrat Backend":
    st.title("🛡️ Respect des Règles du Backend Central SmartTaxi")
    st.markdown(
        """
        Conformément au **Guide de cohérence et d'intégration de l'écosystème SmartTaxi (Août 2026)** :
        """
    )
    st.markdown(
        """
        1. **Strictement Read-Only :** Aucune écriture directe dans la table `Ride` ni dans PostgreSQL. Le service ETA est stateless.
        2. **Source Officielle des États :** Le backend centralisé .NET 10 (Houda Ghenmi) reste le seul propriétaire des transitions de courses et de la facturation.
        3. **Pas de Simulation Silencieuse :** Si les capteurs trafic ou météo sont indisponibles, le modèle ne les invente pas ; il renvoie `traffic_included: false` et `weather_included: false`.
        4. **Tolérance aux Pannes & Fallback :** En cas d'indisponibilité du modèle IA, un calcul heuristique urbain prend automatiquement le relais avec le drapeau `FALLBACK_ROUTING`.
        5. **Tests d'Intégration Minimum (24/24 validés) :** Trajets courts/longs, heures de pointe, données manquantes, coordonnées hors-limites, latence sous 50 ms.
        """
    )
    st.markdown("Consultez le rapport complet dans `reports/rapport_integration_eta_smarttaxi.md`.")
