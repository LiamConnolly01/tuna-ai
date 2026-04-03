import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ============================================================
# Tuna AI MVP
# ------------------------------------------------------------
# A starter prototype for a tuna-detection decision engine.
# This app does NOT use real vessel data yet.
# It uses synthetic data shaped around realistic signals:
# - sea surface temperature
# - chlorophyll
# - current speed
# - time of day
# - bird activity
# - surface disturbance
# - sonar biomass estimate
# - fish depth
#
# Goal:
# Convert fragmented sensor signals into a Tuna Probability Score.
# ============================================================


@dataclass
class TunaSignals:
    sst_c: float
    chlorophyll_mg_m3: float
    current_speed_kts: float
    hour_local: int
    bird_count: int
    surface_disturbance: float
    sonar_biomass_score: float
    fish_depth_m: float
    fad_nearby: int


# -----------------------------
# Synthetic Data Generator
# -----------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tuna_probability_formula(row: pd.Series) -> float:
    """
    Hand-built scoring logic to generate synthetic labels.
    Later, you can replace this with real outcomes from vessel trips.
    Output is probability 0-1.
    """
    score = 0.0

    # Tuna often prefer certain SST ranges depending on species/region.
    # We use a broad warm-water preference curve.
    temp_center = 24.5
    temp_penalty = abs(row["sst_c"] - temp_center) * 0.35
    score += 2.8 - temp_penalty

    # Moderate chlorophyll can indicate productive feeding zones.
    chl = row["chlorophyll_mg_m3"]
    if 0.15 <= chl <= 0.75:
        score += 1.6
    elif 0.08 <= chl <= 1.0:
        score += 0.8
    else:
        score -= 0.5

    # Some current can be good; too much can hurt aggregation.
    cur = row["current_speed_kts"]
    if 0.5 <= cur <= 2.2:
        score += 1.0
    elif cur > 3.0:
        score -= 0.7

    # Dawn/morning often better surface signs.
    hr = row["hour_local"]
    if 5 <= hr <= 10:
        score += 1.2
    elif 11 <= hr <= 15:
        score += 0.3
    else:
        score -= 0.2

    # Birds are a strong indirect signal.
    score += min(row["bird_count"] / 12.0, 2.0)

    # Surface disturbance from drone/camera analysis.
    score += row["surface_disturbance"] * 2.2

    # Sonar biomass is a major signal.
    score += row["sonar_biomass_score"] * 3.0

    # Depth: shallower/mid-depth schools are easier to act on.
    depth = row["fish_depth_m"]
    if 10 <= depth <= 60:
        score += 1.3
    elif 60 < depth <= 120:
        score += 0.6
    else:
        score -= 0.8

    # FAD presence can increase probability of aggregation.
    if row["fad_nearby"] == 1:
        score += 1.0

    # Convert score to probability with sigmoid.
    prob = 1 / (1 + math.exp(-score + 4.5))
    return float(clamp(prob, 0.0, 1.0))


def generate_synthetic_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    data = []
    for _ in range(n):
        sst_c = np.random.normal(24.5, 2.8)
        chlorophyll = abs(np.random.normal(0.35, 0.22))
        current_speed = abs(np.random.normal(1.2, 0.9))
        hour_local = np.random.randint(0, 24)
        bird_count = max(0, int(np.random.normal(18, 14)))
        surface_disturbance = clamp(np.random.beta(2.2, 3.3), 0.0, 1.0)
        sonar_biomass_score = clamp(np.random.beta(2.0, 2.5), 0.0, 1.0)
        fish_depth_m = clamp(abs(np.random.normal(55, 35)), 1, 250)
        fad_nearby = np.random.binomial(1, 0.42)

        row = {
            "sst_c": round(float(sst_c), 2),
            "chlorophyll_mg_m3": round(float(chlorophyll), 3),
            "current_speed_kts": round(float(current_speed), 2),
            "hour_local": int(hour_local),
            "bird_count": int(bird_count),
            "surface_disturbance": round(float(surface_disturbance), 3),
            "sonar_biomass_score": round(float(sonar_biomass_score), 3),
            "fish_depth_m": round(float(fish_depth_m), 1),
            "fad_nearby": int(fad_nearby),
        }
        data.append(row)

    df = pd.DataFrame(data)
    df["tuna_prob_true"] = df.apply(tuna_probability_formula, axis=1)

    # Simulated ground-truth catchable-tuna label.
    # Some noise is added to avoid a perfectly clean synthetic dataset.
    noise = np.random.normal(0, 0.08, size=len(df))
    df["tuna_prob_noisy"] = np.clip(df["tuna_prob_true"] + noise, 0, 1)
    df["tuna_present"] = (df["tuna_prob_noisy"] >= 0.55).astype(int)

    return df


# -----------------------------
# Model Training
# -----------------------------
def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    features = [
        "sst_c",
        "chlorophyll_mg_m3",
        "current_speed_kts",
        "hour_local",
        "bird_count",
        "surface_disturbance",
        "sonar_biomass_score",
        "fish_depth_m",
        "fad_nearby",
    ]
    X = df[features]
    y = df["tuna_present"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train.values, y_test.values


# -----------------------------
# Prediction Helpers
# -----------------------------
def probability_bucket(prob: float) -> str:
    if prob >= 0.8:
        return "HIGH"
    if prob >= 0.55:
        return "MEDIUM"
    return "LOW"


def action_recommendation(prob: float, signals: TunaSignals) -> str:
    if prob >= 0.8:
        if signals.fish_depth_m <= 70:
            return "High-probability tuna zone. Prioritize this location now."
        return "High tuna probability, but fish are deeper. Monitor and prepare for movement upward."
    if prob >= 0.55:
        return "Moderate-probability zone. Validate with more drone or sonar passes before committing."
    return "Low-probability zone. Skip for now and reallocate search time elsewhere."


def make_prediction(model: RandomForestClassifier, signals: TunaSignals) -> dict:
    row = pd.DataFrame([
        {
            "sst_c": signals.sst_c,
            "chlorophyll_mg_m3": signals.chlorophyll_mg_m3,
            "current_speed_kts": signals.current_speed_kts,
            "hour_local": signals.hour_local,
            "bird_count": signals.bird_count,
            "surface_disturbance": signals.surface_disturbance,
            "sonar_biomass_score": signals.sonar_biomass_score,
            "fish_depth_m": signals.fish_depth_m,
            "fad_nearby": signals.fad_nearby,
        }
    ])

    prob = float(model.predict_proba(row)[0][1])
    bucket = probability_bucket(prob)
    recommendation = action_recommendation(prob, signals)
    return {
        "probability": prob,
        "bucket": bucket,
        "recommendation": recommendation,
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Tuna AI MVP", layout="wide")
st.title("Tuna AI MVP Dashboard")
st.caption("Prototype decision engine using synthetic tuna-search signals.")

with st.sidebar:
    st.header("Model Setup")
    sample_size = st.slider("Synthetic training rows", min_value=500, max_value=10000, value=3000, step=500)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)


@st.cache_data
def get_data(sample_size: int, seed: int) -> pd.DataFrame:
    return generate_synthetic_dataset(sample_size, seed)


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


df = get_data(sample_size, int(seed))
model, X_train, X_test, y_train, y_test = train_model(df)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training rows", f"{len(df):,}")
col2.metric("Model accuracy", f"{acc:.3f}")
col3.metric("Positive tuna zones", f"{df['tuna_present'].mean() * 100:.1f}%")
col4.metric("Avg true tuna probability", f"{df['tuna_prob_true'].mean() * 100:.1f}%")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Live Prediction Input")

    sst_c = st.slider("Sea Surface Temperature (°C)", 15.0, 32.0, 24.8, 0.1)
    chlorophyll = st.slider("Chlorophyll (mg/m³)", 0.01, 2.00, 0.32, 0.01)
    current_speed = st.slider("Current Speed (knots)", 0.0, 5.0, 1.1, 0.1)
    hour_local = st.slider("Local Hour", 0, 23, 7)
    bird_count = st.slider("Bird Count", 0, 120, 24)
    surface_disturbance = st.slider("Surface Disturbance Score", 0.0, 1.0, 0.55, 0.01)
    sonar_biomass_score = st.slider("Sonar Biomass Score", 0.0, 1.0, 0.62, 0.01)
    fish_depth_m = st.slider("Detected Fish Depth (m)", 1.0, 250.0, 42.0, 1.0)
    fad_nearby = st.selectbox("FAD Nearby", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    signals = TunaSignals(
        sst_c=sst_c,
        chlorophyll_mg_m3=chlorophyll,
        current_speed_kts=current_speed,
        hour_local=hour_local,
        bird_count=bird_count,
        surface_disturbance=surface_disturbance,
        sonar_biomass_score=sonar_biomass_score,
        fish_depth_m=fish_depth_m,
        fad_nearby=fad_nearby,
    )

    result = make_prediction(model, signals)
    st.success(f"Tuna Probability Score: {format_pct(result['probability'])}")
    st.write(f"**Zone Rating:** {result['bucket']}")
    st.write(f"**Action:** {result['recommendation']}")

with right:
    st.subheader("Model Feature Importance")
    importances = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    st.bar_chart(importances.set_index("feature"))

st.divider()

st.subheader("Synthetic Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Search Zones Simulation")
zone_count = st.slider("Number of candidate zones", 5, 25, 10)

zone_rows = []
for i in range(zone_count):
    candidate = TunaSignals(
        sst_c=round(random.uniform(18.0, 30.0), 2),
        chlorophyll_mg_m3=round(random.uniform(0.03, 1.1), 3),
        current_speed_kts=round(random.uniform(0.0, 4.0), 2),
        hour_local=random.randint(0, 23),
        bird_count=random.randint(0, 80),
        surface_disturbance=round(random.uniform(0.0, 1.0), 3),
        sonar_biomass_score=round(random.uniform(0.0, 1.0), 3),
        fish_depth_m=round(random.uniform(5.0, 180.0), 1),
        fad_nearby=random.randint(0, 1),
    )
    pred = make_prediction(model, candidate)
    zone_rows.append(
        {
            "zone_id": f"Z{i+1}",
            "lat": round(random.uniform(-8.0, 8.0), 4),
            "lon": round(random.uniform(145.0, 165.0), 4),
            "sst_c": candidate.sst_c,
            "chlorophyll_mg_m3": candidate.chlorophyll_mg_m3,
            "bird_count": candidate.bird_count,
            "surface_disturbance": candidate.surface_disturbance,
            "sonar_biomass_score": candidate.sonar_biomass_score,
            "fish_depth_m": candidate.fish_depth_m,
            "fad_nearby": candidate.fad_nearby,
            "tuna_probability": pred["probability"],
            "zone_rating": pred["bucket"],
            "recommendation": pred["recommendation"],
        }
    )

zone_df = pd.DataFrame(zone_rows).sort_values("tuna_probability", ascending=False)
st.dataframe(
    zone_df.style.format({"tuna_probability": "{:.1%}"}),
    use_container_width=True,
)

st.subheader("Top Recommended Zone")
best_zone = zone_df.iloc[0]
st.info(
    f"Prioritize {best_zone['zone_id']} | Probability: {best_zone['tuna_probability']:.1%} | "
    f"Depth: {best_zone['fish_depth_m']} m | FAD Nearby: {'Yes' if best_zone['fad_nearby'] == 1 else 'No'}"
)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)

st.markdown("""
### How to upgrade this MVP
1. Replace synthetic data with real trip logs, buoy data, or public ocean data.
2. Add drone computer vision inputs such as bird counts and surface-disturbance detection.
3. Replace the simple probability engine with a time-series model.
4. Add zone ranking by fuel cost, distance, and vessel route.
5. Add API ingestion for satellite SST, chlorophyll, and current maps.

### Run locally
```bash
pip install streamlit pandas numpy scikit-learn
streamlit run tuna_ai_mvp.py
```
""")
