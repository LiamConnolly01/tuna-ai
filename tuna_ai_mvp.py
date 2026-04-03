import math
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# Tuna AI Live Ocean Zones MVP
# ------------------------------------------------------------
# This version uses LIVE marine/weather data from Open-Meteo
# to score candidate ocean zones around a selected center point.
#
# Live inputs used:
# - sea surface temperature
# - ocean current velocity
# - wave height
# - wind speed
# - cloud cover
#
# Optional manual overlays:
# - bird activity
# - sonar biomass
# - FAD nearby
#
# Goal:
# Rank nearby search zones on an interactive map.
# ============================================================


st.set_page_config(page_title="Tuna AI Live Ocean Map", layout="wide")
st.title("Tuna AI Live Ocean Map")
st.caption("Interactive tuna-zone ranking using live marine and weather data.")


# -----------------------------
# Helpers
# -----------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_sst(sst_c: float) -> float:
    # Broad warm-water tuna preference curve centered near 24.5 C
    penalty = abs(sst_c - 24.5) / 6.0
    return clamp(1.0 - penalty, 0.0, 1.0)


def score_current(current_kmh: float) -> float:
    # Mild-to-moderate currents are often more favorable than dead calm or extreme current
    if 2.0 <= current_kmh <= 8.0:
        return 1.0
    if 1.0 <= current_kmh < 2.0 or 8.0 < current_kmh <= 12.0:
        return 0.7
    if 0.2 <= current_kmh < 1.0 or 12.0 < current_kmh <= 16.0:
        return 0.4
    return 0.15


def score_wave_height(wave_height_m: float) -> float:
    # Lower wave heights are easier for spotting and operations
    if wave_height_m <= 1.0:
        return 1.0
    if wave_height_m <= 2.0:
        return 0.75
    if wave_height_m <= 3.0:
        return 0.45
    return 0.15


def score_wind(wind_kmh: float) -> float:
    if wind_kmh <= 15:
        return 1.0
    if wind_kmh <= 25:
        return 0.7
    if wind_kmh <= 35:
        return 0.4
    return 0.15


def score_cloud(cloud_cover_pct: float) -> float:
    # Lower cloud cover can help surface/drone visibility
    if cloud_cover_pct <= 25:
        return 1.0
    if cloud_cover_pct <= 50:
        return 0.7
    if cloud_cover_pct <= 75:
        return 0.45
    return 0.2


def tuna_probability_from_live_data(row: pd.Series, bird_count: int, sonar_score: float, fad_nearby: int) -> float:
    sst = score_sst(row["sea_surface_temperature"])
    current = score_current(row["ocean_current_velocity"])
    wave = score_wave_height(row["wave_height"])
    wind = score_wind(row["wind_speed_10m"])
    cloud = score_cloud(row["cloud_cover"])

    birds = clamp(bird_count / 40.0, 0.0, 1.0)
    sonar = clamp(sonar_score, 0.0, 1.0)
    fad = 1.0 if fad_nearby == 1 else 0.0

    score = (
        0.28 * sst
        + 0.16 * current
        + 0.12 * wave
        + 0.12 * wind
        + 0.08 * cloud
        + 0.12 * birds
        + 0.08 * sonar
        + 0.04 * fad
    )
    return clamp(score, 0.0, 1.0)


def probability_bucket(prob: float) -> str:
    if prob >= 0.75:
        return "HIGH"
    if prob >= 0.5:
        return "MEDIUM"
    return "LOW"


def recommendation_from_prob(prob: float) -> str:
    if prob >= 0.75:
        return "High-priority zone. Check immediately."
    if prob >= 0.5:
        return "Worth scouting. Validate with birds or sonar."
    return "Lower priority. Re-check later or skip for now."


def generate_zone_grid(center_lat: float, center_lon: float, zone_count: int, spacing_deg: float) -> pd.DataFrame:
    side = int(math.sqrt(zone_count))
    if side * side < zone_count:
        side += 1

    start_offset = -(side // 2)
    rows = []
    zone_id = 1
    for i in range(side):
        for j in range(side):
            if zone_id > zone_count:
                break
            lat = center_lat + (start_offset + i) * spacing_deg
            lon = center_lon + (start_offset + j) * spacing_deg
            rows.append(
                {
                    "zone_id": f"Z{zone_id}",
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                }
            )
            zone_id += 1
    return pd.DataFrame(rows)


def parse_multi_location_response(payload):
    if isinstance(payload, list):
        return payload
    return [payload]


@st.cache_data(ttl=900)
def fetch_live_zone_data(latitudes: List[float], longitudes: List[float]) -> pd.DataFrame:
    lat_param = ",".join(str(x) for x in latitudes)
    lon_param = ",".join(str(x) for x in longitudes)

    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": lat_param,
        "longitude": lon_param,
        "current": "sea_surface_temperature,ocean_current_velocity,wave_height",
        "cell_selection": "sea",
    }

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat_param,
        "longitude": lon_param,
        "current": "wind_speed_10m,cloud_cover",
        "wind_speed_unit": "kmh",
        "cell_selection": "sea",
    }

    marine_resp = requests.get(marine_url, params=marine_params, timeout=30)
    marine_resp.raise_for_status()
    marine_data = parse_multi_location_response(marine_resp.json())

    weather_resp = requests.get(weather_url, params=weather_params, timeout=30)
    weather_resp.raise_for_status()
    weather_data = parse_multi_location_response(weather_resp.json())

    rows = []
    for marine_item, weather_item in zip(marine_data, weather_data):
        current_marine = marine_item.get("current", {})
        current_weather = weather_item.get("current", {})
        rows.append(
            {
                "latitude": round(float(marine_item["latitude"]), 4),
                "longitude": round(float(marine_item["longitude"]), 4),
                "time": current_marine.get("time") or current_weather.get("time"),
                "sea_surface_temperature": float(current_marine.get("sea_surface_temperature", float("nan"))),
                "ocean_current_velocity": float(current_marine.get("ocean_current_velocity", float("nan"))),
                "wave_height": float(current_marine.get("wave_height", float("nan"))),
                "wind_speed_10m": float(current_weather.get("wind_speed_10m", float("nan"))),
                "cloud_cover": float(current_weather.get("cloud_cover", float("nan"))),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Map Setup")
  center_lat = st.number_input("Center latitude", value=20.0, format="%.4f")
center_lon = st.number_input("Center longitude", value=-155.0, format="%.4f")
    zone_count = st.select_slider("Number of zones", options=[4, 9, 16, 25], value=9)
    spacing_deg = st.slider("Zone spacing (degrees)", min_value=0.1, max_value=1.0, value=0.35, step=0.05)

    st.header("Manual Search Overlays")
    bird_count = st.slider("Bird activity score (count)", 0, 80, 20)
    sonar_score = st.slider("Sonar biomass score", 0.0, 1.0, 0.45, 0.01)
    fad_nearby = st.selectbox("FAD nearby", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.header("About the Live Inputs")
    st.write("This version uses live sea surface temperature, ocean current velocity, wave height, wind speed, and cloud cover.")


# -----------------------------
# Build Zones + Fetch Live Data
# -----------------------------
zone_grid = generate_zone_grid(center_lat, center_lon, zone_count, spacing_deg)
live_df = fetch_live_zone_data(zone_grid["latitude"].tolist(), zone_grid["longitude"].tolist())
zone_df = zone_grid.merge(live_df, on=["latitude", "longitude"], how="left")
zone_df = zone_df.fillna({
    "sea_surface_temperature": 24.0,
    "ocean_current_velocity": 2.0,
    "wave_height": 1.0,
    "wind_speed_10m": 12.0,
    "cloud_cover": 30.0,
})
zone_df["tuna_probability"] = zone_df.apply(
    lambda row: tuna_probability_from_live_data(row, bird_count, sonar_score, fad_nearby),
    axis=1,
)
zone_df["zone_rating"] = zone_df["tuna_probability"].apply(probability_bucket)
zone_df["recommendation"] = zone_df["tuna_probability"].apply(recommendation_from_prob)
zone_df["probability_pct"] = (zone_df["tuna_probability"] * 100).round(1)
zone_df["marker_size"] = 16 + (zone_df["tuna_probability"] * 26)
zone_df = zone_df.sort_values("tuna_probability", ascending=False).reset_index(drop=True)


# -----------------------------
# Top Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Zones scanned", len(zone_df))
col2.metric("Best zone", zone_df.iloc[0]["zone_id"])
col3.metric("Top tuna score", f"{zone_df.iloc[0]['probability_pct']:.1f}%")
col4.metric("Data timestamp", str(zone_df.iloc[0]["time"]))

st.divider()


# -----------------------------
# Interactive Map
# -----------------------------
st.subheader("Active Zone Map")
fig = px.scatter_map(
    zone_df,
    lat="latitude",
    lon="longitude",
    color="zone_rating",
    size="marker_size",
    size_max=30,
    hover_name="zone_id",
    hover_data={
        "probability_pct": True,
        "sea_surface_temperature": ":.2f",
        "ocean_current_velocity": ":.2f",
        "wave_height": ":.2f",
        "wind_speed_10m": ":.1f",
        "cloud_cover": ":.1f",
        "latitude": ":.4f",
        "longitude": ":.4f",
        "marker_size": False,
    },
    zoom=5,
    height=650,
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

st.divider()

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Ranked Live Zones")
    display_cols = [
        "zone_id",
        "probability_pct",
        "zone_rating",
        "sea_surface_temperature",
        "ocean_current_velocity",
        "wave_height",
        "wind_speed_10m",
        "cloud_cover",
        "latitude",
        "longitude",
        "recommendation",
    ]
    st.dataframe(zone_df[display_cols], use_container_width=True)

with right:
    st.subheader("Top Zone Breakdown")
    best = zone_df.iloc[0]
    st.success(f"Prioritize {best['zone_id']} — {best['probability_pct']:.1f}% tuna score")
    st.write(f"**Recommendation:** {best['recommendation']}")
    st.write(f"**Sea Surface Temp:** {best['sea_surface_temperature']:.2f} °C")
    st.write(f"**Ocean Current:** {best['ocean_current_velocity']:.2f} km/h")
    st.write(f"**Wave Height:** {best['wave_height']:.2f} m")
    st.write(f"**Wind Speed:** {best['wind_speed_10m']:.1f} km/h")
    st.write(f"**Cloud Cover:** {best['cloud_cover']:.1f}%")
    st.write(f"**Coordinates:** {best['latitude']:.4f}, {best['longitude']:.4f}")

st.divider()

st.subheader("How this live version works")
st.markdown(
    """
1. You choose a center point and number of candidate zones.
2. The app builds a grid of ocean search zones around that point.
3. It pulls live marine and weather conditions for each zone.
4. It scores zones with a tuna-likelihood model.
5. It ranks and maps the best zones.

### Files needed for deployment
Create a `requirements.txt` file with:
```text
streamlit
pandas
plotly
requests
```

### Run locally
```bash
pip install streamlit pandas plotly requests
streamlit run tuna_ai_mvp.py
```
"""
)
