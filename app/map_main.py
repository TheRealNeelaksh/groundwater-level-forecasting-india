import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster

st.title("🗺️ Groundwater Map Test (Folium)")

# 1. Load CSV
try:
    df = pd.read_csv("dataset/groundwater-DATASET.csv")
except Exception as e:
    st.error("❌ Failed to load CSV")
    st.exception(e)
    st.stop()

# 2. Ensure columns exist
required = {'latitude', 'longitude', 'state_name', 'district_name', 'currentlevel'}
if not required.issubset(df.columns):
    st.error(f"❌ Missing columns: {required - set(df.columns)}")
    st.write(df.columns.tolist())
    st.stop()

# 3. Clean & filter valid rows
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude', 'currentlevel'])
st.write(f"✅ {len(df)} valid data rows")

# 4. Show spinner, build map
with st.spinner("🔃 Generating map..."):
    m = folium.Map(location=[22.5937, 78.9629], zoom_start=5)
    mc = MarkerCluster().add_to(m)
    for _, r in df.head(200).iterrows():  # limit to 200 for test
        folium.Marker(
            location=(r.latitude, r.longitude),
            popup=f"{r.state_name} | {r.district_name} | {r.currentlevel} m"
        ).add_to(mc)

    st.components.v1.html(folium.Figure().add_child(m).render(), height=600)
