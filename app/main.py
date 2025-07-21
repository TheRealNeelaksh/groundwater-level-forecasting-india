import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Groundwater Level Dashboard", layout="wide")
st.title("🧠 Groundwater Level Forecasting Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/groundwater_clean.csv")
    df.columns = df.columns.str.strip().str.lower()
    df['district_name'] = df['district_name'].str.title()
    df['state_name'] = df['state_name'].str.title()
    return df

data = load_data()

# Load Predictions
@st.cache_data
def load_predictions():
    predictions_path = Path(__file__).parent.parent / "models" / "results" / "predictions.csv"
    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
        df.columns = df.columns.str.strip().str.lower()
        df['district_name'] = df['district_name'].str.title()
        df['state_name'] = df['state_name'].str.title()
        return df
    else:
        return pd.DataFrame()

pred_df = load_predictions()

# Sidebar Filters
st.sidebar.header("Filter the data")
common_states = sorted(set(data['state_name']) | set(pred_df['state_name']))
selected_state = st.sidebar.selectbox("Select State", common_states)

districts_data = data[data['state_name'] == selected_state]
districts_pred = pred_df[pred_df['state_name'] == selected_state]
common_districts = sorted(set(districts_data['district_name']) | set(districts_pred['district_name']))
selected_district = st.sidebar.selectbox("Select District", ["All"] + common_districts)

# Filter Data
filtered_data = data[data['state_name'] == selected_state]
filtered_pred = pred_df[pred_df['state_name'] == selected_state]

if selected_district != "All":
    filtered_data = filtered_data[filtered_data['district_name'] == selected_district]
    filtered_pred = filtered_pred[filtered_pred['district_name'] == selected_district]

# Tabs
viz = st.tabs(["Data Overview", "State-Level Map", "District-Level Trends", "Model Prediction"])

# Tab 1: Data Overview
with viz[0]:
    st.subheader("📊 Raw Groundwater Data Overview")
    st.dataframe(filtered_data, use_container_width=True)

# Tab 2: State-Level Map
with viz[1]:
    st.subheader("🗺️ Groundwater Levels by State")
    map_df = filtered_data.groupby("district_name")["groundwater_level_m"].mean().reset_index()
    map_df["District"] = map_df["district_name"]
    fig = px.choropleth(map_df, 
                        locations="District",
                        locationmode="geojson-id",  # You'll need proper geojson
                        color="groundwater_level_m",
                        color_continuous_scale="Viridis",
                        title=f"Average Groundwater Level – {selected_state}")
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: District-Level Trends
with viz[2]:
    st.subheader("📈 Groundwater Trends Over Time")
    if selected_district != "All":
        trend_data = filtered_data.copy()
        fig2 = px.line(trend_data, 
                       x="date",
                       y="groundwater_level_m",
                       title=f"Trend – {selected_district}, {selected_state}",
                       labels={"groundwater_level_m": "Groundwater Level (m)"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Please select a district to view trends.")

# Tab 4: Model Prediction
with viz[3]:
    st.subheader("🤖 Actual vs Predicted Groundwater Levels")
    if pred_df.empty:
        st.error("Prediction file not found or could not be loaded.")
        st.stop()

    if not filtered_pred.empty:
        fig4 = px.line(
            filtered_pred,
            x=filtered_pred.index,
            y=["actual_level", "predicted_level"],
            labels={"value": "Groundwater Level (m)", "index": "Sample Index"},
            title=f"Actual vs Predicted – {selected_district}, {selected_state}" if selected_district != "All" else f"{selected_state}"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("ℹ️ No prediction data for selected filters.")
