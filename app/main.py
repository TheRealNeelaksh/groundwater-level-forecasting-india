import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Groundwater EDA Dashboard", layout="wide")
st.title("\U0001F4CA Groundwater Level EDA – India")

# Load Data (assuming cleaned CSV is placed in app folder)
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "dataset" / "groundwater-DATASET.csv"
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'currentlevel'], inplace=True)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'

    df['season'] = df['month'].apply(get_season)
    return df

data = load_data()

# Sidebar Filters
st.sidebar.header("Filter the data")
states = sorted(data['state_name'].dropna().str.title().unique())
selected_state = st.sidebar.selectbox("Select State", states)

filtered_data = data[data['state_name'].str.title() == selected_state]
districts = sorted(filtered_data['district_name'].dropna().str.title().unique())
selected_district = st.sidebar.selectbox("Select District", ["All"] + districts)

if selected_district != "All":
    filtered_data = filtered_data[filtered_data['district_name'].str.title() == selected_district]

# Tabs for Visualization
viz = st.tabs(["\U0001F4CD District Analysis", "\U0001F30D State Comparison", "\U0001F300 Seasonal Trend", "\U0001F4C8 Model Prediction","\U0001F5FA️ Geo Distribution"])

# --- District View
with viz[0]:
    st.subheader(f"Average Groundwater Levels – Top Districts in {selected_state}")
    top_districts = filtered_data.groupby('district_name')['currentlevel'].mean().sort_values(ascending=False).head(20).reset_index()
    fig1 = px.bar(top_districts, x='currentlevel', y='district_name', orientation='h',
                 color='currentlevel', color_continuous_scale='viridis',
                 labels={'currentlevel': 'Avg Water Level (m)', 'district_name': 'District'},
                 title="Top 20 Districts by Average Water Level")
    fig1.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"\U0001F3FD District Water Level Change – {selected_state}")

    if 'year' in filtered_data.columns:
        top_districts_yearly = filtered_data.groupby(['year', 'district_name'])['currentlevel'].mean().reset_index()
        top10_districts = top_districts_yearly['district_name'].value_counts().head(10).index.tolist()
        top_districts_yearly = top_districts_yearly[top_districts_yearly['district_name'].isin(top10_districts)]

        fig_anim = px.bar(
            top_districts_yearly,
            x="currentlevel", y="district_name",
            color="district_name",
            animation_frame="year",
            orientation="h",
            title="Animated District Groundwater Levels by Year",
            labels={"currentlevel": "Water Level (m)", "district_name": "District"}
        )
        fig_anim.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_anim, use_container_width=True)

# --- State Comparison
with viz[1]:
    st.subheader("Average Groundwater Levels by State")
    state_avg = data.groupby('state_name')['currentlevel'].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(state_avg, x='currentlevel', y='state_name', orientation='h',
                 color='currentlevel', color_continuous_scale='plasma',
                 labels={'currentlevel': 'Avg Water Level (m)', 'state_name': 'State'},
                 title="States by Average Groundwater Level")
    fig2.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

# --- Seasonal Trend
with viz[2]:
    st.subheader(f"Seasonal Trends in {selected_state}")
    season_avg = filtered_data.groupby('season')['currentlevel'].mean().reset_index()
    fig3 = px.bar(season_avg, x='season', y='currentlevel', color='season',
                  title="Average Groundwater Level by Season",
                  labels={'currentlevel': 'Avg Water Level (m)'})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("\U0001F3FD Seasonal Line Animation (District vs Year)")
    st.subheader("Under Construction")

    district_filtered = filtered_data[filtered_data['district_name'].str.title() == selected_district]

    line_anim = px.line(
        filtered_data,
        x="season", y="currentlevel",
        color="district_name",
        animation_frame="year",
        title="Seasonal Trend per District (Animated)",
        labels={"currentlevel": "Water Level (m)"},
        height=900,
        width=1100
    )

# --- Model Prediction
with viz[3]:
    st.subheader("\U0001F4C8 Actual vs Predicted Groundwater Levels")

    predictions_path = Path(__file__).parent.parent / "models" / "results" / "predictions.csv"

    if not predictions_path.exists():
        st.error(f"Prediction file not found at: {predictions_path}")
        st.write("\U0001F4C2 Current working directory:", Path.cwd())
        st.write("\U0001F50D Expected prediction path:", predictions_path.resolve())
        st.stop()

    pred_df = pd.read_csv(predictions_path)
    pred_df.columns = pred_df.columns.str.strip().str.lower()

    required_cols = {'district_name', 'state_name', 'Actual_Level', 'Predicted_Level'}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        st.error(f"❌ Missing columns in predictions.csv: {', '.join(missing_cols)}")
        st.dataframe(pred_df.head())
        st.stop()

    pred_df['district_name'] = pred_df['district_name'].str.title()
    pred_df['state_name'] = pred_df['state_name'].str.title()

    state_filter = pred_df[pred_df['state_name'] == selected_state]
    if selected_district != "All":
        state_filter = state_filter[state_filter['district_name'] == selected_district]

    if not state_filter.empty:
        fig4 = px.line(
            state_filter,
            x=state_filter.index,
            y=["Actual_Level", "Predicted_Level"],
            labels={"value": "Groundwater Level (m)", "index": "Sample Index"},
            title=f"Actual vs Predicted Groundwater Levels – {selected_district}, {selected_state}"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("ℹ️ No data available for selected filters.")

# --- Geo Distribution Map with toggle
with viz[4]:
    st.subheader("\U0001F5FA️ Groundwater Level Map (log scale)")

    show_full_map = st.checkbox("Show Full India Map", value=True)

    if show_full_map:
        map_df = data.copy()
    else:
        map_df = filtered_data.copy()

    fig_map, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        map_df['longitude'],
        map_df['latitude'],
        c=np.log1p(map_df['currentlevel']),
        cmap='viridis',
        s=30, alpha=0.7
    )

    cbar = fig_map.colorbar(scatter, ax=ax, label='log(Groundwater Level + 1)')
    ax.set_title('📍 Groundwater Level (log scale)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    fig_map.tight_layout()

    st.pyplot(fig_map)

st.markdown("---")
st.caption("Built as a project · Groundwater Level Dashboard 🇮🇳")
