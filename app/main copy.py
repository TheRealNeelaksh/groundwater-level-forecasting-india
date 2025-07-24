import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import numpy as np
import folium
from folium import plugins

st.set_page_config(page_title="Groundwater EDA Dashboard", layout="wide")
st.title("📊 Groundwater Level EDA – India")

# ----------------- LOAD & CLEAN DATA ---------------------
@st.cache_data
def load_data():
    # Adjust path if necessary based on your deployment structure
    # For local development, ensure 'dataset/groundwater-DATASET.csv' is correct relative to your script
    data_path = Path(__file__).parent.parent / "dataset" / "groundwater-DATASET.csv"
    
    # Add a check for file existence
    if not data_path.exists():
        st.error(f"Error: Dataset file not found at {data_path}")
        st.stop() # Stop the app if data file is missing

    df = pd.read_csv(data_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Normalize string casing for filters
    df['state_name'] = df['state_name'].str.strip().str.title()
    df['district_name'] = df['district_name'].str.strip().str.title()

    # Date parsing
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

# ----------------- SIDEBAR FILTERS ---------------------
st.sidebar.header("Filter the data")
states = sorted(data['state_name'].dropna().unique())
selected_state = st.sidebar.selectbox("Select State", states)

filtered_data = data[data['state_name'] == selected_state]
districts = sorted(filtered_data['district_name'].dropna().unique())
selected_district = st.sidebar.selectbox("Select District", ["All"] + districts)

if selected_district != "All":
    filtered_data = filtered_data[filtered_data['district_name'] == selected_district]

# ----------------- TABS ---------------------
viz = st.tabs(["📍 District Analysis", "🌐 State Comparison", "🌀 Seasonal Trend", "📈 Model Prediction", "🗺️ Geo Distribution"])

# ----------------- TAB 1: District Analysis ---------------------
with viz[0]:
    st.subheader(f"Average Groundwater Levels – Top Districts in {selected_state}")
    top_districts = filtered_data.groupby('district_name')['currentlevel'].mean().sort_values(ascending=False).head(20).reset_index()

    fig1 = px.bar(top_districts, x='currentlevel', y='district_name', orientation='h',
                  color='currentlevel', color_continuous_scale='viridis',
                  labels={'currentlevel': 'Avg Water Level (m)', 'district_name': 'District'},
                  title="Top 20 Districts by Average Water Level")
    fig1.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"📽️ District Water Level Change – {selected_state}")
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

# ----------------- TAB 2: State Comparison ---------------------
with viz[1]:
    st.subheader("Average Groundwater Levels by State")
    state_avg = data.groupby('state_name')['currentlevel'].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(state_avg, x='currentlevel', y='state_name', orientation='h',
                  color='currentlevel', color_continuous_scale='plasma',
                  labels={'currentlevel': 'Avg Water Level (m)', 'state_name': 'State'},
                  title="States by Average Groundwater Level")
    fig2.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

# ----------------- TAB 3: Seasonal Trend ---------------------
with viz[2]:
    st.subheader(f"Seasonal Trends in {selected_state}")
    season_avg = filtered_data.groupby('season')['currentlevel'].mean().reset_index()
    fig3 = px.bar(season_avg, x='season', y='currentlevel', color='season',
                   title="Average Groundwater Level by Season",
                   labels={'currentlevel': 'Avg Water Level (m)'})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📽️ Seasonal Line Animation (District vs Year)")
    st.info("Under Construction")

# ----------------- TAB 4: Model Prediction ---------------------
with viz[3]:
    st.subheader("📈 Actual vs Predicted Groundwater Levels")

    # --- CHANGE THIS LINE ---
    # Original: predictions_path = Path(__file__).resolve().parent.parent / "models" / "trainingNotebook" / "models" / "results" / "groundwater_predictions.csv"
    # New: Point to the main dataset
    predictions_path = Path(__file__).parent.parent / "dataset" / "groundwater-DATASET.csv" # Changed path

    if not predictions_path.exists():
        st.error(f"Prediction file not found at: {predictions_path}")
        st.stop()

    pred_df = pd.read_csv(predictions_path)
    pred_df.columns = pred_df.columns.str.strip().str.lower()
    
    # Robustly convert to string and apply string operations
    pred_df['state_name'] = pred_df['state_name'].fillna('').astype(str).str.strip().str.title()
    pred_df['district_name'] = pred_df['district_name'].fillna('').astype(str).str.strip().str.title()

    # The required_cols check below will likely cause an error or stop the app
    # because 'actual_level' and 'predicted_level' are not in groundwater-DATASET.csv
    required_cols = {'district_name', 'state_name', 'actual_level', 'predicted_level'}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        st.error(f"Missing columns required for prediction visualization: {', '.join(missing_cols)}")
        st.dataframe(pred_df.head())
        st.stop() # This will stop the app if the required columns are missing

    state_filter = pred_df[pred_df['state_name'] == selected_state]
    if selected_district != "All":
        state_filter = state_filter[state_filter['district_name'] == selected_district]

    if not state_filter.empty:
        fig4 = px.line(
            state_filter,
            x=state_filter.index,
            y=["actual_level", "predicted_level"], # These columns won't exist in groundwater-DATASET.csv
            labels={"value": "Groundwater Level (m)", "index": "Sample Index"},
            title=f"Actual vs Predicted Groundwater Levels – {selected_district}, {selected_state}"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data available for selected filters.")

# ----------------- TAB 5: Geo Distribution (MAP) ---------------------
with viz[4]:
    st.subheader("🗺️ Groundwater Level Map – India (Folium)")
    st.write("🧭 State Filter:", selected_state)
    st.write("🏙️ District Filter:", selected_district)

    with st.spinner("🌀 Loading groundwater map..."):
        try:
            # Debugging: Check initial data size
            st.write(f"📊 Initial data shape: {data.shape}")

            # Ensure latitude and longitude are numeric
            data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
            data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')

            # Drop rows with invalid coordinates or missing water level
            map_data = data.dropna(subset=['latitude', 'longitude', 'currentlevel'])
            # Debugging: Check data shape after dropping NaNs
            st.write(f"📊 Data shape after dropping NaNs (for map): {map_data.shape}")

            # Ensure state/district are string (for filtering)
            map_data['state_name'] = map_data['state_name'].astype(str).str.strip().str.title()
            map_data['district_name'] = map_data['district_name'].astype(str).str.strip().str.title()

            # Apply filters
            filtered_map_data = map_data[map_data['state_name'] == selected_state]
            if selected_district != "All":
                filtered_map_data = filtered_map_data[filtered_map_data['district_name'] == selected_district]

            st.write(f"📍 Filtered map points: {len(filtered_map_data)}")

            # Debugging: Show first few rows of filtered data if available
            if not filtered_map_data.empty:
                st.write("First 5 rows of data for mapping:")
                st.dataframe(filtered_map_data[['state_name', 'district_name', 'latitude', 'longitude', 'currentlevel']].head())

                # Initialize Folium map centered on India
                m = folium.Map(location=[22.5937, 78.9629], zoom_start=5)
                marker_cluster = plugins.MarkerCluster().add_to(m)

                for _, row in filtered_map_data.iterrows():
                    popup_html = f"""
                        <b>State:</b> {row['state_name']}<br>
                        <b>District:</b> {row['district_name']}<br>
                        <b>Water Level:</b> {row['currentlevel']} m
                    """
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color='blue', icon='tint', prefix='fa')
                    ).add_to(marker_cluster)

                # Render map in Streamlit
                st.components.v1.html(folium.Figure().add_child(m).render(), height=600)

            else:
                st.warning("⚠️ No valid groundwater data for the selected filters. Try a different state or district.")

        except Exception as e:
            st.error("❌ An error occurred while rendering the map.")
            st.exception(e)

# ----------------- FOOTER ---------------------
