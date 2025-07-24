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
st.title("Groundwater Level EDA - India")

# Placeholder for initial loading messages
initial_loading_placeholder = st.empty()

# ----------------- LOAD & CLEAN DATA ---------------------
@st.cache_data
def load_data_and_status():
    messages = []
    try:
        messages.append(("info", "Attempting to load data..."))
        
        # Get the absolute path of the current script
        script_path = Path(__file__).resolve()
        # Navigate up two directories from the script to the project root, then to 'dataset'
        data_path = script_path.parent.parent / "dataset" / "groundwater-DATASET.csv"
        
        messages.append(("info", f"Attempting to load from absolute path: `{data_path}`"))
        messages.append(("info", f"Current working directory: `{Path.cwd()}`"))

        if not data_path.exists():
            messages.append(("error", f"🚨 Fatal Error: Dataset file not found at: `{data_path}`"))
            messages.append(("error", "Please ensure 'groundwater-DATASET.csv' is in the 'dataset' folder, which should be a sibling to your 'app' folder (where this script is located)."))
            return None, None, messages

        df = pd.read_csv(data_path)
        
        # --- NEW CHECK FOR EMPTY DATAFRAME ---
        if df.empty:
            messages.append(("error", f"❌ Error: The file '{data_path.name}' was loaded but appears to be empty or unreadable by pandas."))
            messages.append(("error", "Please check if the CSV file contains data and is not corrupted."))
            return None, None, messages
        # --- END NEW CHECK ---

        messages.append(("success", f"Dataset '{data_path.name}' loaded successfully! Initial shape: {df.shape}"))

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        messages.append(("info", "Column names normalized."))

        # Ensure latitude and longitude are numeric right after loading
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        messages.append(("info", "Latitude and Longitude columns converted to numeric."))

        # Normalize string casing for filters and create mapping
        df['state_name'] = df['state_name'].astype(str).str.strip().str.title()
        df['district_name'] = df['district_name'].astype(str).str.strip().str.title()
        messages.append(("info", "State and District names normalized."))

        # Create mapping dictionaries from the main dataset (groundwater-DATASET.csv)
        # Ensure 'state_code' and 'district_code' columns exist in groundwater-DATASET.csv
        state_code_to_name_map = {}
        district_code_to_name_map = {}

        if 'state_code' in df.columns and 'state_name' in df.columns:
            # Create a mapping from state_code (as string) to state_name
            temp_df_state_map = df[['state_code', 'state_name']].drop_duplicates().dropna()
            state_code_to_name_map = dict(zip(temp_df_state_map['state_code'].astype(str), temp_df_state_map['state_name']))
            messages.append(("info", f"Created state code to name map with {len(state_code_to_name_map)} entries."))
        else:
            messages.append(("warning", "Could not create state code to name map: 'state_code' or 'state_name' column missing in main dataset."))

        if 'district_code' in df.columns and 'district_name' in df.columns:
            # Create a mapping from district_code (as string) to district_name
            temp_df_district_map = df[['district_code', 'district_name']].drop_duplicates().dropna()
            district_code_to_name_map = dict(zip(temp_df_district_map['district_code'].astype(str), temp_df_district_map['district_name']))
            messages.append(("info", f"Created district code to name map with {len(district_code_to_name_map)} entries."))
        else:
            messages.append(("warning", "Could not create district code to name map: 'district_code' or 'district_name' column missing in main dataset."))


        # Date parsing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows where essential columns for analysis/mapping are missing
        df.dropna(subset=['date', 'currentlevel', 'latitude', 'longitude', 'state_name', 'district_name'], inplace=True)
        messages.append(("info", f"Dropped rows with missing essential data. Remaining rows: {df.shape[0]}"))

        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        messages.append(("info", "Month and Year columns extracted."))

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
        messages.append(("info", "Season column added."))
        
        messages.append(("success", "Data loading and initial cleaning complete."))
        return df, {'state_map': state_code_to_name_map, 'district_map': district_code_to_name_map}, messages
    except Exception as e:
        messages.append(("error", f"❌ An error occurred during data loading or initial cleaning: {e}"))
        return None, None, messages

# Display loading messages and load data
with initial_loading_placeholder.container():
    st.info("Initializing dashboard...")
    data, mapping_dicts, loading_messages = load_data_and_status()
    
    for msg_type, msg_text in loading_messages:
        if msg_type == "info":
            st.info(msg_text)
        elif msg_type == "success":
            st.success(msg_text)
        elif msg_type == "error":
            st.error(msg_text)
            # If there's an error, stop the app after displaying it
            st.stop()

# If data loaded successfully, set a session state flag and clear the placeholder
if data is not None and mapping_dicts is not None:
    initial_loading_placeholder.empty()
    if 'dashboard_ready_message_shown' not in st.session_state:
        st.session_state.dashboard_ready_message_shown = False
    
    if not st.session_state.dashboard_ready_message_shown:
        ready_message_placeholder = st.empty()
        ready_message_placeholder.success("Dashboard ready! Select an analysis tab from the sidebar.")
        st.session_state.dashboard_ready_message_shown = True
else:
    st.stop() # Stop if data loading failed

# ----------------- SIDEBAR FILTERS ---------------------
st.sidebar.header("Filter the data")
if not data['state_name'].dropna().empty:
    states = sorted(data['state_name'].dropna().unique())
    selected_state = st.sidebar.selectbox("Select State", states)
else:
    st.sidebar.warning("No states available in the data for selection.")
    selected_state = None 

filtered_data = data[data['state_name'] == selected_state] if selected_state else pd.DataFrame()

if not filtered_data['district_name'].dropna().empty:
    districts = sorted(filtered_data['district_name'].dropna().unique())
    selected_district = st.sidebar.selectbox("Select District", ["All"] + districts)
else:
    st.sidebar.info("No districts available for the selected state.")
    selected_district = "All"

if selected_district != "All":
    filtered_data = filtered_data[filtered_data['district_name'] == selected_district]

# ----------------- NAVIGATION (using sidebar radio) ---------------------
page_selection = st.sidebar.radio(
    "Select Analysis Tab",
    ["District Analysis", "State Comparison", "Seasonal Trend", "Model Prediction", "Geo Distribution"]
)

# ----------------- TAB 1: District Analysis ---------------------
if page_selection == "District Analysis":
    st.subheader(f"Average Groundwater Levels – Top Districts in {selected_state if selected_state else 'Selected State'}")
    if not filtered_data.empty:
        top_districts = filtered_data.groupby('district_name')['currentlevel'].mean().sort_values(ascending=False).head(20).reset_index()
        fig1 = px.bar(top_districts, x='currentlevel', y='district_name', orientation='h',
                      color='currentlevel', color_continuous_scale='viridis',
                      labels={'currentlevel': 'Avg Water Level (m)', 'district_name': 'District'},
                      title="Top 20 Districts by Average Water Level")
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data available for district analysis with current filters.")

    st.subheader(f"District Water Level Change – {selected_state if selected_state else 'Selected State'}")
    if 'year' in filtered_data.columns and not filtered_data.empty:
        top_districts_yearly = filtered_data.groupby(['year', 'district_name'])['currentlevel'].mean().reset_index()
        if not top_districts_yearly.empty and len(top_districts_yearly['district_name'].unique()) > 0:
            top10_districts = top_districts_yearly['district_name'].value_counts().head(10).index.tolist()
            top_districts_yearly = top_districts_yearly[top_districts_yearly['district_name'].isin(top10_districts)]

            if not top_districts_yearly.empty:
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
            else:
                st.info("Not enough data to create an animated district water level chart for the selected filters.")
        else:
            st.info("Not enough district data to create an animated district water level chart.")
    else:
        st.info("No data available for animated district water level chart with current filters.")

# ----------------- TAB 2: State Comparison ---------------------
if page_selection == "State Comparison":
    st.subheader("Average Groundwater Levels by State")
    if not data.empty:
        state_avg = data.groupby('state_name')['currentlevel'].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(state_avg, x='currentlevel', y='state_name', orientation='h',
                      color='currentlevel', color_continuous_scale='plasma',
                      labels={'currentlevel': 'Avg Water Level (m)', 'state_name': 'State'},
                      title="States by Average Groundwater Level")
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data available for state comparison.")

# ----------------- TAB 3: Seasonal Trend ---------------------
if page_selection == "Seasonal Trend":
    st.subheader(f"Seasonal Trends in {selected_state if selected_state else 'Selected State'}")
    if not filtered_data.empty:
        season_avg = filtered_data.groupby('season')['currentlevel'].mean().reset_index()
        fig3 = px.bar(season_avg, x='season', y='currentlevel', color='season',
                       title="Average Groundwater Level by Season",
                       labels={'currentlevel': 'Avg Water Level (m)'})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data available for seasonal trend analysis with current filters.")

    st.subheader("Seasonal Line Animation (District vs Year)")
    st.info("Under Construction")

# ----------------- TAB 4: Model Prediction ---------------------
if page_selection == "Model Prediction":
    st.subheader("Actual vs Predicted Groundwater Levels")

    predictions_path = Path(__file__).resolve().parent.parent / "models" / "trainingNotebook" / "models" / "results" / "groundwater_predictions.csv"
    
    if not predictions_path.exists():
        st.error(f"❌ Prediction file not found at: `{predictions_path}`")
        st.warning("Please ensure 'groundwater_predictions.csv' is located at "
                   "`groundwater-level-forecasting-india/models/trainingNotebook/models/results/` "
                   "relative to your project's root directory.")
        st.info("The Model Prediction tab cannot function without this file. "
                "Please upload or place the file in the correct location.")
        st.stop() 


    pred_df = pd.read_csv(predictions_path)
    pred_df.columns = pred_df.columns.str.strip().str.lower()
    
    # Apply mapping from state_code/district_code to state_name/district_name
    if 'state_code' in pred_df.columns and 'state_map' in mapping_dicts:
        pred_df['state_name_mapped'] = pred_df['state_code'].astype(str).map(mapping_dicts['state_map']).fillna(pred_df['state_name'])
        pred_df['state_name'] = pred_df['state_name_mapped']
    else:
        pred_df['state_name'] = pred_df['state_name'].fillna('').astype(str).str.strip().str.title()

    if 'district_code' in pred_df.columns and 'district_map' in mapping_dicts:
        pred_df['district_name_mapped'] = pred_df['district_code'].astype(str).map(mapping_dicts['district_map']).fillna(pred_df['district_name'])
        pred_df['district_name'] = pred_df['district_name_mapped']
    else:
        pred_df['district_name'] = pred_df['district_name'].fillna('').astype(str).str.strip().str.title()


    # Updated required columns to match the actual columns found in your prediction file
    required_cols = {'district_name', 'state_name', 'currentlevel', 'predicted_currentlevel'}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        st.error(f"Missing columns in prediction file: {', '.join(missing_cols)}")
        st.warning(f"Expected columns: {required_cols}. Found columns: {set(pred_df.columns)}")
        st.dataframe(pred_df.head())
        st.stop()

    # Filtering using the correct column names: 'state_name' and 'district_name'
    state_filter = pred_df[pred_df['state_name'] == selected_state]
    if selected_district != "All":
        state_filter = state_filter[state_filter['district_name'] == selected_district]

    # --- Start of Plotting Logic for Model Prediction ---
    if not state_filter.empty:
        # Ensure relevant columns are numeric and handle potential NaNs
        state_filter['currentlevel'] = pd.to_numeric(state_filter['currentlevel'], errors='coerce')
        state_filter['predicted_currentlevel'] = pd.to_numeric(state_filter['predicted_currentlevel'], errors='coerce')
        
        # Drop rows where plotting values are NaN after conversion
        state_filter.dropna(subset=['currentlevel', 'predicted_currentlevel'], inplace=True)

        if state_filter.empty:
            st.info("No valid numeric data points remain for plotting after cleaning in the prediction dataset.")
            # Do not return here, let the outer 'else' handle the "No data available" message.
        else:
            # Create a simple numerical index for plotting to avoid issues with original index
            state_filter = state_filter.reset_index(drop=True)
            state_filter['plot_index'] = state_filter.index # Use a simple integer index for x-axis

            fig4 = px.line(
                state_filter,
                x="plot_index", # Use the new simple index column for x-axis
                y=["currentlevel", "predicted_currentlevel"], 
                labels={
                    "value": "Groundwater Level (m)", 
                    "plot_index": "Data Point Index", # Label for the new x-axis
                    "currentlevel": "Actual Level", # Custom label for legend/hover
                    "predicted_currentlevel": "Predicted Level" # Custom label for legend/hover
                },
                title=f"Actual vs Predicted Groundwater Levels – {selected_district}, {selected_state}",
                markers=True, # Add markers for data points
                template="plotly_white", # Use a clean white background template
                hover_data={ # Customize hover information
                    "currentlevel": ":.2f",
                    "predicted_currentlevel": ":.2f",
                    "state_name": True,
                    "district_name": True,
                    "plot_index": False # Hide the new index from hover if not needed
                }
            )
            fig4.update_layout(hovermode="x unified") # Show unified hover for all traces at an x-position
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data available for selected filters in the prediction dataset.")

# ----------------- TAB 5: Geo Distribution (MAP) ---------------------
if page_selection == "Geo Distribution":
    st.subheader("Groundwater Level Map - India (Plotly Express)")

    map_display_data = filtered_data 

    if not map_display_data.empty:
        with st.spinner("🌀 Generating map..."):
            try:
                map_title = f"Groundwater Levels in {selected_state}"
                if selected_district != "All":
                    map_title = f"Groundwater Levels in {selected_district}, {selected_state}"

                fig_map = px.scatter_map(
                    map_display_data,
                    lat="latitude",
                    lon="longitude",
                    color="currentlevel", 
                    hover_name="district_name",
                    hover_data={"state_name": True, "currentlevel": ":.2f"}, 
                    color_continuous_scale=px.colors.sequential.Viridis, 
                    title=map_title, 
                    height=600,
                )
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                
                fig_map.update_geos(
                    fitbounds="locations", 
                    visible=False, 
                ) 

                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as map_err:
                st.error(f"❌ An error occurred during Plotly map generation: {map_err}")
                st.exception(map_err)
    else:
        st.warning("⚠️ No data available for the selected filters to render the map. Try selecting a different state or district.")
    
# ----------------- FOOTER ---------------------