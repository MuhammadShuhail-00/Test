import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
from datetime import timedelta

st.set_page_config(page_title="Road Roughness Detection", layout="wide")
st.title("📱 Road Surface Roughness Detection App")

# Upload section
st.sidebar.header("📤 Upload Sensor Files")
uploaded_file = st.sidebar.file_uploader("Upload merged CSV file", type=["csv"])

# Load model
@st.cache_resource
def load_model():
    return joblib.load("iri_rf_classifier_cleaned.pkl")

model = load_model()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Rename columns
    df.rename(columns={
        'x_x': 'accel_x', 'y_x': 'accel_y', 'z_x': 'accel_z',
        'x_y': 'gyro_x', 'y_y': 'gyro_y', 'z_y': 'gyro_z'
    }, inplace=True)

    # Drop missing values
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    # Feature Extraction
    window_size = 200
    segments = []
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i+window_size]
        if window['speed'].mean() >= 5:
            y_filtered = window['accel_y'].rolling(window=5, center=True).mean().bfill().ffill()
            features = {
                'start_time': window['timestamp'].iloc[0],
                'end_time': window['timestamp'].iloc[-1],
                'mean_accel_y': y_filtered.mean(),
                'std_accel_y': y_filtered.std(),
                'rms_accel_y': np.sqrt(np.mean(y_filtered**2)),
                'peak2peak_accel_y': y_filtered.max() - y_filtered.min(),
                'mean_speed': window['speed'].mean() * 3.6,
                'elevation_change': window['altitude'].iloc[-1] - window['altitude'].iloc[0],
                'gyro_y_std': window['gyro_y'].std(),
                'gyro_x_std': window['gyro_x'].std(),
                'latitude': window['latitude'].mean(),
                'longitude': window['longitude'].mean()
            }
            segments.append(features)

    features_df = pd.DataFrame(segments)

    # Predict labels
    predictions = model.predict(features_df.drop(columns=['start_time', 'end_time', 'latitude', 'longitude']))
    features_df['Prediction'] = predictions

    # UI display
    st.subheader("📋 Prediction Results")
    st.dataframe(features_df[['start_time', 'end_time', 'mean_speed', 'rms_accel_y', 'Prediction']])

    # Pie Chart
    st.subheader("📊 Road Condition Distribution")
    pie_data = features_df['Prediction'].value_counts().reset_index()
    pie_data.columns = ['Condition', 'Count']
    st.bar_chart(pie_data.set_index('Condition'))

    # Map with filter
    st.subheader("🗺️ Map View of Road Segments")
    labels = features_df['Prediction'].unique().tolist()
    selected_labels = st.sidebar.multiselect("Filter by label", labels, default=labels)
    map_df = features_df[features_df['Prediction'].isin(selected_labels)]

    if not map_df.empty:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[longitude, latitude]',
            get_fill_color="[255 * (Prediction == 'Rough'), 255 * (Prediction == 'Fair'), 255 * (Prediction == 'Smooth'), 140]",
            get_radius=10,
        )
        view_state = pdk.ViewState(
            latitude=map_df['latitude'].mean(),
            longitude=map_df['longitude'].mean(),
            zoom=15,
            pitch=0,
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
    else:
        st.warning("No segments match the selected labels.")
