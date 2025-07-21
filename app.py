import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import joblib
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import timedelta

# --- UI ---
st.set_page_config(page_title="Road Roughness Detection", layout="wide")
st.title("🚗 Road Surface Roughness Detection App")

st.markdown("Upload a **.zip** file containing Accelerometer.csv, Gyroscope.csv, and Location.csv.")

# --- Upload zip file ---
zip_file = st.file_uploader("Upload sensor ZIP file", type="zip")

if zip_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find sensor files
        def find_file(keywords):
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    fname = file.lower()
                    if all(k in fname for k in keywords):
                        return os.path.join(root, file)
            return None

        accel_path = find_file(["accelerometer"])
        gyro_path = find_file(["gyroscope"])
        gps_path = find_file(["location"])

        if not accel_path or not gyro_path or not gps_path:
            st.error("One or more required files not found in the ZIP. Make sure it includes Accelerometer.csv, Gyroscope.csv, and Location.csv.")
        else:
            # --- Merge ---
            accel = pd.read_csv(accel_path)
            gyro = pd.read_csv(gyro_path)
            gps = pd.read_csv(gps_path)

            for df in [accel, gyro, gps]:
                df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                if df["timestamp"].max() > 1e15:
                    df["timestamp"] = df["timestamp"] / 1_000_000
                df.sort_values("timestamp", inplace=True)
                df.drop(columns=[col for col in df.columns if "elapsed" in col.lower()], inplace=True)

            merged = pd.merge_asof(accel, gyro, on="timestamp", direction="nearest", tolerance=50)
            merged = pd.merge_asof(merged, gps, on="timestamp", direction="nearest", tolerance=200)
            merged.dropna(inplace=True)
            merged.insert(0, "readable_time", pd.to_datetime(merged["timestamp"], unit='ms') + timedelta(hours=8))
            raw_df = merged.copy()

            # --- Feature Extraction ---
            df = raw_df.rename(columns={
                'x_x': 'accel_x',
                'y_x': 'accel_y',
                'z_x': 'accel_z',
                'x_y': 'gyro_x',
                'y_y': 'gyro_y',
                'z_y': 'gyro_z'
            })
            df = df.dropna().sort_values("timestamp").reset_index(drop=True)

            segments = []
            window_size = 200
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
                        'elevation_change': window['altitude'].iloc[-1] - window['altitude'].iloc[0] if 'altitude' in window else 0,
                        'gyro_y_std': window['gyro_y'].std(),
                        'gyro_x_std': window['gyro_x'].std(),
                        'latitude': window['latitude'].mean(),
                        'longitude': window['longitude'].mean()
                    }
                    segments.append(features)

            features_df = pd.DataFrame(segments)
            if features_df.empty:
                st.warning("No valid segments extracted (check if speed > 5 km/h).")
                st.stop()

            # --- Prediction ---
            model = joblib.load("iri_rf_classifier_cleaned.pkl")
            X_pred = features_df.drop(columns=["start_time", "end_time", "latitude", "longitude"], errors="ignore")
            features_df["prediction"] = model.predict(X_pred)

            # --- Add readable_time ---
            if "readable_time" in raw_df.columns:
                time_lookup = raw_df[["timestamp", "readable_time"]].drop_duplicates()
                features_df = features_df.merge(time_lookup, left_on="start_time", right_on="timestamp", how="left")
                features_df.drop(columns=["timestamp"], inplace=True)

            # --- Sidebar display options ---
            st.sidebar.title("🧭 Display Options")
            show_data = st.sidebar.checkbox("Show Extracted Data", value=True)
            show_pie = st.sidebar.checkbox("Show Pie Chart", value=True)
            show_map = st.sidebar.checkbox("Show Map View", value=True)

            # --- Show extracted data ---
            if show_data:
                st.subheader("📊 Segment-wise Prediction")
                available_cols = [c for c in ["readable_time", "mean_speed", "prediction"] if c in features_df.columns]
                st.dataframe(features_df[available_cols])

            # --- Pie chart ---
            if show_pie:
                st.subheader("📎 Road Condition Distribution")
                fig, ax = plt.subplots()
                features_df["prediction"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
                ax.set_ylabel('')
                ax.set_title("Predicted Segment Labels")
                st.pyplot(fig)

            # --- Map view ---
            if show_map and "latitude" in features_df.columns and "longitude" in features_df.columns:
                st.subheader("🗺️ Map View of Road Segments")
                color_map = {
                    "Smooth": "green",
                    "Fair": "blue",
                    "Rough": "red"
                }
                m = folium.Map(location=[features_df["latitude"].mean(), features_df["longitude"].mean()], zoom_start=15, tiles="CartoDB dark_matter")

                for _, row in features_df.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=5,
                        color=color_map.get(row["prediction"], "gray"),
                        fill=True,
                        fill_opacity=0.8
                    ).add_to(m)

                st_folium(m, width=1100, height=500)
