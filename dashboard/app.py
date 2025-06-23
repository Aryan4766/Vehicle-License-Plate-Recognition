import streamlit as st
st.set_page_config(page_title="Indian Vehicle Recognition Dashboard", layout="wide")

import pandas as pd
import os
from PIL import Image
from streamlit_autorefresh import st_autorefresh

# 🔄 Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="refresh")

# Set CSV path
csv_path = "logs/detections.csv"

# App Header
st.title("🚙 Indian Vehicle Recognition Dashboard")
st.markdown("Live feed detection from multi-CCTV using **YOLOv8 + Color + License Plate Recognition**")

# Check if detections exist
if not os.path.exists(csv_path) or os.stat(csv_path).st_size < 100:
    st.warning("🚫 No detections yet. System is initializing or still processing...")
    st.stop()

# Load CSV data
df = pd.read_csv(csv_path)
df = df.sort_values(by=["Frame"], ascending=False)

# Show table
st.success("✅ Showing latest detections")
st.dataframe(df.tail(50), use_container_width=True)

# Show thumbnails
st.subheader("🖼️ Recent Vehicle Snapshots")
latest_images = df.tail(10)
cols = st.columns(5)

for i, (_, row) in enumerate(latest_images.iterrows()):
    with cols[i % 5]:
        try:
            img = Image.open(row["ImagePath"])
            st.image(img, caption=f"{row['Camera']} | Color: {row.get('Color', 'N/A')}", use_column_width=True)
        except Exception as e:
            st.error(f"Image load error: {e}")
