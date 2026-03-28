import streamlit as st
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from ultralytics import YOLO
import tempfile
from PIL import Image

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🧵",
    layout="wide"
)

# -----------------------------
# PROJECT TITLE
# -----------------------------
st.title("DGAN_YOLOv8n: A Fine-Grained Fabric Defect Detection Using Enhancement Techniques")

# -----------------------------
# BATCH INFORMATION
# -----------------------------
st.subheader("Batch A3")

# -----------------------------
# PROJECT DESCRIPTION
# -----------------------------
st.write("""
Upload a fabric image to detect defects such as **Hole** and **Stain** using  **Denoise GAN enhancement followed by YOLOv8n detection**.
""")

# -----------------------------
# TEAM MEMBERS
# -----------------------------
st.write("""
Dr. Vamsi Bandi - 69003108  
T. Anitha - 23695A3101  
G. Bhavitha - 22691A3121  
S. Inthiyaz - 22691A354  
M. Jalandar - 22691A358
""")

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
st.subheader("Upload Fabric Image (JPG / PNG)")

uploaded_file = st.file_uploader(
    "Drag and drop a fabric image here or click Browse",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# PROCESS IMAGE
# -----------------------------
if uploaded_file is not None:

    st.success("Image uploaded successfully!")

    image = Image.open(uploaded_file)

    # Save temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)

    # Run YOLO prediction
    results = model.predict(temp_file.name, conf=0.25)

    # Get plotted image
    result_img = results[0].plot()

    # -----------------------------
    # EXTRACT DETECTED DEFECTS
    # -----------------------------
    detected_classes = []

    if results[0].boxes is not None:
        for cls in results[0].boxes.cls:
            detected_classes.append(results[0].names[int(cls)])

    detected_classes = list(set(detected_classes))

    # -----------------------------
    # DISPLAY DEFECT INFORMATION
    # -----------------------------
    if len(detected_classes) == 0:
        st.error("No defects detected in the fabric.")
    else:
        st.success(f"Detected Defects: {', '.join(detected_classes)}")

    # -----------------------------
    # DISPLAY IMAGES SIDE BY SIDE
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Fabric Image", width=400)

    with col2:
        st.image(result_img, caption="Detected Fabric Defects", width=400)