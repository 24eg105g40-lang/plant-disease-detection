import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------
# CSS (ONLY UI)
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fff4, #e6f7f1);
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 20px;
    color: #14532d;
}

.result-card {
    background-color: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.markdown("<div class='main-title'>🌱 Plant Leaf Disease Detection</div>", unsafe_allow_html=True)
st.caption("Upload a tomato leaf image to detect disease.")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------------------
# FULL classes (how model was trained)
# ---------------------------
CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew",
    "Cherry Healthy",
    "Corn Gray Leaf Spot",
    "Corn Common Rust",
    "Corn Northern Leaf Blight",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Esca",
    "Grape Leaf Blight",
    "Grape Healthy",
    "Orange Haunglongbing",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Strawberry Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))

    predicted_label = CLASS_NAMES[predicted_index]

    # ---------------------------
    # Tomato filter
    # ---------------------------
    if not predicted_label.lower().startswith("tomato"):
        predicted_label = "Not a Tomato Leaf"

    st.success("✅ Prediction completed!")

    # ---------------------------
    # Result display
    # ---------------------------
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader(f"🌿 Disease: {predicted_label}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    "<div class='footer'>🚀 Tomato Leaf Disease Detection | Streamlit</div>",
    unsafe_allow_html=True
)
