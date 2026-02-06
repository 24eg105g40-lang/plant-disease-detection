import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #f4fff8;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #4b5563;
    margin-bottom: 30px;
}
.result-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    margin-top: 20px;
}
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title & description
# ---------------------------
st.markdown("<div class='main-title'>🌱 Tomato Leaf Disease Detection</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload a tomato leaf image to detect possible diseases. "
    "This model is trained only on tomato leaves.</div>",
    unsafe_allow_html=True
)

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------------------
# Tomato-only classes
# ---------------------------
CLASS_NAMES = [
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

DISEASE_INFO = {
    "Tomato Bacterial Spot": "Bacterial infection causing dark, water-soaked spots on leaves.",
    "Tomato Early Blight": "Fungal disease causing brown spots with concentric rings on older leaves.",
    "Tomato Late Blight": "Serious fungal disease causing dark, water-soaked lesions.",
    "Tomato Leaf Mold": "Fungal disease common in humid conditions, causing yellow patches.",
    "Tomato Septoria Leaf Spot": "Causes small circular spots with dark borders.",
    "Tomato Spider Mites": "Pest infestation causing stippling and webbing under leaves.",
    "Tomato Target Spot": "Fungal disease with target-like spots on leaves.",
    "Tomato Yellow Leaf Curl Virus": "Viral disease causing leaf curling and yellowing.",
    "Tomato Mosaic Virus": "Viral infection causing mottled leaf patterns.",
    "Tomato Healthy": "The tomato leaf appears healthy with no visible disease symptoms."
}

# ---------------------------
# File uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # Preprocess image
    # ---------------------------
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))

    # Safety check
    if predicted_index < len(CLASS_NAMES):
        predicted_label = CLASS_NAMES[predicted_index]
    else:
        predicted_label = "Unknown Tomato Condition"

    st.success("✅ Prediction completed!")

    # ---------------------------
    # Display result
    # ---------------------------
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    st.subheader(f"🌿 Disease: {predicted_label}")

    description = DISEASE_INFO.get(
        predicted_label,
        "The model could not confidently identify this tomato leaf condition."
    )
    st.write(description)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    "<div class='footer'>🚀 Tomato Leaf Disease Detection | TensorFlow + Streamlit</div>",
    unsafe_allow_html=True
)
