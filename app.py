import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------
# Title & description
# ---------------------------
st.title("🌱 Plant Disease Detection")
st.write(
    "Upload a leaf image to detect whether it is healthy or affected by a plant disease. "
    "This model is trained on the PlantVillage dataset."
)

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------------------
# Class names + descriptions
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

DISEASE_INFO = {
    "Tomato Early Blight": "Fungal disease causing brown spots on older leaves. Use fungicides and avoid overhead watering.",
    "Tomato Healthy": "The leaf appears healthy with no visible disease symptoms.",
}

# ---------------------------
# File uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # Preprocess image
    # ---------------------------
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    predicted_label = CLASS_NAMES[predicted_index]

    st.success("✅ Prediction completed!")

    # ---------------------------
    # Display result
    # ---------------------------
    st.subheader(f"🌿 Disease: {predicted_label}")

    description = DISEASE_INFO.get(
        predicted_label,
        "No detailed description available for this disease."
    )
    st.write(description)

   
