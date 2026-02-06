import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱")

st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to detect the plant disease.")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# -------------------------------
# Class names (PlantVillage order)
# -------------------------------
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

# -------------------------------
# Disease descriptions
# -------------------------------
DISEASE_INFO = {
    "Tomato Early Blight": "Fungal disease causing brown spots on older leaves. Use fungicides and avoid overhead watering.",
    "Tomato Late Blight": "Serious disease causing dark lesions. Remove infected plants immediately.",
    "Tomato Healthy": "The plant is healthy with no visible disease symptoms.",
    "Potato Early Blight": "Dry brown spots caused by fungus. Ensure proper crop rotation.",
    "Potato Late Blight": "Rapid spreading disease that can destroy crops.",
    "Apple Scab": "Fungal disease causing dark scabs on leaves and fruit."
}

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = np.max(prediction) * 100

    description = DISEASE_INFO.get(
        predicted_label,
        "No detailed description available for this disease."
    )

    # Output
    st.success("✅ Prediction completed!")
    st.subheader(f"🦠 Disease: {predicted_label}")
    st.write(description)
    st.write(f"**Confidence:** {confidence:.2f}%")
    if confidence < 70:
    st.warning("⚠️ Low confidence prediction. Image may be healthy or unclear.")

