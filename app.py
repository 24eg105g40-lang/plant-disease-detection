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
st.info("🌿 This model is trained only for **Tomato plant leaves**. Uploading other plant leaves may give incorrect results.")

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
    # 🍅 Tomato
    "Tomato Early Blight": "Fungal disease causing brown spots on older leaves with concentric rings. Control using fungicides and proper crop rotation.",
    "Tomato Late Blight": "Serious fungal disease causing dark, water-soaked lesions. Avoid overhead watering and use resistant varieties.",
    "Tomato Leaf Mold": "Causes yellow spots on upper leaf surface and mold growth underneath. Improve air circulation and reduce humidity.",
    "Tomato Septoria Leaf Spot": "Small circular spots with dark borders. Remove infected leaves and apply fungicides.",
    "Tomato Spider Mites": "Tiny pests causing yellowing and webbing on leaves. Control using miticides or neem oil.",
    "Tomato Yellow Leaf Curl Virus": "Viral disease causing leaf curling and stunted growth. Spread by whiteflies; remove infected plants.",
    "Tomato Mosaic Virus": "Causes mottled leaf patterns and distorted growth. Spread through contact; disinfect tools.",
    "Tomato Healthy": "The leaf appears healthy with normal green color and no visible disease symptoms.",

    # 🥔 Potato
    "Potato Early Blight": "Fungal disease causing dark brown spots on leaves. Use crop rotation and fungicides.",
    "Potato Late Blight": "Highly destructive disease causing rapid leaf decay. Avoid wet conditions and use resistant varieties.",
    "Potato Healthy": "Healthy potato leaf with no signs of infection or damage.",

    # 🍎 Apple
    "Apple Scab": "Fungal disease causing dark scabby lesions on leaves and fruit. Prune affected areas and apply fungicides.",
    "Apple Black Rot": "Causes circular brown lesions on leaves and fruit rot. Remove infected branches and improve airflow.",
    "Apple Cedar Rust": "Orange-yellow spots on leaves caused by fungal infection. Remove nearby cedar trees if possible.",
    "Apple Healthy": "Apple leaf appears healthy with smooth surface and uniform green color.",

    # 🌽 Corn
    "Corn Gray Leaf Spot": "Fungal disease causing rectangular gray lesions. Practice crop rotation and residue management.",
    "Corn Common Rust": "Reddish-brown pustules on leaves. Usually mild but can reduce yield.",
    "Corn Northern Leaf Blight": "Long gray-green lesions that reduce photosynthesis. Use resistant hybrids.",
    "Corn Healthy": "Corn leaf shows healthy green color and intact structure.",

    # 🍇 Grape
    "Grape Black Rot": "Brown spots with black margins on leaves. Apply fungicides and remove infected debris.",
    "Grape Esca": "Chronic disease causing tiger-striped leaves. Improve vineyard sanitation.",
    "Grape Leaf Blight": "Leaf browning and premature leaf drop. Maintain proper irrigation.",
    "Grape Healthy": "Grape leaf is green and free from spots or discoloration.",

    # 🍓 Strawberry
    "Strawberry Leaf Scorch": "Dark purple spots on leaves that merge. Remove infected leaves and avoid overhead watering.",
    "Strawberry Healthy": "Strawberry leaf appears healthy with no signs of scorch or spots.",

    # 🌶 Pepper
    "Pepper Bell Bacterial Spot": "Water-soaked lesions turning brown. Use certified seeds and avoid wet foliage.",
    "Pepper Bell Healthy": "Pepper leaf is green and healthy with no lesions.",

    # 🫐 Blueberry
    "Blueberry Healthy": "Blueberry leaf appears healthy with uniform color and no disease symptoms.",

    # 🌱 General
    "Soybean Healthy": "Soybean leaf shows normal color and growth with no visible disease.",
    "Raspberry Healthy": "Healthy raspberry leaf with no discoloration or spotting.",
    "Squash Powdery Mildew": "White powdery fungal growth on leaves. Improve air circulation and apply fungicides."
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

    

