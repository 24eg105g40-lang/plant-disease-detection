import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Title
# -------------------------------
st.title("🌱 Plant Disease Detection")

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model("plant_disease_model.keras")

# -------------------------------
# Upload image
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction logic
# -------------------------------
if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)

    # Show result
    st.success("✅ Prediction completed!")
    st.write("Predicted class index:", predicted_index)

