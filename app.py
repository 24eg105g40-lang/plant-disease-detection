import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main {
    background-color: #f6fff8;
}
.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}
.disease {
    font-size: 24px;
    font-weight: 700;
}
.desc {
    font-size: 16px;
    color: #444;
}
.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("## 🌱 Plant Disease Detection")
st.warning("⚠️ This model is trained **only for tomato leaves**")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# -------------------- CLASS NAMES --------------------
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

# -------------------- DISEASE INFO --------------------
DISEASE_INFO = {
    "Tomato Bacterial Spot": "Bacterial infection causing small water-soaked spots. Avoid overhead irrigation.",
    "Tomato Early Blight": "Fungal disease causing brown concentric rings on older leaves.",
    "Tomato Late Blight": "Serious fungal disease with dark water-soaked lesions.",
    "Tomato Leaf Mold": "Yellow spots on upper leaf surface with mold underneath.",
    "Tomato Septoria Leaf Spot": "Small circular spots with dark borders.",
    "Tomato Spider Mites": "Tiny pests causing yellowing and webbing on leaves.",
    "Tomato Target Spot": "Brown lesions with target-like rings.",
    "Tomato Yellow Leaf Curl Virus": "Virus causing leaf curling and yellowing.",
    "Tomato Mosaic Virus": "Mosaic pattern and distorted leaves.",
    "Tomato Healthy": "The tomato leaf appears healthy 🌿"
}

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------- PREPROCESS --------------------
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------- PREDICT --------------------
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))

    st.success("Prediction completed!")

    # -------------------- SAFE CLASS CHECK --------------------
    if predicted_index < len(CLASS_NAMES):
        predicted_label = CLASS_NAMES[predicted_index]

        # -------------------- NON-TOMATO HANDLING --------------------
        if "Tomato" not in predicted_label:
            st.markdown("""
            <div class="result-box">
                <div class="disease">🌿 Disease: Unknown / Non-Tomato Leaf</div>
                <div class="desc">This plant is not supported by the model.</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            description = DISEASE_INFO.get(
                predicted_label,
                "No description available."
            )

            st.markdown(f"""
            <div class="result-box">
                <div class="disease">🍃 Disease: {predicted_label}</div>
                <div class="desc">{description}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("Prediction out of range. Please upload a clear tomato leaf image.")

# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<div class="footer">
🚀 Built using TensorFlow + Streamlit | Hackathon Ready
</div>
""", unsafe_allow_html=True)
