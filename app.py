import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# 🎨 CSS STYLING
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fdf4, #ecfeff);
}

.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

h1 {
    text-align: center;
    color: #065f46;
    font-weight: 800;
}

.prediction-box {
    padding: 20px;
    border-radius: 16px;
    background: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.confidence {
    font-size: 16px;
    font-weight: 600;
    color: #0f766e;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🌱 TITLE
# -------------------------------
st.title("🌱 Plant Disease Detection")
st.caption("AI-powered Tomato Leaf Disease Detection using Deep Learning")

# -------------------------------
# 📦 LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# -------------------------------
# 🏷️ CLASS NAMES (PlantVillage – Tomato only)
# -------------------------------
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
    "Healthy Tomato Leaf"
]

# -------------------------------
# 📖 DISEASE DESCRIPTIONS
# -------------------------------
DISEASE_INFO = {
    "Tomato Early Blight":
        "Fungal disease causing brown concentric spots on older leaves. Use fungicides and avoid overhead watering.",

    "Tomato Late Blight":
        "Serious fungal disease causing dark, water-soaked lesions. Avoid overhead watering and use resistant varieties.",

    "Tomato Leaf Mold":
        "Yellow spots on upper leaf surface with mold underneath. Improve air circulation and reduce humidity.",

    "Tomato Septoria Leaf Spot":
        "Small circular spots with dark borders. Remove infected leaves and apply fungicide.",

    "Tomato Bacterial Spot":
        "Water-soaked spots turning dark brown. Avoid overhead irrigation and use copper sprays.",

    "Tomato Spider Mites":
        "Tiny pests causing yellow stippling. Use insecticidal soap or neem oil.",

    "Tomato Target Spot":
        "Brown lesions with concentric rings. Apply fungicide and rotate crops.",

    "Tomato Yellow Leaf Curl Virus":
        "Viral disease causing yellow curled leaves. Control whiteflies.",

    "Tomato Mosaic Virus":
        "Mottled leaves and stunted growth. Remove infected plants.",

    "Healthy Tomato Leaf":
        "The leaf appears healthy with no visible disease symptoms."
}

# -------------------------------
# 📤 FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image (Tomato leaves only)",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# 🔍 PREDICTION LOGIC
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions)) * 100
    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]

    st.success("Prediction completed!")

    # -------------------------------
    # ⚠️ LOW CONFIDENCE HANDLING
    # -------------------------------
    if confidence < 40:
        st.warning("⚠️ Low confidence prediction. Image may be healthy, unclear, or not a tomato leaf.")

    # -------------------------------
    # 🧾 RESULT DISPLAY
    # -------------------------------
    description = DISEASE_INFO.get(
        predicted_label,
        "This model is trained only on tomato leaves. Other plant leaves may give incorrect results."
    )

    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>🌿 Disease: {predicted_label}</h3>
            <p>{description}</p>
            <p class="confidence">Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------
    # 🌱 IMPORTANT DISCLAIMER
    # -------------------------------
    st.info(
        "ℹ️ This model is trained **only on tomato leaves**. "
        "Uploading other plant leaves (e.g., strawberry, potato) may produce incorrect results."
    )
