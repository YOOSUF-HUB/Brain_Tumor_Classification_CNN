import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brain_tumor_custom_cnn.keras')
    return model

try:
    with st.spinner('Loading Model...'):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Class Labels ---
# Ensure this matches the order printed in your training notebook!
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- UI Layout ---
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload an MRI scan to detect if a tumor is present.")

# File Uploader
file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:
    # 1. Load Image
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded MRI', width=300)
    
    # 2. Preprocess (EXACTLY as in your working debug script)
    # Resize to 256x256
    image = image.resize((256, 256))
    
    # Convert to array (Values are 0-255)
    img_array = np.asarray(image)
    
    # --- CRITICAL FIX: DO NOT DIVIDE BY 255.0 ---
    # The model has a Rescaling layer that handles this.
    
    # Expand dimensions to create batch: (1, 256, 256, 3)
    img_reshape = img_array[np.newaxis, ...]
    
    # 3. Predict
    if st.button("Analyze Image"):
        prediction = model.predict(img_reshape)
        score = tf.nn.softmax(prediction[0])
        
        # Get results
        class_index = np.argmax(score)
        confidence = 100 * np.max(score)
        predicted_label = CLASS_NAMES[class_index]
        
        # 4. Display Results
        st.divider()
        st.subheader(f"Prediction: {predicted_label}")
        
        if predicted_label == "No Tumor":
            st.success(f"Confidence: {confidence:.2f}%")
        else:
            st.error(f"Confidence: {confidence:.2f}%")
        
        # Chart
        st.bar_chart(dict(zip(CLASS_NAMES, score.numpy())))