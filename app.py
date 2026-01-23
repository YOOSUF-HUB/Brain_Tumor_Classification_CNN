import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
# Removed unused openai import

# --- Page Config ---
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
# FIX APPLIED HERE: Added 'color: #333333;' to .gpt-box
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    .gpt-box {
        background-color: #ffffff;
        color: #333333; /* <-- THIS FIXES THE INVISIBLE TEXT */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #7c3aed;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Vision Model ---
@st.cache_resource
def load_model():
    # Make sure filename matches exactly what you saved
    model = tf.keras.models.load_model('brain_tumor_custom_cnn.keras')
    return model

with st.spinner('Initializing AI Systems...'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Error loading vision model: {e}")
        st.stop()

# --- Medical Info Dictionary (The "Free Fix") ---
TUMOR_INFO = {
    "Glioma": (
        "**Glioma** is a type of tumor that starts in the glial cells of the brain or the spine. "
        "It is one of the most common types of primary brain tumors. Treatment often involves surgery, "
        "radiation therapy, or chemotherapy depending on the grade."
    ),
    "Meningioma": (
        "**Meningioma** is a tumor that forms on the membranes that cover the brain and spinal cord. "
        "Most are slow-growing and often benign (not cancer). Many do not require immediate treatment, "
        "but larger ones may need surgery."
    ),
    "Pituitary": (
        "**Pituitary tumors** are abnormal growths in the pituitary gland. "
        "Most are benign (adenomas). They can cause hormone imbalances or vision problems. "
        "Treatment may include medication or surgery."
    ),
    "No Tumor": (
        "**Great news!** No abnormalities consistent with the trained tumor classes were detected in this scan. "
        "However, this AI is not a doctor. If you are experiencing headaches, vision issues, or nausea, "
        "please consult a medical professional."
    )
}

# Helper function to get the info
def get_explanation(label):
    return TUMOR_INFO.get(label, "No information available.")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=100)
    st.title("NeuroScan AI")
    st.markdown("### 🧬 Medical Analysis System")
    st.info("This system uses a CNN Deep Learning model to detect tumor patterns in MRI scans and provides standard medical definitions.")

# --- Main Interface ---
st.markdown("## 🏥 MRI Analysis Dashboard")
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1. Upload Scan")
    file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])
    
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Source Image', use_container_width=True)

with col2:
    st.markdown("### 2. Analysis Results")
    
    if file:
        # Preprocessing (Safe Mode - No manual division)
        image_resized = image.resize((256, 256))
        img_array = np.asarray(image_resized)
        img_reshape = img_array[np.newaxis, ...]
        
        analyze_btn = st.button("🔍 Analyze MRI Scan")
        
        if analyze_btn:
            # 1. Visual Analysis Animation
            progress_text = "Analyzing neural patterns..."
            my_bar = st.progress(0, text=progress_text)
            
            for p in range(70):
                time.sleep(0.01)
                my_bar.progress(p + 1, text=progress_text)
            
            # 2. Actual Prediction
            prediction = model.predict(img_reshape)
            score = tf.nn.softmax(prediction[0])
            class_index = np.argmax(score)
            confidence = 100 * np.max(score)
            predicted_label = CLASS_NAMES[class_index]
            
            # 3. Get Explanation Text
            my_bar.progress(90, text="Retrieving medical information...")
            explanation_text = get_explanation(predicted_label)
            time.sleep(0.3)
            my_bar.progress(100, text="Report Ready.")
            time.sleep(0.2)
            my_bar.empty()
            
            # --- Display Results ---
            st.divider()
            
            # Header
            color = "green" if predicted_label == "No Tumor" else "red"
            icon = "✅" if predicted_label == "No Tumor" else "⚠️"
            
            st.markdown(f"""
            <h3 style="color: {color}; margin-bottom: 0;">{icon} Prediction: {predicted_label}</h3>
            <p style="font-size: 1.1em;">Confidence: <strong>{confidence:.2f}%</strong></p>
            """, unsafe_allow_html=True)
            
            # Probability Bars
            st.caption("Confidence Breakdown:")
            cols = st.columns(4)
            for i, name in enumerate(CLASS_NAMES):
                with cols[i]:
                    # Using progress bar instead of just metric for better visuals
                    st.write(f"**{name}**")
                    st.progress(int(score[i]*100))

            # Explanation Box (Text should now be visible)
            st.markdown("#### 📝 Medical Report")
            st.markdown(f"""
            <div class="gpt-box">
                {explanation_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("Note: This is an AI-generated report for educational purposes. Always verify with a doctor.")