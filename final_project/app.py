import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Dehydration Detection in Infants",
    layout="wide",
    page_icon="🧒💧",
    initial_sidebar_state="expanded"
)

# Dark mode custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #121212 !important;
        color: #f5f5f5 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0077b6;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-top: 2rem;
    }
    .footer {
        margin-top: 4rem;
        text-align: center;
        font-size: 0.9rem;
        color: #888;
        border-top: 1px solid #333;
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4360/4360413.png", width=120)
    st.title("Dehydration Scan")
    st.markdown("AI-powered tool to detect **dehydration in infants** from facial images.")
    st.markdown("### 📖 How it works:")
    st.markdown("""
    - Upload a clear photo of the infant's face.
    - The AI model analyzes features for signs of dehydration.
    - Get a prediction instantly.
    """)
    st.markdown("---")
    st.info("Made by Labhansh Pal | B.Tech CSE - AIDS")

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dehydration_model.h5")

model = load_model()

# Title and description
st.markdown("<h1 style='text-align: center; color: #90e0ef;'>Dehydration Detection in Infants 💧</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and receive an instant AI diagnosis.</p>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)


    with col2:
        st.markdown("### ⏳ Processing Image...")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = " Healthy 😊" if prediction > 0.65 else " Dehydrated 😟"
        bg_color = "#2a9d8f" if prediction > 0.65 else "#e63946"

        st.markdown(f"""
        <div class='result-box' style='background-color: {bg_color}; color: white;'>
            Prediction: {label}<br>
            
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        Made by Labhansh Pal | B.Tech CSE - Artificial Intelligence & Data Science
    </div>
""", unsafe_allow_html=True)
