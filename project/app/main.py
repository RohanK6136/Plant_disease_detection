import os
import json
from PIL import Image
import time

import numpy as np
import tensorflow as tf
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem 0;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        margin-bottom: 3rem;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .upload-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: #764ba2;
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3);
    }
    
    .disease-name {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Image container styling */
    .image-container {
        text-align: center;
        margin: 2rem 0;
    }
    
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        max-width: 100%;
        height: auto;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content .block-container {
        padding: 2rem 1rem;
    }
    
    /* Stats cards */
    .stats-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_indices():
    return json.load(open(f"{working_dir}/class_indices.json"))

# Load model and class indices
model = load_model()
class_indices = load_class_indices()

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = float(np.max(predictions) * 100)
    return predicted_class_name, confidence

# Function to format disease name
def format_disease_name(disease_name):
    if "___" in disease_name:
        plant, disease = disease_name.split("___", 1)
        plant = plant.replace("_", " ").title()
        disease = disease.replace("_", " ").title()
        return f"{plant} - {disease}"
    return disease_name.replace("_", " ").title()

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🌱 Plant Disease Classifier</h1>
        <p class="main-subtitle">AI-Powered Plant Health Diagnosis • 38 Disease Classes • Real-time Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with stats
    with st.sidebar:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">38</div>
            <div class="stats-label">Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">15+</div>
            <div class="stats-label">Plant Species</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">CNN</div>
            <div class="stats-label">Deep Learning Model</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🚀 How to Use")
        st.markdown("""
        1. **Upload** a clear image of a plant leaf
        2. **Click** the Classify button
        3. **View** the disease prediction and confidence
        4. **Get** instant health diagnosis
        """)
        
        st.markdown("### 📱 Supported Formats")
        st.markdown("- JPG, JPEG, PNG")
        st.markdown("- High resolution recommended")
        st.markdown("- Clear leaf images work best")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📸 Upload Plant Image</h3>
            <p style="color: #666; margin-bottom: 1.5rem;">Choose a clear image of a plant leaf for disease analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader(
            "Choose an image file...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease classification"
        )
    
    with col2:
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.markdown("""
            <div class="image-container">
                <h4 style="color: #667eea; margin-bottom: 1rem;">📷 Your Image</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Resize image for display
            display_img = image.resize((300, 300))
            st.image(display_img, use_column_width=True, caption="Uploaded Plant Image")
            
            # Classification button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                classify_btn = st.button(
                    "🔍 Analyze Disease", 
                    use_container_width=True,
                    help="Click to analyze the plant for diseases"
                )
            
            if classify_btn:
                # Show loading animation
                with st.spinner("🔬 Analyzing plant health..."):
                    # Add a small delay for better UX
                    time.sleep(1)
                    
                    # Get prediction
                    prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
                    formatted_prediction = format_disease_name(prediction)
                    
                    # Store results in session state
                    st.session_state.prediction = formatted_prediction
                    st.session_state.confidence = confidence
                    st.session_state.is_healthy = "healthy" in prediction.lower()
        
        # Display results
        if hasattr(st.session_state, 'prediction'):
            st.markdown("---")
            
            # Result card
            if st.session_state.is_healthy:
                result_color = "#4CAF50"
                result_icon = "✅"
                result_title = "Healthy Plant"
            else:
                result_color = "#FF9800"
                result_icon = "⚠️"
                result_title = "Disease Detected"
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: {result_color}; text-align: center; margin-bottom: 1.5rem;">
                    {result_icon} {result_title}
                </h3>
                <div class="prediction-result">
                    <div style="font-size: 1.1rem; opacity: 0.9;">Predicted Disease:</div>
                    <div class="disease-name">{st.session_state.prediction}</div>
                </div>
                <div style="text-align: center; margin-top: 1rem;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: #667eea;">
                        Confidence: {st.session_state.confidence:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional information
            if st.session_state.is_healthy:
                st.success("🎉 Great news! Your plant appears to be healthy.")
            else:
                st.warning("⚠️ Disease detected. Please consult with a plant expert for treatment recommendations.")
            
            # Reset button
            col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
            with col_reset2:
                if st.button("🔄 Analyze Another Image", use_container_width=True):
                    # Clear session state
                    for key in ['prediction', 'confidence', 'is_healthy']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

if __name__ == "__main__":
    main()
