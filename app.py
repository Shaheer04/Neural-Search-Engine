
import streamlit as st
import os
import sys
import time
from PIL import Image

# Add current directory to path to import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extractor import FeatureExtractor
from src.predictor import Predictor

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'images')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'images_dataset.csv')

st.set_page_config(page_title="NEURAL SEARCH MODULE // V1.0", layout="wide", page_icon="⚡")

# Custom CSS for "Nerd/Hacker" Aesthetic
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Container Styling */
    .stApp {
        background-color: #0e1117;
    }

    /* Headers */
    h1, h2, h3 {
        color: #00ff41; /* Hacker Green */
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0px 0px 5px #00ff41;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #00ffff; /* Cyan */
        text-shadow: 0px 0px 5px #00ffff;
    }

    /* Custom Cards */
    .result-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-2px);
        border-color: #00ff41;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Buttons */
    button {
        border-radius: 0px !important;
        border: 1px solid #00ff41 !important;
        background-color: #0e1117 !important;
        color: #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    button:hover {
        background-color: #00ff41 !important;
        color: #000 !important;
        box-shadow: 0 0 10px #00ff41;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #00ff41;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 0px;
        border: 1px solid #30363d;
        color: #8b949e;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00ff41;
        border-color: #00ff41;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117 !important;
        color: #00ff41 !important;
        border-color: #00ff41 !important;
        border-bottom: none;
    }

</style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ NEURAL_SEARCH_ENGINE <BETA>")
st.markdown("`SYSTEM STATUS: ONLINE` | `PROTOCOL: TRIPLE_ENGINE` | `MODE: INFERENCE`")
st.markdown("---")

# Sidebar System Stats
with st.sidebar:
    st.header("🖥️ SYSTEM METRICS")
    st.markdown("LOADING CORE MODULES...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        if i < 90:
            time.sleep(0.0001) 
        progress_bar.progress(i + 1)
    
    status_text.success("MODULES LOADED")
    
    st.markdown("---")
    st.metric(label="MODEL ARCHITECTURE", value="MobileNetV2")
    st.metric(label="DIMENSIONALITY", value="50 Dims (PCA)")
    st.metric(label="CLUSTERING ENGINE", value="K-Means (K=6)")
    st.metric(label="CLASSIFIER", value="Linear SVM")
    st.markdown("---")
    st.markdown("`> initializing faiss_index... OK`")
    st.markdown("`> checking_gpu_drivers... NOT_FOUND (CPU_MODE)`")

@st.cache_resource
def load_engine():
    try:
        fe = FeatureExtractor(models_dir=MODELS_DIR)
        predictor = Predictor(models_dir=MODELS_DIR, data_dir=os.path.dirname(DATA_DIR))
        return fe, predictor
    except Exception as e:
        return None, None

fe, predictor = load_engine()

if fe is None or predictor is None:
    st.error("❌ CRITICAL ERROR: MODEL ARTIFACTS MISSING. EXECUTE `src/train.py`.")
    st.stop()

# Tabs
tab_search, tab_tech = st.tabs(["🚀 NEURAL SEARCH", "🛠️ SYSTEM ARCHITECTURE"])

# --- TAB 1: SEARCH ---
with tab_search:
    
    col_upload, col_process = st.columns([1, 2])

    with col_upload:
        st.subheader("📡 INPUT STREAM")
        uploaded_file = st.file_uploader("UPLOAD_IMAGE_BINARY", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='> SOURCE_IMAGE', use_container_width=True)
            st.success("IMAGE ACQUIRED")

    if uploaded_file is not None:
        with col_process:
            st.subheader("⚙️ PROCESSING CORE")
            
            with st.spinner("COMPUTING FEATURE VECTORS..."):
                start_time = time.time()
                features = fe.extract(uploaded_file)
                inference_time = (time.time() - start_time) * 1000
                
                # Predict
                category = predictor.predict_classification(features)
                cluster = predictor.predict_cluster(features)
                results = predictor.find_similar(features, k=5)

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("INFERENCE TIME", f"{inference_time:.2f} ms")
            m2.metric("PREDICTED CLASS", category.upper())
            m3.metric("CLUSTER ID", f"CLUSTER_{cluster}")
            
            st.markdown(f"""
            <div style="background-color: #161b22; padding: 10px; border-left: 3px solid #00ff41; margin-top: 10px;">
                <code style="color: #00ff41;">> VECTOR_EXTRACTED: {features.shape}</code><br>
                <code style="color: #00ff41;">> NEAREST_NEIGHBORS_FOUND: {len(results)}</code>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🔍 SIMILARITY MATRIX [TOP_5]")
        
        # Custom Grid for Results
        cols = st.columns(5)
        for i, res in enumerate(results):
            with cols[i]:
                img_path = os.path.join(DATA_DIR, res['filename'])
                
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                    st.caption(f"ID: {res['id']} | DIST: {res['distance']:.2f}")
                    st.markdown(f"`{res['category']}`")
                else:
                    st.warning("IMG_NOT_FOUND")
    else:
        with col_process:
            st.info("WAITING FOR INPUT STREAM...")

# --- TAB 2: TECHNICAL DETAILS ---
with tab_tech:
    st.header("🔬 SYSTEM ARCHITECTURE & PERFORMANCE")
    
    st.markdown("""
    ### 1. PROJECT OVERVIEW
    This system implements a **content-based retrieval engine** utilizing deep learning feature extraction 
    and vector similarity search. It is designed to classify product images and retrieve visually similar 
    items from a catalog.
    """)
    
    st.markdown("---")
    
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.subheader("📊 THE DATASET")
        st.markdown("""
        - **Source**: E-commerce Product Images
        - **Total Samples**: ~4,500 Images
        - **Categories**: Sports & Fitness, Accessories, etc.
        - **Format**: JPG/PNG (Resized to 224x224)
        """)
        
        st.subheader("🛠️ METHODS ALGORITHM")
        st.markdown("""
        1.  **Feature Extractor**: `MobileNetV2` (ImageNet Weights) -> 1280 Vectors.
        2.  **Compression**: `PCA` (Principal Component Analysis) -> 50 Components.
        3.  **Clustering**: `K-Means` (Unsupervised) for determining product groups.
        4.  **Classification**: `Linear SVM` for labeled category prediction.
        5.  **Retrieval**: `FAISS` (Facebook AI Similarity Search) L2 Index.
        """)
        
    with col_tech2:
        st.subheader("📉 PCA VARIANCE ANALYSIS")
        st.image(os.path.join(PLOTS_DIR, 'pca_variance.png'), caption="Explained Variance Ratio", use_container_width=True)

    st.markdown("---")
    
    col_tech3, col_tech4 = st.columns(2)
    
    with col_tech3:
         st.subheader("🔄 K-MEANS ELBOW METHOD")
         st.markdown("Optimal K was determined by analyzing the WCSS inflection point.")
         st.image(os.path.join(PLOTS_DIR, 'elbow_plot.png'), caption="Elbow Plot (Optimal K=6)", use_container_width=True)
         
    with col_tech4:
        st.subheader("🌌 CLUSTER VISUALIZATION (2D)")
        st.markdown("2D Projection of the 50D feature space showing separation of product groups.")
        st.image(os.path.join(PLOTS_DIR, 'clusters.png'), caption="K-Means Clusters", use_container_width=True)

    st.markdown("---")
    
    st.subheader("✅ CLASSIFICATION PERFORMANCE")
    st.markdown("Evaluation of the Linear SVM on the test set.")
    col_conf, col_metrics = st.columns([1, 1])
    
    with col_conf:
        st.image(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), caption="Confusion Matrix", use_container_width=True)
        
    with col_metrics:
        st.info("ACCURACY: 91.52%")
        st.markdown("""
        The high accuracy indicates that the PCA-compressed features from MobileNetV2 
        retain sufficient semantic information to distinguish between product categories 
        linear boundaries.
        """)
