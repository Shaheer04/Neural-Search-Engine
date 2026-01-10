# ⚡ Neural Search Engine

![Status](https://img.shields.io/badge/Status-Beta-orange)
![Mode](https://img.shields.io/badge/Mode-Inference-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

A robust **Content-Based Image Retrieval (CBIR)** system designed to classify product images and retrieve visually similar items from a catalog using deep learning feature extraction and vector similarity search.

## 🚀 Overview

This project implements a "Triple Engine" protocol for neural search:
1.  **Feature Extraction**: Uses **MobileNetV2** (pre-trained on ImageNet) to extract 1280-dimensional feature vectors from images.
2.  **Dimensionality Reduction**: Applies **PCA** (Principal Component Analysis) to compress vectors to 50 dimensions while retaining semantic information.
3.  **Indexing & Retrieval**: Utilizes **FAISS** (Facebook AI Similarity Search) for high-performance L2 distance similarity search.

Additionally, the system includes:
*   **Unsupervised Clustering**: K-Means clustering (K=6) to identify product groups.
*   **Supervised Classification**: Linear SVM for predicting product categories.

## ✨ Features

*   **⚡ Real-time Inference**: Upload an image and get classification and similarity results in milliseconds.
*   **🧠 Deep Learning Core**: Powered by MobileNetV2 and TensorFlow.
*   **🔍 Vector Search**: Fast nearest-neighbor search using FAISS.
*   **📊 Insightful Metrics**: View inference time, predicted class, cluster ID, and visual similarity matches.
*   **🛠️ Tech Dashboard**: Explore system architecture, PCA variance, K-Means elbow plots, and confusion matrices directly in the UI.
*   **🎨 Cyberpunk/Hacker Aesthetic**: A custom-styled Dark Mode UI using Streamlit.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Deep Learning**: TensorFlow / Keras (MobileNetV2)
*   **Machine Learning**: Scikit-Learn (PCA, K-Means, SVM)
*   **Vector Search**: FAISS-CPU
*   **Data Processing**: Pandas, NumPy, Pillow

## 📦 Installation

1.  **Clone the repository** (or download usage files):
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    # using pip
    pip install -r requirements.txt
    ```

    *Alternatively, if you use `uv`:*
    ```bash
    uv sync
    ```

## 🖥️ Usage

1.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

2.  **Interact**:
    *   Open the provided local URL (usually `http://localhost:8501`).
    *   Go to the **"NEURAL SEARCH"** tab.
    *   Upload a product image (JPG/PNG).
    *   View predicted category, cluster, and visually similar images from the dataset.

3.  **Explore Architecture**:
    *   Switch to the **"SYSTEM ARCHITECTURE"** tab to view PCA variance, cluster visualizations, and model performance metrics.

## 📂 Project Structure

```text
├── app.py                 # Main Streamlit Application entry point
├── src/
│   ├── feature_extractor.py # MobileNetV2 feature extraction logic
│   ├── predictor.py         # Inference logic (SVM, K-Means, FAISS)
│   └── train.py             # Script to retrain models (optional)
├── data/
│   ├── images/              # Image dataset folder
│   ├── images_dataset.csv   # Metadata CSV
│   └── features.pickle      # Pre-computed features (if cached)
├── models/                # Saved models (PCA, SVM, KMeans, FAISS index)
├── plots/                 # Generated plots for analysis (PCA, Confusion Matrix)
├── requirements.txt       # Python dependencies
└── README.md              # Project Documentation
```

## ⚠️ Requirements

*   **Python 3.8+**
*   The system runs in **CPU Mode** by default if GPU drivers are not found, which is sufficient for inference on this dataset scale.

---
*Created for Complex Computing Activity.*
