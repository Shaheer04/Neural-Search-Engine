
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from src.utils import load_and_preprocess_image

class FeatureExtractor:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.mobilenet = None
        self.pca = None
        self._load_models()

    def _load_models(self):
        # Load MobileNetV2
        self.mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        self.mobilenet.trainable = False
        
        # Load PCA
        pca_path = os.path.join(self.models_dir, 'pca.pkl')
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
        else:
            raise FileNotFoundError(f"PCA model not found at {pca_path}. Please run training first.")

    def extract(self, image_source):
        """
        Extract features from an image source using MobileNetV2 and reduce dims with PCA.
        """
        # Preprocess
        img_data = load_and_preprocess_image(image_source)
        
        # Extract features (1280d)
        features = self.mobilenet.predict(img_data, verbose=0)
        
        # Reduce dimensions (50d)
        pca_features = self.pca.transform(features)
        
        return pca_features.flatten()
