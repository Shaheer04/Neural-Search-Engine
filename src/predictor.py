
import os
import pickle
import pandas as pd
import faiss
import numpy as np

class Predictor:
    def __init__(self, models_dir, data_dir):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.svm = None
        self.kmeans = None
        self.label_encoder = None
        self.index = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        # Load SVM
        with open(os.path.join(self.models_dir, 'svm.pkl'), 'rb') as f:
            self.svm = pickle.load(f)
            
        # Load K-Means
        with open(os.path.join(self.models_dir, 'kmeans.pkl'), 'rb') as f:
            self.kmeans = pickle.load(f)
            
        # Load Label Encoder
        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        # Load FAISS Index
        self.index = faiss.read_index(os.path.join(self.models_dir, 'faiss_index.bin'))
        
        # Load Metadata
        # We saved it as 'processed_metadata.pkl' in train.py
        meta_path = os.path.join(self.data_dir, 'processed_metadata.pkl')
        # Fallback to CSV if pickle not found (though pickle is preferred for index alignment)
        if os.path.exists(meta_path):
             self.metadata = pd.read_pickle(meta_path)
        else:
             print("Warning: Processed metadata not found. Similarity results might be misaligned if dataset changed.")
             # Fallback logic could go here but it's risky
             raise FileNotFoundError(f"Metadata not found at {meta_path}")

    def predict_classification(self, feature_vector):
        """Predict category using SVM."""
        # SVM expects (n_samples, n_features)
        vector_2d = feature_vector.reshape(1, -1)
        dataset_idx = self.svm.predict(vector_2d)[0]
        label = self.label_encoder.inverse_transform([dataset_idx])[0]
        return label

    def predict_cluster(self, feature_vector):
        """Predict cluster using K-Means."""
        vector_2d = feature_vector.reshape(1, -1)
        cluster = self.kmeans.predict(vector_2d)[0]
        return cluster

    def find_similar(self, feature_vector, k=5):
        """Find k similar images using FAISS."""
        vector_2d = feature_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(vector_2d, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                row = self.metadata.iloc[idx]
                results.append({
                    'id': idx,
                    'distance': dist,
                    'filename': row['filename'],
                    'category': row['main_category'],
                    'cluster': row.get('cluster', -1),
                    'image_url': row['image']
                })
        return results
