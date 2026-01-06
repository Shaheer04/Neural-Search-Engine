
import os
import sys
import numpy as np

# Add project root to path
# __file__ is src/test_backend.py, so dirname is src/. We need the parent of src/ i.e., project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.predictor import Predictor

# Config
# __file__ is src/test_backend.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'images')

def test_backend():
    print("Testing FeatureExtractor...")
    try:
        fe = FeatureExtractor(models_dir=MODELS_DIR)
        print("FeatureExtractor initialized.")
    except Exception as e:
        print(f"FAILED to init FeatureExtractor: {e}")
        return

    print("Testing Predictor...")
    try:
        # Note: Predictor expects `data_dir` to contain `processed_metadata.pkl` IF it fails to load CSV, 
        # but in our code we pass `os.path.dirname(DATA_DIR)` which is `data/`.
        predictor = Predictor(models_dir=MODELS_DIR, data_dir=os.path.dirname(DATA_DIR))
        print("Predictor initialized.")
    except Exception as e:
        print(f"FAILED to init Predictor: {e}")
        return

    # Create dummy image (random noise)
    # 224x224x3
    dummy_img_path = os.path.join(BASE_DIR, 'dummy_test.jpg')
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(dummy_img_path)

    try:
        print("Extracting features from dummy image...")
        features = fe.extract(dummy_img_path)
        print(f"Features shape: {features.shape}")
        
        print("Predicting category...")
        cat = predictor.predict_classification(features)
        print(f"Category: {cat}")
        
        print("Predicting cluster...")
        clust = predictor.predict_cluster(features)
        print(f"Cluster: {clust}")
        
        print("Finding similar...")
        sim = predictor.find_similar(features)
        print(f"Found {len(sim)} similar items.")
        
        print("Top 1 similar:", sim[0] if sim else "None")
        
        print("\nSUCCESS: Backend logic verified!")
        
    except Exception as e:
        print(f"FAILED during inference: {e}")
    finally:
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)

if __name__ == "__main__":
    test_backend()
