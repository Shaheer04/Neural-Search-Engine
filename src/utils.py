
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

def load_and_preprocess_image(image_source, target_size=(224, 224)):
    """
    Load an image from a path or file-like object and preprocess it for MobileNetV2.
    
    Args:
        image_source: File path (str) or file-like object (e.g., UploadedFile).
        target_size: Tuple (height, width).
    
    Returns:
        np.array: Preprocessed image batch of shape (1, 224, 224, 3).
    """
    # Handle file-like object from Streamlit
    if hasattr(image_source, 'read'):
        img = Image.open(image_source)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
    else:
        # Assume it's a file path
        img = keras_image.load_img(image_source, target_size=target_size)

    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
