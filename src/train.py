
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import faiss

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'images')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'images_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots') # Save plots here
IMG_SIZE = (224, 224)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Starting Training Pipeline...")

# 1. Load Data
print("Loading Data...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_PATH}")
    exit(1)

def extract_filename(url):
    return url.split('/')[-1]

df['filename'] = df['image'].apply(extract_filename)

if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory not found at {DATA_DIR}")
    exit(1)

existing_files = set(os.listdir(DATA_DIR))
df = df[df['filename'].isin(existing_files)].reset_index(drop=True)
print(f"Found {len(df)} images.")

if len(df) == 0:
    print("No images found! Exiting.")
    exit(1)

# 2. Feature Extraction
print("Extracting Features with MobileNetV2...")
mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
mobilenet.trainable = False

def load_and_preprocess_image(filename):
    path = os.path.join(DATA_DIR, filename)
    img = keras_image.load_img(path, target_size=IMG_SIZE)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

features = []
valid_indices = []

# Batch processing would be faster but let's keep it simple for now or use a small batch
# doing one by one for safety against memory OOM if dataset is huge, though it seems small (6k rows)
for i, row in df.iterrows():
    try:
        img_data = load_and_preprocess_image(row['filename'])
        feat = mobilenet.predict(img_data, verbose=0)
        features.append(feat.flatten())
        valid_indices.append(i)
        if i % 100 == 0:
            print(f"Processed {i}/{len(df)} images", end='\r')
    except Exception as e:
        print(f"Error processing {row['filename']}: {e}")

print(f"\nFinished extracting features. Valid images: {len(valid_indices)}")
features = np.array(features)
df_clean = df.iloc[valid_indices].reset_index(drop=True)

# 3. PCA
print("Training PCA...")
pca = PCA(n_components=50)
pca_features = pca.fit_transform(features)

# Save PCA plot
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.savefig(os.path.join(PLOTS_DIR, 'pca_variance.png'))
plt.close()

# 4. K-Means (Elbow)
print("Running Elbow Method...")
wcss = []
k_range = range(2, 16)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=SEED, n_init='auto')
    kmeans_temp.fit(pca_features)
    wcss.append(kmeans_temp.inertia_)

plt.figure()
plt.plot(k_range, wcss, marker='o')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.savefig(os.path.join(PLOTS_DIR, 'elbow_plot.png'))
plt.close()

# Train Final KMeans
OPTIMAL_K = 6 # Heuristic or determined from previous runs. Automating this is tricky without complex logic.
print(f"Training K-Means with K={OPTIMAL_K}...")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=SEED, n_init='auto')
clusters = kmeans.fit_predict(pca_features)
df_clean['cluster'] = clusters

# Save Cluster Plot
pca_2d = PCA(n_components=2)
features_2d = pca_2d.fit_transform(features)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=features_2d[:,0], y=features_2d[:,1], hue=clusters, palette='tab10', s=50, alpha=0.7)
plt.title('K-Means Clusters')
plt.savefig(os.path.join(PLOTS_DIR, 'clusters.png'))
plt.close()

# 5. SVM Classification
print("Training SVM...")
target_column = 'main_category'
le = LabelEncoder()
y = le.fit_transform(df_clean[target_column])

X_train, X_test, y_train, y_test = train_test_split(pca_features, y, test_size=0.2, random_state=SEED, stratify=y)

svm_model = LinearSVC(random_state=SEED, dual='auto')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {acc:.4f}")

# Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.close()

# 6. FAISS
print("Building FAISS Index...")
d = pca_features.shape[1]
index = faiss.IndexFlatL2(d)
index.add(pca_features.astype('float32'))
faiss.write_index(index, os.path.join(MODELS_DIR, 'faiss_index.bin'))

# 7. Save Models
print("Saving Models...")
with open(os.path.join(MODELS_DIR, 'pca.pkl'), 'wb') as f:
    pickle.dump(pca, f)
with open(os.path.join(MODELS_DIR, 'kmeans.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)
with open(os.path.join(MODELS_DIR, 'svm.pkl'), 'wb') as f:
    pickle.dump(svm_model, f)
with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

# Save processed df for app
df_clean.to_pickle(os.path.join(DATA_DIR, '../processed_metadata.pkl'))

print("Training Pipeline Full/Completed Successfully!")
