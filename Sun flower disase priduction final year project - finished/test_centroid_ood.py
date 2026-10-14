import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

print("Loading MobileNetV2 feature extractor...")
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features[0]

# Let's collect 30 sunflower leaf images from each category to build a centroid
categories = ['Downy mildew', 'Fresh Leaf', 'Gray mold', 'Leaf scars']
sunflower_features = []

print("Extracting sunflower leaf features...")
for cat in categories:
    cat_dir = os.path.join('original', cat)
    if not os.path.isdir(cat_dir):
        continue
    files = [f for f in os.listdir(cat_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:30]
    for filename in files:
        filepath = os.path.join(cat_dir, filename)
        try:
            feat = get_features(filepath)
            sunflower_features.append(feat)
        except Exception as e:
            pass

sunflower_features = np.array(sunflower_features)
centroid = np.mean(sunflower_features, axis=0)
# Normalize centroid for cosine similarity
centroid_norm = centroid / np.linalg.norm(centroid)

print(f"Centroid calculated. Shape: {centroid_norm.shape}")

# Function to compute cosine similarity
def get_similarity(img_path):
    feat = get_features(img_path)
    feat_norm = feat / np.linalg.norm(feat)
    similarity = np.dot(centroid_norm, feat_norm)
    return similarity

# Test on some other sunflower leaves (not in the training subset)
print("\nTesting on other sunflower leaves:")
for cat in categories:
    cat_dir = os.path.join('original', cat)
    if not os.path.isdir(cat_dir):
        continue
    files = [f for f in os.listdir(cat_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][40:43]
    for filename in files:
        filepath = os.path.join(cat_dir, filename)
        sim = get_similarity(filepath)
        print(f"  Sunflower Leaf ({cat} - {filename}): Cosine Similarity = {sim:.4f}")

# Test on human face
face_path = 'test_images/human_face.jpg'
if os.path.exists(face_path):
    sim = get_similarity(face_path)
    print(f"\n  Human Face (Barack Obama): Cosine Similarity = {sim:.4f}")
