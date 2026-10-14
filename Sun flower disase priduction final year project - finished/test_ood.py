import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load models
print("Loading models...")
model = tf.keras.models.load_model("image_classifier.h5")
ood_model = MobileNetV2(weights='imagenet')

print("Evaluating sunflower leaf images...")
# Let's get a few images from each category in original/
categories = ['Downy mildew', 'Fresh Leaf', 'Gray mold', 'Leaf scars']
for cat in categories:
    cat_dir = os.path.join('original', cat)
    if not os.path.isdir(cat_dir):
        print(f"Directory {cat_dir} does not exist!")
        continue
    files = [f for f in os.listdir(cat_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:3]
    for filename in files:
        filepath = os.path.join(cat_dir, filename)
        
        # Test MobileNetV2
        img_ood = image.load_img(filepath, target_size=(224, 224))
        img_array_ood = image.img_to_array(img_ood)
        img_array_ood = np.expand_dims(img_array_ood, axis=0)
        img_array_ood = preprocess_input(img_array_ood)
        ood_preds = ood_model.predict(img_array_ood)
        decoded_ood = decode_predictions(ood_preds, top=5)[0]
        
        # Test custom CNN
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        conf_score = round(float(np.max(preds)) * 100, 2)
        
        print(f"\n--- {filepath} ---")
        print(f"Custom Model: {categories[class_idx]} ({conf_score}%)")
        print(f"MobileNet predictions:")
        for _, label, score in decoded_ood:
            print(f"  {label}: {score:.4f}")
