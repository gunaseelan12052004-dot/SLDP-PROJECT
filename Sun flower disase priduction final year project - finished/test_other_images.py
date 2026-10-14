import os
import urllib.request
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Create test folder
os.makedirs('test_images', exist_ok=True)

# URLs of test images
urls = {
    'tomato_leaf.jpg': 'https://upload.wikimedia.org/wikipedia/commons/8/88/Tomato_leaf_m.JPG',
    'maple_leaf.jpg': 'https://upload.wikimedia.org/wikipedia/commons/a/a5/Maple_Leaf.jpg',
    'rose_leaf.jpg': 'https://upload.wikimedia.org/wikipedia/commons/e/e0/Rosa_rubiginosa_leaf_5.jpg',
    'car.jpg': 'https://upload.wikimedia.org/wikipedia/commons/4/49/2015_Ford_Mustang_GT_USA_2.jpg',
    'human_face.jpg': 'https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg'
}

print("Downloading test images with user agent...")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for filename, url in urls.items():
    path = os.path.join('test_images', filename)
    if not os.path.exists(path):
        try:
            print(f"Downloading {filename}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                with open(path, 'wb') as f:
                    f.write(response.read())
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

# Load models
print("Loading models...")
model = tf.keras.models.load_model("image_classifier.h5")
ood_model = MobileNetV2(weights='imagenet')

class_names = ['Downy mildew', 'Fresh Leaf', 'Gray mold', 'Leaf scars']

print("\nEvaluating test images...")
for filename in urls.keys():
    path = os.path.join('test_images', filename)
    if not os.path.exists(path):
        print(f"Skipping {filename} because it doesn't exist.")
        continue
        
    # MobileNetV2
    img_ood = image.load_img(path, target_size=(224, 224))
    img_array_ood = image.img_to_array(img_ood)
    img_array_ood = np.expand_dims(img_array_ood, axis=0)
    img_array_ood = preprocess_input(img_array_ood)
    ood_preds = ood_model.predict(img_array_ood)
    decoded_ood = decode_predictions(ood_preds, top=5)[0]
    
    # Custom CNN
    img = image.load_img(path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    conf_score = round(float(np.max(preds)) * 100, 2)
    
    print(f"\n====================================")
    print(f"Image: {filename}")
    print(f"Custom Model: {class_names[class_idx]} ({conf_score}%)")
    print(f"Raw Softmax: {np.round(preds[0], 4)}")
    print(f"MobileNet predictions:")
    for _, label, score in decoded_ood:
        print(f"  {label}: {score:.4f}")
