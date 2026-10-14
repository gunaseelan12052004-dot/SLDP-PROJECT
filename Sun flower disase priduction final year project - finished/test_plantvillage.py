import os
import urllib.request
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# URLs of test images from PlantVillage
urls = {
    'tomato_healthy.jpg': 'https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato-Healthy/0001df45-ad4f-43e1-ae00-75be6b0d911b___GH_HL%20Leaf%208.1.JPG',
    'apple_rust.jpg': 'https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Cedar_apple_rust/007320b6-68fb-4fd5-b1ab-dfafebc7c252___FREC_C.Rust%203673.JPG',
    'corn_healthy.jpg': 'https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Corn_(maize)___healthy/00140228-5690-4a87-b9fa-3f8251f28b7e___R.S_HL%207908%20copy.jpg'
}

os.makedirs('test_images', exist_ok=True)
headers = {'User-Agent': 'Mozilla/5.0'}

print("Downloading PlantVillage test images...")
for filename, url in urls.items():
    path = os.path.join('test_images', filename)
    if not os.path.exists(path):
        try:
            print(f"Downloading {filename}...")
            # urllib.request.urlretrieve can fail if user-agent is blocked, let's use Request
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

print("\nEvaluating PlantVillage test images...")
for filename in urls.keys():
    path = os.path.join('test_images', filename)
    if not os.path.exists(path):
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
