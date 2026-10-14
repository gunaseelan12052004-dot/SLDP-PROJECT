import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

m = MobileNetV2(weights='imagenet')

def check_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = m.predict(x)
    decoded = decode_predictions(preds, top=5)[0]
    print(f"Predictions for {path}:")
    for d in decoded:
        print(d)

check_image('static/uploads/GUNASEELAN.jpg')
try:
    check_image('original/Downy mildew/1.jpg') # guessing the dataset directory
except Exception as e:
    print("No leaf image found to test:", e)
