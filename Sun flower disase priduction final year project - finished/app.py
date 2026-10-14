from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = tf.keras.models.load_model("image_classifier.h5")


from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import urllib.request
ood_model = MobileNetV2(weights='imagenet')


plant_keywords = [
    'leaf', 'plant', 'flower', 'daisy', 'pot', 'vase', 'sunflower', 'greenhouse', 
    'earthstar', 'cabbage', 'vegetable', 'fruit', 'tree', 'grass', 'mushroom', 
    'bell_pepper', 'cucumber', 'fig', 'strawberry', 'acorn', 'artichoke', 
    'buckeye', 'cardoon', 'chestnut', 'clover', 'fern', 'thistle', 'moss', 'lichen', 
    'corn', 'maize', 'hay', 'straw', 'garden', 'forest', 'nature', 'broccoli', 
    'cauliflower', 'zucchini', 'lettuce', 'spinach', 'herbs', 'vine', 'ivy', 
    'park', 'meadow', 'field', 'landscape', 'agriculture', 'honeycomb', 'slug', 
    'velvet', 'stinkhorn', 'bubble', 'terrapin', 'stinkhorn', 'wood'
]


class_names = ['Downy mildew', 'Fresh Leaf', 'Gray mold', 'Leaf scars']


CONFIDENCE_THRESHOLD = 80.0


suggestions = {
    'Downy mildew': [
        "Remove the infected leaf immediately.",
        "Spray copper or a systemic fungicide.",
        "Avoid water stagnation and improve airflow."
    ],
    'Gray mold': [
        "Cut and destroy the affected parts.",
        "Stop over-watering and maintain dry conditions.",
        "Apply the appropriate fungicide spray."
    ],
    'Leaf scars': [
        "Check if there is any insect attack.",
        "Use neem oil or a mild pesticide.",
        "Monitor the plant regularly."
    ],
    'Fresh Leaf': [
        "The plant is healthy.",
        "Maintain regular watering and sunlight.",
        "Continue periodic inspection."
    ],
    'Invalid Image': [
        "Please upload a clear image of a plant leaf.",
        "Ensure the lighting is sufficient.",
        "Try capturing the leaf from a top-down angle."
    ],
    'Unknown Leaf': [
        "The uploaded leaf is not in our dataset (Sunflower).",
        "Make sure you are uploading a sunflower leaf.",
        "If it is a sunflower leaf, try a clearer photo."
    ]
}


disease_symptoms = {
    'Downy mildew': {
        'title': 'Fungal Infection (Plasmopara)',
        'severity': 'High',
        'spread': 'Rapid',
        'details': [
            "Yellowish or pale green angular spots on the upper leaf surface.",
            "Purplish to gray downy growth of fungus on the leaf underside.",
            "Leaves may eventually turn brown and die (necrosis).",
            "Stunted growth of the plant if infection is severe."
        ]
    },
    'Gray mold': {
        'title': 'Botrytis Fungal Pathogen',
        'severity': 'Medium-High',
        'spread': 'Moderate',
        'details': [
            "Water-soaked spots appearing on leaves or petals.",
            "Masses of fuzzy gray to brown spores covering the affected area.",
            "Wilting and rapid decay of the leaf tissue.",
            "Soft rot in the base of the plant in humid conditions."
        ]
    },
    'Leaf scars': {
        'title': 'Bacterial/Physical Lesions',
        'severity': 'Medium',
        'spread': 'Low-Moderate',
        'details': [
            "Small, circular to irregular brown spots with dark borders.",
            "Older spots may show a 'shot-hole' appearance (tissue falls out).",
            "Yellow halo often surrounding the necrotic lesions.",
            "Multiple spots merging to cause large-scale leaf drying."
        ]
    },
    'Fresh Leaf': {
        'title': 'Physiologically Normal Tissue',
        'severity': 'None',
        'spread': 'N/A',
        'details': [
            "Uniform green pigment across the entire leaf surface.",
            "Leaf margins are smooth and intact without signs of insect damage.",
            "Surface texture is firm and hydrated without fuzzy growths.",
            "No internal spotting or chlorotic (yellow) patches present."
        ]
    },
    'Invalid Image': {
        'title': 'Non-Plant Data Detected',
        'severity': 'N/A',
        'spread': 'N/A',
        'details': [
            "The system was unable to identify a leaf structure in this image.",
            "Object detection failed to find relevant physiological markers.",
            "Please ensure the image contains only a single sunflower leaf."
        ]
    },
    'Unknown Leaf': {
        'title': 'Out-of-Distribution Sample',
        'severity': 'N/A',
        'spread': 'N/A',
        'details': [
            "Plant detected, but does not match any class in the Sunflower Expert System.",
            "Specific symptoms could not be correlated with the target dataset.",
            "Analysis inconclusive due to lack of target biomarkers."
        ]
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None
    disease_suggestions = None
    symptoms_data = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename != "":
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # preprocess image
            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # OOD (Out Of Distribution) check using MobileNetV2
            img_ood = image.load_img(image_path, target_size=(224, 224))
            img_array_ood = image.img_to_array(img_ood)
            img_array_ood = np.expand_dims(img_array_ood, axis=0)
            img_array_ood = preprocess_input(img_array_ood)
            
            ood_preds = ood_model.predict(img_array_ood)
            decoded_ood = decode_predictions(ood_preds, top=10)[0]
            
            is_plant = False
            for _, label, _ in decoded_ood:
                label_lower = label.lower()
                if any(k in label_lower for k in plant_keywords):
                    is_plant = True
                    break
            
            # Disease prediction
            preds = model.predict(img_array)
            class_index = np.argmax(preds)
            conf_score = round(float(np.max(preds)) * 100, 2)
            
            # if MobileNet says not a plant, but our model is 95%+ sure, we trust our model
            if not is_plant and conf_score < 95.0:
                prediction = "Invalid Image"
                confidence = 0.0
            else:
                # plant detected or model is extremely sure
                prediction = class_names[class_index]
                confidence = conf_score
                
            # get suggestions and symptoms
            prediction = prediction.strip()
            disease_suggestions = suggestions.get(prediction, ["No suggestions available."])
            symptoms_data = disease_symptoms.get(prediction, {})
            
            # DEBUG LOGS (mapla check panna)
            print(f"DEBUG: Selected Prediction -> '{prediction}'")
            print(f"DEBUG: Symptoms Data Found -> {bool(symptoms_data)}")
            if not symptoms_data:
                print(f"DEBUG: Available Keys -> {list(disease_symptoms.keys())}")

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        suggestions=disease_suggestions,
        symptoms=symptoms_data
    )

if __name__ == '__main__':
    app.run(debug=True)
