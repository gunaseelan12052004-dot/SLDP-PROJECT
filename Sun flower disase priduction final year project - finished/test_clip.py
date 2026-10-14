import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("Loading CLIP model...")
try:
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("CLIP model loaded successfully!")
    
    # Let's test it on human_face.jpg if it exists
    image_path = 'test_images/human_face.jpg'
    if os.path.exists(image_path):
        image = Image.open(image_path)
        
        texts = [
            "a photo of a sunflower leaf",
            "a photo of another type of leaf",
            "a photo of a human face",
            "a photo of an object"
        ]
        
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get probabilities
        
        print("\nTest on human face:")
        for text, prob in zip(texts, probs[0]):
            print(f"  {text}: {prob.item():.4f}")
            
    # Let's test on a sunflower leaf image from the dataset
    sunflower_leaf_path = 'original/Fresh Leaf/fresh. (10).jpg'
    if os.path.exists(sunflower_leaf_path):
        image = Image.open(sunflower_leaf_path)
        
        texts = [
            "a photo of a sunflower leaf",
            "a photo of another type of leaf",
            "a photo of a human face",
            "a photo of an object"
        ]
        
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        print("\nTest on sunflower leaf:")
        for text, prob in zip(texts, probs[0]):
            print(f"  {text}: {prob.item():.4f}")
            
except Exception as e:
    print(f"Error during CLIP test: {e}")
