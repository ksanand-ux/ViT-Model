import io
import json

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained ViT model and feature extractor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load ImageNet class labels
with open("imagenet_classes.json", "r") as f:
    class_labels = json.load(f)

# Debug: Print the number of class labels loaded
print(f"Class Labels Loaded: {len(class_labels)} labels")  # Add this debug line here

@app.route('/')
def home():
    return "Welcome to the Vision Transformer (ViT) Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Read the uploaded file
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # Perform prediction
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # Get the label for the predicted class
        predicted_label = class_labels[predicted_class_idx]

        # Debug: Print predicted index and label
        print(f"Predicted Index: {predicted_class_idx}")
        print(f"Predicted Label: {predicted_label}")

        return jsonify({
            "predicted_class_index": predicted_class_idx,
            "predicted_label": predicted_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
