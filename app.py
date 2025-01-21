import io
import json
import os

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Initialize Flask app
app = Flask(__name__)

# Print the current working directory to debug file paths
print(f"Current Working Directory: {os.getcwd()}")

# Load Vision Transformer model and feature extractor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load ImageNet class labels
try:
    with open("imagenet_classes.json", "r") as f:
        class_labels = json.load(f)
    print(f"Class Labels Loaded: {len(class_labels)} labels")
except FileNotFoundError:
    print("Error: imagenet_classes.json file not found. Ensure it's in the correct directory.")
    class_labels = []

# Define routes
@app.route('/')
def home():
    return "Welcome to the Vision Transformer (ViT) API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file is in the request
        if 'file' not in request.files:
            print("Error: No file provided in the request.")
            return jsonify({"error": "No file provided"}), 400

        # Read the uploaded file
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        print(f"Image Loaded Successfully: {file.filename}")

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")
        print(f"Inputs Processed: {inputs}")

        # Perform prediction
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Predicted Index: {predicted_class_idx}")

        # Map predicted index to label
        if 0 <= predicted_class_idx < len(class_labels):
            predicted_label = class_labels[predicted_class_idx]
            print(f"Predicted Label: {predicted_label}")
        else:
            predicted_label = "Unknown"
            print("Index Out of Range: Predicted class index is out of bounds.")

        # Return response
        response = {
            "predicted_class_index": predicted_class_idx,
            "predicted_label": predicted_label
        }
        print(f"Response Sent: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
