import io
import json
import sys

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification


# Debug function to flush output
def debug_print(message):
    print(message)
    sys.stdout.flush()

# Initialize Flask app
app = Flask(__name__)

# Load model and feature extractor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load class labels
try:
    with open("imagenet_classes.json", "r") as f:
        class_labels = json.load(f)
    debug_print(f"Class Labels Loaded: {len(class_labels)} labels")
    debug_print(f"First 5 Class Labels: {class_labels[:5]}")
except Exception as e:
    debug_print(f"Error loading class labels: {e}")
    class_labels = []

# Define routes
@app.route('/')
def home():
    return "Welcome to the Vision Transformer (ViT) API!"

@app.route('/predict', methods=['POST'])
def predict():
    debug_print("=== Predict endpoint accessed ===")  # Debug 1

    # Ensure an image file is in the request
    if 'file' not in request.files:
        debug_print("No file provided in the request.")  # Debug 2
        return jsonify({"error": "No file provided"}), 400
    debug_print("File received.")  # Debug 3

    # Process the uploaded file
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        debug_print(f"Image processed: {image.format}, {image.size}, {image.mode}")  # Debug 4
    except Exception as e:
        debug_print(f"Error processing image: {e}")
        return jsonify({"error": "Invalid image"}), 400

    # Preprocess the image
    try:
        debug_print("Debug 5: Preprocessing image...")
        inputs = feature_extractor(images=image, return_tensors="pt")
        debug_print(f"Inputs Shape: {inputs['pixel_values'].shape}")  # Debug 6
    except Exception as e:
        debug_print(f"Error preprocessing image: {e}")
        return jsonify({"error": "Image preprocessing failed"}), 500

    # Perform prediction
    try:
        debug_print("Debug 7: Performing model prediction...")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        debug_print(f"Logits Shape: {logits.shape}")
        debug_print(f"Predicted Index: {predicted_class_idx}")
    except Exception as e:
        debug_print(f"Error during model prediction: {e}")
        return jsonify({"error": "Model prediction failed"}), 500

    # Map predicted index to label
    try:
        debug_print("Debug 8: Mapping predicted index to label...")
        predicted_label = class_labels[predicted_class_idx] if predicted_class_idx < len(class_labels) else "Unknown"
        debug_print(f"Predicted Label: {predicted_label}")
    except Exception as e:
        debug_print(f"Error mapping predicted index to label: {e}")
        predicted_label = "Unknown"

    # Prepare and return response
    response = {
        "predicted_class_index": predicted_class_idx,
        "predicted_label": predicted_label
    }
    debug_print("Debug 9: Preparing response...")
    debug_print(f"Response to be returned: {response}")
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    import os
    debug_print("=== Updated app.py Loaded ===")
    debug_print(f"Running app from: {__file__}")
    debug_print(f"Current Working Directory: {os.getcwd()}")
    app.run(host='0.0.0.0', port=5000, debug=True)
