import io
import json

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Initialize Flask app
app = Flask(__name__)

# Load model and feature extractor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load class labels
try:
    with open("imagenet_classes.json", "r") as f:
        class_labels = json.load(f)
    print(f"Class Labels Loaded: {len(class_labels)} labels")
    print(f"First 5 Class Labels: {class_labels[:5]}")
except Exception as e:
    print(f"Error loading class labels: {e}")
    class_labels = []

# Define routes
@app.route('/')
def home():
    return "Welcome to the Vision Transformer (ViT) API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict endpoint hit.")  # Confirm endpoint is accessed

        # Ensure an image file is in the request
        if 'file' not in request.files:
            print("No file provided in the request.")
            return jsonify({"error": "No file provided"}), 400

        # Read the uploaded file
        file = request.files['file']
        print(f"File Received: {file.filename}")  # Debug file name
        image = Image.open(io.BytesIO(file.read()))
        print(f"Image Loaded: {image.format}, {image.size}, {image.mode}")  # Debug: Image info

        # Preprocess the image
        print("Debug 1: Preprocessing image...")
        inputs = feature_extractor(images=image, return_tensors="pt")
        print(f"Inputs Shape: {inputs['pixel_values'].shape}")  # Debug: Input tensor shape

        # Perform prediction
        print("Debug 2: Performing model prediction...")
        try:
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            print(f"Logits Shape: {logits.shape}")
            print(f"Predicted Index: {predicted_class_idx}")
      
