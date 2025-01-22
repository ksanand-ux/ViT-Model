import io

import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 class labels
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load the trained model
print("=== Loading Trained Model ===")
vit_model = vit_b_16(weights=None)  # Do not load ImageNet pretrained weights
vit_model.heads = torch.nn.Linear(vit_model.heads[0].in_features, len(class_labels))  # Adjust for 10 classes
vit_model.load_state_dict(torch.load("vit_cifar10.pth", map_location=device))  # Load trained weights
vit_model = vit_model.to(device)
vit_model.eval()
print("=== Model Loaded Successfully ===")

# Image preprocessing transforms (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/')
def home():
    return "Welcome to the Vision Transformer CIFAR-10 Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug: Confirm route is hit
        print("=== /predict endpoint accessed ===")

        # Ensure an image file is in the request
        if 'file' not in request.files:
            print("Error: No file provided")
            return jsonify({"error": "No file provided"}), 400

        # Read the uploaded image
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0).to(device)
        print("=== Image Preprocessed ===")

        # Perform prediction
        with torch.no_grad():
            outputs = vit_model(input_tensor)
            predicted_class_idx = outputs.argmax(dim=1).item()
        print(f"Predicted Index: {predicted_class_idx}")

        # Map index to class label
        predicted_label = class_labels[predicted_class_idx]
        print(f"Predicted Label: {predicted_label}")

        # Return response
        response = {
            "predicted_class_index": predicted_class_idx,
            "predicted_label": predicted_label
        }
        print(f"Response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    import os
    print(f"Running app from: {os.path.abspath(__file__)}")
    print(f"Current Working Directory: {os.getcwd()}")
    app.run(host='0.0.0.0', port=5000, debug=True)
