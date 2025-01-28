from flask import Flask, request
from torchvision import transforms
from PIL import Image
import torch
from torchvision.models import vit_b_16

app = Flask(__name__)

# Define model path and class labels
model_path = "C:/Users/hello/OneDrive/Documents/Python/ViT Model/ViT-Model/vit_cifar10.pth"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load the model
print("=== Loading Trained Model ===")
try:
    # Initialize the ViT architecture
    model = vit_b_16(pretrained=False, num_classes=len(classes))

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Handle any prefix issues in state_dict keys
    if any(key.startswith("module.") for key in state_dict.keys()):
        print("Stripping 'module.' prefix from state_dict keys...")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return {"error": "Model could not be loaded. Please check your model file."}, 500

    try:
        file = request.files['file']
        img = Image.open(file).convert('RGB')
        preprocess = transforms.Compose([
