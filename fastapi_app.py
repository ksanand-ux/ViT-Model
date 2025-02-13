import io

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models

app = FastAPI()

# Define CIFAR-10 class labels
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load Vision Transformer (ViT) model architecture
model = models.vit_b_16(weights=None)  # Initialize ViT model

# Modify model for CIFAR-10 (10 output classes)
num_classes = 10
in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, num_classes)

# Load trained model weights
model_path = "fine_tuned_vit.pth"

try:
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Handle renamed keys if necessary
    if "heads.weight" in state_dict and "heads.bias" in state_dict:
        state_dict["heads.head.weight"] = state_dict.pop("heads.weight")
        state_dict["heads.head.bias"] = state_dict.pop("heads.bias")

    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Image Preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

@app.get("/")
def read_root():
    return {"message": "ViT Model API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)

        # Get model prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = output.argmax(dim=1).item()  # Get class index

        # Map index to label
        predicted_label = class_labels[predicted_class]

        return JSONResponse(content={"prediction": predicted_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
