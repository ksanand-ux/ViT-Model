import io

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

# Load your model from S3
model_path = "vit_cifar10.pth"
model_state = torch.load(model_path, map_location=torch.device("cpu"))

# Define the model architecture to match the saved state
from torchvision import models

model = models.vit_b_16(pretrained=False)  # Ensure this matches your trained model
model.load_state_dict(model_state)
model.eval()

@app.get("/")
def read_root():
    return {"message": "ViT Model API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    output = model(image_tensor.unsqueeze(0))  # Add batch dimension
    probabilities = F.softmax(output, dim=1)  # Convert to probabilities
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return {"prediction": predicted_class}

# Function to preprocess image (Implement based on your model's requirements)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)
