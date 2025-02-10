import io
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

# Load the model
model_path = "vit_cifar10.pth"
model_state = torch.load(model_path, map_location=torch.device("cpu"))

# Fix mismatched keys
new_state_dict = OrderedDict()
for key, value in model_state.items():
    new_key = key.replace("heads.", "heads.head.")  # Adjust key names
    new_state_dict[new_key] = value

# Initialize the model (adjust based on your architecture)
from torchvision.models import vit_b_16  # Example ViT model

model = vit_b_16(pretrained=False, num_classes=10)  # Adjust if needed
model.load_state_dict(new_state_dict, strict=False)
model.eval()

@app.get("/")
def read_root():
    return {"message": "ViT Model API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    output = model(image_tensor.unsqueeze(0))  # Add batch dimension
    _, predicted_class = torch.max(output, 1)
    return {"prediction": predicted_class.item()}

# Function to preprocess image (adjust based on model requirements)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)

# Run API: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
