import io

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

# Load the model
model_path = "vit_cifar10.pth"
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Define class labels for CIFAR-10
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

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
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)

    # Get model prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()  # Get class index
    
    # Map index to label
    predicted_label = class_labels[predicted_class]

    return {"prediction": predicted_label}
