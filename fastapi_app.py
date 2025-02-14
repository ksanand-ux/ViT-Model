import io
import os

import boto3
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import models, transforms

app = FastAPI()

# S3 Configuration
s3_bucket = "e-see-vit-model"
s3_key = "models/fine_tuned_vit_imagenet100.pth"
local_model_path = "fine_tuned_vit_imagenet100.pth"

# Initialize S3 Client
s3 = boto3.client("s3")

def is_model_updated():
    """Check if a newer model exists in S3."""
    s3_metadata = s3.head_object(Bucket=s3_bucket, Key=s3_key)
    s3_last_modified = s3_metadata["LastModified"].timestamp()
    
    if os.path.exists(local_model_path):
        local_last_modified = os.path.getmtime(local_model_path)
        return s3_last_modified > local_last_modified  # True if S3 model is newer
    return True  # If model doesn't exist locally, download it

def download_model():
    """Download the latest model from S3."""
    print("🔄 Downloading new model from S3...")
    s3.download_file(s3_bucket, s3_key, local_model_path)

def load_model():
    """Load the model into memory."""
    global model, class_labels
    state_dict = torch.load(local_model_path, map_location="cpu")
    
    # Load ViT model with updated classification head
    model = models.vit_b_16(pretrained=False)  # Don't load default weights
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, 100)  # 100 classes for ImageNet-100
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model Loaded & Ready for Inference!")

    # Define ImageNet-100 class labels
    class_labels = [str(i) for i in range(100)]  # Replace with actual class names if available

# Step 1: Ensure we have the latest model on startup
if is_model_updated():
    download_model()
load_model()

# Image Preprocessing
def preprocess_image(image_bytes):
    """Preprocess uploaded image before feeding into the model."""
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
    """Check for new model before making predictions"""
    if is_model_updated():
        download_model()
        load_model()

    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)

        # Get model prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = output.argmax(dim=1).item()  # Get class index

        # Map index to label
        predicted_label = class_labels[predicted_class]

        return {"prediction": predicted_label}
    except Exception as e:
        return {"error": str(e)}
