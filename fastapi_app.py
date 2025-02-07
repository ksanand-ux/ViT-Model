import io

import torch
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Load your model from S3
model_path = "vit_cifar10.pth"
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

@app.get("/")
def read_root():
    return {"message": "ViT Model API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)  # Implement preprocessing
    output = model(image_tensor)
    return {"prediction": output.tolist()}

# Function to preprocess image (Implement based on your model's requirements)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    # Apply your preprocessing steps (resize, normalize, etc.)
    return image

# Run the API with: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
