import io
import os

import boto3
import onnxruntime as ort
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Constants for S3 Bucket
BUCKET_NAME = "e-see-vit-model"
MODEL_KEY = "models/fine_tuned_vit_imagenet100.onnx"
LOCAL_MODEL_PATH = "fine_tuned_vit_imagenet100.onnx"

# Global variable for ONNX session
ort_session = None
app = FastAPI()

# ImageNet-100 Class Labels (Shortened for Readability)
CLASS_NAMES = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 
    'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling'
]

# Download the latest ONNX model from S3
def download_model():
    print("Downloading new ONNX model from S3...")
    s3 = boto3.client('s3')
    s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
    print("Model Download Complete.")

# Load the ONNX model
def load_model():
    global ort_session
    try:
        print("Starting model loading process...")
        if not os.path.exists(LOCAL_MODEL_PATH):
            download_model()
        print("Model Download Complete. Loading ONNX Model...")

        # Load ONNX Model with GPU Support if Available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(LOCAL_MODEL_PATH, providers=providers)
        print(f"ONNX Model Loaded & Ready for Inference! Using Providers: {providers}")

        # Debug: Input and Output Names and Types
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        input_type = ort_session.get_inputs()[0].type
        output_name = ort_session.get_outputs()[0].name
        output_type = ort_session.get_outputs()[0].type
        print(f"ONNX Model Input Name: {input_name}, Type: {input_type}, Shape: {input_shape}")
        print(f"ONNX Model Output Name: {output_name}, Type: {output_type}")

    except Exception as e:
        print(f"Error Loading Model: {e}")
        raise RuntimeError(f"Model Loading Error: {e}")

# Call the function to load the model at startup
load_model()

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    print("Preprocessing Image...")

    # 🔥 Directly Use PyTorch to Load Image
    # Completely Bypass NumPy to Avoid Float64
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_tensor = torch.frombuffer(image.tobytes(), dtype=torch.uint8)  # Load as uint8 to prevent float64
    image_tensor = image_tensor.view(224, 224, 3).permute(2, 0, 1).unsqueeze(0)  # CHW and add batch dimension

    # Convert to float32 and Normalize
    image_tensor = image_tensor.to(torch.float32) / 255.0  # Convert to float32 here

    # Normalize Using ImageNet Mean & Std
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    print(f"Final Input Tensor Shape: {image_tensor.shape}")
    print(f"Final Input Tensor Data Type: {image_tensor.dtype}")

    return image_tensor

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # 🔥 Directly Use OrtValue to Bypass NumPy
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: input_tensor.cpu().numpy()}
        outputs = ort_session.run(None, ort_inputs)

        output_tensor = outputs[0]
        predicted_class_index = torch.argmax(torch.tensor(output_tensor), dim=1).item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        print(f"Predicted Class: {predicted_class_name}")

        return JSONResponse({"predicted_class": predicted_class_name})

    except Exception as e:
        print(f"Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

@app.get("/health")
def health_check():
    return JSONResponse({"status": "healthy"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
