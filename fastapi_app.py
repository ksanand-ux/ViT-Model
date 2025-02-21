import os

import boto3
import numpy as np
import onnxruntime as ort
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

# ImageNet-100 Class Labels (Partial for brevity)
CLASS_NAMES = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
    'electric_ray', 'stingray', 'cock', 'hen', 'ostrich'
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
        print("Starting model loading process...")  # Debug Start
        if not os.path.exists(LOCAL_MODEL_PATH):
            download_model()
        print("Model Download Complete. Loading ONNX Model...")
        ort_session = ort.InferenceSession(LOCAL_MODEL_PATH)
        print("ONNX Model Loaded & Ready for Inference!")  # Debug End
        
        # Debug: Input and Output Names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        print(f"ONNX Model Input Name: {input_name}")
        print(f"ONNX Model Output Name: {output_name}")
    except Exception as e:
        print(f"Error Loading Model: {e}")

# Call the function to load the model at startup
load_model()

# Preprocess image for ONNX model
def preprocess_image(image: Image.Image) -> np.ndarray:
    print("Preprocessing Image...")

    # Resize to 224x224
    image = image.resize((224, 224))
    print(f"Image Size After Resize: {image.size}")
    
    # Convert image to NumPy array and Normalize to [0, 1]
    image = np.array(image).astype(np.float32) / 255.0  
    print(f"Image Array Shape Before Transpose: {image.shape}")
    print(f"Image Array Values (Sample): {image[0][0]}")
    print(f"Data Type After Normalization: {image.dtype}")

    # Handle Grayscale or RGBA images
    if image.ndim == 2:  # Grayscale to RGB
        print("Converting Grayscale to RGB...")
        image = np.stack([image] * 3, axis=-1).astype(np.float32)
    elif image.shape[2] == 4:  # RGBA to RGB
        print("Converting RGBA to RGB...")
        image = image[..., :3].astype(np.float32)
    print(f"Data Type After Handling Channels: {image.dtype}")

    # Normalize Using ImageNet Mean & Std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    # Ultimate Fix: Wrap with np.array() to Force float32
    image = np.array(image, dtype=np.float32, copy=False)
    print(f"Image Array After Normalization (Sample): {image[0][0]}")
    print(f"Data Type After ImageNet Normalization: {image.dtype}")

    # Transpose HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    # Ultimate Fix: Wrap with np.array() to Force float32
    image = np.array(image, dtype=np.float32, copy=False)
    print(f"Image Array Shape After Transpose: {image.shape}")
    print(f"Data Type After Transpose: {image.dtype}")
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    # Ultimate Fix: Wrap with np.array() to Force float32
    image = np.array(image, dtype=np.float32, copy=False)
    print(f"Final Input Tensor Shape: {image.shape}")
    print(f"Final Input Tensor Data Type: {image.dtype}")
    print(f"Final Input Tensor Values (Sample): {image[0][0][0]}")

    return image

# Predict using ONNX model
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"Received File: {file.filename}")
        
        # Read image
        image = Image.open(file.file).convert("RGB")
        
        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Ultimate Fix: Wrap with np.array() to Force float32
        input_tensor = np.array(input_tensor, dtype=np.float32, copy=False)
        print(f"Final Input Tensor Data Type Before Inference (Ultimate Fix): {input_tensor.dtype}")
        
        # Run inference
        print("Starting Inference...")
        outputs = ort_session.run(None, {"input": input_tensor})
        
        # Debugging Output Details
        print("ONNX Model Output:", outputs)
        print("ONNX Model Output Type:", type(outputs))
        print("ONNX Model Output Shape:", outputs[0].shape if outputs else "Empty Output")
        print("ONNX Model Output Values:", outputs[0] if outputs else "No Output")
        
        # Get prediction
        output_tensor = outputs[0]
        predicted_class_index = np.argmax(output_tensor, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        print(f"Predicted Class: {predicted_class_name}")
        
        return JSONResponse({"predicted_class": predicted_class_name})
    
    except Exception as e:
        print(f"Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

# Health Check Endpoint
@app.get("/health/")
async def health_check():
    return JSONResponse(content={"status": "healthy"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
