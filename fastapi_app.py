import os

import boto3
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Constants
BUCKET_NAME = "e-see-vit-model"
MODEL_KEY = "models/fine_tuned_vit_imagenet100.onnx"
LOCAL_MODEL_PATH = "fine_tuned_vit_imagenet100.onnx"

# Global variable for ONNX session
ort_session = None

app = FastAPI()

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
        download_model()
        print("Model Download Complete. Loading ONNX Model...")
        ort_session = ort.InferenceSession(LOCAL_MODEL_PATH)
        print("ONNX Model Loaded & Ready for Inference!")  # Debug End
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
    
    # Convert image to NumPy array
    image = np.array(image).astype(np.float32)
    print(f"Image Array Shape Before Transpose: {image.shape}")

    # Handle Grayscale or RGBA images
    if image.ndim == 2:  # Grayscale to RGB
        print("Converting Grayscale to RGB...")
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA to RGB
        print("Converting RGBA to RGB...")
        image = image[..., :3]

    # Transpose HWC to CHW
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    print(f"Image Array Shape After Transpose: {image.shape}")
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print(f"Final Input Tensor Shape: {image.shape}")
    print(f"Final Input Tensor Data Type: {image.dtype}")
    print(f"Final Input Tensor Values: {image}")

    return image

# Predict using ONNX model
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"Received File: {file.filename}")
        
        # Read image
        image = Image.open(file.file).convert("RGB")
        
        # Debugging Statements - Image Details
        print(f"Image Mode After Conversion: {image.mode}")
        print(f"Image Size After Resize: {image.size}")
        print(f"Image Data Type Before Preprocess: {type(image)}")
        
        # Preprocess the image
        input_tensor = preprocess_image(image)
        print(f"Input Tensor Shape Before Inference: {input_tensor.shape}")
        print(f"Input Tensor Data Type: {input_tensor.dtype}")
        print(f"Input Tensor Values: {input_tensor}")
        
        # Run inference
        outputs = ort_session.run(None, {"input": input_tensor})
        print(f"ONNX Model Output: {outputs}")
        print("ONNX Model Output Shape:", outputs[0].shape if outputs else "Empty Output")
        
        # Get prediction
        pred = int(np.argmax(outputs[0]))
        print(f"Predicted Class: {pred}")
        
        return JSONResponse({"prediction": pred})
    
    except Exception as e:
        print(f"Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

# Root endpoint to check if API is running
@app.get("/")
def root():
    return {"message": "ViT Model API is running with ONNX!"}
