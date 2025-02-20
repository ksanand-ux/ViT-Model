import os
from io import BytesIO

import numpy as np
import onnxruntime as ort
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

# Model Path and S3 Config
MODEL_NAME = "fine_tuned_vit_imagenet100.onnx"
MODEL_URL = f"https://e-see-vit-model.s3.amazonaws.com/models/{MODEL_NAME}"
LOCAL_MODEL_PATH = f"./{MODEL_NAME}"

# Auto-download model if not present
def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Downloading new ONNX model from S3...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(LOCAL_MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            print("Failed to download the model.")
            raise HTTPException(status_code=500, detail="Model download failed.")
    else:
        print("Model already exists locally.")

# 🔄 Load the ONNX model
def load_model():
    global ort_session
    download_model()
    ort_session = ort.InferenceSession(LOCAL_MODEL_PATH)
    print("ONNX Model Loaded & Ready for Inference!")

load_model()

# 🔍 Preprocess image
def preprocess_image(image: Image.Image):
    print(f"Original Image Size: {image.size}")
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize to 224x224
    input_tensor = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # Change to (C, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    print(f"Preprocessed Input Tensor Shape: {input_tensor.shape}")
    return input_tensor

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Ensure file is an image
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="File must be an image (PNG, JPG, or JPEG).")
    
    # Load image
    image = Image.open(BytesIO(await file.read()))
    input_tensor = preprocess_image(image)
    
    # Debug: Check input tensor shape
    print(f"Input Tensor Shape: {input_tensor.shape}")

    # Check Model Input Name
    input_name = ort_session.get_inputs()[0].name
    print(f"Model Input Name: {input_name}")
    
    # Run Inference
    outputs = ort_session.run(None, {input_name: input_tensor})

    # Debug: Check raw model output
    print(f"ONNX Model Raw Output: {outputs}")

    # Check if output is not empty
    if len(outputs) == 0:
        raise HTTPException(status_code=500, detail="Model output is empty.")
    
    # Check Output Shape
    print(f"ONNX Model Output Shape: {outputs[0].shape}")
    
    # Get prediction
    predictions = outputs[0]
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    
    return JSONResponse({"predicted_class": predicted_class})
