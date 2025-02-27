import hashlib  # For consistent cache key generation
import io
import os

import boto3
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from redis import Redis  # Using Redis for caching

# Constants for S3 Bucket
BUCKET_NAME = "e-see-vit-model"
MODEL_KEY = "models/fine_tuned_vit_imagenet100_scripted.pt"
LOCAL_MODEL_PATH = "fine_tuned_vit_imagenet100_scripted.pt"

# Redis Configuration
REDIS_HOST = "redis-service"  # ClusterIP service name
REDIS_PORT = 6379
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Global Cache for Model
model_cache = None
app = FastAPI()

# Check Redis Connection
try:
    redis_client.ping()
    print("[REDIS] Connection Successful!")
except Exception as e:
    print(f"[REDIS] Connection Error: {e}")

# Download the latest TorchScript model from S3
def download_model():
    print("Downloading new TorchScript model from S3...")
    s3 = boto3.client('s3')
    s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
    print("Model Download Complete.")

# Load the TorchScript model with Caching
def load_model():
    global model_cache
    if model_cache is None:
        print("Loading TorchScript Model...")
        model_cache = torch.jit.load(LOCAL_MODEL_PATH)
        model_cache.eval()
    else:
        print("Using Cached Model")
    return model_cache

# Call the function to load the model at startup
load_model()

# Generate Consistent Cache Key Using Image Hash
def generate_cache_key(image_bytes: bytes) -> str:
    image_hash = hashlib.md5(image_bytes).hexdigest()
    cache_key = f"prediction:{image_hash}"
    print(f"[CACHE] Generated Cache Key: {cache_key}")
    return cache_key

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        cache_key = generate_cache_key(image_bytes)
        
        # Check Cache
        cached_prediction = redis_client.get(cache_key)
        if cached_prediction:
            print(f"[CACHE] Cache Hit! Key: {cache_key}, Value: {cached_prediction}")
            return JSONResponse({"predicted_class": cached_prediction})
        
        print("[CACHE] Cache Miss! Running Inference...")
        input_tensor = preprocess_image(image_bytes)

        # Using Cached Model
        model = load_model()
        with torch.no_grad():
            outputs = model(input_tensor)
        
        predicted_class_index = torch.argmax(outputs, dim=1).item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        # Store in Cache
        redis_client.set(cache_key, predicted_class_name)
        print(f"[CACHE] Cached Prediction for Key: {cache_key}, Value: {predicted_class_name}")

        return JSONResponse({"predicted_class": predicted_class_name})

    except Exception as e:
        print(f"Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")
