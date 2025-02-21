import io
import os

import boto3
import numpy as np
import onnx
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from onnx import numpy_helper  # 🚀 NEW: Direct ONNX Helper to Force Float32
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
        print("Starting model loading process...")
        if not os.path.exists(LOCAL_MODEL_PATH):
            download_model()
        print("Model Download Complete. Loading ONNX Model...")
        ort_session = ort.InferenceSession(LOCAL_MODEL_PATH)
        print("ONNX Model Loaded & Ready for Inference!")

        # Debug: Input and Output Names and Types
        input_name = ort_session.get_inputs()[0].name
        input_type = ort_session.get_inputs()[0].type
        output_name = ort_session.get_outputs()[0].name
        output_type = ort_session.get_outputs()[0].type
        print(f"ONNX Model Input Name: {input_name}, Type: {input_type}")
        print(f"ONNX Model Output Name: {output_name}, Type: {output_type}")

        # NEW: Validate ONNX Model Input Details
        input_details = ort_session.get_inputs()[0]
        print(f"ONNX Input Name: {input_details.name}")
        print(f"ONNX Input Shape: {input_details.shape}")
        print(f"ONNX Input Type: {input_details.type}")

    except Exception as e:
        print(f"Error Loading Model: {e}")

# Call the function to load the model at startup
load_model()

# Preprocess image for ONNX model
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    print("Preprocessing Image...")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error opening image: {e}")

    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    # NEW: Force Input Tensor to float32 with ONNX Helper
    final_image = np.zeros(image.shape, dtype=np.float32)
    np.copyto(final_image, image)
    onnx_tensor = numpy_helper.from_array(final_image)
    
    print(f"Final Input Tensor Shape: {final_image.shape}")
    print(f"Final Input Tensor Data Type: {final_image.dtype}")
    print(f"ONNX Tensor Data Type: {onnx_tensor.data_type}")
    print(f"Final Input Tensor Values (Sample): {final_image[0][0][0]}")

    return final_image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        input_name = ort_session.get_inputs()[0].name

        # NEW: Validate Input Tensor Before Inference
        print(f"Input Tensor (Before Inference): Shape={input_tensor.shape}, dtype={input_tensor.dtype}")
        outputs = ort_session.run(None, {input_name: input_tensor})

        output_tensor = outputs[0]
        print(f"Output Tensor Shape: {output_tensor.shape}, dtype={output_tensor.dtype}")

        predicted_class_index = np.argmax(output_tensor, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        print(f"Predicted Class: {predicted_class_name}")

        return JSONResponse({"predicted_class": predicted_class_name})

    except Exception as e:
        print(f"Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

@app.get("/health/")
async def health_check():
    return JSONResponse(content={"status": "healthy"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
