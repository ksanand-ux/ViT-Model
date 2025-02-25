import io
import os

import boto3
import numpy as np
import onnxruntime as ort
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

        # Load ONNX Model with Verbose Logging for Debugging
        options = ort.SessionOptions()
        options.log_severity_level = 0  # Enable verbose logging
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(LOCAL_MODEL_PATH, options, providers=providers)
        print(f"ONNX Model Loaded & Ready for Inference! Using Providers: {providers}")

        # Debugging Input and Output Details
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

def preprocess_image(image_bytes: bytes) -> ort.OrtValue:
    print("Preprocessing Image...")

    # Load and Convert to RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    print(f"Image Size After Resize: {image.size}")

    # Convert Image to Float32 Array Directly
    np_image = np.asarray(image, dtype=np.float32) / 255.0
    print(f"Image Array Shape Before Transpose: {np_image.shape}")
    print(f"Image Array Values (Sample): {np_image[0][0]}")

    # Normalize Using ImageNet Mean & Std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    np_image = (np_image - mean) / std
    print(f"Image Array After Normalization (Sample): {np_image[0][0]}")

    # Transpose to CHW Format
    np_image = np_image.transpose(2, 0, 1)
    print(f"Image Array Shape After Transpose: {np_image.shape}")

    # Add Batch Dimension
    input_tensor = np.expand_dims(np_image, axis=0)
    print(f"Final Input Tensor Shape: {input_tensor.shape}")
    print(f"Final Input Tensor Data Type: {input_tensor.dtype}")
    
    # Create OrtValue with Explicit float32 Specification
    ort_value = ort.OrtValue.ortvalue_from_numpy(input_tensor.astype(np.float32), 'cpu')
    print(f"OrtValue Data Type: {ort_value.data_type()}")

    return ort_value

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor.numpy()})

        output_tensor = outputs[0]
        predicted_class_index = np.argmax(output_tensor, axis=1)[0]
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
