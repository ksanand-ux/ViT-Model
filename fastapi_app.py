import os

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI()

# S3 Bucket Configuration
BUCKET_NAME = "e-see-vit-model"
MODEL_KEY = "models/fine_tuned_vit_imagenet100.onnx"
LOCAL_MODEL_PATH = "fine_tuned_vit_imagenet100.onnx"
ort_session = None

# 🔄 Load the ONNX model
def load_model():
    global ort_session
    try:
        print("🔄 Starting model loading process...")
        print("🔄 Downloading new ONNX model from S3...")
        download_model()
        print("📁 Model Download Complete. Loading ONNX Model...")
        ort_session = ort.InferenceSession(LOCAL_MODEL_PATH)
        print("✅ ONNX Model Loaded & Ready for Inference!")
    except Exception as e:
        print(f"❌ Error Loading Model: {e}")

# 🔄 Download Model from S3
def download_model():
    import boto3
    s3 = boto3.client("s3")
    try:
        print(f"🔄 Downloading model from S3: {MODEL_KEY}")
        s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
        print("✅ Model Downloaded Successfully!")
    except Exception as e:
        print(f"❌ Error Downloading Model from S3: {e}")

load_model()

# 🔄 Preprocess Image
def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        print("🔄 Preprocessing Image...")
        # Resize, Convert to Float32, Normalize
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32)
        print(f"📏 Image Shape after Resize: {image_array.shape}")

        # Check if grayscale and convert to RGB
        if len(image_array.shape) == 2:
            print("⚠️ Grayscale image detected, converting to RGB...")
            image_array = np.stack([image_array] * 3, axis=-1)

        # Convert to Channel-First Format (C, H, W)
        image_array = np.transpose(image_array, (2, 0, 1))
        print(f"🔄 Transposed Image Shape (C, H, W): {image_array.shape}")

        # Add Batch Dimension
        input_tensor = np.expand_dims(image_array, axis=0)
        print(f"📦 Input Tensor Shape with Batch Dimension: {input_tensor.shape}")

        return input_tensor
    except Exception as e:
        print(f"❌ Error in Image Preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Image Preprocessing Error: {e}")

# 🔮 Run Inference
def run_inference(input_tensor: np.ndarray) -> np.ndarray:
    try:
        print(f"🔍 Input Tensor Shape: {input_tensor.shape}")
        input_name = ort_session.get_inputs()[0].name
        print(f"🔍 ONNX Model Input Name: {input_name}")

        outputs = ort_session.run(None, {input_name: input_tensor})
        print("🔍 ONNX Model Output:", outputs)
        print("🔍 ONNX Model Output Shape:", outputs[0].shape if outputs else "Empty Output")

        return outputs[0]
    except Exception as e:
        print(f"❌ Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {e}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print("🔄 Starting Prediction...")
        image = Image.open(file.file).convert("RGB")
        input_tensor = preprocess_image(image)

        # Run Inference
        outputs = run_inference(input_tensor)
        print(f"✅ Inference Completed. Output Shape: {outputs.shape}")

        # Get Top-5 Predictions
        top_5_indices = np.argsort(outputs[0])[::-1][:5]
        top_5_scores = outputs[0][top_5_indices].tolist()
        top_5_classes = top_5_indices.tolist()

        return {
            "predictions": [
                {"class": int(cls), "confidence": float(score)}
                for cls, score in zip(top_5_classes, top_5_scores)
            ]
        }
    except Exception as e:
        print(f"❌ Error in Prediction Endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

# Start FastAPI Server (uncomment when running locally)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)
