import io
import os
import boto3
import redis
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

# ✅ **S3 Configuration**
s3_bucket = "e-see-vit-model"
s3_key = "models/fine_tuned_vit_imagenet100.onnx"
local_model_path = "fine_tuned_vit_imagenet100.onnx"

# ✅ **Initialize AWS S3 Client**
s3 = boto3.client("s3")

# ✅ **Redis Connection (Kubernetes Service)**
redis_client = redis.Redis(host="redis-service", port=6379, db=0, decode_responses=True)

def is_model_updated():
    """🔹 Check if a newer ONNX model exists in S3."""
    try:
        s3_metadata = s3.head_object(Bucket=s3_bucket, Key=s3_key)
        s3_last_modified = s3_metadata["LastModified"].timestamp()

        if os.path.exists(local_model_path):
            local_last_modified = os.path.getmtime(local_model_path)
            return s3_last_modified > local_last_modified  # True if S3 model is newer
        return True  # If model doesn't exist locally, download it
    except Exception as e:
        print(f"⚠️ Error checking S3 model: {e}")
        return False

def download_model():
    """🔄 Download the latest ONNX model from S3."""
    print("🔄 Downloading new ONNX model from S3...")
    s3.download_file(s3_bucket, s3_key, local_model_path)

def load_model():
    """✅ Load the ONNX model into memory."""
    global ort_session, class_labels
    ort_session = ort.InferenceSession(local_model_path)
    print("✅ ONNX Model Loaded & Ready for Inference!")

    # **Class labels (Replace with actual ImageNet-100 labels)**
    class_labels = ["label_1", "label_2", "label_3", "label_4", "label_5"]  # TODO: Add correct labels

# 🔹 **Ensure the latest ONNX model is downloaded on startup**
if is_model_updated():
    download_model()
load_model()

# ✅ **Image Preprocessing**
def preprocess_image(image_bytes):
    """Preprocess uploaded image before feeding into ONNX model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/")
def read_root():
    return {"message": "ViT Model API is running with ONNX!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """✅ Check for new ONNX model before making predictions"""
    if is_model_updated():
        download_model()
        load_model()

    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # **🔹 Caching Mechanism Using Redis**
        cache_key = f"pred_{hash(image_bytes)}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            print("✅ Cache Hit! Returning cached result.")
            return {"prediction": cached_result}

        # **Run ONNX inference**
        outputs = ort_session.run(None, {"input": input_tensor})

        # **Get predicted class**
        predicted_class = np.argmax(outputs[0])

        # **Map index to label**
        predicted_label = class_labels[predicted_class]

        # **Store result in Redis cache (expires in 10 mins)**
        redis_client.setex(cache_key, 600, predicted_label)

        return {"prediction": predicted_label}
    except Exception as e:
        return {"error": str(e)}
