import io
import os

import boto3
import numpy as np
import onnxruntime as ort
import torch  # 🚀 Ultimate Fix: Use PyTorch for Guaranteed float32
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

# ImageNet-100 Class Labels
CLASS_NAMES = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
    'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
    'zabra finch', 'American robin', 'bulbul', 'goldfinch', 'house finch',
    'junco', 'kite', 'bald eagle', 'vulture', 'great grey owl', 'European nightjar',
    'albatross', 'auk', 'bittern', 'American bittern', 'bustard', 'quail',
    'partridge', 'African grey', 'macaw', 'sulphur-crested cockatoo', 'lorikeet',
    'coucal', 'cuckoo', 'yellow billed cuckoo', 'European cuckoo', 'owl',
    'great horned owl', 'hummingbird', 'jacamar', 'kingfisher', 'hoopoe',
    'hornbill', 'pelican', 'king penguin', 'albatross', 'auk', 'bittern',
    'American bittern', 'bustard', 'quail', 'partridge', 'African grey',
    'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'cuckoo',
    'yellow billed cuckoo', 'European cuckoo', 'owl', 'great horned owl',
    'hummingbird', 'jacamar', 'kingfisher', 'hoopoe', 'hornbill', 'pelican',
    'king penguin', 'spoonbill', 'white stork', 'black stork', 'crane bird',
    'common crane', 'blue heron', 'great white heron', 'green heron', 'mallard',
    'American black duck', 'teal duck', 'red-breasted merganser', 'wild turkey',
    'guinea', 'peacock', 'pigeon', 'European turtle dove', 'dove', 'Arctic tern',
    'thick-billed murre', 'long-tailed jaeger', 'skua', 'black-backed gull',
    'herring gull', 'laughing gull', 'tern', 'chickadee', 'nuthatch', 'wren',
    'house wren', 'goldcrest', 'kinglet', 'red-backed shrike', 'loggerhead shrike',
    'starling', 'Northern mockingbird', 'thrush', 'American robin', 'European robin',
    'blackbird', 'magpie', 'jay', 'blue jay', 'crow', 'raven', 'cormorant',
    'cormorant', 'shag', 'bee eater', 'cockatoo', 'grey gull', 'puffin',
    'wild goose', 'snow goose', 'Canada goose', 'Barnacle goose', 'duck',
    'red-necked grebe', 'great crested grebe', 'great egret', 'bittern',
    'crane', 'coot', 'moorhen', 'flamingo', 'ostrich', 'woodpecker',
    'kingfisher', 'pigeon', 'dove', 'parrot'
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

    except Exception as e:
        print(f"Error Loading Model: {e}")

# Call the function to load the model at startup
load_model()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    print("Preprocessing Image...")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    
    # 🔥 Ultimate Fix: Use PyTorch for Reliable float32 Handling
    torch_tensor = torch.tensor(np.array(image) / 255.0, dtype=torch.float32)
    torch_tensor = torch_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW and add batch dimension

    # Normalize Using ImageNet Mean & Std
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    torch_tensor = (torch_tensor - mean) / std
    
    print(f"Final Input Tensor Shape: {torch_tensor.shape}")
    print(f"Final Input Tensor Data Type: {torch_tensor.dtype}")
    
    # Convert back to NumPy
    input_tensor = torch_tensor.numpy()

    return input_tensor

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})

        output_tensor = outputs[0]
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
