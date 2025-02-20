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

# ImageNet-100 Class Labels
CLASS_NAMES = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
    'electric_ray', 'stingray', 'cock', 'hen', 'ostrich',
    'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting',
    'robin', 'bulbul', 'jay', 'magpie', 'chickadee',
    'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl',
    'fire_salamander', 'smooth_newt', 'axolotl', 'bullfrog', 'tree_frog',
    'tailed_frog', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin',
    'box_turtle', 'banded_gecko', 'common_iguana', 'American_chameleon', 'whiptail',
    'agama', 'frilled_lizard', 'alligator_lizard', 'Gila_monster', 'green_lizard',
    'African_chameleon', 'Komodo_dragon', 'triceratops', 'African_crocodile', 'American_alligator'
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

# Debugging and Prediction Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"Received File: {file.filename}")
        
        # Read image
        image = Image.open(file.file).convert("RGB")
        input_tensor = preprocess_image(image)

        # Run inference
        outputs = ort_session.run([output_name], {input_name: input_tensor})

        # Debug: Output Details After Inference
        print("ONNX Model Output:", outputs)
        print("ONNX Model Output Shape:", outputs[0].shape if outputs else "Empty Output")
        print("ONNX Model Output Values:", outputs[0] if outputs else "No Output")

        # Check for NaN or Inf values
        if np.isnan(outputs[0]).any() or np.isinf(outputs[0]).any():
            print("⚠️ Warning: Output contains NaN or Inf values!")

        # Get prediction
        pred = int(np.argmax(outputs[0]))
        predicted_class_name = CLASS_NAMES[pred]
        print(f"Predicted Class: {predicted_class_name}")

        return JSONResponse({"predicted_class": predicted_class_name})
    
    except Exception as e:
        print(f"Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")
