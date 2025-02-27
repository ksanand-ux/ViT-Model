import hashlib
import io
import os

import boto3
import numpy as np
import redis
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Constants for S3 Bucket
BUCKET_NAME = "e-see-vit-model"
MODEL_KEY = "models/fine_tuned_vit_imagenet100_scripted.pt"
LOCAL_MODEL_PATH = "fine_tuned_vit_imagenet100_scripted.pt"

# Global Cache for Model and Redis Connection
model_cache = None
redis_cache = redis.Redis(host='redis-service', port=6379, db=0)
app = FastAPI()

# ImageNet-100 Class Labels
CLASS_NAMES = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich", "brambling",
    "goldfinch", "house_finch", "junco", "indigo_bunting", "robin", 
    "bulbul", "jay", "magpie", "chickadee", "water_ouzel", "kite", 
    "bald_eagle", "vulture", "great_grey_owl", "fire_salamander", 
    "smooth_newt", "tailed_frog", "american_alligator", "green_iguana",
    "african_chameleon", "komodo_dragon", "african_crocodile", 
    "american_chameleon", "agama", "frilled_lizard", "alligator_lizard", 
    "gila_monster", "green_lizard", "african_chameleon", "komodo_dragon",
    "triceratops", "thunder_snake", "ringneck_snake", "hognose_snake",
    "green_snake", "king_snake", "garter_snake", "water_snake",
    "vine_snake", "night_snake", "boa_constrictor", "rock_python", 
    "indian_cobra", "green_mamba", "sea_snake", "horned_viper", 
    "diamondback", "sidewinder", "trilobite", "harvestman", "scorpion", 
    "black_and_gold_garden_spider", "barn_spider", "garden_spider", 
    "black_widow", "tarantula", "wolf_spider", "tick", "centipede", 
    "black_grouse", "ptarmigan", "ruffed_grouse", "prairie_chicken", 
    "peacock", "quail", "partridge", "african_grey", "macaw", "sulphur_crested_cockatoo", 
    "lorikeet", "coucal", "bee_eater", "hornbill", "hummingbird", 
    "jacamar", "toucan", "drake", "red_breasted_merganser", "goose", 
    "black_swan", "white_stork", "black_stork", "spoonbill", "flamingo", 
    "little_blue_heron", "bittern", "crane", "limpkin", "american_coot", 
    "bustard", "ruddy_turnstone", "red_backed_sandpiper", "redshank"
]

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

# Cache Check & Update Functions
def check_cache(image_hash):
    cached_result = redis_cache.get(image_hash)
    if cached_result:
        print("Cache Hit")
        return cached_result.decode('utf-8')
    else:
        print("Cache Miss")
        return None

def update_cache(image_hash, prediction):
    redis_cache.setex(image_hash, 3600, prediction)

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    print("Preprocessing Image...")

    # Load and Convert to RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    print(f"Image Size After Resize: {image.size}")

    # Direct Tensor Creation Without Conversion
    np_image = np.array(image) / 255.0
    print(f"Image Array Shape Before Transpose: {np_image.shape}")

    # Transpose to CHW format
    np_image = np_image.transpose(2, 0, 1)
    print(f"Image Array Shape After Transpose: {np_image.shape}")

    # Normalize Using ImageNet Mean & Std
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    np_image = (np_image - mean) / std

    # Final Conversion to float32 Just Before Inference
    input_tensor = torch.tensor(np_image, dtype=torch.float32).unsqueeze(0)
    return input_tensor

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # Image Hash for Caching
        image_hash = hashlib.md5(image_bytes).hexdigest()
        cached_result = check_cache(image_hash)

        if cached_result:
            return JSONResponse({"predicted_class": cached_result})

        # Using Cached Model
        model = load_model()
        with torch.no_grad():
            outputs = model(input_tensor)
        
        predicted_class_index = torch.argmax(outputs, dim=1).item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        update_cache(image_hash, predicted_class_name)
        print(f"Predicted Class: {predicted_class_name}")

        return JSONResponse({"predicted_class": predicted_class_name})

    except Exception as e:
        print(f"Error During Inference: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

@app.get("/health")
def health_check():
    return JSONResponse({"status": "healthy"})

if __name__ == "__main__":
    import multiprocessing
    workers = (2 * multiprocessing.cpu_count()) + 1
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=workers)