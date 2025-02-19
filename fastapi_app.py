import io
import os

import boto3
import numpy as np
import onnxruntime as ort
import redis
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
    """Check if a newer ONNX model exists in S3."""
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
    """Download the latest ONNX model from S3."""
    print("🔄 Downloading new ONNX model from S3...")
    s3.download_file(s3_bucket, s3_key, local_model_path)

def load_model():
    """Load the ONNX model into memory."""
    global ort_session, class_labels

    if not os.path.exists(local_model_path):
        print("❌ Model file missing! Downloading...")
        download_model()

    ort_session = ort.InferenceSession(local_model_path)
    print("✅ ONNX Model Loaded & Ready for Inference!")

    # ✅ **Class Labels (Ensure these match your dataset)**
    class_labels = [
        "bonnet, poke bonnet", "green mamba", "langur", "Doberman, Doberman pinscher", "gyromitra",
        "Saluki, gazelle hound", "vacuum, vacuum cleaner", "window screen", "cocktail shaker", "garden spider, Aranea diademata",
        "garter snake, grass snake", "carbonara", "pineapple, ananas", "computer keyboard, keypad", "tripod",
        "komondor", "American lobster, Northern lobster, Maine lobster, Homarus americanus", "bannister, banister, balustrade, balusters, handrail",
        "honeycomb", "tile roof", "papillon", "boathouse", "stinkhorn, carrion fungus",
        "jean, blue jean, denim", "Chihuahua", "Chesapeake Bay retriever", "robin, American robin, Turdus migratorius",
        "tub, vat", "Great Dane", "rotisserie", "bottlecap", "throne",
        "little blue heron, Egretta caerulea", "rock crab, Cancer irroratus", "Rottweiler", "lorikeet",
        "Gila monster, Heloderma suspectum", "head cabbage", "car wheel", "coyote, prairie wolf, brush wolf, Canis latrans",
        "moped", "milk can", "mixing bowl", "toy terrier", "chocolate sauce, chocolate syrup",
        "rocking chair, rocker", "wing", "park bench", "ambulance", "football helmet",
        "leafhopper", "cauliflower", "pirate, pirate ship", "purse", "hare",
        "lampshade, lamp shade", "fiddler crab", "standard poodle", "Shih-Tzu", "pedestal, plinth, footstall",
        "gibbon, Hylobates lar", "safety pin", "English foxhound", "chime, bell, gong",
        "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
        "bassinet", "wild boar, boar, Sus scrofa", "theater curtain, theatre curtain", "dung beetle",
        "hognose snake, puff adder, sand viper", "Mexican hairless", "mortarboard", "Walker hound, Walker foxhound",
        "red fox, Vulpes vulpes", "modem", "slide rule, slipstick", "walking stick, walkingstick, stick insect",
        "cinema, movie theater, movie theatre, movie house, picture palace", "meerkat, mierkat",
        "kuvasz", "obelisk", "harmonica, mouth organ, harp, mouth harp", "sarong",
        "mousetrap", "hard disc, hard disk, fixed disk", "American coot, marsh hen, mud hen, water hen, Fulica americana",
        "reel", "pickup, pickup truck", "iron, smoothing iron", "tabby, tabby cat", "ski mask",
        "vizsla, Hungarian pointer", "laptop, laptop computer", "stretcher", "Dutch oven",
        "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus", "boxer", "gasmask, respirator, gas helmet",
        "goose", "borzoi, Russian wolfhound"
    ]

# Ensure latest ONNX model is downloaded on startup
if is_model_updated():
    download_model()
load_model()

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
    """Check for new ONNX model before making predictions"""
    if is_model_updated():
        download_model()
        load_model()

    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        print("DEBUG: Input tensor shape:", input_tensor.shape)  # Debugging

        # Run ONNX inference
        outputs = ort_session.run(None, {"input": input_tensor})

        print("DEBUG: Model output:", outputs)  # Debugging
        print("DEBUG: Model output shape:", outputs[0].shape)

        # **Fix potential indexing error**
        if len(outputs) == 0 or len(outputs[0]) == 0:
            return {"error": "Model output is empty"}

        predicted_class = int(np.argmax(outputs[0][0]))  # Select first batch

        print("DEBUG: Predicted class index:", predicted_class)  # Debugging

        if predicted_class >= len(class_labels):
            return {"error": f"Predicted class index {predicted_class} out of range"}

        predicted_label = class_labels[predicted_class]

        return {"prediction": predicted_label}
    except Exception as e:
        print("ERROR:", str(e))  # Debugging
        return {"error": str(e)}
