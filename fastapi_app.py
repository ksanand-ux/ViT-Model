import io
import os

import boto3
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import models, transforms

app = FastAPI()

# S3 Configuration
s3_bucket = "e-see-vit-model"
s3_key = "models/fine_tuned_vit_imagenet100.pth"
local_model_path = "fine_tuned_vit_imagenet100.pth"

# Initialize S3 Client
s3 = boto3.client("s3")

def is_model_updated():
    """Check if a newer model exists in S3."""
    s3_metadata = s3.head_object(Bucket=s3_bucket, Key=s3_key)
    s3_last_modified = s3_metadata["LastModified"].timestamp()
    
    if os.path.exists(local_model_path):
        local_last_modified = os.path.getmtime(local_model_path)
        return s3_last_modified > local_last_modified  # True if S3 model is newer
    return True  # If model doesn't exist locally, download it

def download_model():
    """Download the latest model from S3."""
    print("🔄 Downloading new model from S3...")
    s3.download_file(s3_bucket, s3_key, local_model_path)

def load_model():
    """Load the model into memory."""
    global model, class_labels
    state_dict = torch.load(local_model_path, map_location="cpu")
    
    # Load ViT model with updated classification head
    model = models.vit_b_16(pretrained=False)  # Don't load default weights
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, 100)  # 100 classes for ImageNet-100
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model Loaded & Ready for Inference!")

    # Load class labels
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
        # Add actual ImageNet-100 labels here...
    ]

# Step 1: Ensure we have the latest model on startup
if is_model_updated():
    download_model()
load_model()

# Image Preprocessing
def preprocess_image(image_bytes):
    """Preprocess uploaded image before feeding into the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

@app.get("/")
def read_root():
    return {"message": "ViT Model API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Check for new model before making predictions"""
    if is_model_updated():
        download_model()
        load_model()

    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)

        # Get model prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = output.argmax(dim=1).item()  # Get class index

        # Map index to label
        predicted_label = class_labels[predicted_class]

        return {"prediction": predicted_label}
    except Exception as e:
        return {"error": str(e)}
