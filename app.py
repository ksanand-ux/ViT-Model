import torch
from flask import Flask, request
from PIL import Image
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__)

# Define model and class labels
model_path = "C:/Users/hello/OneDrive/Documents/Python/ViT Model/ViT-Model/vit_cifar10.pth"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # Update this if needed

# Load the model
print("Loading the model...")
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(model, torch.nn.Module):
        model.eval()  # If it’s a full model
    else:
        from torchvision.models import \
            vit_b_16  # Example architecture, update if needed
        model = vit_b_16(pretrained=False, num_classes=len(classes))  # Adjust num_classes
        model.load_state_dict(model)
        model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return {"error": "Model could not be loaded. Please check your model file."}, 500

    try:
        # Step 1: Receive and process the image
        file = request.files['file']
        img = Image.open(file).convert('RGB')
        print("Image received and converted to RGB.")  # Debug

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(img).unsqueeze(0)

        # Step 2: Get predictions
        with torch.no_grad():
            output = model(input_tensor)
        predicted_class_index = output.argmax().item()
        print(f"Predicted class index: {predicted_class_index}")  # Debug

        # Step 3: Return the result
        return {
            "predicted_class_index": predicted_class_index,
            "predicted_label": classes[predicted_class_index]
        }

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return {"error": str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True)
