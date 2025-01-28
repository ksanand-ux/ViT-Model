import torch
from flask import Flask, request
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

app = Flask(__name__)

# Define model path and class labels
model_path = "C:/Users/hello/OneDrive/Documents/Python/ViT Model/ViT-Model/vit_cifar10.pth"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load the model
print("=== Loading Trained Model ===")
try:
    # Initialize the ViT architecture
    model = vit_b_16(pretrained=False)

    # Replace the final classification head
    num_features = model.heads[-1].in_features  # Access in_features of the last layer
    model.heads[-1] = torch.nn.Linear(num_features, len(classes))  # Update the number of classes

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Handle any prefix issues in state_dict keys
    if any(key.startswith("module.") for key in state_dict.keys()):
        print("Stripping 'module.' prefix from state_dict keys...")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load the state_dict into the adjusted model
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")  # Debug message for terminal
    model = None  # Prevent predictions if model loading fails


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return {"error": "Model could not be loaded. Please check your model file."}, 500

    try:
        file = request.files['file']
        img = Image.open(file).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
        predicted_class_index = output.argmax().item()

        return {
            "predicted_class_index": predicted_class_index,
            "predicted_label": classes[predicted_class_index]
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True)
