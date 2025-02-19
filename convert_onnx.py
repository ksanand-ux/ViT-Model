import torch
import torchvision.models as models

# Load ViT model
model = models.vit_b_16(pretrained=True)
model.eval()

# Dummy input tensor (batch of 1, 3-channel image of size 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Convert to ONNX
torch.onnx.export(model, dummy_input, "vit_model.onnx",
                  export_params=True,  # Store trained parameters
                  opset_version=11,  # ONNX version
                  do_constant_folding=True,  # Optimize graph
                  input_names=['input'], output_names=['output'])

print("ONNX Model Saved: vit_model.onnx")
