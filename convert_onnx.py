import torch
import torchvision.models as models

# Load the fine-tuned ViT model (do NOT use pretrained weights)
model = models.vit_b_16(pretrained=False)  # Don't load default weights
in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, 100)  # 100 classes for ImageNet-100

# Load the fine-tuned weights
model.load_state_dict(torch.load("fine_tuned_vit_imagenet100.pth", map_location="cpu"))
model.eval()

# Dummy input tensor (batch of 1, 3-channel image of size 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Debugging Statements for Verification
print("Model Input Shape:", dummy_input.shape)
print("Model Output Shape (Expected): (1, 100)")

# Export to ONNX Format
print("Exporting Model to ONNX Format...")
torch.onnx.export(
    model,
    dummy_input,
    "fine_tuned_vit_imagenet100.onnx",
    export_params=True,  # Store trained parameters
    opset_version=17,  # Latest ONNX compatibility
    do_constant_folding=True,  # Optimize computation graph
    input_names=["input"], 
    output_names=["output"],  # Naming inputs/outputs
    # Critical Fix: Correct Dynamic Axis
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  
    verbose=True  # Detailed export log for debugging
)

print("ONNX Model Exported Successfully: fine_tuned_vit_imagenet100.onnx")
