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

# Convert to ONNX with correct batch dimension fix
torch.onnx.export(
    model,
    dummy_input,
    "fine_tuned_vit_imagenet100.onnx",
    export_params=True,  # Store trained parameters
    opset_version=17,  # Ensure ONNX compatibility
    do_constant_folding=True,  # Optimize computation graph
    input_names=["input"], 
    output_names=["output"],  # Naming inputs/outputs
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Allow dynamic batching
)

print("ONNX Model Exported: fine_tuned_vit_imagenet100.onnx")
