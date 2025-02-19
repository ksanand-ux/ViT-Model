import torch
from torchvision import models

# Load your ViT model
model = models.vit_b_16(pretrained=True)
model.eval()  # Set to evaluation mode

# Convert to TorchScript
scripted_model = torch.jit.script(model)

# Save the TorchScript model
scripted_model.save("vit_model_scripted.pt")

print("TorchScript Model Saved: vit_model_scripted.pt")
