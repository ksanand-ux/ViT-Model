import onnxruntime as ort
import numpy as np

# Load the ONNX model
ort_session = ort.InferenceSession("vit_model.onnx")

# Create a random test input (batch size 1, 3-channel image of 224x224)
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = ort_session.run(None, {"input": dummy_input})

# Print the output shape
print("ONNX Inference Output Shape:", outputs[0].shape)
