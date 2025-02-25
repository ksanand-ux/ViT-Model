import onnx

# Load the ONNX model
model_path = "fine_tuned_vit_imagenet100.onnx"
model = onnx.load(model_path)

# Get the input tensor and change its data type to float64
input_tensor = model.graph.input[0]
print(f"Original Input Type: {input_tensor.type.tensor_type.elem_type}")

# ONNX TensorProto data type for float64 is 11
input_tensor.type.tensor_type.elem_type = 11

# Save the modified ONNX model
modified_model_path = "fine_tuned_vit_imagenet100_float64.onnx"
onnx.save(model, modified_model_path)
print(f"Modified ONNX Model saved at {modified_model_path}")