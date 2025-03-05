import onnx

# Load the ONNX model
model_path = "fine_tuned_vit_imagenet100_float64.onnx"
model = onnx.load(model_path)

# Get the first input's name and change its type to float64
input_tensor = model.graph.input[0]
input_tensor.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE

# Save the modified model
modified_model_path = "fine_tuned_vit_imagenet100_float64_fixed.onnx"
onnx.save(model, modified_model_path)
print(f"Modified ONNX Model saved at {modified_model_path}")
