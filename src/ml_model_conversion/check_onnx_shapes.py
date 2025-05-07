import onnx
import onnxruntime as ort

# Load the ONNX model
model_path = "tl_detect.onnx"
onnx_model = onnx.load(model_path)

# Create an ONNX Runtime InferenceSession
sess = ort.InferenceSession(model_path)

for input_tensor in onnx_model.graph.input:
    print(input_tensor.name, input_tensor.type.tensor_type.shape)

# Get the output names and their shapes
for output in sess.get_outputs():
    print(f"Output Name: {output.name}, Output Shape: {output.shape}")
