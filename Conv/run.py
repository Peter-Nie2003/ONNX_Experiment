import onnxruntime as ort
import numpy as np
import os

# Get the directory of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))
onxx_file_path = os.path.join(script_dir, "Conv.onnx")

# Load the ONNX model
session = ort.InferenceSession(onxx_file_path)

# Get model input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Generate a random input tensor matching the model's expected shape
input_data = np.random.randn(*input_shape).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})

# Print output
print("Model output shape:", outputs[0].shape)
print("Model output:", outputs[0])