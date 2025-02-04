import onnx
import numpy as np
import onnx.helper as helper
from onnx import TensorProto
import os

# Define input tensor
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])  # Batch=1, Channels=3, H=32, W=32

# Define weight tensor for ConvTranspose (random values for example)
weight_tensor = helper.make_tensor(
    "W", TensorProto.FLOAT, [3, 16, 3, 3],  # (input_channels, output_channels, kernel_H, kernel_W)
    np.random.rand(3, 16, 3, 3).astype(np.float32)
)

# Define bias tensor (random values for example)
bias_tensor = helper.make_tensor(
    "B", TensorProto.FLOAT, [3],  # Bias for each output channel
    np.random.rand(3).astype(np.float32)
)

# Create the ConvTranspose node
conv_transpose_node = helper.make_node(
    "ConvTranspose",
    inputs=["X", "W", "B"],  # Input, Weights, and Bias
    outputs=["Y"],  # Output
    kernel_shape=[3, 3],  # Kernel size: 3x3
    strides=[2, 2],  # Stride: 2x2 (upsampling)
    pads=[1, 1, 1, 1],  # Padding: top, left, bottom, right
    group=1  # Standard transposed convolution
)

# Define the output tensor (upsampled spatial dimensions)
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64, 64])  # Batch=1, Channels=16, H=64, W=64

# Create the graph
graph = helper.make_graph(
    nodes=[conv_transpose_node],
    name="ConvTransposeGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[weight_tensor, bias_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-convtranspose-example")

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "ConvTranspose.onnx")
onnx.save(model, onnx_file_path)

print(f"ONNX model saved at {onnx_file_path}")
