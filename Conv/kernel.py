import onnx
import numpy as np
import onnx.helper as helper
from onnx import TensorProto
import os

# Define input, weights, and bias tensors
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])  # Batch=1, Channels=3, H=32, W=32
weight_tensor = helper.make_tensor("W", TensorProto.FLOAT, [16, 3, 3, 3], np.random.rand(16, 3, 3, 3).astype(np.float32))
bias_tensor = helper.make_tensor("B", TensorProto.FLOAT, [16], np.random.rand(16).astype(np.float32))

# Create the Conv node
conv_node = helper.make_node(
    "Conv",
    inputs=["X", "W", "B"],  # Input, Weights, and Bias
    outputs=["Y"],  # Output
    kernel_shape=[3, 3],  # Kernel size: 3x3
    strides=[1, 1],  # Stride: 1x1
    pads=[1, 1, 1, 1],  # Padding: top, left, bottom, right
    group=1  # Standard convolution (not grouped)
)

# Define the output tensor
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 32, 32])  # Batch=1, Channels=16, H=32, W=32

# Create the graph
graph = helper.make_graph(
    nodes=[conv_node],
    name="ConvGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[weight_tensor, bias_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-conv-example")
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "Conv.onnx")
onnx.save(model, onnx_file_path)