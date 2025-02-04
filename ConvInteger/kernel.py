import onnx
import numpy as np
import onnx.helper as helper
from onnx import TensorProto
import os

# Define input, weight, and zero-point tensors
input_tensor = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 3, 32, 32])  # Batch=1, Channels=3, H=32, W=32
weight_tensor = helper.make_tensor("W", TensorProto.INT8, [16, 3, 3, 3], np.random.randint(-128, 127, (16, 3, 3, 3), dtype=np.int8))
bias_tensor = helper.make_tensor("B", TensorProto.INT32, [16], np.random.randint(-128, 127, 16, dtype=np.int32))

# Zero points for input and weights
input_zero_point = helper.make_tensor("X_zero_point", TensorProto.INT8, [], np.array([0], dtype=np.int8))
weight_zero_point = helper.make_tensor("W_zero_point", TensorProto.INT8, [], np.array([0], dtype=np.int8))

# Create the ConvInteger node
conv_integer_node = helper.make_node(
    "ConvInteger",
    inputs=["X", "W", "X_zero_point", "W_zero_point"],  # Input, Weights, and Zero Points
    outputs=["Y"],  # Output
    kernel_shape=[3, 3],  # Kernel size: 3x3
    strides=[1, 1],  # Stride: 1x1
    pads=[1, 1, 1, 1],  # Padding: top, left, bottom, right
    group=1  # Standard convolution (not grouped)
)

# Define the output tensor (INT32 to store accumulation results)
output_tensor = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 16, 32, 32])  # Batch=1, Channels=16, H=32, W=32

# Create the graph
graph = helper.make_graph(
    nodes=[conv_integer_node],
    name="ConvIntegerGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[weight_tensor, input_zero_point, weight_zero_point]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-convinteger-example")

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "ConvInteger.onnx")
onnx.save(model, onnx_file_path)

print(f"ONNX model saved at {onnx_file_path}")
