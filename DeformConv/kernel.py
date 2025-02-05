import onnx
import numpy as np
import onnx.helper as helper
from onnx import TensorProto
import os

# Define input tensors (feature map, offset map, weights, and bias)
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])  # Input Feature Map
offset_tensor = helper.make_tensor_value_info("Offset", TensorProto.FLOAT, [1, 18, 32, 32])  # Offset Map
weight_tensor = helper.make_tensor("W", TensorProto.FLOAT, [16, 3, 3, 3], np.random.rand(16, 3, 3, 3).astype(np.float32))
bias_tensor = helper.make_tensor("B", TensorProto.FLOAT, [16], np.random.rand(16).astype(np.float32))

# Create the custom Deformable Convolution node
deform_conv_node = helper.make_node(
    "DeformConv2D",  # Custom Op (may not be available in vanilla ONNX)
    inputs=["X", "Offset", "W", "B"],
    outputs=["Y"],
    kernel_shape=[3, 3],  # Kernel Size
    strides=[1, 1],  # Stride
    pads=[1, 1, 1, 1],  # Padding
    group=1,
    deformable_groups=1  # Number of deformable groups
)

# Define the output tensor
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 32, 32])  # Output feature map

# Create the graph
graph = helper.make_graph(
    nodes=[deform_conv_node],
    name="DeformConvGraph",
    inputs=[input_tensor, offset_tensor],
    outputs=[output_tensor],
    initializer=[weight_tensor, bias_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-deformconv-example")

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "DeformConv.onnx")
onnx.save(model, onnx_file_path)

print(f"ONNX model saved at {onnx_file_path}")
