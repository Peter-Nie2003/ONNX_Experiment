import onnx
import onnx.helper as helper
from onnx import TensorProto
import os

# Define the input tensor
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])  # Batch=1, Channels=3, H=32, W=32

# Create the MaxPool node
maxpool_node = helper.make_node(
    "MaxPool",
    inputs=["X"],
    outputs=["Y"],
    kernel_shape=[2, 2],  # Pool size: 2x2
    strides=[2, 2]  # Stride: 2x2 (downsamples by 2)
)

# Define the output tensor (downsampled spatial dimensions)
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 16, 16])  # Batch=1, Channels=3, H=16, W=16

# Create the graph
graph = helper.make_graph(
    nodes=[maxpool_node],
    name="MaxPoolGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-maxpool-example")

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "MaxPool.onnx")
onnx.save(model, onnx_file_path)

print(f"ONNX model saved at {onnx_file_path}")
