import onnx
import onnx.helper as helper
from onnx import TensorProto
import os

# Define the input tensor
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])  # Batch=1, Channels=3, H=32, W=32

# Create the LpPool node
lp_pool_node = helper.make_node(
    "LpPool",
    inputs=["X"],
    outputs=["Y"],
    kernel_shape=[2, 2],  # Pool size: 2x2
    strides=[2, 2],  # Stride: 2x2 (downsampling)
    p=2  # L2 pooling (Euclidean norm)
)

# Define the output tensor (downsampled spatial dimensions)
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 16, 16])  # Batch=1, Channels=3, H=16, W=16

# Create the graph
graph = helper.make_graph(
    nodes=[lp_pool_node],
    name="LpPoolGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-lppool-example")

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "LpPool.onnx")
onnx.save(model, onnx_file_path)

print(f"ONNX model saved at {onnx_file_path}")
