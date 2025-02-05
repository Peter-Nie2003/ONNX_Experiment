import onnx
import onnx.helper as helper
from onnx import TensorProto
import os

# Define the input tensor
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])  # Batch=1, 10 values

# Create the Sigmoid node
sigmoid_node = helper.make_node(
    "Sigmoid",
    inputs=["X"],
    outputs=["Y"]
)

# Define the output tensor (same shape as input)
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])  # Batch=1, 10 values

# Create the graph
graph = helper.make_graph(
    nodes=[sigmoid_node],
    name="SigmoidGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-sigmoid-example")

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "Sigmoid.onnx")
onnx.save(model, onnx_file_path)

print(f"ONNX model saved at {onnx_file_path}")
