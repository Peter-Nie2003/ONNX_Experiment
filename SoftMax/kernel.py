import onnx
import onnx.helper as helper
from onnx import TensorProto
import os

# Define the input tensor
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])  # Batch=1, 10-class logits

# Create the Softmax node
softmax_node = helper.make_node(
    "Softmax",
    inputs=["X"],
    outputs=["Y"],
    axis=1  # Apply softmax across the class dimension
)

# Define the output tensor
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])  # Batch=1, 10-class probabilities

# Create the graph
graph = helper.make_graph(
    nodes=[softmax_node],
    name="SoftmaxGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="onnx-softmax-example")

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_file_path = os.path.join(script_dir, "Softmax.onnx")
onnx.save(model, onnx_file_path)

print(f"ONNX model saved at {onnx_file_path}")
