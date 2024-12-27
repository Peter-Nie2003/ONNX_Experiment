import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto

node = helper.make_node(
    "MaxPool",
    inputs=["X"],
    outputs=["Y"],
    kernel_shape=[2, 2],
    strides=[2, 2]
)

input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])

graph = helper.make_graph(
    nodes=[node],
    name="MaxPoolGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="onnx-maxpool-example", opset_imports=[helper.make_opsetid("", 21)])
onnx.save(model, "MaxPool/maxpool.onnx")
