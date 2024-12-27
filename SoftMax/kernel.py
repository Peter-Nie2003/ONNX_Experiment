import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto

node = helper.make_node(
    "Softmax",
    inputs=["X"],
    outputs=["Y"],
    axis=1  
)

input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5])
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5])

graph = helper.make_graph(
    nodes=[node],
    name="SoftmaxGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="onnx-softmax-example",opset_imports=[helper.make_opsetid("", 21)])
onnx.save(model, "SoftMax/softmax.onnx")
