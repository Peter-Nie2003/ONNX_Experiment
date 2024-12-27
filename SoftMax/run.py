import onnxruntime as ort
import numpy as np
import time

def run_softmax_model(device="CPU"):
    if device == "CUDA":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession("SoftMax/softmax.onnx", providers=providers)

    input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    inputs = {"X": input_data}

    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000

    print(f"Running on {device}")
    print("Input:")
    print(input_data)
    print("\nSoftmax Output:")
    print(outputs[0])
    print(f"\nInference Latency: {latency_ms:.2f} ms")

run_softmax_model(device="CPU")

run_softmax_model(device="CUDA")
