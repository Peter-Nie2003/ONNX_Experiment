import onnxruntime as ort
import numpy as np
import time

def run_model(device="CPU"):
    
    if device == "CUDA":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession("Conv/conv.onnx", providers=providers)

    input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)
    inputs = {"X": input_data}

    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000

    print(f"Running on {device}")
    print("Input:")
    print(input_data)
    print("Conv Output:")
    print(outputs[0])
    print(f"\nInference Latency: {latency_ms:.2f} ms")

run_model(device="CPU")

run_model(device="CUDA")
