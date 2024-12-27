import onnxruntime as ort
import numpy as np
import time

def run_model(device="CPU"):
    
    if device == "CUDA":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession("MaxPool/maxpool.onnx", providers=providers)

    input_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]], dtype=np.float32)
    inputs = {"X": input_data}

    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000

    print(f"Running on {device}")
    print("Input:")
    print(input_data)
    print("\nMaxPool Output:")
    print(outputs[0])
    print(f"\nInference Latency: {latency_ms:.2f} ms")

run_model(device="CPU")

run_model(device="CUDA")
