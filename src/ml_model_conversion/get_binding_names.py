import tensorrt as trt

def print_tensor_names_and_shapes(engine):
    num_io_tensors = engine.num_io_tensors  # Get the number of input/output tensors

    for i in range(num_io_tensors):
        tensor_name = engine.get_tensor_name(i)  # Get the name of the tensor
        tensor_shape = engine.get_tensor_shape(tensor_name)  # Get the shape of the tensor
        tensor_dtype = engine.get_tensor_dtype(tensor_name)  # Get the dtype of the tensor
        
        print(f"Tensor {i} Name: {tensor_name}")
        print(f"Dims: {tensor_shape}")
        print(f"Data Type: {tensor_dtype}")

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)

    if not engine:
        print("Error: Failed to deserialize the engine.")
        return None

    return engine

# Example usage
engine_file_path = "../tl_detect.engine"
engine = load_engine(engine_file_path)

if engine:
    # Print tensor names and dimensions
    print_tensor_names_and_shapes(engine)

