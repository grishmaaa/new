# tensor.py - Refactored Python wrapper with a high-level Tensor class
import ctypes
import os
import random
import time

# Define the C Tensor struct
class C_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int)
    ]

# Load the compiled shared library
try:
    # IMPORTANT: The path below assumes the compiled shared library is in the same directory.
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtensor.so")
    tensor_lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading the shared library: {e}")
    print("Please make sure libtensor.so is compiled and in the same directory.")
    exit()

# Define argument and return types for the C functions
tensor_lib.product.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
tensor_lib.product.restype = ctypes.c_longlong
tensor_lib._tensor_init_host.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
tensor_lib._tensor_init_host.restype = ctypes.POINTER(C_Tensor)
tensor_lib._tensor_init_gpu.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
tensor_lib._tensor_init_gpu.restype = ctypes.POINTER(C_Tensor)
tensor_lib.tensor_free.argtypes = [ctypes.POINTER(C_Tensor)]
tensor_lib.tensor_free.restype = None
tensor_lib.tensor_free_gpu.argtypes = [ctypes.POINTER(C_Tensor)]
tensor_lib.tensor_free_gpu.restype = None
tensor_lib.tensor_copy_h2d.argtypes = [ctypes.POINTER(C_Tensor), ctypes.POINTER(C_Tensor)]
tensor_lib.tensor_copy_h2d.restype = None
tensor_lib.tensor_copy_d2h.argtypes = [ctypes.POINTER(C_Tensor), ctypes.POINTER(C_Tensor)]
tensor_lib.tensor_copy_d2h.restype = None
tensor_lib.tensor_matmul_gpu.argtypes = [ctypes.POINTER(C_Tensor), ctypes.POINTER(C_Tensor)]
tensor_lib.tensor_matmul_gpu.restype = ctypes.POINTER(C_Tensor)
tensor_lib.tensor_print_host.argtypes = [ctypes.POINTER(C_Tensor)]
tensor_lib.tensor_print_host.restype = None


class Tensor:
    """A high-level Python class that wraps the C-level tensor and manages GPU memory."""
    def __init__(self, data, shape):
        """Initializes a new Tensor, allocating host and device memory and copying data."""
        self.shape = shape
        self.ndim = len(shape)
        
        # Explicitly store the C-type shape array to prevent garbage collection
        self.shape_c = (ctypes.c_int * self.ndim)(*self.shape)
        
        # Initialize C-level tensors
        self.c_host_tensor = tensor_lib._tensor_init_host(self.shape_c, self.ndim)
        self.c_device_tensor = tensor_lib._tensor_init_gpu(self.shape_c, self.ndim)
        
        # Copy Python data into the C host tensor
        size = 1
        for dim in self.shape:
            size *= dim
        for i in range(size):
            self.c_host_tensor.contents.data[i] = data[i]
        
        # Copy data from host to device
        tensor_lib.tensor_copy_h2d(self.c_host_tensor, self.c_device_tensor)
        
    def __del__(self):
        """Frees the C-level memory when the Python object is garbage collected."""
        if hasattr(self, 'c_host_tensor') and self.c_host_tensor:
            tensor_lib.tensor_free(self.c_host_tensor)
        if hasattr(self, 'c_device_tensor') and self.c_device_tensor:
            tensor_lib.tensor_free_gpu(self.c_device_tensor)

    def print(self):
        """Prints the tensor contents by copying data from device to host first."""
        # Create a temporary host tensor to copy the result back to
        shape_c = (ctypes.c_int * self.ndim)(*self.shape)
        temp_host_tensor = tensor_lib._tensor_init_host(shape_c, self.ndim)
        
        # Copy the device tensor data to the temporary host tensor
        tensor_lib.tensor_copy_d2h(self.c_device_tensor, temp_host_tensor)
        
        # Print the host tensor and free the temporary memory
        tensor_lib.tensor_print_host(temp_host_tensor)
        tensor_lib.tensor_free(temp_host_tensor)

    @staticmethod
    def matmul(t1, t2):
        """
        Performs matrix multiplication on two Tensor objects and returns a new Tensor.
        This handles all the low-level C calls internally.
        """
        if not isinstance(t1, Tensor) or not isinstance(t2, Tensor):
            raise TypeError("Both arguments must be instances of the Tensor class.")
        
        # Perform matmul on GPU
        c_result_tensor = tensor_lib.tensor_matmul_gpu(t1.c_device_tensor, t2.c_device_tensor)

        # Handle the case of NULL pointer from C++
        if not c_result_tensor:
            return None  # Or raise a custom exception

        # Create a new Python Tensor object to wrap the result
        result_shape = [c_result_tensor.contents.shape[i] for i in range(c_result_tensor.contents.ndim)]
        
        # Allocate a new Python Tensor object
        result_tensor = Tensor.__new__(Tensor)
        result_tensor.shape = result_shape
        result_tensor.ndim = len(result_shape)
        result_tensor.c_host_tensor = tensor_lib._tensor_init_host(
            (ctypes.c_int * result_tensor.ndim)(*result_tensor.shape), result_tensor.ndim
        )
        result_tensor.c_device_tensor = c_result_tensor
        
        return result_tensor

# --- Refactored Test Suite ---
def run_test_case(name, shape_a, data_a, shape_b, data_b):
    """Encapsulates the logic for a single test case."""
    print(f"\n--- Running Test: {name} ---")
    try:
        t_a = Tensor(data_a, shape_a)
        t_b = Tensor(data_b, shape_b)

        # Warm-up (optional): trigger any lazy CUDA init/JIT/etc. without timing it
        _ = Tensor.matmul(t_a, t_b)

        # Time the matmul (ensure GPU sync by copying result back to host once)
        start = time.perf_counter()
        result_tensor = Tensor.matmul(t_a, t_b)

        if result_tensor:
            # Force synchronization by doing a D2H copy into a temporary host tensor
            shape_c = (ctypes.c_int * result_tensor.ndim)(*result_tensor.shape)
            tmp_host = tensor_lib._tensor_init_host(shape_c, result_tensor.ndim)
            tensor_lib.tensor_copy_d2h(result_tensor.c_device_tensor, tmp_host)
            tensor_lib.tensor_free(tmp_host)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"Matmul time: {elapsed_ms:.3f} ms")

            # print(f"Result for {result_tensor.ndim}D Tensor:")
            # result_tensor.print()
        else:
            print("Matmul failed to produce a result.")
    except Exception as e:
        print(f"An error occurred: {e}")
    print("-----------------------------------")



def main():
    """Main function to run the test suite."""
    
    # # Test for basic Tensor functionality (init and print)
    # print("--- Basic Tensor Functionality Test ---")
    # try:
    #     data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    #     shape = [2, 3]
    #     my_tensor = Tensor(data, shape)
    #     print("Initialized Tensor:")
    #     my_tensor.print()
    # except Exception as e:
    #     print(f"Basic functionality test failed: {e}")
    # print("-----------------------------------")
    
    
    # Test cases for N-dimensional matmul
    test_cases = []
    for d in range(2, 8):
        # Dynamically create shapes for N-dimensional batch matrix multiplication
        # The shape is [2, 2, ..., 2, 2, 3] and [2, 2, ..., 2, 3, 2]
        # The number of '2's in the batch dimensions is d - 2
        shape_a = [2] * (d - 2) + [2, 3]
        shape_b = [2] * (d - 2) + [3, 2]

        # Calculate the total size of each tensor
        size_a = 1
        for dim in shape_a:
            size_a *= dim
        
        size_b = 1
        for dim in shape_b:
            size_b *= dim

        # Add the test case to the list
        test_cases.append({
            "name": f"{d}D Matrix Multiplication: {shape_a} @ {shape_b}",
            "shape_a": shape_a,
            "data_a": [1.0] * size_a,
            "shape_b": shape_b,
            "data_b": [random.random() for _ in range(size_b)]
        })

    # Run the generated test cases
    for case in test_cases:
        run_test_case(case["name"], case["shape_a"], case["data_a"], case["shape_b"], case["data_b"])


if __name__ == "__main__":
    main()
