import numpy as np
import ctypes
import os

# p = os.path.join(os.getcwd(), "libvector_cpu.so")
# print("Loading:", p)
# lib = ctypes.CDLL(p)
# print("Found:", lib.vector_add_cpu)
# Force all arrays to be float32 and contiguous
def to_float32(a):
    return np.ascontiguousarray(a, dtype=np.float32)

def as_c_ptr(a: np.ndarray):
    assert a.dtype == np.float32
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Load CPU and CUDA libs
cpu_lib = ctypes.CDLL(os.path.join(os.getcwd(), "libvector_cpu.so"))
cpu_lib.vector_add_cpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
cpu_lib.vector_add_cpu.restype = ctypes.c_int

cuda_lib = ctypes.CDLL(os.path.join(os.getcwd(), "libvector_cuda.so"))
cuda_lib.vector_add_cuda.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
cuda_lib.vector_add_cuda.restype = ctypes.c_int

# Example data (all float32)
N = 10
A = to_float32(np.arange(N))
B = to_float32(np.ones(N) * 5)
C_cpu = to_float32(np.empty(N))
C_cuda = to_float32(np.empty(N))

print("A:", A)
print("B:", B)


# CPU call
err_cpu = cpu_lib.vector_add_cpu(as_c_ptr(A), as_c_ptr(B), as_c_ptr(C_cpu), N)
print("CPU result:", C_cpu, "error:", err_cpu)

# CUDA call
err_cuda = cuda_lib.vector_add_cuda(as_c_ptr(A), as_c_ptr(B), as_c_ptr(C_cuda), N)
print("CUDA result:", C_cuda, "error:", err_cuda)

# Verify
print("Match CPU/CUDA:", np.allclose(C_cpu, C_cuda))
