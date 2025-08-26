# # py_matmul.py
# from ctypes import cdll, c_int, c_float, POINTER
# from pathlib import Path
# import numpy as np

# # load the .so sitting next to this file
# _lib = cdll.LoadLibrary(str(Path(__file__).with_name("libmatmul.so")))
# _lib.matmul.argtypes = [
#     POINTER(c_float), POINTER(c_float), POINTER(c_float),
#     c_int, c_int, c_int
# ]
# _lib.matmul.restype = c_int

# def matmul(A, B, C=None):
#     """
#     A: (M, N) float32 array-like
#     B: (N, K) float32 array-like
#     C: optional (M, K) float32 array to write into
#     Returns the output array (C).
#     """
#     A = np.ascontiguousarray(A, dtype=np.float32)
#     B = np.ascontiguousarray(B, dtype=np.float32)

#     if A.ndim != 2 or B.ndim != 2:
#         raise ValueError("A and B must be 2D")
#     M, N = A.shape
#     N2, K = B.shape
#     if N != N2:
#         raise ValueError(f"shapes not aligned: A{A.shape} @ B{B.shape}")

#     if C is None:
#         C = np.empty((M, K), dtype=np.float32)
#     else:
#         C = np.ascontiguousarray(C, dtype=np.float32)
#         if C.shape != (M, K):
#             raise ValueError(f"C must be shape {(M, K)}")

#     rc = _lib.matmul(
#         A.ctypes.data_as(POINTER(c_float)),
#         B.ctypes.data_as(POINTER(c_float)),
#         C.ctypes.data_as(POINTER(c_float)),
#         c_int(M), c_int(N), c_int(K)
#     )
#     if rc != 0:
#         raise RuntimeError(f"CUDA matmul failed with code {rc}")
#     return C



# py_matmul.py
from ctypes import cdll, c_int, c_float, POINTER
from pathlib import Path
import numpy as np
import time

# load the shared lib sitting next to this file (adjust path if needed)
_lib = cdll.LoadLibrary(str(Path(__file__).with_name("libmatmul.so")))

# int matmul(const float* A, const float* B, float* C, int N1, int N2, int N3);
_lib.matmul.argtypes = [
    POINTER(c_float), POINTER(c_float), POINTER(c_float),
    c_int, c_int, c_int
]
_lib.matmul.restype = c_int

def matmul(A, B, C=None):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    N1, N2 = A.shape
    N2b, N3 = B.shape
    if N2 != N2b:
        raise ValueError(f"shape mismatch: A{A.shape} @ B{B.shape}")

    if C is None:
        C = np.empty((N1, N3), dtype=np.float32)
    else:
        C = np.ascontiguousarray(C, dtype=np.float32)
        if C.shape != (N1, N3):
            raise ValueError(f"C must be shape {(N1, N3)}")

    rc = _lib.matmul(
        A.ctypes.data_as(POINTER(c_float)),
        B.ctypes.data_as(POINTER(c_float)),
        C.ctypes.data_as(POINTER(c_float)),
        c_int(N1), c_int(N2), c_int(N3)
    )
    if rc != 0:
        raise RuntimeError(f"matmul failed with code {rc}")
    return C