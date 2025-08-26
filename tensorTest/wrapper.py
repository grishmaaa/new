# wrapper.py
import ctypes
import cupy as cp
import numpy as np
import os

class TensorCoreGEMM:
    """
    A Python wrapper for the custom CUDA Tensor Core GEMM kernel.
    """
    def __init__(self, lib_path='./libtgemm.so'):
        """
        Loads the shared library and sets up the function signature.

        Args:
            lib_path (str): Path to the compiled .so file.
        """
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Shared library not found at '{lib_path}'. "
                "Please compile the CUDA code first using compile.sh."
            )
            
        # Load the shared library
        self._lib = ctypes.CDLL(lib_path)
        
        # Get a handle to the naive_tensor_tgemm function
        self._func = self._lib.naive_tensor_tgemm
        
        # Define the function signature (argument types) for ctypes
        # Corresponds to: (half*, half*, float*, int, int, int)
        # We use c_void_p for opaque GPU pointers.
        self._func.argtypes = [
            ctypes.c_void_p,  # half *d_A_ptr
            ctypes.c_void_p,  # half *d_B_ptr
            ctypes.c_void_p,  # float *d_C_ptr
            ctypes.c_int,     # int C_n_rows (M)
            ctypes.c_int,     # int C_n_cols (N)
            ctypes.c_int      # int A_n_cols (K)
        ]
        
        # Define the function's return type (void)
        self._func.restype = None

    def compute(self, A: np.ndarray, B: np.ndarray) -> cp.ndarray:
        """
        Performs the matrix multiplication C(MxN) = A(MxK) * B(KxN) on the GPU.

        Args:
            A (np.ndarray): Left-hand matrix of shape (M, K) and dtype float16.
            B (np.ndarray): Right-hand matrix of shape (K, N) and dtype float16.

        Returns:
            cupy.ndarray: The result matrix C on the GPU, with shape (M, N) and dtype float32.
        """
        # 1. Validate inputs
        if A.dtype != np.float16 or B.dtype != np.float16:
            raise TypeError("Input matrices A and B must be of dtype numpy.float16.")
        
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"Matrix dimension mismatch: A is {A.shape} and B is {B.shape}. "
                "The inner dimensions must be equal."
            )

        M, K = A.shape
        _, N = B.shape
        
        # The CUDA kernel has alignment requirements.
        # M and N must be multiples of 64, K must be a multiple of 16
        # (based on the default BLOCK_ROW_TILES=4, BLOCK_COL_TILES=4).
        if M % 64 != 0 or N % 64 != 0 or K % 16 != 0:
            raise ValueError(
                f"Dimension alignment error: M({M}) and N({N}) must be multiples of 64, "
                f"and K({K}) must be a multiple of 16."
            )

        # 2. Allocate GPU memory and copy data from host to device
        d_A = cp.asarray(A)
        
        # IMPORTANT: The CUDA kernel expects the B matrix in column-major format.
        # We achieve this by transposing B and then making it Fortran-contiguous.
        d_B_colmajor = cp.asfortranarray(B)

        # Allocate output matrix C on the device (result is float32)
        d_C = cp.zeros((M, N), dtype=cp.float32)

        # 3. Call the CUDA function
        self._func(
            d_A.data.ptr,       # Pointer to A on GPU
            d_B_colmajor.data.ptr, # Pointer to B (col-major) on GPU
            d_C.data.ptr,       # Pointer to C on GPU
            M,                  # Number of rows in C
            N,                  # Number of columns in C
            K                   # Number of columns in A / rows in B
        )
        
        return d_C