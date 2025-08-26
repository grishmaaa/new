# linear.py
import numpy as np
import ctypes
import os

# ---------- Pure NumPy reference ----------
def linear_numpy(X: np.ndarray, W: np.ndarray, b: np.ndarray | None):
    X = np.asarray(X, dtype=np.float32, order='C')
    W = np.asarray(W, dtype=np.float32, order='C')
    if b is not None:
        b = np.asarray(b, dtype=np.float32, order='C')
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    Y = X @ W
    if b is not None:
        Y += b  # broadcasts over rows
    return Y

# ---------- ctypes bindings ----------

# CPU
try:
    _cpu = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "liblinear_cpu.so"))
    _cpu.linear_forward_cpu.restype = ctypes.c_int
    _cpu.linear_forward_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # W
        ctypes.POINTER(ctypes.c_float),  # b (nullable)
        ctypes.POINTER(ctypes.c_float),  # Y
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
except OSError:
    _cpu = None

# CUDA
try:
    _cuda = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "liblinear_cuda.so"))
    _cuda.linear_forward.restype = ctypes.c_int
    _cuda.linear_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # W
        ctypes.POINTER(ctypes.c_float),  # b (nullable)
        ctypes.POINTER(ctypes.c_float),  # Y
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
except OSError:
    _cuda = None

def _as_c_ptr(a: np.ndarray):
    assert a.dtype == np.float32 and a.flags['C_CONTIGUOUS']
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def linear_cpu(X: np.ndarray, W: np.ndarray, b: np.ndarray | None):
    assert _cpu is not None, "CPU library not loaded"
    X = np.ascontiguousarray(X, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    if b is not None:
        b = np.ascontiguousarray(b, dtype=np.float32)
        assert b.shape == (W.shape[1],)
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    Y = np.empty((M, N), dtype=np.float32)
    err = _cpu.linear_forward_cpu(
        _as_c_ptr(X), _as_c_ptr(W),
        (_as_c_ptr(b) if b is not None else ctypes.POINTER(ctypes.c_float)()),
        _as_c_ptr(Y),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
    )
    if err != 0:
        raise RuntimeError(f"CPU linear_forward_cpu error code {err}")
    return Y

def linear_cuda(X: np.ndarray, W: np.ndarray, b: np.ndarray | None):
    assert _cuda is not None, "CUDA library not loaded"
    X = np.ascontiguousarray(X, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    if b is not None:
        b = np.ascontiguousarray(b, dtype=np.float32)
        assert b.shape == (W.shape[1],)
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    Y = np.empty((M, N), dtype=np.float32)
    err = _cuda.linear_forward(
        _as_c_ptr(X), _as_c_ptr(W),
        (_as_c_ptr(b) if b is not None else ctypes.POINTER(ctypes.c_float)()),
        _as_c_ptr(Y),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
    )
    if err != 0:
        raise RuntimeError(f"CUDA linear_forward error code {err}")
    return Y

# ---------- quick self-test ----------
if __name__ == "__main__":
    M, K, N = 4, 3, 5
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    ref = linear_numpy(X, W, b)
    if _cpu:
        out_cpu = linear_cpu(X, W, b)
        print("CPU max abs diff:", np.max(np.abs(ref - out_cpu)))
    if _cuda:
        out_cuda = linear_cuda(X, W, b)
        print("CUDA max abs diff:", np.max(np.abs(ref - out_cuda)))
