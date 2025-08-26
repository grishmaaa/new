# tensorcore_py.py
# Minimal Python wrapper using only standard library (ctypes, array, struct).

import os
import sys
import subprocess
import ctypes as C
from array import array
from time import perf_counter

LIB_NAME = "libtensor.so"
LIB_PATH = os.path.join(os.path.dirname(__file__), LIB_NAME)

def _build_if_needed():
    if os.path.exists(LIB_PATH):
        return
    cu = os.path.join(os.path.dirname(__file__), "tensor_core_kernels.cu")
    out = LIB_PATH
    cmd = [
    "nvcc", "-O3", "-Xcompiler", "-fPIC", "-std=c++17",
    "-arch=sm_86", "-shared",            # <-- add -shared
    "-o", out, cu
    ]   

    print("Building CUDA shared library:")
    print(" ", " ".join(cmd))
    subprocess.check_call(cmd)

_build_if_needed()

_lib = C.CDLL(LIB_PATH)

# ----- ctypes signatures -----
# helpers
_lib.gpu_malloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
_lib.gpu_malloc.restype = C.c_int
_lib.gpu_free.argtypes = [C.c_void_p]
_lib.gpu_free.restype = C.c_int
_lib.h2d.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
_lib.h2d.restype = C.c_int
_lib.d2h.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
_lib.d2h.restype = C.c_int

# math ops
_lib.matmul_tiled_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                  C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int]
_lib.matmul_tiled_f32.restype = C.c_int

_lib.bias_add_row_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                  C.c_int, C.c_int, C.c_int]
_lib.bias_add_row_f32.restype = C.c_int

def _set_ew_sig(fn):
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_longlong]
    fn.restype = C.c_int

_set_ew_sig(_lib.ew_add_f32)
_set_ew_sig(_lib.ew_sub_f32)
_set_ew_sig(_lib.ew_mul_f32)
_set_ew_sig(_lib.ew_div_f32)

def _set_un_sig(fn):
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_longlong]
    fn.restype = C.c_int

_set_un_sig(_lib.relu_f32)
_set_un_sig(_lib.sigmoid_f32)
_set_un_sig(_lib.tanh_f32)
_set_un_sig(_lib.exp_f32)
_set_un_sig(_lib.log_f32)
_set_un_sig(_lib.softplus_f32)

_lib.reduce_max_rows_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int]
_lib.reduce_max_rows_f32.restype = C.c_int
_lib.reduce_sum_rows_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int]
_lib.reduce_sum_rows_f32.restype = C.c_int
_lib.softmax_rows_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
_lib.softmax_rows_f32.restype = C.c_int
_lib.transpose_2d_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
_lib.transpose_2d_f32.restype = C.c_int

# ===== Tensor (host+device handle) =====
class Tensor:
    """
    Lightweight tensor. Only float32 and 2-D for simplicity here.
    Host buffer: array('f') length rows*cols.
    Device ptr: stored as int handle (C.c_void_p).
    """
    def __init__(self, rows, cols, data=None, device="cpu"):
        assert rows >= 0 and cols >= 0
        self.shape = (int(rows), int(cols))
        self.ndim = 2
        self.dtype = "float32"
        n = rows * cols
        if data is None:
            self._h = array('f', [0.0] * n)
        else:
            assert len(data) == n
            self._h = array('f', data)
        self._d = None
        if device == "cuda" and n > 0:
            self.gpu()

    def numel(self):
        return self.shape[0] * self.shape[1]

    def cpu(self):
        if self._d is not None and self.numel() > 0:
            bytes_ = self.numel() * 4
            _check(_lib.d2h(self._h_buffer_ptr(), self._d, bytes_))
        return self

    def gpu(self):
        if self._d is None and self.numel() > 0:
            ptr = C.c_void_p()
            _check(_lib.gpu_malloc(C.byref(ptr), self.numel() * 4))
            self._d = ptr
            _check(_lib.h2d(self._d, self._h_buffer_ptr(), self.numel() * 4))
        return self

    def free(self):
        if self._d is not None:
            _lib.gpu_free(self._d)
            self._d = None

    def _h_buffer_ptr(self):
        # Return raw pointer to host array
        return self._h.buffer_info()[0]

    # ----- ops -----
    def matmul(self, other):
        a = self.gpu()
        b = other.gpu()
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        out = Tensor(M, N, device="cuda")
        _check(_lib.matmul_tiled_f32(a._d, b._d, out._d, M, N, K, K, N, N))
        return out

    def bias_add_row(self, bias):  # bias shape (1, C) or (C,)
        x = self.gpu()
        rows, cols = self.shape
        assert bias.shape == (1, cols) or bias.shape == (cols, 1) or bias.shape == (1, cols)
        # force shape (1, C)
        if bias.shape != (1, cols):
            b = Tensor(1, cols, device="cuda")
            # copy bias flat into b
            bias.cpu()
            b._h = array('f', list(bias._h))
            b.gpu()
            bias = b
        out = Tensor(rows, cols, device="cuda")
        _check(_lib.bias_add_row_f32(x._d, bias._d, out._d, rows, cols, cols))
        return out

    def add(self, other):
        assert self.numel() == other.numel()
        x = self.gpu(); y = other.gpu()
        out = Tensor(self.shape[0], self.shape[1], device="cuda")
        _check(_lib.ew_add_f32(x._d, y._d, out._d, self.numel()))
        return out

    def relu(self):
        x = self.gpu()
        out = Tensor(self.shape[0], self.shape[1], device="cuda")
        _check(_lib.relu_f32(x._d, out._d, self.numel()))
        return out

    def sigmoid(self):
        x = self.gpu()
        out = Tensor(self.shape[0], self.shape[1], device="cuda")
        _check(_lib.sigmoid_f32(x._d, out._d, self.numel()))
        return out

    def softplus(self):
        x = self.gpu()
        out = Tensor(self.shape[0], self.shape[1], device="cuda")
        _check(_lib.softplus_f32(x._d, out._d, self.numel()))
        return out

    def softmax_rows(self):
        x = self.gpu()
        rows, cols = self.shape
        out = Tensor(rows, cols, device="cuda")
        _check(_lib.softmax_rows_f32(x._d, out._d, rows, cols, cols, cols))
        return out

    def sum_rows(self):
        x = self.gpu()
        rows, cols = self.shape
        out = Tensor(rows, 1, device="cuda")
        _check(_lib.reduce_sum_rows_f32(x._d, out._d, rows, cols, cols))
        return out

    def max_rows(self):
        x = self.gpu()
        rows, cols = self.shape
        out = Tensor(rows, 1, device="cuda")
        _check(_lib.reduce_max_rows_f32(x._d, out._d, rows, cols, cols))
        return out

    def tolist(self):
        self.cpu()
        return list(self._h)

def _check(code):
    if code != 0:
        raise RuntimeError("CUDA backend error code: %d" % code)
