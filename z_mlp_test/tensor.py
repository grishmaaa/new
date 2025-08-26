# tensor.py - Python wrapper with autodiff functionality.
# This file translates the user's provided blueprint into a functional
# high-level API that calls low-level C/CUDA functions.

import ctypes
import os
import numpy as np
from typing import Any, Tuple, Optional, Callable, List

# Define the C Tensor struct
class C_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int)
    ]

# Define the pointer type for C_Tensor once
C_Tensor_Pointer = ctypes.POINTER(C_Tensor)

# Load the compiled shared library
try:
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
tensor_lib._tensor_init_host.restype = C_Tensor_Pointer
tensor_lib._tensor_init_gpu.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
tensor_lib._tensor_init_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_free.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_free.restype = None
tensor_lib.tensor_free_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_free_gpu.restype = None
tensor_lib.tensor_copy_h2d.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_copy_h2d.restype = None
tensor_lib.tensor_copy_d2h.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_copy_d2h.restype = None

# Forward pass functions
tensor_lib.tensor_add_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_add_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_sub_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_sub_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_mul_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_mul_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_div_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_div_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_pow_gpu.argtypes = [C_Tensor_Pointer, ctypes.c_double]
tensor_lib.tensor_pow_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_neg_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_neg_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_matmul_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_matmul_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_sum_gpu.argtypes = [C_Tensor_Pointer, ctypes.c_int]
tensor_lib.tensor_sum_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_transpose_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_transpose_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_exp_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_exp_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_log_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_log_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_tanh_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_tanh_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_sigmoid_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_sigmoid_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_relu_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_relu_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_softplus_gpu.argtypes = [C_Tensor_Pointer]
tensor_lib.tensor_softplus_gpu.restype = C_Tensor_Pointer
tensor_lib.tensor_maximum_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_maximum_gpu.restype = C_Tensor_Pointer

# Backward pass functions
tensor_lib.tensor_add_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_sub_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_mul_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_div_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_pow_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, ctypes.c_double, C_Tensor_Pointer]
tensor_lib.tensor_neg_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_matmul_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_sum_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, ctypes.c_int]
tensor_lib.tensor_transpose_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_reshape_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
tensor_lib.tensor_exp_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_log_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_tanh_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_sigmoid_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_relu_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_softplus_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]
tensor_lib.tensor_maximum_backward_gpu.argtypes = [C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer, C_Tensor_Pointer]


class Tensor:
    """A minimal reverse‑mode autodiff tensor.

    - Supports broadcasting, 0D/1D/2D arrays
    - Keep graph via parents + backward closures
    """
    
    # Store C-level pointers for cleanup
    _c_host_tensor: Optional[C_Tensor_Pointer] = None # type: ignore
    _c_device_tensor: Optional[C_Tensor_Pointer] = None # pyright: ignore[reportInvalidTypeForm]

    def __init__(self, data: Any, requires_grad: bool = False, name: Optional[str] = None):
        if isinstance(data, self.__class__):
            self.data: np.ndarray = data.data
            self.requires_grad: bool = bool(data.requires_grad)
            self._c_host_tensor = data._c_host_tensor
            self._c_device_tensor = data._c_device_tensor
        else:
            self.data: np.ndarray = np.array(data, dtype=np.float64)
            self.requires_grad: bool = bool(requires_grad)
            self.shape_c = (ctypes.c_int * self.data.ndim)(*self.data.shape)
            self._c_host_tensor = tensor_lib._tensor_init_host(self.shape_c, self.data.ndim)
            self._c_device_tensor = tensor_lib._tensor_init_gpu(self.shape_c, self.data.ndim)
            
            size = self.data.size
            for i in range(size):
                self._c_host_tensor.contents.data[i] = self.data.ravel()[i]
            tensor_lib.tensor_copy_h2d(self._c_host_tensor, self._c_device_tensor)
        
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._parents: Tuple["Tensor", ...] = tuple()
        self.name = name

    def __del__(self):
        if self._c_host_tensor:
            tensor_lib.tensor_free(self._c_host_tensor)
        if self._c_device_tensor:
            tensor_lib.tensor_free_gpu(self._c_device_tensor)

    # --- convenience ---
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        req = ", requires_grad=True" if self.requires_grad else ""
        nm = f", name={self.name}" if self.name else ""
        return f"Tensor(shape={self.data.shape}{req}{nm})\n{self.data}"

    def clone(self) -> "Tensor":
        out = Tensor(self.data.copy(), self.requires_grad)
        return out

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    # --- graph utilities ---
    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
            
    def _set_parents(self, *parents: "Tensor"):
        self._parents = tuple(p for p in parents if isinstance(p, Tensor))

    # --- autograd core ---
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        # Build topological order
        topo: List[Tensor] = []
        visited = set()

        def build(v: Tensor):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._parents:
                    build(p)
                topo.append(v)

        build(self)

        # Seed gradient
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be provided for non-scalar outputs")
            seed = np.ones_like(self.data)
        else:
            seed = np.array(grad, dtype=np.float64)

        self._ensure_grad()
        self.grad = self.grad + seed

        # Backward pass
        for v in reversed(topo):
            v._backward()

    # --- helpers for broadcasting in backward ---
    @staticmethod
    def _unbroadcast(g: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        if g.shape == target_shape:
            return g
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)
        for i, (gs, ts) in enumerate(zip(g.shape, target_shape)):
            if ts == 1 and gs != 1:
                g = g.sum(axis=i, keepdims=True)
        return g

    # ------------------------------
    # Elementwise ops
    # ------------------------------
    def __add__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        # Note: The C-level GPU call will eventually need to be added here.
        # out._c_device_tensor = tensor_lib.tensor_add_gpu(self._c_device_tensor, other._c_device_tensor)
        out._set_parents(self, other)

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += self._unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self._unbroadcast(out.grad, other.shape)
        out._backward = _bw
        return out
    
    def __mul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        # Note: The C-level GPU call will eventually need to be added here.
        # out._c_device_tensor = tensor_lib.tensor_mul_gpu(self._c_device_tensor, other._c_device_tensor)
        out._set_parents(self, other)

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += self._unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self._unbroadcast(out.grad * self.data, other.shape)
        out._backward = _bw
        return out
    
    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)
    
    def __pow__(self, other: Any) -> "Tensor":
        out = Tensor(self.data ** other, requires_grad=self.requires_grad)
        # Note: The C-level GPU call will eventually need to be added here.
        # out._c_device_tensor = tensor_lib.tensor_pow_gpu(self._c_device_tensor, other)
        out._set_parents(self)

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                grad_val = other * (self.data ** (other - 1))
                self.grad += self._unbroadcast(out.grad * grad_val, self.shape)
        out._backward = _bw
        return out

    # ------------------------------
    # Full Test Suite
    # ------------------------------
    @staticmethod
    def run_tests():
        print("--- Running Tests ---")
        
        # Test basic tensor creation and printing
        print("\nTest 1: Tensor Initialization")
        x = Tensor([[1, 2], [3,  4]], requires_grad=True)
        print(x)
        
        # Test a simple forward pass
        print("\nTest 2: Forward Pass (x * 2 + 1)")
        y = x * 2 + 1
        print(y)
        
        # Test backward pass
        print("\nTest 3: Backward Pass")
        y.backward(grad=np.array([[1,1], [1,1]]))
        print("x.grad:")
        print(x.grad)

if __name__ == "__main__":
    Tensor.run_tests()
