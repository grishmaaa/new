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
    # Full Test Suite (Drop-in)
    # ------------------------------
    @staticmethod
    def run_tests(verbose: bool = False):
        """
        Runs a suite of forward/grad & property tests with tabular output.
        Uses only stdlib + NumPy. No extra deps.
        """
        import time
        import math
        rng = np.random.default_rng(42)

        # ---------- small pretty table printer ----------
        def _table(headers, rows, title=None):
            col_widths = [len(h) for h in headers]
            for r in rows:
                for j, cell in enumerate(r):
                    col_widths[j] = max(col_widths[j], len(str(cell)))
            def sep(ch='-'):
                return '+' + '+'.join(ch * (w + 2) for w in col_widths) + '+'
            def row(vals):
                return '| ' + ' | '.join(f"{str(v):<{w}}" for v, w in zip(vals, col_widths)) + ' |'
            if title:
                print(title)
            print(sep('-'))
            print(row(headers))
            print(sep('='))
            for r in rows:
                print(row(r))
            print(sep('-'))

        def _fmt_bool(b):
            return "PASS" if b else "FAIL"

        def _cmp_arrays(a, b, rtol=1e-6, atol=1e-8):
            diff = np.asarray(a) - np.asarray(b)
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            ok = np.allclose(a, b, rtol=rtol, atol=atol)
            return ok, max_abs

        def _finite_diff_grad(f_builder: Callable[[Tensor], Tensor],
                              x_np: np.ndarray,
                              eps: float = 1e-6) -> np.ndarray:
            """
            Compute finite-difference gradient of scalar function:
            f(x) = sum( f_builder(Tensor(x)).data )
            """
            grad = np.zeros_like(x_np, dtype=np.float64)
            flat = x_np.reshape(-1)
            for i in range(flat.size):
                orig = flat[i]
                flat[i] = orig + eps
                y_plus  = f_builder(Tensor(x_np.copy())).data.sum()
                flat[i] = orig - eps
                y_minus = f_builder(Tensor(x_np.copy())).data.sum()
                flat[i] = orig
                grad.reshape(-1)[i] = (y_plus - y_minus) / (2 * eps)
            return grad

        # ---------- registry of tests ----------
        results = []  # for summary table
        details = []  # per-failure details

        def record(name, f_ok, g_ok, f_err, g_err, secs, notes=""):
            results.append([name, _fmt_bool(f_ok), _fmt_bool(g_ok),
                            f"{f_err:.3e}", f"{g_err:.3e}", f"{secs*1e3:.1f} ms", notes])

        # ===== Test 1: add (no broadcast) =====
        def test_add_basic():
            name = "add/basic"
            x_np = rng.normal(size=(2, 3))
            y_np = rng.normal(size=(2, 3))
            x = Tensor(x_np, requires_grad=True, name="x")
            y = Tensor(y_np, requires_grad=True, name="y")
            t0 = time.time()
            out = x + y
            upstream = np.ones_like(out.data)
            out.backward(grad=upstream)
            f_ok, f_err = _cmp_arrays(out.data, x_np + y_np)
            g_ok1, g1_err = _cmp_arrays(x.grad, upstream)
            g_ok2, g2_err = _cmp_arrays(y.grad, upstream)
            g_ok = g_ok1 and g_ok2
            g_err = max(g1_err, g2_err)
            secs = time.time() - t0
            if not (f_ok and g_ok) and verbose:
                details.append((
                    name,
                    ("forward", out.data, x_np + y_np),
                    ("grad_x", x.grad, upstream),
                    ("grad_y", y.grad, upstream),
                ))
            record(name, f_ok, g_ok, f_err, g_err, secs)

        # ===== Test 2: add (broadcast) =====
        def test_add_broadcast():
            name = "add/broadcast"
            x_np = rng.normal(size=(2, 3))
            b_np = rng.normal(size=(1, 3))
            x = Tensor(x_np, requires_grad=True, name="x")
            b = Tensor(b_np, requires_grad=True, name="b")
            t0 = time.time()
            out = x + b
            upstream = np.ones_like(out.data)
            out.backward(grad=upstream)
            f_ok, f_err = _cmp_arrays(out.data, x_np + b_np)
            # grads
            dx_expect = upstream         # (2,3)
            db_expect = upstream.sum(axis=0, keepdims=True)  # (1,3)
            g_ok1, g1_err = _cmp_arrays(x.grad, dx_expect)
            g_ok2, g2_err = _cmp_arrays(b.grad, db_expect)
            g_ok = g_ok1 and g_ok2
            g_err = max(g1_err, g2_err)
            secs = time.time() - t0
            if not (f_ok and g_ok) and verbose:
                details.append((
                    name,
                    ("forward", out.data, x_np + b_np),
                    ("grad_x", x.grad, dx_expect),
                    ("grad_b", b.grad, db_expect),
                ))
            record(name, f_ok, g_ok, f_err, g_err, secs, notes="Broadcast axis=0")

        # ===== Test 3: mul (no broadcast) =====
        def test_mul_basic():
            name = "mul/basic"
            x_np = rng.normal(size=(2, 3))
            y_np = rng.normal(size=(2, 3))
            x = Tensor(x_np, requires_grad=True, name="x")
            y = Tensor(y_np, requires_grad=True, name="y")
            t0 = time.time()
            out = x * y
            upstream = np.ones_like(out.data)
            out.backward(grad=upstream)
            f_ok, f_err = _cmp_arrays(out.data, x_np * y_np)
            dx_expect = upstream * y_np
            dy_expect = upstream * x_np
            g_ok1, g1_err = _cmp_arrays(x.grad, dx_expect)
            g_ok2, g2_err = _cmp_arrays(y.grad, dy_expect)
            g_ok = g_ok1 and g_ok2
            g_err = max(g1_err, g2_err)
            secs = time.time() - t0
            if not (f_ok and g_ok) and verbose:
                details.append((
                    name,
                    ("forward", out.data, x_np * y_np),
                    ("grad_x", x.grad, dx_expect),
                    ("grad_y", y.grad, dy_expect),
                ))
            record(name, f_ok, g_ok, f_err, g_err, secs)

        # ===== Test 4: mul (broadcast) =====
        def test_mul_broadcast():
            name = "mul/broadcast"
            x_np = rng.normal(size=(2, 3))
            b_np = rng.normal(size=(1, 3))
            x = Tensor(x_np, requires_grad=True, name="x")
            b = Tensor(b_np, requires_grad=True, name="b")
            t0 = time.time()
            out = x * b
            upstream = np.ones_like(out.data)
            out.backward(grad=upstream)
            f_ok, f_err = _cmp_arrays(out.data, x_np * b_np)
            dx_expect = upstream * b_np
            db_expect = (upstream * x_np).sum(axis=0, keepdims=True)
            g_ok1, g1_err = _cmp_arrays(x.grad, dx_expect)
            g_ok2, g2_err = _cmp_arrays(b.grad, db_expect)
            g_ok = g_ok1 and g_ok2
            g_err = max(g1_err, g2_err)
            secs = time.time() - t0
            if not (f_ok and g_ok) and verbose:
                details.append((
                    name,
                    ("forward", out.data, x_np * b_np),
                    ("grad_x", x.grad, dx_expect),
                    ("grad_b", b.grad, db_expect),
                ))
            record(name, f_ok, g_ok, f_err, g_err, secs, notes="Broadcast axis=0")

        # ===== Test 5: power (elementwise) =====
        def test_pow3():
            name = "pow/^3"
            x_np = rng.normal(size=(2, 3))
            x = Tensor(x_np, requires_grad=True, name="x")
            p = 3.0
            t0 = time.time()
            out = x ** p
            upstream = np.ones_like(out.data)
            out.backward(grad=upstream)
            f_ok, f_err = _cmp_arrays(out.data, x_np ** p)
            dx_expect = upstream * (p * (x_np ** (p - 1)))
            g_ok, g_err = _cmp_arrays(x.grad, dx_expect)
            secs = time.time() - t0
            if not (f_ok and g_ok) and verbose:
                details.append((
                    name,
                    ("forward", out.data, x_np ** p),
                    ("grad_x", x.grad, dx_expect),
                ))
            record(name, f_ok, g_ok, f_err, g_err, secs)

        # ===== Test 6: chain rule ( (3x+2) * (x^2) ) =====
        def test_chain_poly():
            name = "chain/(3x+2)*x^2"
            x_np = rng.normal(size=(2, 3))
            x = Tensor(x_np, requires_grad=True, name="x")
            t0 = time.time()
            out = (x * 3.0 + 2.0) * (x ** 2.0)
            upstream = np.ones_like(out.data)
            out.backward(grad=upstream)
            f_ok, f_err = _cmp_arrays(out.data, (3.0 * x_np + 2.0) * (x_np ** 2.0))
            # d/dx [(3x+2)*x^2] = 3*x^2 + (3x+2)*2x
            dx_expect = 3.0 * (x_np ** 2.0) + (3.0 * x_np + 2.0) * (2.0 * x_np)
            g_ok, g_err = _cmp_arrays(x.grad, dx_expect)
            secs = time.time() - t0
            if not (f_ok and g_ok) and verbose:
                details.append((
                    name,
                    ("forward", out.data, (3.0 * x_np + 2.0) * (x_np ** 2.0)),
                    ("grad_x", x.grad, dx_expect),
                ))
            record(name, f_ok, g_ok, f_err, g_err, secs, notes="Product rule check")

        # ===== Test 7: scalar * tensor grad =====
        def test_scalar_mul_grad():
            name = "scalar*tensor grad"
            x_np = rng.normal(size=(2, 3))
            k = 7.5
            x = Tensor(x_np, requires_grad=True, name="x")
            t0 = time.time()
            out = k * x
            upstream = np.ones_like(out.data)
            out.backward(grad=upstream)
            f_ok, f_err = _cmp_arrays(out.data, k * x_np)
            dx_expect = upstream * k
            g_ok, g_err = _cmp_arrays(x.grad, dx_expect)
            secs = time.time() - t0
            if not (f_ok and g_ok) and verbose:
                details.append((name, ("forward", out.data, k * x_np), ("grad_x", x.grad, dx_expect)))
            record(name, f_ok, g_ok, f_err, g_err, secs)

        # ===== Test 8: properties (commutativity, distributivity) =====
        def test_properties():
            name = "algebraic properties"
            a_np = rng.normal(size=(2, 3))
            b_np = rng.normal(size=(2, 3))
            c_np = rng.normal(size=(2, 3))
            a = Tensor(a_np); b = Tensor(b_np); c = Tensor(c_np)
            t0 = time.time()
            # (a + b) == (b + a)
            ok1, e1 = _cmp_arrays((a + b).data, (b + a).data)
            # (a * b) == (b * a)
            ok2, e2 = _cmp_arrays((a * b).data, (b * a).data)
            # a*(b + c) == a*b + a*c
            ok3, e3 = _cmp_arrays((a * (b + c)).data, (a * b + a * c).data)
            ok = ok1 and ok2 and ok3
            emax = max(e1, e2, e3)
            secs = time.time() - t0
            if not ok and verbose:
                details.append((name,
                                ("comm add", (a + b).data, (b + a).data),
                                ("comm mul", (a * b).data, (b * a).data),
                                ("distrib",   (a * (b + c)).data, (a * b + a * c).data)))
            record(name, ok, True, emax, 0.0, secs, notes="Commutativity + Distributivity")

        # ===== Test 9: finite-difference grad check =====
        def test_finite_diff():
            name = "finite-diff grad check"
            x_np = rng.normal(size=(2, 3))
            # define f(T) = sum( (T*2 + 1) ** 3 )  (scalarized by sum)
            def f_builder(T):
                return (T * 2.0 + 1.0) ** 3.0
            # autodiff
            x = Tensor(x_np.copy(), requires_grad=True, name="x")
            t0 = time.time()
            y = f_builder(x)
            y.backward(grad=np.ones_like(y.data))
            grad_auto = x.grad
            # finite diff
            grad_fd = _finite_diff_grad(f_builder, x_np.copy(), eps=1e-6)
            secs = time.time() - t0
            g_ok, g_err = _cmp_arrays(grad_auto, grad_fd, rtol=1e-5, atol=1e-6)
            if not g_ok and verbose:
                details.append((name, ("grad_auto", grad_auto, "grad_fd", grad_fd)))
            # forward equivalence is trivial here (we didn't compare out.data), mark as True
            record(name, True, g_ok, 0.0, g_err, secs, notes="eps=1e-6")

        # ---------- run all ----------
        print("\n=== Tensor Test Suite ===")
        start = time.time()
        test_add_basic()
        test_add_broadcast()
        test_mul_basic()
        test_mul_broadcast()
        test_pow3()
        test_chain_poly()
        test_scalar_mul_grad()
        test_properties()
        test_finite_diff()
        total = time.time() - start

        # ---------- summary table ----------
        _table(
            headers=["Test", "Forward", "Grad", "Forward max|Δ|", "Grad max|Δ|", "Time", "Notes"],
            rows=results,
            title="Summary"
        )
        print(f"Total time: {total*1e3:.1f} ms\n")

        # ---------- (optional) failure details ----------
        if verbose and details:
            print("Details (failures):")
            for item in details:
                name, *pairs = item
                print(f"\n-- {name} --")
                for p in pairs:
                    if len(p) == 3:
                        tag, got, exp = p
                        print(f"[{tag}] got:\n{got}\n[expected]:\n{exp}")
                    else:
                        print(p)

if __name__ == "__main__":
    # Set verbose=True to print per-test matrices when a test fails
    Tensor.run_tests(verbose=False)
