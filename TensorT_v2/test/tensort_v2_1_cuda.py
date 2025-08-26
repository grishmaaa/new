# tensort_v2_1_cuda.py — GPU-backed version of your TensorT (forward on CUDA, Python autograd)
import ctypes, os
from typing import Tuple, List, Sequence, Any

# ===== ctypes glue to libtensort.so =====
LIB = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libtensort.so'))

class C_TT_Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_longlong),
    ]

# prototypes
LIB.tt_product.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB.tt_product.restype  = ctypes.c_longlong

LIB.tt_tensor_init_host.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB.tt_tensor_init_host.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_tensor_init_gpu.argtypes  = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB.tt_tensor_init_gpu.restype   = ctypes.POINTER(C_TT_Tensor)

LIB.tt_tensor_free_host.argtypes = [ctypes.POINTER(C_TT_Tensor)]
LIB.tt_tensor_free_host.restype  = None
LIB.tt_tensor_free_gpu.argtypes  = [ctypes.POINTER(C_TT_Tensor)]
LIB.tt_tensor_free_gpu.restype   = None

LIB.tt_copy_h2d.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.POINTER(C_TT_Tensor)]
LIB.tt_copy_h2d.restype  = None
LIB.tt_copy_d2h.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.POINTER(C_TT_Tensor)]
LIB.tt_copy_d2h.restype  = None

LIB.tt_add_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.POINTER(C_TT_Tensor)]
LIB.tt_add_gpu.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_sub_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.POINTER(C_TT_Tensor)]
LIB.tt_sub_gpu.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_mul_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.POINTER(C_TT_Tensor)]
LIB.tt_mul_gpu.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_div_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.POINTER(C_TT_Tensor)]
LIB.tt_div_gpu.restype  = ctypes.POINTER(C_TT_Tensor)

LIB.tt_matmul2d_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.POINTER(C_TT_Tensor)]
LIB.tt_matmul2d_gpu.restype  = ctypes.POINTER(C_TT_Tensor)

LIB.tt_print_host.argtypes = [ctypes.POINTER(C_TT_Tensor)]
LIB.tt_print_host.restype  = None

LIB.tt_add_scalar_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.c_float]
LIB.tt_add_scalar_gpu.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_sub_scalar_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.c_float]
LIB.tt_sub_scalar_gpu.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_rsub_scalar_gpu.argtypes= [ctypes.c_float, ctypes.POINTER(C_TT_Tensor)]
LIB.tt_rsub_scalar_gpu.restype = ctypes.POINTER(C_TT_Tensor)
LIB.tt_mul_scalar_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.c_float]
LIB.tt_mul_scalar_gpu.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_div_scalar_gpu.argtypes = [ctypes.POINTER(C_TT_Tensor), ctypes.c_float]
LIB.tt_div_scalar_gpu.restype  = ctypes.POINTER(C_TT_Tensor)
LIB.tt_rdiv_scalar_gpu.argtypes= [ctypes.c_float, ctypes.POINTER(C_TT_Tensor)]
LIB.tt_rdiv_scalar_gpu.restype = ctypes.POINTER(C_TT_Tensor)


# ===== small helpers =====
def _is_seq(x): return isinstance(x, (list, tuple))

def _shape_of(data) -> Tuple[int, ...]:
    if _is_seq(data):
        return (len(data),) + _shape_of(data[0]) if len(data)>0 else (0,)
    return ()

def _flatten(data, out: List[float]):
    if _is_seq(data):
        for y in data: _flatten(y, out)
    else:
        out.append(float(data))

# ===== CUDA-backed TensorT =====
class TensorT:
    def __init__(self, data: Any, _op=None, _parent: tuple=()):
        # accept nested lists or scalars
        self.shape = _shape_of(data) if _is_seq(data) else ()
        flat: List[float] = []
        _flatten(data if _is_seq(data) else [data], flat)
        self.ndim = len(self.shape)

        # keep host+device C tensors
        if self.ndim == 0:
            shape_list = [1]
        else:
            shape_list = list(self.shape)
        shape_c = (ctypes.c_int * len(shape_list))(*shape_list)

        self.c_host = LIB.tt_tensor_init_host(shape_c, len(shape_list))
        self.c_dev  = LIB.tt_tensor_init_gpu(shape_c, len(shape_list))

        # fill host data (pad scalar case)
        n = self.c_host.contents.size
        for i in range(n):
            self.c_host.contents.data[i] = flat[0] if len(flat)==1 else flat[i]
        LIB.tt_copy_h2d(self.c_host, self.c_dev)

        self.grad = None
        self._op = _op
        self._parent = _parent
        self.backward_fn = None

    # ---- lifecycle ----
    def __del__(self):
        try:
            if hasattr(self, 'c_host') and self.c_host: LIB.tt_tensor_free_host(self.c_host)
            if hasattr(self, 'c_dev')  and self.c_dev:  LIB.tt_tensor_free_gpu(self.c_dev)
        except Exception:
            pass

    # ---- debug ----
    def print(self):
        # Copy device -> host temp and print
        tmp = LIB.tt_tensor_init_host(self.c_host.contents.shape, self.c_host.contents.ndim)
        LIB.tt_copy_d2h(self.c_dev, tmp)
        LIB.tt_print_host(tmp)
        LIB.tt_tensor_free_host(tmp)

    # ---- math (GPU forward; same-shaped for elementwise) ----
    def _ew(self, other: 'TensorT', which: str):
    # normalize
        if not isinstance(other, TensorT):
            # scalar RHS
            a = self.c_dev
            s = ctypes.c_float(float(other))
            fn_scalar = {
                'add': LIB.tt_add_scalar_gpu,
                'sub': LIB.tt_sub_scalar_gpu,
                'mul': LIB.tt_mul_scalar_gpu,
                'div': LIB.tt_div_scalar_gpu,
            }[which]
            c_dev = fn_scalar(a, s)
            out = TensorT.__new__(TensorT)
            out.shape = self.shape; out.ndim = self.ndim
            out.c_host = LIB.tt_tensor_init_host(self.c_host.contents.shape, self.ndim)
            out.c_dev = c_dev; out.grad=None; out._op=which; out._parent=(self,)
            out.backward_fn=None
            return out

        # tensor-tensor path (sizes must match for now)
        if self.c_host.contents.size != other.c_host.contents.size:
            raise NotImplementedError("Broadcasted elementwise on GPU not implemented yet")
    ...


    def __add__(self, other):
        out = self._ew(other, 'add')
        a_shape = self.shape; b_shape = other.shape if isinstance(other, TensorT) else ()
        def backward_fn(grad_out):
            return (grad_out, grad_out)
        out.backward_fn = backward_fn
        return out

    def __sub__(self, other):
        out = self._ew(other, 'sub')
        def backward_fn(grad_out):
            return (grad_out, TensorT(-1.0) * grad_out)
        out.backward_fn = backward_fn
        return out

    def __mul__(self, other):
        out = self._ew(other, 'mul')
        self_ref, other_ref = self, other if isinstance(other, TensorT) else TensorT(other)
        def backward_fn(grad_out):
            # dA = grad_out * B ; dB = grad_out * A
            return (grad_out._ew(other_ref, 'mul'), grad_out._ew(self_ref, 'mul'))
        out.backward_fn = backward_fn
        return out
    
    def rscalar(self, s, which):
        if which=='rsub':
            c_dev = LIB.tt_rsub_scalar_gpu(ctypes.c_float(float(s)), self.c_dev)
        else:  # 'rdiv'
            c_dev = LIB.tt_rdiv_scalar_gpu(ctypes.c_float(float(s)), self.c_dev)
        out = TensorT.__new__(TensorT)
        out.shape=self.shape; out.ndim=self.ndim
        out.c_host = LIB.tt_tensor_init_host(self.c_host.contents.shape, self.ndim)
        out.c_dev = c_dev; out.grad=None; out._op=which; out._parent=(self,)
        return out

    def __rsub__(self, other):  # other - self
        if not isinstance(other, TensorT):
            return self.rscalar(other, 'rsub')
        return TensorT(other).__sub__(self)

    def __rtruediv__(self, other):  # other / self
        if not isinstance(other, TensorT):
            return self.rscalar(other, 'rdiv')
        return TensorT(other).__truediv__(self)


    def __truediv__(self, other):
        out = self._ew(other, 'div')
        other_ref = other if isinstance(other, TensorT) else TensorT(other)
        def backward_fn(grad_out):
            # dA = grad_out / other; dB = -grad_out * A / (other*other)
            dA = grad_out._ew(other_ref, 'div')
            dB_num = (grad_out._ew(self, 'mul'))
            dB_den = (other_ref._ew(other_ref, 'mul'))
            dB = TensorT(-1.0) * (dB_num._ew(dB_den, 'div'))
            return (dA, dB)
        out.backward_fn = backward_fn
        return out

    # ---- matmul (2D; batch via Python loop if needed) ----
    def tmatmul(self, other: 'TensorT'):
        if not isinstance(other, TensorT): other = TensorT(other)
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise NotImplementedError("Only 2D matmul on GPU here; add batch loop on top if needed")
        c_dev = LIB.tt_matmul2d_gpu(self.c_dev, other.c_dev)
        out = TensorT.__new__(TensorT)
        out.shape = (self.shape[0], other.shape[1])
        out.ndim  = 2
        shape_c = (ctypes.c_int * 2)(*out.shape)
        out.c_host = LIB.tt_tensor_init_host(shape_c, 2)
        out.c_dev  = c_dev
        out.grad = None
        out._op = 'matmul'
        out._parent = (self, other)

        def backward_fn(grad):
            # grad wrt A = grad @ B^T ; wrt B = A^T @ grad
            # Implement using two more mm calls and ad-hoc transposes by reshaping on host for now
            # Pull shapes
            M,K = self.shape; K2,N = other.shape
            # Build B^T and A^T on host via temporary copies (small helper)
            # Copy B to host, transpose on CPU, send back to GPU
            def _transpose_host(c_tensor):
                # c_tensor: POINTER(C_TT_Tensor) for device; create host tmp, D2H, transpose flat
                d = c_tensor.contents
                # read to host
                h = LIB.tt_tensor_init_host(d.shape, d.ndim)
                LIB.tt_copy_d2h(c_tensor, h)
                Mx,Nx = d.shape[0], d.shape[1]
                flat = [0.0]*(Mx*Nx)
                for i in range(Mx):
                    for j in range(Nx):
                        flat[j*Mx + i] = h.contents.data[i*Nx + j]
                # write transposed back to host tensor with swapped shape
                shape_c2 = (ctypes.c_int*2)(Nx, Mx)
                hT = LIB.tt_tensor_init_host(shape_c2, 2)
                for i in range(Mx*Nx): hT.contents.data[i] = ctypes.c_float(flat[i])
                dT = LIB.tt_tensor_init_gpu(shape_c2, 2)
                LIB.tt_copy_h2d(hT, dT)
                LIB.tt_tensor_free_host(h)
                LIB.tt_tensor_free_host(hT)
                return dT

            B_T = _transpose_host(other.c_dev)
            A_T = _transpose_host(self.c_dev)
            dA_dev = LIB.tt_matmul2d_gpu(grad.c_dev, B_T)
            dB_dev = LIB.tt_matmul2d_gpu(A_T, grad.c_dev)

            # wrap grads
            def wrap_dev(dev_ptr, shape):
                g = TensorT.__new__(TensorT)
                g.shape = shape; g.ndim = 2
                shape_cg = (ctypes.c_int*2)(*shape)
                g.c_host = LIB.tt_tensor_init_host(shape_cg, 2)
                g.c_dev  = dev_ptr
                g.grad = None; g._op=None; g._parent=()
                return g
            return wrap_dev(dA_dev, (M,K)), wrap_dev(dB_dev, (K,N))

        out.backward_fn = backward_fn
        return out

    # ---- autograd driver ----
    @staticmethod
    def unit_tensor(unit: float, shape: Sequence[int]):
        flat = [float(unit)]
        for _ in range(int(__import__('functools').reduce(lambda x,y: x*y, shape, 1))-1): flat.append(float(unit))
        return TensorT([flat[i] for i in range(len(flat))] if len(shape)==1 else [[0.0]*shape[-1]]*0)  # not used here

    def backward(self, grad=None):
        # Incoming grad: if None, use ones-like (not implemented fully here)
        if grad is None:
            raise NotImplementedError("Provide grad for now (e.g., TensorT(1.0))")
        if isinstance(grad, TensorT): g = grad
        else: g = TensorT(grad)
        # set my grad (device-only gradient object for simplicity)
        self.grad = g
        if not self._parent or self.backward_fn is None:
            return
        g_parents = self.backward_fn(g)
        for parent, g_p in zip(self._parent, g_parents):
            if g_p is None: continue
            parent.backward(g_p)