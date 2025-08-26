"""
Neural Networks from First Principles — Stage 3

Single-file, no frameworks. Implements:
  • Reverse‑mode autodiff Tensor (add/mul/div/matmul/power/exp/log/tanh/sigmoid/relu/softplus
    + sum/mean/reshape/transpose + broadcasting-aware backprop)
  • Layers: Linear, simple MLP(Sequential)
  • Losses: MSELoss, BCEWithLogitsLoss, CrossEntropyLoss (one‑hot)
  • Optimizers: SGD, Adam (with weight decay & gradient clipping)
  • Tests: numerical grad check; XOR classification demo

Only dependency: NumPy (for array math). No DL libraries are used.
"""
from __future__ import annotations
import math
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray

# ------------------------------
# Autodiff Tensor
# ------------------------------
class Tensor:
    """A minimal reverse‑mode autodiff tensor.

    - Supports broadcasting, 0D/1D/2D arrays
    - Keep graph via parents + backward closures
    """

    def __init__(self, data: Any, requires_grad: bool = False, name: Optional[str] = None):
        self.data: Array = np.array(data, dtype=np.float64)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[Array] = None
        self._backward: Callable[[], None] = lambda: None
        self._parents: Tuple[Tensor, ...] = tuple()
        self.name = name

    # ---- convenience ----
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

    # ---- graph utilities ----
    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def _set_parents(self, *parents: "Tensor"):
        self._parents = tuple(p for p in parents if isinstance(p, Tensor))

    # ---- autograd core ----
    def backward(self, grad: Optional[Array] = None) -> None:
        """Backprop from this scalar (or tensor with given upstream grad)."""
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

    # ---- helpers for broadcasting in backward ----
    @staticmethod
    def _unbroadcast(g: Array, target_shape: Tuple[int, ...]) -> Array:
        """Sum grad g to match target_shape (reverse of broadcasting)."""
        if g.shape == target_shape:
            return g
        # Reduce extra leading dims
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)
        # Sum along axes where target has 1
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
        out._set_parents(self, other)

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + Tensor._unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad = other.grad + Tensor._unbroadcast(out.grad, other.shape)
        out._backward = _bw
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        return self.__add__(-other)

    def __rsub__(self, other: Any) -> "Tensor":
        return (-self).__add__(other)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad - out.grad
        out._backward = _bw
        return out

    def __mul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._set_parents(self, other)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + Tensor._unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad = other.grad + Tensor._unbroadcast(out.grad * self.data, other.shape)
        out._backward = _bw
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._set_parents(self, other)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + Tensor._unbroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad = other.grad - Tensor._unbroadcast(out.grad * self.data / (other.data**2), other.shape)
        out._backward = _bw
        return out

    def __pow__(self, p: float) -> "Tensor":
        out = Tensor(self.data ** p, requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad * (p * (self.data ** (p - 1)))
        out._backward = _bw
        return out

    # ------------------------------
    # Reductions & shaping
    # ------------------------------
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self.grad = self.grad + np.ones_like(self.data) * g
        out._backward = _bw
        return out

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        denom = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad.reshape(self.shape)
        out._backward = _bw
        return out

    @property
    def T(self) -> "Tensor":
        out = Tensor(self.data.T, requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad.T
        out._backward = _bw
        return out

    # ------------------------------
    # Matrix ops
    # ------------------------------
    def __matmul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._set_parents(self, other)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad = other.grad + self.data.T @ out.grad
        out._backward = _bw
        return out

    # ------------------------------
    # Nonlinearities
    # ------------------------------
    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad * e
        out._backward = _bw
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad / self.data
        out._backward = _bw
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad * (1 - t**2)
        out._backward = _bw
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad)
        out._set_parents(self)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad * (s * (1 - s))
        out._backward = _bw
        return out

    def relu(self) -> "Tensor":
        y = np.maximum(self.data, 0.0)
        out = Tensor(y, requires_grad=self.requires_grad)
        out._set_parents(self)
        mask = (self.data > 0.0)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad * mask
        out._backward = _bw
        return out

    def softplus(self) -> "Tensor":
        # Stable softplus
        x = self.data
        y = np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
        out = Tensor(y, requires_grad=self.requires_grad)
        out._set_parents(self)
        s = 1.0 / (1.0 + np.exp(-x))  # sigmoid(x)
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad = self.grad + out.grad * s
        out._backward = _bw
        return out

    # ------------------------------
    # Utilities
    # ------------------------------
    @staticmethod
    def maximum(a: "Tensor", b: "Tensor") -> "Tensor":
        a = a if isinstance(a, Tensor) else Tensor(a)
        b = b if isinstance(b, Tensor) else Tensor(b)
        y = np.maximum(a.data, b.data)
        out = Tensor(y, requires_grad=a.requires_grad or b.requires_grad)
        out._set_parents(a, b)
        mask_a = (a.data >= b.data)
        mask_b = ~mask_a
        def _bw():
            if a.requires_grad:
                a._ensure_grad()
                a.grad = a.grad + Tensor._unbroadcast(out.grad * mask_a, a.shape)
            if b.requires_grad:
                b._ensure_grad()
                b.grad = b.grad + Tensor._unbroadcast(out.grad * mask_b, b.shape)
        out._backward = _bw
        return out

    @staticmethod
    def logsumexp(x: "Tensor", axis: int) -> "Tensor":
        # lse(x) = m + log(sum(exp(x - m))) for stability
        m = Tensor(np.max(x.data, axis=axis, keepdims=True), requires_grad=False)
        z = (x - m).exp().sum(axis=axis, keepdims=True).log() + m
        return z


# Alias
Parameter = Tensor


# ------------------------------
# Modules & layers
# ------------------------------
class Module:
    def parameters(self) -> List[Parameter]:
        ps: List[Parameter] = []
        for k, v in self.__dict__.items():
            if isinstance(v, Parameter) and v.requires_grad:
                ps.append(v)
            elif isinstance(v, Module):
                ps.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        ps.extend(item.parameters())
                    elif isinstance(item, Parameter) and item.requires_grad:
                        ps.append(item)
        return ps

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # override
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # Kaiming/He uniform-ish: scale ~ 1/sqrt(in)
        limit = math.sqrt(2.0 / in_features)
        W = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.W = Parameter(W, requires_grad=True, name="W")
        self.b = Parameter(np.zeros((1, out_features)), requires_grad=True, name="b") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.W
        if self.b is not None:
            y = y + self.b  # broadcast over batch
        return y


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ------------------------------
# Activations (as Modules for convenience)
# ------------------------------
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


# ------------------------------
# Losses
# ------------------------------
class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()

class BCEWithLogitsLoss(Module):
    """Binary cross‑entropy given logits and targets in {0,1}. Stable via softplus.
       L = mean( softplus(x) - y*x )
    """
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return logits.softplus() - (logits * targets)
        # average over all elements
        
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return (logits.softplus() - (logits * targets)).mean()

class CrossEntropyLoss(Module):
    """Multi-class cross entropy with one‑hot targets of shape (N, C).
       Uses log-softmax constructed from primitives, so gradients flow.
    """
    def forward(self, logits: Tensor, targets_one_hot: Tensor) -> Tensor:
        # logits: (N,C), targets_one_hot: (N,C)
        lse = Tensor.logsumexp(logits, axis=1)  # shape (N,1)
        log_softmax = logits - lse  # broadcast
        nll = -(targets_one_hot * log_softmax).sum(axis=1)  # (N,)
        return nll.mean()


# ------------------------------
# Optimizers
# ------------------------------
class Optimizer:
    def __init__(self, params: Iterable[Parameter]):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0, grad_clip: Optional[float] = None):
        super().__init__(params)
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.grad_clip = grad_clip
        self._vel = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0.0:
                g = g + self.wd * p.data
            if self.grad_clip is not None:
                norm = np.linalg.norm(g)
                if norm > self.grad_clip:
                    g = g * (self.grad_clip / (norm + 1e-12))
            self._vel[i] = self.m * self._vel[i] + (1 - self.m) * g
            p.data = p.data - self.lr * self._vel[i]


class Adam(Optimizer):
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3, betas: Tuple[float,float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, grad_clip: Optional[float] = None):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.grad_clip = grad_clip
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0.0:
                g = g + self.wd * p.data
            if self.grad_clip is not None:
                norm = np.linalg.norm(g)
                if norm > self.grad_clip:
                    g = g * (self.grad_clip / (norm + 1e-12))
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data = p.data - self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ------------------------------
# Utilities
# ------------------------------

def one_hot(y: Array, num_classes: int) -> Array:
    y = y.astype(int)
    out = np.zeros((y.size, num_classes), dtype=np.float64)
    out[np.arange(y.size), y] = 1.0
    return out


def numerical_grad(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-6) -> Array:
    """Finite-diff gradient of scalar function f at tensor x."""
    base = x.data.copy()
    grad = np.zeros_like(base)
    it = np.nditer(base, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = base[idx]
        base[idx] = old + eps
        x.data = base
        y1 = f(x).data.item()
        base[idx] = old - eps
        x.data = base
        y2 = f(x).data.item()
        grad[idx] = (y1 - y2) / (2 * eps)
        base[idx] = old
        x.data = base
        it.iternext()
    return grad


# ------------------------------
# Demo: XOR with a tiny MLP
# ------------------------------
class MLP(Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int, activation: str = 'relu'):
        acts = {'relu': ReLU, 'tanh': Tanh, 'sigmoid': Sigmoid}
        layers: List[Module] = []
        d = in_dim
        for h in hidden:
            layers.append(Linear(d, h))
            layers.append(acts[activation]())
            d = h
        layers.append(Linear(d, out_dim))
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def demo_xor(epochs: int = 2000, lr: float = 0.05) -> None:
    # Data (XOR)
    X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=np.float64)
    y = np.array([[0.],[1.],[1.],[0.]], dtype=np.float64)

    model = MLP(2, [8], 1)
    criterion = BCEWithLogitsLoss()
    optim = Adam(model.parameters(), lr=lr)

    for t in range(1, epochs + 1):
        # forward
        logits = model(Tensor(X, requires_grad=False))
        loss = criterion(logits, Tensor(y))
        # backward
        model.zero_grad()
        loss.backward()
        # step
        optim.step()

        if t % 200 == 0 or t == 1:
            # compute binary accuracy
            probs = logits.sigmoid().data
            acc = ((probs > 0.5) == y).mean()
            print(f"step {t:4d} | loss {loss.data.item():.6f} | acc {acc*100:.1f}%")

    print("\nFinal probs:")
    print(logits.sigmoid().data)


# ------------------------------
# Self-test: basic gradient check
# ------------------------------

def _self_test_grad():
    np.random.seed(0)
    x = Tensor(np.random.randn(3,3), requires_grad=True)
    W = Tensor(np.random.randn(3,3), requires_grad=True)
    def f(z: Tensor) -> Tensor:
        return ((z @ W).relu().sum())
    # analytic
    y = f(x)
    x.grad = None; W.grad = None
    y.backward()
    gx = x.grad.copy()
    # numerical
    ng = numerical_grad(lambda z: f(z), x)
    err = np.abs(gx - ng).max()
    print(f"gradcheck max abs error: {err:.3e}")


if __name__ == "__main__":
    _self_test_grad()
    print("\nTraining XOR:")
    demo_xor()
