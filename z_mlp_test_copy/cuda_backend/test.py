# test.py
# No third-party packages. Pure-Python refs, build and verify kernels.

import os, sys, math, time, subprocess
from array import array

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
from tensorcore_py import Tensor, LIB_PATH

def build_if_needed():
    if os.path.exists(LIB_PATH):
        return
    print("libtensor.so missing; building via tensorcore_py.py on import.")

# ----- Pure-Python reference ops (no NumPy) -----
def matmul_py(A, B, M, N, K):  # row-major lists
    C = [0.0]*(M*N)
    for i in range(M):
        for k in range(K):
            aik = A[i*K + k]
            baseB = k*N
            baseC = i*N
            for j in range(N):
                C[baseC + j] += aik * B[baseB + j]
    return C

def relu_py(x):
    return [v if v > 0.0 else 0.0 for v in x]

def sigmoid_py(x):
    out = []
    for v in x:
        out.append(1.0/(1.0+math.exp(-v)))
    return out

def softmax_rows_py(X, rows, cols):
    Y = [0.0]*(rows*cols)
    for i in range(rows):
        row = X[i*cols:(i+1)*cols]
        m = max(row) if cols>0 else -1e30
        exps = [math.exp(v - m) for v in row]
        s = sum(exps) if cols>0 else 1.0
        for j in range(cols):
            Y[i*cols + j] = exps[j]/s
    return Y

def sum_rows_py(X, rows, cols):
    out = [0.0]*rows
    for i in range(rows):
        s = 0.0
        base = i*cols
        for j in range(cols):
            s += X[base + j]
        out[i] = s
    return out

def max_abs_err(a, b):
    return max(abs(x-y) for x,y in zip(a,b)) if a and b else 0.0

def rel_err(a, b):
    mnum = 0.0
    mden = 1e-12
    for x,y in zip(a,b):
        mnum = max(mnum, abs(x-y))
        mden = max(mden, max(1.0, abs(y)))
    return mnum/mden

def check_close(name, got, ref, atol=1e-4, rtol=1e-4):
    mae = max_abs_err(got, ref)
    rre = rel_err(got, ref)
    print(f"{name:20s} | max_abs_err={mae:.3e} rel_err={rre:.3e}")
    if not (mae <= atol or rre <= rtol):
        raise AssertionError(f"{name} mismatch")

def test_correctness():
    # Matmul small
    M,N,K = 5,7,3
    A = [(i*K + j + 1)*0.1 for i in range(M) for j in range(K)]
    B = [(i*N + j + 1)*0.05 for i in range(K) for j in range(N)]
    Cref = matmul_py(A,B,M,N,K)

    At = Tensor(M,K,A,device="cuda")
    Bt = Tensor(K,N,B,device="cuda")
    Ct = At.matmul(Bt).cpu()
    check_close("matmul 5x7x3", Ct.tolist(), Cref, atol=1e-4, rtol=1e-4)

    # Non-multiple-of-16: 37x29 x 29x13
    M,N,K = 37,13,29
    A = [math.sin(0.3*(i*K+j)) for i in range(M) for j in range(K)]
    B = [math.cos(0.2*(i*N+j)) for i in range(K) for j in range(N)]
    Cref = matmul_py(A,B,M,N,K)
    Ct = Tensor(M,K,A,"cuda").matmul(Tensor(K,N,B,"cuda")).cpu()
    check_close("matmul 37x29x13", Ct.tolist(), Cref, atol=2e-4, rtol=2e-4)

    # ReLU
    x = [(-1.0)**i * (i*0.1) for i in range(64)]
    tref = relu_py(x)
    t = Tensor(8,8,x,"cuda").relu().cpu()
    check_close("relu", t.tolist(), tref)

    # Sigmoid
    tref = sigmoid_py(x)
    t = Tensor(8,8,x,"cuda").sigmoid().cpu()
    check_close("sigmoid", t.tolist(), tref, atol=2e-4, rtol=2e-4)

    # Softmax rows
    rows, cols = 6, 11
    X = [math.sin(0.7*(i*cols+j)) for i in range(rows) for j in range(cols)]
    Yref = softmax_rows_py(X, rows, cols)
    Y = Tensor(rows, cols, X, "cuda").softmax_rows().cpu()
    check_close("softmax rows", Y.tolist(), Yref, atol=2e-5, rtol=2e-5)

    # Sum rows
    Sref = sum_rows_py(X, rows, cols)
    S = Tensor(rows, cols, X, "cuda").sum_rows().cpu()
    check_close("sum rows", S.tolist(), Sref, atol=2e-5, rtol=2e-5)

def test_timing():
    # Simple wall-clock check
    shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 512, 512),
        (2048,2048,2048)
    ]
    for M,N,K in shapes:
        A = [0.001*(i%97) for i in range(M*K)]
        B = [0.002*(i%89) for i in range(K*N)]
        At = Tensor(M,K,A,"cuda")
        Bt = Tensor(K,N,B,"cuda")
        t0 = time.time()
        Ct = At.matmul(Bt)  # stays on GPU
        C = Ct.cpu().tolist()
        t1 = time.time()
        print(f"matmul {M}x{K} @ {K}x{N} -> {M}x{N} : {t1-t0:.3f} s")

if __name__ == "__main__":
    print("Using:", LIB_PATH)
    test_correctness()
    test_timing()
    print("ALL TESTS PASSED")
