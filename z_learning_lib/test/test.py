import random, time
import fastmatmul
# import fastmatmul1 as fastmatmul
import numpy as np
import random, time

def make_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]

sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# header
print(f"{'Size':>10} | {'Naive fastmatmul1 (s)':>25} | {'NumPy (OpenBLAS) (s)':>25}")
print("-" * 65)

for n in sizes:
    # naive C extension
    A = make_matrix(n)
    B = make_matrix(n)
    t0 = time.time()
    fastmatmul.matmul(A, B)
    naive_time = time.time() - t0

    # numpy/OpenBLAS
    A_np = np.random.rand(n, n)
    B_np = np.random.rand(n, n)
    t0 = time.time()
    A_np @ B_np
    numpy_time = time.time() - t0

    # row
    print(f"{n:>10} | {naive_time:>25.6f} | {numpy_time:>25.6f}")
