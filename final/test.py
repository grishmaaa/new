# import numpy as np
# from py_matmul import matmul

# A = np.random.rand(4, 3).astype(np.float32)
# B = np.random.rand(3, 5).astype(np.float32)
# C = np.empty((4, 5), dtype=np.float32)

# x = matmul(A, B, C)

# # pretty print the full matrix
# np.set_printoptions(precision=4, suppress=True)
# print(x)        # or: print(C) — same array

######################################################################

# import numpy as np
# import time
# from py_matmul import matmul

# A = np.random.rand(4000, 3000).astype(np.float32)
# B = np.random.rand(3000, 5000).astype(np.float32)
# C = np.empty((4000, 5000), dtype=np.float32)

# t0 = time.perf_counter()
# x = matmul(A, B, C)
# dt = time.perf_counter() - t0

# np.set_printoptions(precision=4, suppress=True)
# print(x)
# print(f"Elapsed: {dt:.6f} s")


######################################################################

import numpy as np
import time
from py_matmul import matmul

A = np.random.rand(2048, 2048).astype(np.float32)
B = np.random.rand(2048, 2048).astype(np.float32)
C = np.empty((2048, 2048), dtype=np.float32)
print(f'A = {A}')
print(f'B = {B}')
# print(f'a = {A}')

# Warm-up (stabilize clocks/first-call overheads)
matmul(A, B, C)

t0 = time.perf_counter()
x = matmul(A, B, C)   # writes into C and returns it
dt = time.perf_counter() - t0

np.set_printoptions(precision=4, suppress=True)
# print('A @ B ')
print(f"A @ B  = {x}")                    # or print(C)
print(f"Elapsed: {dt:.6f} s")
