# python3 - <<'PY'
import numpy as np
import matmul

M, K, N = 64, 128, 32
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

# GPU matmul
C_gpu = matmul.matmul(A, B)

# NumPy CPU matmul
C_np = A @ B

# Compare
diff = np.max(np.abs(C_gpu - C_np))
print("Max abs diff:", diff)
print("Match?", diff < 1e-4)
# PY
