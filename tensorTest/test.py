# test.py
import numpy as np
import cupy as cp
from wrapper import TensorCoreGEMM

def main():
    """
    Initializes matrices, runs the Tensor Core GEMM, and verifies the result.
    """
    print("🚀 Starting Tensor Core GEMM Test...")

    # Define matrix dimensions that satisfy the kernel's alignment requirements.
    # With default settings (BLOCK_ROW_TILES=4, BLOCK_COL_TILES=4):
    # - M must be a multiple of 4 * 16 = 64
    # - N must be a multiple of 4 * 16 = 64
    # - K must be a multiple of 16
    M, N, K = 256, 512, 128
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")

    # 1. Create host matrices with random data
    # Input matrices must be float16 (half precision)
    print("-> Creating random host matrices (A, B) with dtype=float16...")
    h_A = np.random.randn(M, K).astype(np.float16)
    h_B = np.random.randn(K, N).astype(np.float16)

    try:
        # 2. Instantiate the wrapper and run the computation
        print("-> Initializing GEMM wrapper and calling CUDA kernel...")
        gemm_runner = TensorCoreGEMM()
        # The compute method returns the result as a CuPy array on the GPU
        d_C_gpu = gemm_runner.compute(h_A, h_B)
        
        # Copy the result from GPU back to CPU for verification
        h_C_gpu = d_C_gpu.get()
        print("-> GPU computation finished and result copied to host.")

        # 3. Perform computation on CPU for verification
        # Note: NumPy performs matmul in float32 or float64 for higher precision.
        # We cast inputs to float32 to match the GPU kernel's output accumulator type.
        print("-> Calculating reference result on CPU using NumPy...")
        h_C_cpu = np.matmul(h_A.astype(np.float32), h_B.astype(np.float32))

        # 4. Compare results
        print("-> Comparing GPU and CPU results...")
        if np.allclose(h_C_gpu, h_C_cpu, atol=1e-2):
            print("✅ Success! The GPU result matches the CPU result.")
        else:
            print("❌ Failure! The results do not match.")
            # Optional: print differences for debugging
            diff = np.abs(h_C_gpu - h_C_cpu)
            print(f"   Max difference: {np.max(diff)}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run ./compile.sh first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()