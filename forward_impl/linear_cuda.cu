#include <cuda_runtime.h>

static __global__ void linear_kernel(const float* __restrict__ X, // [M,K]
                                     const float* __restrict__ W, // [N,K]
                                     const float* __restrict__ b, // [N] (nullable)
                                     float* __restrict__ Y,       // [M,N]
                                     int M, int K, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x; // out feature
  int m = blockIdx.y * blockDim.y + threadIdx.y; // batch row
  if (m >= M || n >= N) return;

  const float* xrow = X + (size_t)m * K;   // X[m,:]
  const float* wrow = W + (size_t)n * K;   // W[n,:]
  float acc = 0.0f;

  // naive dot; good enough to start (tile later if needed)
  for (int k = 0; k < K; ++k) {
    acc += xrow[k] * wrow[k];
  }
  if (b) acc += b[n];
  Y[(size_t)m * N + n] = acc;
}

extern "C" void linear_forward_cuda(const float* dX,
                                    const float* dW,
                                    const float* db,
                                    float* dY,
                                    int M, int K, int N) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  linear_kernel<<<grid, block>>>(dX, dW, db, dY, M, K, N);
  // Optionally check for launch errors:
  // cudaDeviceSynchronize();
}
