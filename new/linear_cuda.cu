// linear_cuda.cu
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(expr) do {                             \
    cudaError_t _err = (expr);                            \
    if (_err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error %s at %s:%d\n",       \
                cudaGetErrorString(_err), __FILE__, __LINE__); \
        return 1;                                         \
    }                                                     \
} while(0)

template<int TILE = 16>
__global__ void linear_forward_kernel(
    const float* __restrict__ X, // M x K
    const float* __restrict__ W, // K x N
    const float* __restrict__ b, // N (nullable)
    float* __restrict__ Y,       // M x N
    int M, int K, int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int m0 = blockIdx.y * TILE;
    int n0 = blockIdx.x * TILE;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int m = m0 + ty;
    int n = n0 + tx;

    float acc = 0.f;

    // Loop over tiles along K
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Load a TILE from X (M x K)
        if (m < M && (k0 + tx) < K)
            As[ty][tx] = X[m * K + (k0 + tx)];
        else
            As[ty][tx] = 0.f;

        // Load a TILE from W (K x N)
        if ((k0 + ty) < K && n < N)
            Bs[ty][tx] = W[(k0 + ty) * N + n];
        else
            Bs[ty][tx] = 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        Y[m * N + n] = acc + (b ? b[n] : 0.f);
    }
}

// Host wrapper that manages device buffers (simple API for ctypes)
extern "C" int linear_forward(
    const float* hX,  // host ptrs
    const float* hW,
    const float* hB,  // nullable
    float* hY,
    int M, int K, int N
) {
    if (!hX || !hW || !hY || M <= 0 || K <= 0 || N <= 0) return 1;

    size_t bytesX = (size_t)M * K * sizeof(float);
    size_t bytesW = (size_t)K * N * sizeof(float);
    size_t bytesY = (size_t)M * N * sizeof(float);
    size_t bytesB = (size_t)N * sizeof(float);

    float *dX=nullptr, *dW=nullptr, *dB=nullptr, *dY=nullptr;

    CUDA_CHECK(cudaMalloc((void**)&dX, bytesX));
    CUDA_CHECK(cudaMalloc((void**)&dW, bytesW));
    CUDA_CHECK(cudaMalloc((void**)&dY, bytesY));
    if (hB) CUDA_CHECK(cudaMalloc((void**)&dB, bytesB));

    CUDA_CHECK(cudaMemcpy(dX, hX, bytesX, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW, bytesW, cudaMemcpyHostToDevice));
    if (hB) CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid( (N + block.x - 1) / block.x,
               (M + block.y - 1) / block.y );

    linear_forward_kernel<16><<<grid, block>>>(dX, dW, dB, dY, M, K, N);
    cudaError_t kernErr = cudaGetLastError();
    if (kernErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(kernErr));
        cudaFree(dX); cudaFree(dW); if (dB) cudaFree(dB); cudaFree(dY);
        return 2;
    }

    CUDA_CHECK(cudaMemcpy(hY, dY, bytesY, cudaMemcpyDeviceToHost));

    cudaFree(dX); cudaFree(dW); if (dB) cudaFree(dB); cudaFree(dY);
    return 0;
}
