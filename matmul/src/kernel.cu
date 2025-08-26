// src/kernel.cu
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__
void matmul_kernel(const float* __restrict__ A,
                   const float* __restrict__ B,
                   float* __restrict__ C,
                   int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float s = 0.0f;
        for (int e = 0; e < K; ++e) {
            s += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = s;
    }
}

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// Exported C ABI for the binding TU.
extern "C"
int matmul_cuda(const float* A, const float* B, float* C, int M, int K, int N)
{
    if (!A || !B || !C || M <= 0 || K <= 0 || N <= 0) {
        std::fprintf(stderr, "matmul_cuda: invalid args\n");
        return -1;
    }

    cudaError_t err = cudaSuccess;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // ✅ Declare launch config BEFORE any possible `goto` to avoid "bypasses initialization".
    dim3 block(16, 16);
    dim3 grid(divUp(N, block.x), divUp(M, block.y));

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    if ((err = cudaMalloc((void**)&dA, bytesA)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc((void**)&dB, bytesB)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc((void**)&dC, bytesC)) != cudaSuccess) goto cleanup;

    if ((err = cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;

    matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) goto cleanup;

    if ((err = cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;

cleanup:
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    return (err == cudaSuccess) ? 0 : (int)err;
}
