#include "matmul.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <climits>


#define CUDA_OK(x) do {                             \
    cudaError_t err__ = (x);                        \
    if (err__ != cudaSuccess) {                     \
        std::fprintf(stderr, "CUDA: %s at %s:%d\n", \
                     cudaGetErrorString(err__),     \
                     __FILE__, __LINE__);           \
        return 1;                                   \
    }                                               \
} while (0)

__global__ void matmul_kernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C,
                              int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        int a_base = row * K;
        for (int p = 0; p < K; ++p) {
            sum += A[a_base + p] * B[p * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Exposed C API (no name mangling)
extern "C"
int matmul_cuda(const double* A_h, const double* B_h, double* C_h,
                size_t m, size_t k, size_t n)
{

    // Guard against size_t > INT_MAX for kernel params
    if (m > INT_MAX || k > INT_MAX || n > INT_MAX) {
        std::fprintf(stderr, "Dims too large for this simple kernel.\n");
        return 2;
    }
    int M = static_cast<int>(m);
    int K = static_cast<int>(k);
    int N = static_cast<int>(n);

    size_t bytesA = m * k * sizeof(double);
    size_t bytesB = k * n * sizeof(double);
    size_t bytesC = m * n * sizeof(double);

    double *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    CUDA_OK(cudaMalloc(&A_d, bytesA));
    CUDA_OK(cudaMalloc(&B_d, bytesB));
    CUDA_OK(cudaMalloc(&C_d, bytesC));

    CUDA_OK(cudaMemcpy(A_d, A_h, bytesA, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B_h, bytesB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_kernel<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n",
                     cudaGetErrorString(launch_err));
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
        return 3;
    }
    cudaEventRecord(stop);
    CUDA_OK(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA time: %f milliseconds\n", milliseconds);

    CUDA_OK(cudaMemcpy(C_h, C_d, bytesC, cudaMemcpyDeviceToHost));

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    return 0;
}
