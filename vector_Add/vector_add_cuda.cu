#include <cuda_runtime.h>
#include <cstdio>

__global__ void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" int vector_add_cuda(
    const float* hA,
    const float* hB,
    float* hC,
    int N
) {
    if (!hA || !hB || !hC || N <= 0) return 1;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytes = N * sizeof(float);

    cudaError_t err;
    err = cudaMalloc((void**)&dA, bytes); if (err) return 2;
    err = cudaMalloc((void**)&dB, bytes); if (err) return 3;
    err = cudaMalloc((void**)&dC, bytes); if (err) return 4;

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(dA, dB, dC, N);
    err = cudaGetLastError(); if (err) return 5;

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
