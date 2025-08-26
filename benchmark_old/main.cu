#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>  // <-- added: for cudaDeviceSynchronize()

#include "matmul_cpu.h"
#include "matmul_gpu.h"
#include "tiled.h"
#include "utils.h"

#define MAX_NUM 10 
#define MIN_NUM -10 

int main(int argc, char const *argv[])
{
    // Matrix A size: N1 x N2
    // Matrix B size: N2 x N3
    int N1 = 2678;
    int N2 = 2678;
    int N3 = 2678;

    // Generate N1xN2 matrix A
    float* A = (float*)malloc(N1*N2*sizeof(float));
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++)
            A[i*N2+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }

    // Generate N2xN3 matrix B
    float* B = (float*)malloc(N2*N3*sizeof(float));
    for (int i = 0; i < N2; i++)
    {
        for (int j = 0; j < N3; j++)
            B[i*N3+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }

    // ---- Untiled GPU ----
    float* C_gpu = (float*)malloc(N1*N3*sizeof(float));
    const int WARMUPS = 3;  // <-- added: number of warm-up runs

    // Warm-up runs (not timed)
    for (int w = 0; w < WARMUPS; ++w) {
        matmul_gpu(A, B, C_gpu, N1, N2, N3);
        cudaDeviceSynchronize(); // <-- make sure the GPU finished
    }

    // Timed run
    double t1_gpu = myCPUTimer();
    matmul_gpu(A, B, C_gpu, N1, N2, N3);
    cudaDeviceSynchronize();      // <-- ensure timing is accurate
    double t2_gpu = myCPUTimer();
    double elapsed_ms1 = (t2_gpu - t1_gpu) / 1000.0;
    printf("GPU execution time (N1: %d; N2: %d; N3: %d): %.3f ms \n", N1, N2, N3, elapsed_ms1);
    printf("\n");

    // ---- Tiled GPU ----
    float* C_tiled_gpu = (float*)malloc(N1*N3*sizeof(float));

    // Warm-up runs (not timed)
    for (int w = 0; w < WARMUPS; ++w) {
        tiled_gpu(A, B, C_tiled_gpu, N1, N2, N3);
        cudaDeviceSynchronize(); // <-- ensure completion before next warm-up
    }

    // Timed run
    double t1_tiled_gpu = myCPUTimer();
    tiled_gpu(A, B, C_tiled_gpu, N1, N2, N3);
    cudaDeviceSynchronize();      // <-- ensure timing is accurate
    double t2_tiled_gpu = myCPUTimer();
    double elapsed_ms2 = (t2_tiled_gpu - t1_tiled_gpu) / 1000.0;
    printf("Tiled GPU execution time (N1: %d; N2: %d; N3: %d): %.3f ms \n", N1, N2, N3, elapsed_ms2);
    printf("\n");

    // Speedup
    printf("Speed-up with tiled GPU from untiled GPU (N1: %d; N2: %d; N3: %d): %.3f x\n",
           N1, N2, N3, (double)(elapsed_ms1)/(elapsed_ms2));
    printf("\n");

    // Optional assertion (left commented as in your code)
    // for (int i = 0; i < N1; i++)
    //   for (int j = 0; j < N3; j++)
    //     assert(fabs(C_gpu[i*N3+j] - C_tiled_gpu[i*N3+j]) < 1e-8);

    // Free memory
    free(A);
    free(B);
    free(C_gpu);
    free(C_tiled_gpu);
    return 0;
}