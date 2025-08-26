// #include <stdio.h>
// #include <stdlib.h>
// #include <assert.h>
// #include <cuda_runtime.h>  // <-- added: for cudaDeviceSynchronize()

// #include "matmul_cpu.h"
// #include "matmul_gpu.h"
// #include "tensorcore.h"
// #include "tiled.h"
// #include "utils.h"

// #define MAX_NUM 10 
// #define MIN_NUM -10 

// int main(int argc, char const *argv[])
// {
//     // Matrix A size: N1 x N2
//     // Matrix B size: N2 x N3
//     int N1 = 2678;
//     int N2 = 2678;
//     int N3 = 2678;

//     // Generate N1xN2 matrix A
//     float* A = (float*)malloc(N1*N2*sizeof(float));
//     for (int i = 0; i < N1; i++)
//     {
//         for (int j = 0; j < N2; j++)
//             A[i*N2+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
//     }

//     // Generate N2xN3 matrix B
//     float* B = (float*)malloc(N2*N3*sizeof(float));
//     for (int i = 0; i < N2; i++)
//     {
//         for (int j = 0; j < N3; j++)
//             B[i*N3+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
//     }

//     // ---- Untiled GPU ----
//     float* C_gpu = (float*)malloc(N1*N3*sizeof(float));
//     const int WARMUPS = 3;  // <-- added: number of warm-up runs

//     // Warm-up runs (not timed)
//     for (int w = 0; w < WARMUPS; ++w) {
//         matmul_gpu(A, B, C_gpu, N1, N2, N3);
//         cudaDeviceSynchronize(); // <-- make sure the GPU finished
//     }

//     // Timed run
//     double t1_gpu = myCPUTimer();
//     matmul_gpu(A, B, C_gpu, N1, N2, N3);
//     cudaDeviceSynchronize();      // <-- ensure timing is accurate
//     double t2_gpu = myCPUTimer();
//     double elapsed_ms1 = (t2_gpu - t1_gpu) / 1000.0;
//     printf("GPU execution time (N1: %d; N2: %d; N3: %d): %.3f ms \n", N1, N2, N3, elapsed_ms1);
//     printf("\n");

//     // ---- Tiled GPU ----
//     float* C_tiled_gpu = (float*)malloc(N1*N3*sizeof(float));

//     // Warm-up runs (not timed)
//     for (int w = 0; w < WARMUPS; ++w) {
//         tiled_gpu(A, B, C_tiled_gpu, N1, N2, N3);
//         cudaDeviceSynchronize(); // <-- ensure completion before next warm-up
//     }

//     // Timed run
//     double t1_tiled_gpu = myCPUTimer();
//     tiled_gpu(A, B, C_tiled_gpu, N1, N2, N3);
//     cudaDeviceSynchronize();      // <-- ensure timing is accurate
//     double t2_tiled_gpu = myCPUTimer();
//     double elapsed_ms2 = (t2_tiled_gpu - t1_tiled_gpu) / 1000.0;
//     printf("Tiled GPU execution time (N1: %d; N2: %d; N3: %d): %.3f ms \n", N1, N2, N3, elapsed_ms2);
//     printf("\n");

//     float* C_tensor = (float*)malloc(N1*N3*sizeof(float));

//     // Warm-up runs (not timed)
//     for (int w = 0; w < WARMUPS; ++w) {
//         matmul_gpu(A, B, C_tensor, N1, N2, N3);
//         cudaDeviceSynchronize(); // <-- make sure the GPU finished
//     }

//     // Tensor
//     double t1_tensor = myCPUTimer();
//     naive_tensor_tgemm(A, B, C_tensor, N1, N2, N3);
//     cudaDeviceSynchronize();      // <-- ensure timing is accurate
//     double t2_tensor = myCPUTimer();
//     double elapsed_ms3 = (t2_tensor - t1_tensor) / 1000.0;
//     printf("GPU execution time (N1: %d; N2: %d; N3: %d): %.3f ms \n", N1, N2, N3, elapsed_ms3);
//     printf("\n");

    
//     // Speedup
//     printf("Speed-up with tiled GPU from untiled GPU (N1: %d; N2: %d; N3: %d): %.3f x\n",
//            N1, N2, N3, (double)(elapsed_ms1)/(elapsed_ms2));
//     printf("\n");

//     // Optional assertion (left commented as in your code)
//     // for (int i = 0; i < N1; i++)
//     //   for (int j = 0; j < N3; j++)
//     //     assert(fabs(C_gpu[i*N3+j] - C_tiled_gpu[i*N3+j]) < 1e-8);

//     // Free memory
//     free(A);
//     free(B);
//     free(C_gpu);
//     free(C_tiled_gpu);
//     free(C_tensor);
//     return 0;
// }


//////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "matmul_cpu.h"   // assumes: void matmul_cpu(float*, float*, float*, int,int,int);
#include "matmul_gpu.h"   // assumes: void matmul_gpu(float*, float*, float*, int,int,int);
#include "tiled.h"        // has:     void tiled_gpu(float*, float*, float*, int,int,int);
#include "tensorcore.h"   // has:     void naive_tensor_tgemm(half*,half*,float*,int,int,int);
#include "utils.h"        // has:     unsigned long long myCPUTimer();

#define MAX_NUM 10
#define MIN_NUM -10

// ---------- Helpers for Tensor Core path (padding + type conversion) ----------
__global__ void float_to_half_pad_A(const float* __restrict__ A, half* __restrict__ Aout,
                                    int M, int K, int Mp, int Kp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Mp * Kp;
    if (idx >= total) return;
    int i = idx / Kp;
    int k = idx % Kp;
    float v = 0.0f;
    if (i < M && k < K) v = A[i * K + k];
    Aout[idx] = __float2half(v);
}

__global__ void float_to_half_pad_B(const float* __restrict__ B, half* __restrict__ Bout,
                                    int K, int N, int Kp, int Np) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Kp * Np;
    if (idx >= total) return;
    int k = idx / Np;
    int j = idx % Np;
    float v = 0.0f;
    if (k < K && j < N) v = B[k * N + j];
    Bout[idx] = __float2half(v);
}

__global__ void crop_C(const float* __restrict__ Cpad, float* __restrict__ C,
                       int M, int N, int Mp, int Np) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int i = idx / N;
    int j = idx % N;
    C[i * N + j] = Cpad[i * Np + j];
}

// Max abs error vs reference
static float max_abs_diff(const float* ref, const float* out, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(ref[i] - out[i]);
        if (d > m) m = d;
    }
    return m;
}

// GFLOP/s assuming 2*M*N*K flops
static double gflops(long long M, long long N, long long K, double ms) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    return (flops / 1.0e9) / (ms / 1.0e3);
}

int main(int argc, char const *argv[]) {
    // Matrix A size: N1 x N2
    // Matrix B size: N2 x N3
    int N1 = 2678, N2 = 2678, N3 = 2678;
    if (argc == 4) {
        N1 = atoi(argv[1]);
        N2 = atoi(argv[2]);
        N3 = atoi(argv[3]);
    }

    // Host allocations
    float *A = (float*)malloc((size_t)N1 * N2 * sizeof(float));
    float *B = (float*)malloc((size_t)N2 * N3 * sizeof(float));
    float *C_cpu   = (float*)malloc((size_t)N1 * N3 * sizeof(float));
    float *C_gpu   = (float*)malloc((size_t)N1 * N3 * sizeof(float));
    float *C_tiled = (float*)malloc((size_t)N1 * N3 * sizeof(float));
    float *C_tensor= (float*)malloc((size_t)N1 * N3 * sizeof(float));

    if (!A || !B || !C_cpu || !C_gpu || !C_tiled || !C_tensor) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    // Init A,B
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            A[i * N2 + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        }
    }
    for (int i = 0; i < N2; ++i) {
        for (int j = 0; j < N3; ++j) {
            B[i * N3 + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        }
    }

    // warmup
    for (int w = 0; w < 3; ++w) {
        matmul_cpu(A, B, C_tensor, N1, N2, N3);
        cudaDeviceSynchronize(); // <-- make sure the GPU finished
    }
    // ----------------------- CPU baseline -----------------------
    unsigned long long t0 = myCPUTimer();
    matmul_cpu(A, B, C_cpu, N1, N2, N3);
    unsigned long long t1 = myCPUTimer();
    double ms_cpu = (double)(t1 - t0) / 1000.0;
    printf("[CPU]        %dx%dx%d  time = %.3f ms  (%.2f GFLOP/s)\n",
           N1, N2, N3, ms_cpu, gflops(N1, N3, N2, ms_cpu));

    // ----------------------- Naive GPU --------------------------
    // warmup
    for (int w = 0; w < 3; ++w) {
        matmul_gpu(A, B, C_tensor, N1, N2, N3);
        cudaDeviceSynchronize(); // <-- make sure the GPU finished
    }
    t0 = myCPUTimer();
    matmul_gpu(A, B, C_gpu, N1, N2, N3);
    cudaDeviceSynchronize();
    t1 = myCPUTimer();
    double ms_gpu = (double)(t1 - t0) / 1000.0;
    printf("[GPU naive]  %dx%dx%d  time = %.3f ms  (%.2f GFLOP/s)  max|Δ|=%.3e\n",
           N1, N2, N3, ms_gpu, gflops(N1, N3, N2, ms_gpu), max_abs_diff(C_cpu, C_gpu, N1*N3));

    // ----------------------- Tiled GPU --------------------------
    // warmup
    for (int w = 0; w < 3; ++w) {
        tiled_gpu(A, B, C_tensor, N1, N2, N3);
        cudaDeviceSynchronize(); // <-- make sure the GPU finished
    }
    t0 = myCPUTimer();
    tiled_gpu(A, B, C_tiled, N1, N2, N3);
    cudaDeviceSynchronize();
    t1 = myCPUTimer();
    double ms_tiled = (double)(t1 - t0) / 1000.0;
    printf("[GPU tiled]  %dx%dx%d  time = %.3f ms  (%.2f GFLOP/s)  max|Δ|=%.3e\n",
           N1, N2, N3, ms_tiled, gflops(N1, N3, N2, ms_tiled), max_abs_diff(C_cpu, C_tiled, N1*N3));

    // ----------------------- Tensor Core (WMMA) -----------------
    // Pad dimensions to multiples of 16 (WMMA requirement).
    int Mp = ((N1 + 15) / 16) * 16;
    int Np = ((N3 + 15) / 16) * 16;
    int Kp = ((N2 + 15) / 16) * 16;

    float *d_Af = nullptr, *d_Bf = nullptr;
    half  *d_Ah = nullptr, *d_Bh = nullptr;
    float *d_Cpad = nullptr, *d_C = nullptr;

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_Af,  (size_t)N1 * N2 * sizeof(float));  if (err) { fprintf(stderr,"cudaMalloc d_Af failed\n"); return 1; }
    err = cudaMalloc(&d_Bf,  (size_t)N2 * N3 * sizeof(float));  if (err) { fprintf(stderr,"cudaMalloc d_Bf failed\n"); return 1; }
    err = cudaMalloc(&d_Ah,  (size_t)Mp * Kp * sizeof(half));   if (err) { fprintf(stderr,"cudaMalloc d_Ah failed\n"); return 1; }
    err = cudaMalloc(&d_Bh,  (size_t)Kp * Np * sizeof(half));   if (err) { fprintf(stderr,"cudaMalloc d_Bh failed\n"); return 1; }
    err = cudaMalloc(&d_Cpad,(size_t)Mp * Np * sizeof(float));  if (err) { fprintf(stderr,"cudaMalloc d_Cpad failed\n"); return 1; }
    err = cudaMalloc(&d_C,   (size_t)N1 * N3 * sizeof(float));  if (err) { fprintf(stderr,"cudaMalloc d_C failed\n"); return 1; }

    cudaMemcpy(d_Af, A, (size_t)N1 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bf, B, (size_t)N2 * N3 * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocksA = (Mp * Kp + threads - 1) / threads;
    int blocksB = (Kp * Np + threads - 1) / threads;
    float_to_half_pad_A<<<blocksA, threads>>>(d_Af, d_Ah, N1, N2, Mp, Kp);
    float_to_half_pad_B<<<blocksB, threads>>>(d_Bf, d_Bh, N2, N3, Kp, Np);
    cudaDeviceSynchronize();

    // warmup
    for (int w = 0; w < 3; ++w) {
        naive_tensor_tgemm(d_Ah, d_Bh, d_Cpad, Mp, Np, Kp);
        cudaDeviceSynchronize(); // <-- make sure the GPU finished
    }
    t0 = myCPUTimer();
    naive_tensor_tgemm(d_Ah, d_Bh, d_Cpad, Mp, Np, Kp);
    cudaDeviceSynchronize();
    t1 = myCPUTimer();

    int blocksC = (N1 * N3 + threads - 1) / threads;
    crop_C<<<blocksC, threads>>>(d_Cpad, d_C, N1, N3, Mp, Np);
    cudaMemcpy(C_tensor, d_C, (size_t)N1 * N3 * sizeof(float), cudaMemcpyDeviceToHost);

    double ms_tc = (double)(t1 - t0) / 1000.0;
    printf("[TensorCore] %dx%dx%d  (padded to %d,%d,%d) time = %.3f ms  (%.2f GFLOP/s)  max|Δ|=%.3e\n",
           N1, N2, N3, Mp, Kp, Np, ms_tc, gflops(N1, N3, N2, ms_tc), max_abs_diff(C_cpu, C_tensor, N1*N3));

    // Cleanup
    cudaFree(d_Af); cudaFree(d_Bf);
    cudaFree(d_Ah); cudaFree(d_Bh);
    cudaFree(d_Cpad); cudaFree(d_C);

    free(A); free(B);
    free(C_cpu); free(C_gpu); free(C_tiled); free(C_tensor);

    return 0;
}
