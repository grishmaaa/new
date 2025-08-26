// #include <cuda_runtime.h>
// #include <mma.h>
// #include <cuda_fp16.h>
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>

// // forward declaration from your kernel file
// extern "C" void naive_tensor_tgemm(half *d_A_rowmajor,
//                                    half *d_B_colmajor,
//                                    float *d_C_rowmajor,
//                                    int C_n_rows, int C_n_cols, int A_n_cols);

// int main() {
//     int M = 2048; // must be multiple of BLOCK_ROW_TILES*16
//     int N = 2048; // must be multiple of BLOCK_COL_TILES*16
//     int K = 2048; // must be multiple of 16

//     std::cout << "Matrix size: " << M << "x" << N << " (K=" << K << ")\n";

//     size_t sizeA = M * K * sizeof(half);
//     size_t sizeB = K * N * sizeof(half);
//     size_t sizeC = M * N * sizeof(float);

//     // Host allocations
//     std::vector<half> h_A(M * K), h_Bcol(K * N);
//     std::vector<float> h_C(M * N);

//     // Random number generator in [-1, 1]
//     std::mt19937 rng(42);  // fixed seed for reproducibility
//     std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

//     // Fill A (row-major)
//     for (int i = 0; i < M * K; i++) {
//         h_A[i] = __float2half(dist(rng));
//     }
//     // Fill B (col-major)
//     for (int j = 0; j < N; j++) {
//         for (int i = 0; i < K; i++) {
//             h_Bcol[j*K + i] = __float2half(dist(rng));
//         }
//     }

//     // Device allocations
//     half *d_A, *d_B;
//     float *d_C;
//     cudaMalloc(&d_A, sizeA);
//     cudaMalloc(&d_B, sizeB);
//     cudaMalloc(&d_C, sizeC);

//     cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_Bcol.data(), sizeB, cudaMemcpyHostToDevice);
//     cudaMemset(d_C, 0, sizeC);

//     // Warmup run
//     naive_tensor_tgemm(d_A, d_B, d_C, M, N, K);
//     cudaDeviceSynchronize();

//     // Timed run
//     auto start = std::chrono::high_resolution_clock::now();
//     naive_tensor_tgemm(d_A, d_B, d_C, M, N, K);
//     cudaDeviceSynchronize();
//     auto end = std::chrono::high_resolution_clock::now();
//     double ms = std::chrono::duration<double, std::milli>(end-start).count();

//     cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

//     // Print a few output elements
//     std::cout << "C[0..9] = ";
//     for (int i = 0; i < 10; i++) {
//         std::cout << h_C[i] << " ";
//     }
//     std::cout << "\n";
//     std::cout << "Execution time = " << ms << " ms\n";

//     // Compute effective GFLOPS
//     double ops = 2.0 * M * N * K; // FMA = 2 ops
//     double gflops = ops / (ms*1e6);
//     std::cout << "Performance = " << gflops << " GFLOP/s\n";

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     return 0;
// }
// #######################################################################################################

// main.cu — Python-callable wrapper (no kernel changes)
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Device-pointer launcher from tensorcut.cu (unchanged there)
extern "C" void naive_tensor_tgemm(half *d_A_rowmajor,
                                   half *d_B_colmajor,
                                   float *d_C_rowmajor,
                                   int M, int N, int K);

// Exported entry for ctypes: takes HOST pointers, does H2D/D2H
extern "C" int tensorcut(const half* A_host, const half* Bcol_host, float* C_host,
                         int M, int N, int K)
{
    half  *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;

    const size_t bytesA = (size_t)M * (size_t)K * sizeof(half);   // A: MxK (row-major)
    const size_t bytesB = (size_t)K * (size_t)N * sizeof(half);   // Bcol: KxN (col-major)
    const size_t bytesC = (size_t)M * (size_t)N * sizeof(float);  // C: MxN (row-major)

    if (cudaMalloc(&dA, bytesA) != cudaSuccess) return 1;
    if (cudaMalloc(&dB, bytesB) != cudaSuccess) { cudaFree(dA); return 2; }
    if (cudaMalloc(&dC, bytesC) != cudaSuccess) { cudaFree(dA); cudaFree(dB); return 3; }

    if (cudaMemcpy(dA, A_host,   bytesA, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return 4; }
    if (cudaMemcpy(dB, Bcol_host,bytesB, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return 5; }
    cudaMemset(dC, 0, bytesC);

    // Launch your original device-pointer API
    naive_tensor_tgemm(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    if (cudaMemcpy(C_host, dC, bytesC, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return 6; }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

