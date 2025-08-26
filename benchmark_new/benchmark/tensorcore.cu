#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include "tensorcore.h"

// We will refer to WMMA names with full qualification for clarity.
namespace wmma = nvcuda::wmma;

// WMMA fragment dimensions (16x16x16)
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Each block (one warp: 32 threads) computes one 16x16 tile of C.
// Grid dims should be (C_cols/16, C_rows/16).
__global__ void wmma_gemm_kernel(const half* __restrict__ A,   // [M x K] row-major
                                 const half* __restrict__ B,   // [K x N] row-major
                                 float* __restrict__ C,        // [M x N] row-major
                                 int M, int N, int K)
{
    // One warp per block
    int warp_row = blockIdx.y; // which 16-row tile of C
    int warp_col = blockIdx.x; // which 16-col tile of C

    // Top-left of this C tile
    int row = warp_row * WMMA_M;
    int col = warp_col * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K in 16-wide tiles
    for (int k = 0; k < K; k += WMMA_K) {
        const half* a_tile = A + row * K + k; // row-major, ld = K
        const half* b_tile = B + k * N + col; // row-major, ld = N

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the result
    float* c_tile = C + row * N + col;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}

// Host wrapper. Expects M,N,K to be multiples of 16. (Caller can pad if needed.)
extern "C" void naive_tensor_tgemm(half *d_A_ptr,
                                   half *d_B_ptr,
                                   float *d_C_ptr,
                                   int C_n_rows, int C_n_cols, int A_n_cols)
{
    int M = C_n_rows;
    int N = C_n_cols;
    int K = A_n_cols;

    // Sanity: require multiples of 16
    if ((M % 16) || (N % 16) || (K % 16)) {
        // For simplicity, just return; caller should pad. (Or assert in debug.)
        // fprintf(stderr, "naive_tensor_tgemm: dimensions must be multiples of 16 (got %d x %d, K=%d)\n", M, N, K);
        return;
    }

    dim3 block(32, 1, 1); // one warp
    dim3 grid(N / 16, M / 16, 1);
    wmma_gemm_kernel<<<grid, block>>>(d_A_ptr, d_B_ptr, d_C_ptr, M, N, K);
    cudaDeviceSynchronize();
}
