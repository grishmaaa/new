// #include <cuda_runtime.h>
// #include <mma.h>
// #include <cuda_fp16.h>
// #include <stdio.h>

// #include "tensorcore.h"

// namespace wmma = nvcuda::wmma;

// static constexpr int WMMA_M = 16;
// static constexpr int WMMA_N = 16;
// static constexpr int WMMA_K = 16;

// // A: row-major (M x K), Bcol: col-major (K x N), C: row-major (M x N)
// __global__ void wmma_gemm_kernel(const half* __restrict__ A,     // row-major
//                                  const half* __restrict__ Bcol,  // col-major
//                                  float* __restrict__ C,          // row-major
//                                  int M, int N, int K)
// {
//     int warp_row = blockIdx.y; // tile row
//     int warp_col = blockIdx.x; // tile col

//     int row = warp_row * WMMA_M;
//     int col = warp_col * WMMA_N;

//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

//     wmma::fill_fragment(c_frag, 0.0f);

//     // Sweep K in 16-wide chunks
//     for (int k0 = 0; k0 < K; k0 += WMMA_K) {
//         // A tile: top-left at (row, k0), ld = K (row-major)
//         const half* a_tile = A + row * K + k0;
//         // B tile (col-major): top-left at (k0, col), ld = K (rows = K)
//         const half* b_tile = Bcol + col * K + k0;

//         wmma::load_matrix_sync(a_frag, a_tile, K);
//         wmma::load_matrix_sync(b_frag, b_tile, K);
//         wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//     }

//     float* c_tile = C + row * N + col;
//     wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
// }

// // Expects M,N,K multiples of 16. (Caller pads.)
// extern "C" void naive_tensor_tgemm(half *d_A_rowmajor,
//                                    half *d_B_colmajor,
//                                    float *d_C_rowmajor,
//                                    int C_n_rows, int C_n_cols, int A_n_cols)
// {
//     int M = C_n_rows, N = C_n_cols, K = A_n_cols;
//     if ((M % 16) || (N % 16) || (K % 16)) return;

//     dim3 block(64, 2, 1);        // one warp
//         dim3 grid(N / (WMMA_N * 2),
//               M / (WMMA_M * 2),
//               1);
//     wmma_gemm_kernel<<<grid, block>>>(d_A_rowmajor, d_B_colmajor, d_C_rowmajor, M, N, K);
//     cudaDeviceSynchronize();
// }

//////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "tensorcore.h"

namespace wmma = nvcuda::wmma;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// ===== TUNABLES =====
// CTA macro tile = (BLOCK_ROW_TILES * 16) x (BLOCK_COL_TILES * 16)
// Defaults: 64 x 64 with 8 warps (256 threads) — usually fast & high occupancy on sm_86.
#ifndef BLOCK_ROW_TILES
#define BLOCK_ROW_TILES 4   // rows of 16x16 tiles per block => 64 rows
#endif
#ifndef BLOCK_COL_TILES
#define BLOCK_COL_TILES 4   // cols of 16x16 tiles per block => 64 cols
#endif

// One warp computes one 16x16 tile
static constexpr int WARPS_PER_BLOCK = BLOCK_ROW_TILES * BLOCK_COL_TILES;
static_assert(WARPS_PER_BLOCK * 32 <= 1024, "Too many threads per block");

#if __CUDACC_VER_MAJOR__ >= 11
#define HAS_CP_ASYNC 1
#else
#define HAS_CP_ASYNC 0
#endif

// ---- Helpers ----
static __device__ __forceinline__ int lane_id() {
    int lid;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(lid));
    return lid;
}

// Convert a generic shared pointer to a 32-bit shared address for cp.async
static __device__ __forceinline__ unsigned smem_ptr(const void* p) {
#if __CUDA_ARCH__ >= 800
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
#else
    return 0u;
#endif
}

#if __CUDA_ARCH__ >= 800
// Issue one 16B cp.async (cached) from global to shared
static __device__ __forceinline__ void cp_async_16B(void* dst_smem, const void* src_gmem) {
    unsigned smem_addr = smem_ptr(dst_smem);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(src_gmem));
}
static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
#endif // __CUDA_ARCH__ >= 800

// A (row-major): MxK, Bcol (col-major): KxN, C (row-major): MxN
__global__ void wmma_gemm_kernel_tc_shared(const half* __restrict__ A,     // row-major
                                           const half* __restrict__ Bcol,  // col-major
                                           float* __restrict__ C,          // row-major
                                           int M, int N, int K)
{
    // ---- CTA macro tile origin in C ----
    const int Mblk = BLOCK_ROW_TILES * WMMA_M; // rows per CTA
    const int Nblk = BLOCK_COL_TILES * WMMA_N; // cols per CTA
    const int block_row0 = blockIdx.y * Mblk;
    const int block_col0 = blockIdx.x * Nblk;

    // ---- Warp tile coordinates inside CTA ----
    const int warp_x = threadIdx.x / 32;    // [0 .. BLOCK_COL_TILES-1]
    const int warp_y = threadIdx.y;         // [0 .. BLOCK_ROW_TILES-1]
    const int tile_row = block_row0 + warp_y * WMMA_M; // top-left row of warp's tile
    const int tile_col = block_col0 + warp_x * WMMA_N; // top-left col of warp's tile

    if (tile_row >= M || tile_col >= N) return; // (should be padded already)

    // ---- Shared memory: double-buffered slices ----
    // A_s[buffer]: (Mblk x WMMA_K) row-major
    // B_s[buffer]: (WMMA_K x Nblk) col-major (ld = WMMA_K)
    extern __shared__ half shared[];
    half* A_s0 = shared;
    half* B_s0 = A_s0 + (size_t)Mblk * WMMA_K;
    half* A_s1 = B_s0 + (size_t)WMMA_K * Nblk;
    half* B_s1 = A_s1 + (size_t)Mblk * WMMA_K;

    // ---- Accumulator fragment (per warp tile) ----
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Total threads per CTA
    const int tpb = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Preload k-slice 0 into buffer 0
    int buf = 0;

    auto stage_A = [&](half* As, int k0) {
        // Each A row has 16 halfs (=32B): we copy it in two 16B chunks.
        // Number of 16B copies for A: Mblk * 2
        const int A_copies = Mblk * 2;
#if __CUDA_ARCH__ >= 800
        for (int idx = tid; idx < A_copies; idx += tpb) {
            int row = idx >> 1;              // 0..Mblk-1
            int seg = idx & 1;               // 0 or 1 (first/second 16B)
            const half* gptr = A + (block_row0 + row) * K + k0 + seg * 8; // +8 halfs
            void* sptr = (void*)(&As[row * WMMA_K + seg * 8]);
            cp_async_16B(sptr, gptr);
        }
#else
        // Fallback: coalesced 128-bit loads/stores via int4
        for (int row = tid; row < Mblk; row += tpb) {
            const int4* src = reinterpret_cast<const int4*>(A + (block_row0 + row) * K + k0);
            int4 v0 = src[0], v1 = src[1];
            int4* dst = reinterpret_cast<int4*>(&As[row * WMMA_K]);
            dst[0] = v0; dst[1] = v1;
        }
#endif
    };

    auto stage_B = [&](half* Bs, int k0) {
        // Each B column (length 16) is 32B contiguous in column-major (ld = K).
        // We store in SMEM as true column-major with ld = WMMA_K so WMMA can load directly.
        // Number of 16B copies for B: Nblk * 2
        const int B_copies = Nblk * 2;
#if __CUDA_ARCH__ >= 800
        for (int idx = tid; idx < B_copies; idx += tpb) {
            int col = idx >> 1;              // 0..Nblk-1
            int seg = idx & 1;               // 0 or 1
            const half* gptr = Bcol + (block_col0 + col) * K + k0 + seg * 8; // col*K + row
            void* sptr = (void*)(&Bs[col * WMMA_K + seg * 8]);
            cp_async_16B(sptr, gptr);
        }
#else
        for (int col = tid; col < Nblk; col += tpb) {
            const int4* src = reinterpret_cast<const int4*>(Bcol + (block_col0 + col) * K + k0);
            int4 v0 = src[0], v1 = src[1];
            int4* dst = reinterpret_cast<int4*>(&Bs[col * WMMA_K]);
            dst[0] = v0; dst[1] = v1;
        }
#endif
    };

    // --- Preload first slice ---
#if __CUDA_ARCH__ >= 800
    stage_A(A_s0, /*k0=*/0);
    stage_B(B_s0, /*k0=*/0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
#else
    stage_A(A_s0, 0);
    stage_B(B_s0, 0);
    __syncthreads();
#endif

    // ---- K loop with double buffering ----
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // Prefetch next slice into the other buffer while we compute
        const int next_k = k0 + WMMA_K;
        if (next_k < K) {
#if __CUDA_ARCH__ >= 800
            if (buf == 0) {
                stage_A(A_s1, next_k);
                stage_B(B_s1, next_k);
            } else {
                stage_A(A_s0, next_k);
                stage_B(B_s0, next_k);
            }
            cp_async_commit();
#endif
        }

        // Select current buffers
        half* As = (buf == 0) ? A_s0 : A_s1;
        half* Bs = (buf == 0) ? B_s0 : B_s1;

        // Load fragments from SMEM (coalesced, no bank conflicts in these shapes)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,  wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,  wmma::col_major> b_frag;

        const half* a_tile_s = &As[(warp_y * WMMA_M) * WMMA_K];   // top-left of this warp's A tile
        const half* b_tile_s = &Bs[(warp_x * WMMA_N) * WMMA_K];   // top-left of this warp's B tile

        wmma::load_matrix_sync(a_frag, a_tile_s, WMMA_K);
        wmma::load_matrix_sync(b_frag, b_tile_s, WMMA_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();    // ensure no one uses As/Bs while we're about to overwrite

        if (next_k < K) {
#if __CUDA_ARCH__ >= 800
            cp_async_wait_all(); // make sure the next buffers are ready
#endif
            __syncthreads();
            buf ^= 1;            // flip buffers
        }
    }

    // ---- Store the 16x16 tile back to C ----
    float* c_ptr = C + tile_row * N + tile_col;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

// Public wrapper — same signature you already call.
// Requires/pads: M multiple of (BLOCK_ROW_TILES*16), N multiple of (BLOCK_COL_TILES*16), K multiple of 16.
extern "C" void naive_tensor_tgemm(half *d_A_rowmajor,
                                   half *d_B_colmajor,
                                   float *d_C_rowmajor,
                                   int C_n_rows, int C_n_cols, int A_n_cols)
{
    const int M = C_n_rows, N = C_n_cols, K = A_n_cols;
    const int Mblk = BLOCK_ROW_TILES * WMMA_M;
    const int Nblk = BLOCK_COL_TILES * WMMA_N;

    if ((M % Mblk) || (N % Nblk) || (K % WMMA_K)) {
        // Caller should pad; we silently return to keep interface unchanged.
        return;
    }

    // 32 threads per warp along x; WARPS laid out as (x = BLOCK_COL_TILES warps, y = BLOCK_ROW_TILES warps)
    dim3 block(32 * BLOCK_COL_TILES, BLOCK_ROW_TILES, 1);  // e.g., (128,4,1) => 512 threads/CTA
    dim3 grid(N / Nblk, M / Mblk, 1);

    // Dynamic shared memory: double-buffered A and B slices
    size_t shmem_halfs = (size_t)(Mblk * WMMA_K + WMMA_K * Nblk) * 2;  // *2 for double buffer
    size_t shmem_bytes = shmem_halfs * sizeof(half);

    wmma_gemm_kernel_tc_shared<<<grid, block, shmem_bytes>>>(d_A_rowmajor, d_B_colmajor, d_C_rowmajor, M, N, K);
    cudaDeviceSynchronize();
}
