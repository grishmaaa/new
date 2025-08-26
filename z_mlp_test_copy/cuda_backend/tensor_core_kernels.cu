// tensor_core_kernels.cu  (fixed: no host lambdas, no std::min in launches)
// ASCII-only comments. Row-major everywhere.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <mma.h>   // WMMA for Tensor Cores
using namespace nvcuda;

// ----- Error handling -----
#define CUDA_CHECK(call) do {                         \
    cudaError_t _e = (call);                          \
    if (_e != cudaSuccess) {                          \
        return (int)_e;                               \
    }                                                 \
} while (0)

// ----- Small helpers -----
static inline unsigned int grid_1d_from_n(int64_t n, unsigned int block) {
    if (n <= 0) return 1u;
    unsigned long long g = ((unsigned long long)(n) + block - 1ull) / (unsigned long long)block;
    if (g > 65535ull) g = 65535ull;
    return (unsigned int)g;
}

static __device__ __forceinline__ float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
static __device__ __forceinline__ float softplus_scalar(float x) {
    // Stable softplus
    return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

// ===========================
// Memory utilities
// ===========================
extern "C" int gpu_malloc(void** ptr, size_t bytes) { return (int)cudaMalloc(ptr, bytes); }
extern "C" int gpu_free(void* ptr) { return (int)cudaFree(ptr); }
extern "C" int h2d(void* dst_device, const void* src_host, size_t bytes) {
    return (int)cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice);
}
extern "C" int d2h(void* dst_host, const void* src_device, size_t bytes) {
    return (int)cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);
}

// ===========================
// Elementwise kernels (no lambdas)
// ===========================
__global__ void ew_add_f32_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                  float* __restrict__ Z, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Z[idx] = X[idx] + Y[idx];
}
__global__ void ew_sub_f32_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                  float* __restrict__ Z, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Z[idx] = X[idx] - Y[idx];
}
__global__ void ew_mul_f32_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                  float* __restrict__ Z, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Z[idx] = X[idx] * Y[idx];
}
__global__ void ew_div_f32_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                  float* __restrict__ Z, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Z[idx] = X[idx] / Y[idx];
}
__global__ void relu_f32_kernel(const float* __restrict__ X, float* __restrict__ Y, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float a = X[idx];
        Y[idx] = (a > 0.0f) ? a : 0.0f;
    }
}
__global__ void sigmoid_f32_kernel(const float* __restrict__ X, float* __restrict__ Y, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Y[idx] = sigmoid_scalar(X[idx]);
}
__global__ void tanh_f32_kernel(const float* __restrict__ X, float* __restrict__ Y, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Y[idx] = tanhf(X[idx]);
}
__global__ void exp_f32_kernel(const float* __restrict__ X, float* __restrict__ Y, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Y[idx] = __expf(X[idx]);
}
__global__ void log_f32_kernel(const float* __restrict__ X, float* __restrict__ Y, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Y[idx] = __logf(X[idx]);
}
__global__ void softplus_f32_kernel(const float* __restrict__ X, float* __restrict__ Y, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) Y[idx] = softplus_scalar(X[idx]);
}

extern "C" int ew_add_f32(const float* X, const float* Y, float* Z, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); ew_add_f32_kernel<<<grd, blk>>>(X, Y, Z, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int ew_sub_f32(const float* X, const float* Y, float* Z, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); ew_sub_f32_kernel<<<grd, blk>>>(X, Y, Z, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int ew_mul_f32(const float* X, const float* Y, float* Z, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); ew_mul_f32_kernel<<<grd, blk>>>(X, Y, Z, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int ew_div_f32(const float* X, const float* Y, float* Z, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); ew_div_f32_kernel<<<grd, blk>>>(X, Y, Z, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int relu_f32(const float* X, float* Y, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); relu_f32_kernel<<<grd, blk>>>(X, Y, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int sigmoid_f32(const float* X, float* Y, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); sigmoid_f32_kernel<<<grd, blk>>>(X, Y, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int tanh_f32(const float* X, float* Y, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); tanh_f32_kernel<<<grd, blk>>>(X, Y, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int exp_f32(const float* X, float* Y, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); exp_f32_kernel<<<grd, blk>>>(X, Y, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int log_f32(const float* X, float* Y, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); log_f32_kernel<<<grd, blk>>>(X, Y, n);
    return (int)cudaPeekAtLastError();
}
extern "C" int softplus_f32(const float* X, float* Y, int64_t n) {
    dim3 blk(256); dim3 grd(grid_1d_from_n(n, blk.x)); softplus_f32_kernel<<<grd, blk>>>(X, Y, n);
    return (int)cudaPeekAtLastError();
}

// ===========================
// Bias add (row-wise)
// ===========================
__global__ void bias_add_row_kernel(const float* __restrict__ X, const float* __restrict__ b,
                                    float* __restrict__ Y, int rows, int cols, int ldx) {
    int row = blockIdx.y;
    int col0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int c = col0; c < cols; c += blockDim.x * gridDim.x) {
        Y[(size_t)row * (size_t)cols + c] = X[(size_t)row * (size_t)ldx + c] + b[c];
    }
}
extern "C" int bias_add_row_f32(const float* X, const float* bias, float* Y,
                                int rows, int cols, int ldx) {
    dim3 blk(256);
    dim3 grd((cols + blk.x - 1) / blk.x, rows);
    bias_add_row_kernel<<<grd, blk>>>(X, bias, Y, rows, cols, ldx);
    return (int)cudaPeekAtLastError();
}

// ===========================
// Reductions per row
// ===========================
__inline__ __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ void reduce_max_rows_kernel(const float* __restrict__ X, float* __restrict__ max_out,
                                       int rows, int cols, int ldx) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float vmax = -CUDART_INF_F;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        vmax = fmaxf(vmax, X[(size_t)row * (size_t)ldx + c]);
    }
    __shared__ float smem[32];
    vmax = warp_reduce_max(vmax);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = vmax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : -CUDART_INF_F;
        v = warp_reduce_max(v);
        if (threadIdx.x == 0) max_out[row] = v;
    }
}
extern "C" int reduce_max_rows_f32(const float* X, float* max_out,
                                   int rows, int cols, int ldx) {
    dim3 blk(256); dim3 grd(rows);
    reduce_max_rows_kernel<<<grd, blk>>>(X, max_out, rows, cols, ldx);
    return (int)cudaPeekAtLastError();
}

__global__ void reduce_sum_rows_kernel(const float* __restrict__ X, float* __restrict__ sum_out,
                                       int rows, int cols, int ldx) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float vsum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        vsum += X[(size_t)row * (size_t)ldx + c];
    }
    __shared__ float smem[32];
    vsum = warp_reduce_sum(vsum);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = vsum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.0f;
        v = warp_reduce_sum(v);
        if (threadIdx.x == 0) sum_out[row] = v;
    }
}
extern "C" int reduce_sum_rows_f32(const float* X, float* sum_out,
                                   int rows, int cols, int ldx) {
    dim3 blk(256); dim3 grd(rows);
    reduce_sum_rows_kernel<<<grd, blk>>>(X, sum_out, rows, cols, ldx);
    return (int)cudaPeekAtLastError();
}

// ===========================
// Row-wise softmax
// ===========================
__global__ void softmax_rows_kernel(const float* __restrict__ X, float* __restrict__ Y,
                                    int rows, int cols, int ldx, int ldy) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // 1) Row max
    float vmax = -CUDART_INF_F;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        vmax = fmaxf(vmax, X[(size_t)row * (size_t)ldx + c]);
    __shared__ float smem_max[32];
    vmax = warp_reduce_max(vmax);
    if ((threadIdx.x & 31) == 0) smem_max[threadIdx.x >> 5] = vmax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem_max[threadIdx.x] : -CUDART_INF_F;
        v = warp_reduce_max(v);
        if (threadIdx.x == 0) smem_max[0] = v;
    }
    __syncthreads();
    vmax = smem_max[0];

    // 2) Shifted exp and sum
    float lsum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = __expf(X[(size_t)row * (size_t)ldx + c] - vmax);
        Y[(size_t)row * (size_t)ldy + c] = e;
        lsum += e;
    }
    __shared__ float smem_sum[32];
    lsum = warp_reduce_sum(lsum);
    if ((threadIdx.x & 31) == 0) smem_sum[threadIdx.x >> 5] = lsum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem_sum[threadIdx.x] : 0.0f;
        v = warp_reduce_sum(v);
        if (threadIdx.x == 0) smem_sum[0] = v;
    }
    __syncthreads();
    float denom = smem_sum[0];

    // 3) Normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        Y[(size_t)row * (size_t)ldy + c] /= denom;
    }
}
extern "C" int softmax_rows_f32(const float* X, float* Y,
                                int rows, int cols, int ldx, int ldy) {
    dim3 blk(256); dim3 grd(rows);
    softmax_rows_kernel<<<grd, blk>>>(X, Y, rows, cols, ldx, ldy);
    return (int)cudaPeekAtLastError();
}

// ===========================
// Transpose (copy) Y = X^T
// ===========================
__global__ void transpose_2d_f32_kernel(const float* __restrict__ X, float* __restrict__ Y,
                                        int M, int N, int ldx, int ldy) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (y < M && x < N) tile[threadIdx.y][threadIdx.x] = X[(size_t)y * (size_t)ldx + x];
    __syncthreads();
    int xt = blockIdx.y * 32 + threadIdx.x;
    int yt = blockIdx.x * 32 + threadIdx.y;
    if (yt < N && xt < M) Y[(size_t)yt * (size_t)ldy + xt] = tile[threadIdx.x][threadIdx.y];
}
extern "C" int transpose_2d_f32(const float* X, float* Y, int M, int N, int ldx, int ldy) {
    dim3 blk(32, 32);
    dim3 grd((N + 31) / 32, (M + 31) / 32);
    transpose_2d_f32_kernel<<<grd, blk>>>(X, Y, M, N, ldx, ldy);
    return (int)cudaPeekAtLastError();
}

// ===========================
// FP32 tiled matmul
// ===========================
#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

__global__ void matmul_tiled_f32_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K, int lda, int ldb, int ldc) {
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int rowBase = blockRow * TILE_M;
    const int colBase = blockCol * TILE_N;

    const int ty = threadIdx.y; // 0..15
    const int tx = threadIdx.x; // 0..15

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    float acc[4][4] = {0};

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        for (int i = ty; i < TILE_M; i += blockDim.y) {
            int r = rowBase + i;
            for (int j = tx; j < TILE_K; j += blockDim.x) {
                int c = k0 + j;
                As[i][j] = (r < M && c < K) ? A[(size_t)r * (size_t)lda + c] : 0.0f;
            }
        }
        for (int i = ty; i < TILE_K; i += blockDim.y) {
            int r = k0 + i;
            for (int j = tx; j < TILE_N; j += blockDim.x) {
                int c = colBase + j;
                Bs[i][j] = (r < K && c < N) ? B[(size_t)r * (size_t)ldb + c] : 0.0f;
            }
        }
        __syncthreads();

        for (int kk = 0; kk < TILE_K; ++kk) {
            float a0 = As[ty * 4 + 0][kk];
            float a1 = As[ty * 4 + 1][kk];
            float a2 = As[ty * 4 + 2][kk];
            float a3 = As[ty * 4 + 3][kk];

            float b0 = Bs[kk][tx * 4 + 0];
            float b1 = Bs[kk][tx * 4 + 1];
            float b2 = Bs[kk][tx * 4 + 2];
            float b3 = Bs[kk][tx * 4 + 3];

            acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
            acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
            acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
            acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        int r = rowBase + ty * 4 + i;
        if (r >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int c = colBase + tx * 4 + j;
            if (c < N) C[(size_t)r * (size_t)ldc + c] = acc[i][j];
        }
    }
}

extern "C" int matmul_tiled_f32(const float* A, const float* B, float* C,
                                int M, int N, int K, int lda, int ldb, int ldc) {
    dim3 blk(16, 16);
    dim3 grd((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    matmul_tiled_f32_kernel<<<grd, blk>>>(A, B, C, M, N, K, lda, ldb, ldc);
    return (int)cudaPeekAtLastError();
}

// ===========================
// WMMA FP16->FP32 matmul (multiples of 16)
// ===========================
__global__ void matmul_wmma16_kernel(const __half* __restrict__ A,
                                     const __half* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K, int lda, int ldb, int ldc) {
    // One warp computes one 16x16 tile C[tileRow, tileCol]
    int tileCol = blockIdx.x; // along N
    int tileRow = blockIdx.y; // along M
    if (threadIdx.x >= 32) return;

    wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        const __half* Ap = A + (tileRow * 16) * lda + k0;
        const __half* Bp = B + (k0) * ldb + (tileCol * 16);

        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, Ap, lda);
        wmma::load_matrix_sync(b_frag, Bp, ldb);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* Cp = C + (tileRow * 16) * ldc + (tileCol * 16);
    wmma::store_matrix_sync(Cp, c_frag, ldc, wmma::mem_row_major);
}

extern "C" int matmul_wmma_f16_f32(const void* A_half, const void* B_half, float* C,
                                   int M, int N, int K, int lda, int ldb, int ldc) {
    if ((M % 16) || (N % 16) || (K % 16)) return (int)cudaErrorInvalidValue;
    dim3 blk(32);                    // one warp per tile
    dim3 grd(N / 16, M / 16);
    matmul_wmma16_kernel<<<grd, blk>>>((const __half*)A_half, (const __half*)B_half, C,
                                       M, N, K, lda, ldb, ldc);
    return (int)cudaPeekAtLastError();
}
