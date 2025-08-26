// tensor_core.h
// Minimal C API for CUDA backend (row-major). ASCII-only comments.

#ifndef TENSOR_CORE_H_
#define TENSOR_CORE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== Memory utilities =====
// Return 0 on success, non-zero on failure.
int gpu_malloc(void** ptr, size_t bytes);
int gpu_free(void* ptr);
int h2d(void* dst_device, const void* src_host, size_t bytes);
int d2h(void* dst_host, const void* src_device, size_t bytes);

// ===== FP32 Matmul (row-major) =====
// C[M,N] = A[M,K] @ B[K,N]
// lda, ldb, ldc are leading dimensions in elements (for row-major contiguous use lda=K, ldb=N, ldc=N)
int matmul_tiled_f32(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc);

// ===== Optional FP16->FP32 WMMA (requires compute capability >= 70) =====
int matmul_wmma_f16_f32(const void* A_half, const void* B_half, float* C,
                        int M, int N, int K, int lda, int ldb, int ldc);

// ===== Bias add (row-wise): Y[i,j] = X[i,j] + bias[j] =====
int bias_add_row_f32(const float* X, const float* bias, float* Y,
                     int rows, int cols, int ldx);

// ===== Elementwise ops over n elements (contiguous 1-D view) =====
int ew_add_f32(const float* X, const float* Y, float* Z, int64_t n);
int ew_sub_f32(const float* X, const float* Y, float* Z, int64_t n);
int ew_mul_f32(const float* X, const float* Y, float* Z, int64_t n);
int ew_div_f32(const float* X, const float* Y, float* Z, int64_t n);
int relu_f32(const float* X, float* Y, int64_t n);
int sigmoid_f32(const float* X, float* Y, int64_t n);
int tanh_f32(const float* X, float* Y, int64_t n);
int exp_f32(const float* X, float* Y, int64_t n);
int log_f32(const float* X, float* Y, int64_t n);
int softplus_f32(const float* X, float* Y, int64_t n); // softplus(x) = log(1 + exp(x)), stable

// ===== Reductions by rows on a 2-D matrix (row-major) =====
// ldx is leading dimension (for contiguous use ldx=cols)
int reduce_max_rows_f32(const float* X, float* max_out,
                        int rows, int cols, int ldx);
int reduce_sum_rows_f32(const float* X, float* sum_out,
                        int rows, int cols, int ldx);

// ===== Row-wise softmax, numerically stable =====
// Y[i,:] = softmax(X[i,:]) with stability: subtract row max before exp
int softmax_rows_f32(const float* X, float* Y,
                     int rows, int cols, int ldx, int ldy);

// ===== Transpose copy: Y = X^T =====
// X is MxN (ldx=N), Y is NxM (ldy=M) in row-major
int transpose_2d_f32(const float* X, float* Y,
                     int M, int N, int ldx, int ldy);

#ifdef __cplusplus
}
#endif
#endif  // TENSOR_CORE_H_
