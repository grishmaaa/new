#ifndef CORE_LINALG_H
#define CORE_LINALG_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RtStream RtStream; // opaque; ignored on CPU

typedef enum {
  LINALG_ALGO_AUTO = 0,
  LINALG_ALGO_NAIVE = 1,
  LINALG_ALGO_TILED = 2
} LinalgGemmAlgo;

// ---- BLAS-2/3 core ----
int linalg_gemm_f32(bool transA, bool transB,
                    int M, int N, int K,
                    float alpha, const float* A, int ldA,
                    const float* B, int ldB,
                    float beta, float* C, int ldC,
                    LinalgGemmAlgo algo, RtStream* s);

// ---- Elementwise helpers ----
int linalg_bias_add_row_f32(float* Z, const float* b, int M, int N, int ldZ, RtStream* s); // b: len M
int linalg_bias_add_col_f32(float* Z, const float* b, int M, int N, int ldZ, RtStream* s); // b: len N

int linalg_ewise_relu_f32(const float* Z, float* A, int64_t size, RtStream* s);
int linalg_ewise_relu_bw_f32(const float* Z, const float* dA, float* dZ, int64_t size, RtStream* s);
int linalg_ewise_sigmoid_f32(const float* Z, float* A, int64_t size, RtStream* s);
int linalg_ewise_sigmoid_bw_f32(const float* A, const float* dA, float* dZ, int64_t size, RtStream* s);

// ---- Reductions (row/col sums) ----
int linalg_row_sum_f32(const float* A, int M, int N, int ldA, float* outN, RtStream* s); // len N
int linalg_col_sum_f32(const float* A, int M, int N, int ldA, float* outM, RtStream* s); // len M

// ---- Softmax & CE ----
int linalg_softmax_row_f32(const float* Z, int M, int N, int ldZ, float* P, int ldP, RtStream* s);
int linalg_ce_loss_from_logits_f32(const float* logits, int M, int N, int ldLogits,
                                   const int32_t* labels, float* mean_loss, RtStream* s);
int linalg_softmax_ce_logits_bw_f32(const float* logits, const int32_t* labels,
                                    int M, int N, int ldLogits,
                                    float* dZ, RtStream* s);

#ifdef __cplusplus
}
#endif

#endif // CORE_LINALG_H
