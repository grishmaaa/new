#include "linalg.h"
#include <math.h>
#include <string.h>

#ifndef LINALG_RESTRICT
#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#    define LINALG_RESTRICT restrict
#  else
#    define LINALG_RESTRICT
#  endif
#endif

#define LINALG_OK 0
#define LINALG_BAD_ARG -1

static inline int valid_dims(int M,int N,int K,int ldA,int ldB,int ldC){
  return (M>=0 && N>=0 && K>=0 && ldA>0 && ldB>0 && ldC>0);
}

int linalg_gemm_f32(bool transA, bool transB,
                    int M, int N, int K,
                    float alpha, const float* A, int ldA,
                    const float* B, int ldB,
                    float beta, float* C, int ldC,
                    LinalgGemmAlgo algo, RtStream* s) {
  (void)s; (void)algo; // CPU ignores stream/algorithm hint for now
  if (!valid_dims(M,N,K,ldA,ldB,ldC) || !A || !B || !C) return LINALG_BAD_ARG;

  // Scale C by beta first (safe if C alias occurs)
  if (beta == 0.0f) {
    for (int i=0;i<M;i++) {
      float* Ci = C + (size_t)i*ldC;
      for (int j=0;j<N;j++) Ci[j] = 0.0f;
    }
  } else if (beta != 1.0f) {
    for (int i=0;i<M;i++) {
      float* Ci = C + (size_t)i*ldC;
      for (int j=0;j<N;j++) Ci[j] *= beta;
    }
  }

  // Naive 3-loop GEMM with transA/transB handling
  for (int i=0;i<M;i++) {
    for (int k=0;k<K;k++) {
      float aik = (!transA) ? A[(size_t)i*ldA + k] : A[(size_t)k*ldA + i];
      float alpha_aik = alpha * aik;
      const float* Bk = (!transB) ? (B + (size_t)k*ldB) : (B + k);
      float* Ci = C + (size_t)i*ldC;
      if (!transB) {
        for (int j=0;j<N;j++) Ci[j] += alpha_aik * Bk[j];
      } else {
        for (int j=0;j<N;j++) Ci[j] += alpha_aik * Bk[(size_t)j*ldB];
      }
    }
  }
  return LINALG_OK;
}

int linalg_bias_add_row_f32(float* Z, const float* b, int M, int N, int ldZ, RtStream* s) {
  (void)s; if (!Z || !b || M<0 || N<0 || ldZ<=0) return LINALG_BAD_ARG;
  for (int i=0;i<M;i++) {
    float bi = b[i];
    float* Zi = Z + (size_t)i*ldZ;
    for (int j=0;j<N;j++) Zi[j] += bi;
  }
  return LINALG_OK;
}

int linalg_bias_add_col_f32(float* Z, const float* b, int M, int N, int ldZ, RtStream* s) {
  (void)s; if (!Z || !b || M<0 || N<0 || ldZ<=0) return LINALG_BAD_ARG;
  for (int i=0;i<M;i++) {
    float* Zi = Z + (size_t)i*ldZ;
    for (int j=0;j<N;j++) Zi[j] += b[j];
  }
  return LINALG_OK;
}

int linalg_row_sum_f32(const float* A, int M, int N, int ldA, float* outN, RtStream* s) {
  (void)s; if (!A || !outN || M<0 || N<0 || ldA<=0) return LINALG_BAD_ARG;
  for (int j=0;j<N;j++) outN[j] = 0.0f;
  for (int i=0;i<M;i++) {
    const float* Ai = A + (size_t)i*ldA;
    for (int j=0;j<N;j++) outN[j] += Ai[j];
  }
  return LINALG_OK;
}

int linalg_col_sum_f32(const float* A, int M, int N, int ldA, float* outM, RtStream* s) {
  (void)s; if (!A || !outM || M<0 || N<0 || ldA<=0) return LINALG_BAD_ARG;
  for (int i=0;i<M;i++) {
    const float* Ai = A + (size_t)i*ldA;
    float srow = 0.0f;
    for (int j=0;j<N;j++) srow += Ai[j];
    outM[i] = srow;
  }
  return LINALG_OK;
}

int linalg_ewise_relu_f32(const float* Z, float* A, int64_t size, RtStream* s) {
  (void)s; if (!Z || !A || size<0) return LINALG_BAD_ARG;
  for (int64_t i=0;i<size;i++) A[i] = Z[i] > 0.0f ? Z[i] : 0.0f;
  return LINALG_OK;
}

int linalg_ewise_relu_bw_f32(const float* Z, const float* dA, float* dZ, int64_t size, RtStream* s) {
  (void)s; if (!Z || !dA || !dZ || size<0) return LINALG_BAD_ARG;
  for (int64_t i=0;i<size;i++) dZ[i] = (Z[i] > 0.0f) ? dA[i] : 0.0f;
  return LINALG_OK;
}

static inline float sigmoid_stable(float x) {
  if (x >= 0.0f) {
    float t = expf(-x);
    return 1.0f / (1.0f + t);
  } else {
    float t = expf(x);
    return t / (1.0f + t);
  }
}

int linalg_ewise_sigmoid_f32(const float* Z, float* A, int64_t size, RtStream* s) {
  (void)s; if (!Z || !A || size<0) return LINALG_BAD_ARG;
  for (int64_t i=0;i<size;i++) A[i] = sigmoid_stable(Z[i]);
  return LINALG_OK;
}

int linalg_ewise_sigmoid_bw_f32(const float* A, const float* dA, float* dZ, int64_t size, RtStream* s) {
  (void)s; if (!A || !dA || !dZ || size<0) return LINALG_BAD_ARG;
  for (int64_t i=0;i<size;i++) {
    float a = A[i];
    dZ[i] = dA[i] * a * (1.0f - a);
  }
  return LINALG_OK;
}

int linalg_softmax_row_f32(const float* Z, int M, int N, int ldZ, float* P, int ldP, RtStream* s) {
  (void)s; if (!Z || !P || M<0 || N<=0 || ldZ<=0 || ldP<=0) return LINALG_BAD_ARG;
  for (int i=0;i<M;i++) {
    const float* Zi = Z + (size_t)i*ldZ;
    float* Pi = P + (size_t)i*ldP;
    // 1) max
    float m = Zi[0];
    for (int j=1;j<N;j++) if (Zi[j] > m) m = Zi[j];
    // 2) exp shift & sum
    float sum = 0.0f;
    for (int j=0;j<N;j++) {
      float e = expf(Zi[j] - m);
      Pi[j] = e;
      sum += e;
    }
    // 3) normalize
    float inv = 1.0f / sum;
    for (int j=0;j<N;j++) Pi[j] *= inv;
  }
  return LINALG_OK;
}

int linalg_ce_loss_from_logits_f32(const float* logits, int M, int N, int ldLogits,
                                   const int32_t* labels, float* mean_loss, RtStream* s) {
  (void)s; if (!logits || !labels || !mean_loss || M<=0 || N<=0 || ldLogits<=0) return LINALG_BAD_ARG;
  double total = 0.0; // extra precision for accumulation
  for (int i=0;i<M;i++) {
    const float* Li = logits + (size_t)i*ldLogits;
    // log-sum-exp
    float m = Li[0];
    for (int j=1;j<N;j++) if (Li[j] > m) m = Li[j];
    double sum = 0.0;
    for (int j=0;j<N;j++) sum += exp((double)Li[j] - (double)m);
    double lse = (double)m + log(sum);
    int lbl = labels[i];
    if (lbl < 0 || lbl >= N) return LINALG_BAD_ARG;
    total += (lse - (double)Li[lbl]);
  }
  *mean_loss = (float)(total / (double)M);
  return LINALG_OK;
}

int linalg_softmax_ce_logits_bw_f32(const float* logits, const int32_t* labels,
                                    int M, int N, int ldLogits,
                                    float* dZ, RtStream* s) {
  (void)s; if (!logits || !labels || !dZ || M<0 || N<=0 || ldLogits<=0) return LINALG_BAD_ARG;
  for (int i=0;i<M;i++) {
    const float* Li = logits + (size_t)i*ldLogits;
    float* dZi = dZ + (size_t)i*ldLogits; // output has same ld
    // softmax row into dZi
    float m = Li[0];
    for (int j=1;j<N;j++) if (Li[j] > m) m = Li[j];
    float sum = 0.0f;
    for (int j=0;j<N;j++) { float e = expf(Li[j]-m); dZi[j] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int j=0;j<N;j++) dZi[j] *= inv; // now dZi holds P
    int lbl = labels[i];
    if (lbl < 0 || lbl >= N) return LINALG_BAD_ARG;
    dZi[lbl] -= 1.0f; // dZ = P - onehot(lbl)
  }
  return LINALG_OK;
}
