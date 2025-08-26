#include <stddef.h>
#include "linear.h"

void linear_forward_cpu(const float* X, const float* W, const float* b, float* Y,
                        int M, int K, int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      const float* xrow = X + (size_t)m * K;     // X[m, :]
      const float* wrow = W + (size_t)n * K;     // W[n, :]
      for (int k = 0; k < K; ++k) {
        acc += xrow[k] * wrow[k];                // dot(X[m,:], W[n,:])
      }
      if (b) acc += b[n];
      Y[(size_t)m * N + n] = acc;                // Y[m, n]
    }
  }
}
