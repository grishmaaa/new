#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Forward pass on CPU: Y[M,N] = X[M,K] * W[N,K]^T + b[N]
void linear_forward_cpu(const float* X,  // [M,K]
                       const float* W,   // [N,K]
                       const float* b,   // [N] (can be NULL)
                       float* Y,         // [M,N] (out)
                       int M, int K, int N);

// Forward pass on CUDA (device pointers)
void linear_forward_cuda(const float* X,  // [M,K]
                        const float* W,   // [N,K]
                        const float* b,   // [N] (can be NULL)
                        float* Y,         // [M,N] (out)
                        int M, int K, int N);

#ifdef __cplusplus
}
#endif
