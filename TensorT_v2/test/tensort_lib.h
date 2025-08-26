// tensort_lib.h — CUDA backend for TensorT (float32)
#ifndef TENSORT_LIB_H
#define TENSORT_LIB_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Row-major tensor (float32)
typedef struct {
    float* data;      // pointer to contiguous data (host or device)
    int*   shape;     // length ndim
    int*   strides;   // length ndim (row-major)
    int    ndim;      // rank
    long long size;   // product(shape)
} TT_Tensor;

long long tt_product(const int* a, int n);

// Host/device allocators
TT_Tensor* tt_tensor_init_host(const int* shape, int ndim);
TT_Tensor* tt_tensor_init_gpu (const int* shape, int ndim);

// Free
void tt_tensor_free_host(TT_Tensor* t);
void tt_tensor_free_gpu (TT_Tensor* t);

// Copies
void tt_copy_h2d(const TT_Tensor* h_src, TT_Tensor* d_dst);
void tt_copy_d2h(const TT_Tensor* d_src, TT_Tensor* h_dst);

// Elementwise (same-shaped for now)
TT_Tensor* tt_add_gpu(const TT_Tensor* a, const TT_Tensor* b);
TT_Tensor* tt_sub_gpu(const TT_Tensor* a, const TT_Tensor* b);
TT_Tensor* tt_mul_gpu(const TT_Tensor* a, const TT_Tensor* b);
TT_Tensor* tt_div_gpu(const TT_Tensor* a, const TT_Tensor* b);

// 2D matmul (no broadcast):  (M×K) @ (K×N) -> (M×N)
TT_Tensor* tt_matmul2d_gpu(const TT_Tensor* A, const TT_Tensor* B);

// Optional debug print (host tensor)
void tt_print_host(const TT_Tensor* t);

// ---- scalar elementwise (C = A (op) s) ----
TT_Tensor* tt_add_scalar_gpu(const TT_Tensor* a, float s);
TT_Tensor* tt_sub_scalar_gpu(const TT_Tensor* a, float s);   // A - s
TT_Tensor* tt_rsub_scalar_gpu(float s, const TT_Tensor* a);  // s - A
TT_Tensor* tt_mul_scalar_gpu(const TT_Tensor* a, float s);
TT_Tensor* tt_div_scalar_gpu(const TT_Tensor* a, float s);   // A / s
TT_Tensor* tt_rdiv_scalar_gpu(float s, const TT_Tensor* a);  // s / A


// Error-check
#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    abort(); \
  } \
} while(0)

#ifdef __cplusplus
}
#endif

#endif // TENSORT_LIB_H