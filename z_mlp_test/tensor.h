// tensor.h - Header file for the CUDA-accelerated Tensor library.
// Defines the Tensor struct and declares all public API functions.

#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// A simple tensor struct that holds device-side data and shape information.
typedef struct {
    double* data;
    int* shape;
    int* strides;
    int ndim;
} Tensor;

// --- Utility Functions ---
long long product(const int* shape, int ndim);
Tensor* _tensor_init_host(const int* shape, int ndim);
Tensor* _tensor_init_gpu(const int* shape, int ndim);
void tensor_free(Tensor* t);
void tensor_free_gpu(Tensor* t);
void tensor_copy_h2d(Tensor* h_tensor, Tensor* d_tensor);
void tensor_copy_d2h(Tensor* d_tensor, Tensor* h_tensor);
void tensor_print_host(Tensor* t);

// --- Forward Pass GPU Functions ---
Tensor* tensor_add_gpu(Tensor* t1, Tensor* t2);
Tensor* tensor_sub_gpu(Tensor* t1, Tensor* t2);
Tensor* tensor_mul_gpu(Tensor* t1, Tensor* t2);
Tensor* tensor_div_gpu(Tensor* t1, Tensor* t2);
Tensor* tensor_pow_gpu(Tensor* t, double p);
Tensor* tensor_neg_gpu(Tensor* t);
Tensor* tensor_matmul_gpu(Tensor* t1, Tensor* t2);
Tensor* tensor_sum_gpu(Tensor* t, int axis);
Tensor* tensor_transpose_gpu(Tensor* t);
Tensor* tensor_reshape_gpu(Tensor* t, const int* new_shape, int new_ndim);
Tensor* tensor_exp_gpu(Tensor* t);
Tensor* tensor_log_gpu(Tensor* t);
Tensor* tensor_tanh_gpu(Tensor* t);
Tensor* tensor_sigmoid_gpu(Tensor* t);
Tensor* tensor_relu_gpu(Tensor* t);
Tensor* tensor_softplus_gpu(Tensor* t);
Tensor* tensor_maximum_gpu(Tensor* t1, Tensor* t2);
Tensor* tensor_logsumexp_gpu(Tensor* t, int axis);

// --- Backward Pass (Gradient) GPU Functions ---
void tensor_add_backward_gpu(Tensor* grad_out, Tensor* grad_a, Tensor* grad_b);
void tensor_sub_backward_gpu(Tensor* grad_out, Tensor* grad_a, Tensor* grad_b);
void tensor_mul_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b);
void tensor_div_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b);
void tensor_pow_backward_gpu(Tensor* grad_out, Tensor* t, double p, Tensor* grad_in);
void tensor_neg_backward_gpu(Tensor* grad_out, Tensor* grad_in);
void tensor_matmul_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b);
void tensor_sum_backward_gpu(Tensor* grad_out, Tensor* grad_in, int axis);
void tensor_transpose_backward_gpu(Tensor* grad_out, Tensor* grad_in);
void tensor_reshape_backward_gpu(Tensor* grad_out, Tensor* grad_in, const int* orig_shape, int orig_ndim);
void tensor_exp_backward_gpu(Tensor* grad_out, Tensor* out, Tensor* grad_in);
void tensor_log_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in);
void tensor_tanh_backward_gpu(Tensor* grad_out, Tensor* out, Tensor* grad_in);
void tensor_sigmoid_backward_gpu(Tensor* grad_out, Tensor* out, Tensor* grad_in);
void tensor_relu_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in);
void tensor_softplus_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in);
void tensor_maximum_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b);
void tensor_logsumexp_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
