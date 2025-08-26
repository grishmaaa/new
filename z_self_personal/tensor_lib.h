// tensor_lib.h - Header file for the tensor library

#ifndef TENSOR_LIB_H
#define TENSOR_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

// Define the Tensor struct
typedef struct {
    double* data;
    int* shape;
    int* strides;
    int ndim;
} Tensor;

// Function declarations with C-style linkage
long long product(const int* arr, int n);

Tensor* _tensor_init_host(const int* shape, int ndim);
Tensor* _tensor_init_gpu(const int* shape, int ndim);

void tensor_free(Tensor* t);
void tensor_free_gpu(Tensor* t);

void tensor_copy_h2d(const Tensor* h_src, Tensor* d_dest);
void tensor_copy_d2h(const Tensor* d_src, Tensor* h_dest);

Tensor* tensor_add_gpu(const Tensor* a, const Tensor* b);
Tensor* tensor_matmul_gpu(const Tensor* a, const Tensor* b);
void tensor_print_host(const Tensor* t);

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                \
{                                                                       \
    const cudaError_t error = call;                                     \
    if (error != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(1);                                                        \
    }                                                                   \
}

#ifdef __cplusplus
}
#endif

#endif // TENSOR_LIB_H
