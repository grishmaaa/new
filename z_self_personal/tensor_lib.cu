// tensor_lib.cu - Host-side function implementations and CUDA kernel definitions

#include "tensor_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA Kernels for Element-wise Operations
__global__ void elementwise_add_kernel(double* a, double* b, double* c, long long size) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_sub_kernel(double* a, double* b, double* c, long long size) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void elementwise_mul_kernel(double* a, double* b, double* c, long long size) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// CUDA Kernel for Matrix Multiplication (GEMM)
__global__ void matmul_kernel(double* A, double* B, double* C, int m, int k, int n,
                              int batch_a_stride, int batch_b_stride) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;

    if (row < m && col < n) {
        double sum = 0.0;
        int a_start = batch_idx * batch_a_stride;
        int b_start = batch_idx * batch_b_stride;
        int c_start = batch_idx * m * n;

        for (int i = 0; i < k; ++i) {
            sum += A[a_start + row * k + i] * B[b_start + i * n + col];
        }
        C[c_start + row * n + col] = sum;
    }
}

// Host-side function implementations
long long product(const int* arr, int n) {
    long long p = 1;
    for (int i = 0; i < n; ++i) {
        if (arr[i] <= 0) {
            fprintf(stderr, "Error: Invalid shape dimension.\n");
            return 0;
        }
        p *= arr[i];
    }
    return p;
}

void tensor_free(Tensor* t) {
    if (t) {
        free(t->shape);
        free(t->strides);
        free(t);
    }
}

void tensor_free_gpu(Tensor* t) {
    if (t && t->data) {
        cudaFree(t->data);
        t->data = NULL;
    }
}

Tensor* _tensor_init_host(const int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    long long size = product(shape, ndim);
    if (size == 0) {
        tensor_free(t);
        return NULL;
    }

    t->data = (double*)malloc(size * sizeof(double));
    t->shape = (int*)malloc(ndim * sizeof(int));
    t->strides = (int*)malloc(ndim * sizeof(int));

    if (!t->data || !t->shape || !t->strides) {
        free(t->data);
        tensor_free(t);
        return NULL;
    }

    long long stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        t->shape[i] = shape[i];
        t->strides[i] = stride;
        stride *= shape[i];
    }

    return t;
}

Tensor* _tensor_init_gpu(const int* shape, int ndim) {
    Tensor* t = _tensor_init_host(shape, ndim);
    if (!t) return NULL;
    
    long long size = product(shape, ndim);
    CUDA_CHECK(cudaMalloc((void**)&t->data, size * sizeof(double)));

    return t;
}

void tensor_copy_h2d(const Tensor* h_src, Tensor* d_dest) {
    long long size = product(h_src->shape, h_src->ndim);
    CUDA_CHECK(cudaMemcpy(d_dest->data, h_src->data, size * sizeof(double), cudaMemcpyHostToDevice));
}

void tensor_copy_d2h(const Tensor* d_src, Tensor* h_dest) {
    long long size = product(d_src->shape, d_src->ndim);
    CUDA_CHECK(cudaMemcpy(h_dest->data, d_src->data, size * sizeof(double), cudaMemcpyDeviceToHost));
}

Tensor* tensor_add_gpu(const Tensor* a, const Tensor* b) {
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Error: Tensors must have the same number of dimensions.\n");
        return NULL;
    }
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensors must have the same shape.\n");
            return NULL;
        }
    }

    Tensor* c = _tensor_init_gpu(a->shape, a->ndim);
    if (!c) return NULL;

    long long size = product(a->shape, a->ndim);
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    elementwise_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a->data, b->data, c->data, size);
    CUDA_CHECK(cudaGetLastError());
    return c;
}

Tensor* tensor_matmul_gpu(const Tensor* a, const Tensor* b) {
    // printf("DEBUG: tensor_matmul_gpu called\n");
    // printf("DEBUG: A ndim: %d, B ndim: %d\n", a->ndim, b->ndim);
    // printf("DEBUG: A shape: [");
    for (int i=0; i<a->ndim; ++i) printf("%d%s", a->shape[i], (i == a->ndim - 1) ? "" : ", ");
    // printf("]\n");
    // printf("DEBUG: B shape: [");
    for (int i=0; i<b->ndim; ++i) printf("%d%s", b->shape[i], (i == b->ndim - 1) ? "" : ", ");
    // printf("]\n");

    if (a->ndim < 2 || b->ndim < 2) {
        fprintf(stderr, "Error: Matmul requires at least 2 dimensions.\n");
        return NULL;
    }
    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) {
        fprintf(stderr, "Error: Incompatible shapes for matrix multiplication. Mismatch in inner dimensions: %d vs %d.\n", a->shape[a->ndim - 1], b->shape[b->ndim - 2]);
        return NULL;
    }

    int a_batch_ndim = a->ndim - 2;
    int b_batch_ndim = b->ndim - 2;
    int max_batch_ndim = (a_batch_ndim > b_batch_ndim) ? a_batch_ndim : b_batch_ndim;
    int new_ndim = max_batch_ndim + 2;
    
    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    if (!new_shape) {
        fprintf(stderr, "Error: Memory allocation failed for new_shape.\n");
        return NULL;
    }
    
    // Determine the new shape based on broadcasting rules, from right to left
    for (int i = 0; i < max_batch_ndim; ++i) {
        int a_idx = a_batch_ndim - 1 - i;
        int b_idx = b_batch_ndim - 1 - i;

        int a_dim = (a_idx >= 0) ? a->shape[a_idx] : 1;
        int b_dim = (b_idx >= 0) ? b->shape[b_idx] : 1;
        
        // printf("DEBUG: Batch dim %d: a_dim=%d, b_dim=%d\n", i, a_dim, b_dim);

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            fprintf(stderr, "Error: Incompatible batch shapes for matmul.\n");
            free(new_shape);
            return NULL;
        }
        new_shape[max_batch_ndim - 1 - i] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    // Set the last two dimensions (m, n)
    new_shape[new_ndim - 2] = a->shape[a->ndim - 2];
    new_shape[new_ndim - 1] = b->shape[b->ndim - 1];
    
    // printf("DEBUG: Final output shape: [");
    for (int i=0; i<new_ndim; ++i) printf("%d%s", new_shape[i], (i == new_ndim - 1) ? "" : ", ");
    printf("]\n");


    Tensor* c = _tensor_init_gpu(new_shape, new_ndim);
    free(new_shape); // Free temporary array
    if (!c) return NULL;
    
    int batch_size = product(c->shape, new_ndim - 2);
    // printf("DEBUG: Batch size: %d\n", batch_size);

    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];

    int threadsPerBlock = 32;
    dim3 threads(threadsPerBlock, threadsPerBlock);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y, batch_size);
    // printf("DEBUG: Threads per block: %d x %d\n", threads.x, threads.y);
    // printf("DEBUG: Blocks per grid: %d x %d x %d\n", blocks.x, blocks.y, blocks.z);


    int a_batch_stride = m * k;
    int b_batch_stride = k * n;
    
    // printf("DEBUG: A batch stride: %d\n", a_batch_stride);
    // printf("DEBUG: B batch stride: %d\n", b_batch_stride);

    matmul_kernel<<<blocks, threads>>>(a->data, b->data, c->data, m, k, n, a_batch_stride, b_batch_stride);
    CUDA_CHECK(cudaGetLastError());
    return c;
}

void _tensor_print_host_recursive(const double* data, const int* shape, const int* strides, int ndim, long long offset) {
    printf("[");
    if (ndim == 1) {
        for (int i = 0; i < shape[0]; ++i) {
            printf("%f%s", data[offset + i], (i == shape[0] - 1) ? "" : ", ");
        }
    } else {
        for (int i = 0; i < shape[0]; ++i) {
            _tensor_print_host_recursive(data, shape + 1, strides + 1, ndim - 1, offset + i * strides[0]);
            if (i < shape[0] - 1) {
                printf(", ");
            }
        }
    }
    printf("]");
}

void tensor_print_host(const Tensor* t) {
    if (!t) {
        printf("Tensor is NULL.\n");
        return;
    }
    printf("Tensor(shape=");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%d%s", t->shape[i], (i == t->ndim - 1) ? "" : ", ");
    }
    printf(")\n");
    if (t->ndim > 0) {
        _tensor_print_host_recursive(t->data, t->shape, t->strides, t->ndim, 0);
    }
    printf("\n");
}
