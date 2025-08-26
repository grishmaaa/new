// tensor_lib.cu - CUDA implementation of all tensor operations for autodiff.

#include "tensor.h"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Utility function to calculate the total number of elements
long long product(const int* shape, int ndim) {
    long long size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    return size;
}

// Host-side tensor initialization
Tensor* _tensor_init_host(const int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }
    t->strides = (int*)malloc(ndim * sizeof(int));
    long long current_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        t->strides[i] = current_stride;
        current_stride *= shape[i];
    }
    t->data = (double*)malloc(current_stride * sizeof(double));
    return t;
}

// GPU-side tensor initialization
Tensor* _tensor_init_gpu(const int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }
    t->strides = (int*)malloc(ndim * sizeof(int));
    long long current_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        t->strides[i] = current_stride;
        current_stride *= shape[i];
    }
    CUDA_CHECK(cudaMalloc((void**)&t->data, current_stride * sizeof(double)));
    return t;
}

// Free Host memory
void tensor_free(Tensor* t) {
    if (t) {
        free(t->data);
        free(t->shape);
        free(t->strides);
        free(t);
    }
}

// Free GPU memory
void tensor_free_gpu(Tensor* t) {
    if (t) {
        CUDA_CHECK(cudaFree(t->data));
        free(t->shape);
        free(t->strides);
        free(t);
    }
}

// Copy data from host to device
void tensor_copy_h2d(Tensor* h_tensor, Tensor* d_tensor) {
    long long size = product(h_tensor->shape, h_tensor->ndim);
    CUDA_CHECK(cudaMemcpy(d_tensor->data, h_tensor->data, size * sizeof(double), cudaMemcpyHostToDevice));
}

// Copy data from device to host
void tensor_copy_d2h(Tensor* d_tensor, Tensor* h_tensor) {
    long long size = product(d_tensor->shape, d_tensor->ndim);
    CUDA_CHECK(cudaMemcpy(h_tensor->data, d_tensor->data, size * sizeof(double), cudaMemcpyDeviceToHost));
}

// Print a host tensor
void tensor_print_host(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    printf("Shape: (");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%d%s", t->shape[i], i == t->ndim - 1 ? "" : ", ");
    }
    printf(")\n");
    for (long long i = 0; i < size; ++i) {
        printf("%.4f ", t->data[i]);
    }
    printf("\n");
}


// --- Forward Pass Kernels ---

__global__ void add_kernel(double* a, double* b, double* c, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sub_kernel(double* a, double* b, double* c, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] - b[i];
    }
}

__global__ void mul_kernel(double* a, double* b, double* c, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] * b[i];
    }
}

__global__ void div_kernel(double* a, double* b, double* c, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] / b[i];
    }
}

__global__ void pow_kernel(double* a, double p, double* c, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = pow(a[i], p);
    }
}

__global__ void neg_kernel(double* a, double* c, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = -a[i];
    }
}

__global__ void matmul_kernel(const double* A, const double* B, double* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        double val = 0.0;
        for (int i = 0; i < N; ++i) {
            val += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = val;
    }
}

__global__ void sum_kernel(double* in, double* out, int size, int stride_in, int stride_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double sum = 0.0;
        for(int j = 0; j < stride_in / stride_out; ++j) {
            sum += in[i * (stride_in/stride_out) + j];
        }
        out[i] = sum;
    }
}

__global__ void exp_kernel(double* in, double* out, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = exp(in[i]);
    }
}

__global__ void log_kernel(double* in, double* out, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = log(in[i]);
    }
}

__global__ void tanh_kernel(double* in, double* out, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = tanh(in[i]);
    }
}

__global__ void sigmoid_kernel(double* in, double* out, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = 1.0 / (1.0 + exp(-in[i]));
    }
}

__global__ void relu_kernel(double* in, double* out, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = fmax(in[i], 0.0);
    }
}

__global__ void softplus_kernel(double* in, double* out, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double x = in[i];
        if (x > 0) {
            out[i] = x + log(1.0 + exp(-x));
        } else {
            out[i] = log(1.0 + exp(x));
        }
    }
}

__global__ void maximum_kernel(double* a, double* b, double* out, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = fmax(a[i], b[i]);
    }
}

// --- Backward Pass (Gradient) Kernels ---

__global__ void add_backward_kernel(double* grad_out, double* grad_a, double* grad_b, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Here, we just add the incoming gradient to both.
        // Broadcasting will be handled on the Python side.
        grad_a[i] += grad_out[i];
        grad_b[i] += grad_out[i];
    }
}

__global__ void sub_backward_kernel(double* grad_out, double* grad_a, double* grad_b, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_a[i] += grad_out[i];
        grad_b[i] -= grad_out[i];
    }
}

__global__ void mul_backward_kernel(double* grad_out, double* t1, double* t2, double* grad_a, double* grad_b, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_a[i] += grad_out[i] * t2[i];
        grad_b[i] += grad_out[i] * t1[i];
    }
}

__global__ void div_backward_kernel(double* grad_out, double* t1, double* t2, double* grad_a, double* grad_b, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_a[i] += grad_out[i] / t2[i];
        grad_b[i] -= grad_out[i] * t1[i] / (t2[i] * t2[i]);
    }
}

__global__ void pow_backward_kernel(double* grad_out, double* t, double p, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] += grad_out[i] * (p * pow(t[i], p - 1));
    }
}

__global__ void neg_backward_kernel(double* grad_out, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] -= grad_out[i];
    }
}

__global__ void matmul_backward_kernel_A(const double* grad_out, const double* t2_T, double* grad_in_A,
                                         int M_C, int N_C, int K_C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M_C && col < N_C) {
        double val = 0.0;
        for (int i = 0; i < K_C; ++i) {
            val += grad_out[row * K_C + i] * t2_T[col * K_C + i];
        }
        grad_in_A[row * N_C + col] += val;
    }
}

__global__ void matmul_backward_kernel_B(const double* t1_T, const double* grad_out, double* grad_in_B,
                                         int M_C, int N_C, int K_C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N_C && col < K_C) {
        double val = 0.0;
        for (int i = 0; i < M_C; ++i) {
            val += t1_T[row * M_C + i] * grad_out[i * K_C + col];
        }
        grad_in_B[row * K_C + col] += val;
    }
}

__global__ void sum_backward_kernel(double* grad_out, double* grad_in, long long size_out, long long size_in) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_in) {
        long long out_idx = i % size_out; // Broadcasting
        grad_in[i] += grad_out[out_idx];
    }
}

__global__ void transpose_backward_kernel(double* grad_out, double* grad_in, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        grad_in[col * N + row] += grad_out[row * M + col];
    }
}

__global__ void reshape_backward_kernel(double* grad_out, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] += grad_out[i];
    }
}

__global__ void exp_backward_kernel(double* grad_out, double* out, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] += grad_out[i] * out[i];
    }
}

__global__ void log_backward_kernel(double* grad_out, double* t, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] += grad_out[i] / t[i];
    }
}

__global__ void tanh_backward_kernel(double* grad_out, double* out, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] += grad_out[i] * (1 - out[i] * out[i]);
    }
}

__global__ void sigmoid_backward_kernel(double* grad_out, double* out, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] += grad_out[i] * (out[i] * (1 - out[i]));
    }
}

__global__ void relu_backward_kernel(double* grad_out, double* t, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (t[i] > 0) {
            grad_in[i] += grad_out[i];
        }
    }
}

__global__ void softplus_backward_kernel(double* grad_out, double* t, double* grad_in, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double sigmoid_t = 1.0 / (1.0 + exp(-t[i]));
        grad_in[i] += grad_out[i] * sigmoid_t;
    }
}

__global__ void maximum_backward_kernel(double* grad_out, double* t1, double* t2, double* grad_a, double* grad_b, long long size) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (t1[i] >= t2[i]) {
            grad_a[i] += grad_out[i];
        } else {
            grad_b[i] += grad_out[i];
        }
    }
}

 // Simple transpose kernel for 2D
__global__ void transpose_kernel(const double* in, double* out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        out[row * N + col] = in[col * M + row];
    }
}


// --- Wrapper functions to launch kernels ---

// Forward Pass
Tensor* tensor_add_gpu(Tensor* t1, Tensor* t2) {
    long long size = product(t1->shape, t1->ndim);
    Tensor* result = _tensor_init_gpu(t1->shape, t1->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    add_kernel<<<blocks_per_grid, threads_per_block>>>(t1->data, t2->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_sub_gpu(Tensor* t1, Tensor* t2) {
    long long size = product(t1->shape, t1->ndim);
    Tensor* result = _tensor_init_gpu(t1->shape, t1->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    sub_kernel<<<blocks_per_grid, threads_per_block>>>(t1->data, t2->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_mul_gpu(Tensor* t1, Tensor* t2) {
    long long size = product(t1->shape, t1->ndim);
    Tensor* result = _tensor_init_gpu(t1->shape, t1->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    mul_kernel<<<blocks_per_grid, threads_per_block>>>(t1->data, t2->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_div_gpu(Tensor* t1, Tensor* t2) {
    long long size = product(t1->shape, t1->ndim);
    Tensor* result = _tensor_init_gpu(t1->shape, t1->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    div_kernel<<<blocks_per_grid, threads_per_block>>>(t1->data, t2->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_pow_gpu(Tensor* t, double p) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    pow_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, p, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_neg_gpu(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    neg_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_matmul_gpu(Tensor* t1, Tensor* t2) {
    int m = t1->shape[t1->ndim - 2];
    int n = t1->shape[t1->ndim - 1];
    int k = t2->shape[t2->ndim - 1];
    
    int* result_shape = (int*)malloc(t1->ndim * sizeof(int));
    for (int i = 0; i < t1->ndim - 2; ++i) {
        result_shape[i] = t1->shape[i];
    }
    result_shape[t1->ndim - 2] = m;
    result_shape[t1->ndim - 1] = k;

    Tensor* result = _tensor_init_gpu(result_shape, t1->ndim);
    
    long long batch_size = 1;
    for (int i = 0; i < t1->ndim - 2; ++i) {
        batch_size *= t1->shape[i];
    }
    long long a_stride = m * n;
    long long b_stride = n * k;
    long long c_stride = m * k;
    
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((k + threads_per_block.x - 1) / threads_per_block.x,
                         (m + threads_per_block.y - 1) / threads_per_block.y);

    for (long long i = 0; i < batch_size; ++i) {
        matmul_kernel<<<blocks_per_grid, threads_per_block>>>(
            t1->data + i * a_stride,
            t2->data + i * b_stride,
            result->data + i * c_stride,
            m, n, k
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
    free(result_shape);
    return result;
}

Tensor* tensor_sum_gpu(Tensor* t, int axis) {
    // Simplified for a fixed axis, or you would need a more complex kernel
    long long size_out = 1;
    for (int i = 0; i < t->ndim; ++i) {
        if (i != axis) {
            size_out *= t->shape[i];
        }
    }
    
    int* out_shape = (int*)malloc((t->ndim - 1) * sizeof(int));
    int out_ndim = 0;
    for(int i = 0; i < t->ndim; ++i) {
        if(i != axis) {
            out_shape[out_ndim++] = t->shape[i];
        }
    }
    Tensor* result = _tensor_init_gpu(out_shape, out_ndim);
    
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size_out + threads_per_block.x - 1) / threads_per_block.x);
    
    sum_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size_out, 1, 1); // Simplistic, needs a real implementation
    CUDA_CHECK(cudaGetLastError());
    free(out_shape);
    return result;
}

Tensor* tensor_transpose_gpu(Tensor* t) {
    if (t->ndim != 2) {
        fprintf(stderr, "Error: Transpose only supported for 2D tensors.\n");
        return NULL;
    }
    int new_shape[] = {t->shape[1], t->shape[0]};
    Tensor* result = _tensor_init_gpu(new_shape, 2);
    
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((new_shape[1] + threads_per_block.x - 1) / threads_per_block.x,
                         (new_shape[0] + threads_per_block.y - 1) / threads_per_block.y);
    
   
    transpose_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, t->shape[0], t->shape[1]);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_reshape_gpu(Tensor* t, const int* new_shape, int new_ndim) {
    long long orig_size = product(t->shape, t->ndim);
    long long new_size = product(new_shape, new_ndim);
    if (orig_size != new_size) {
        fprintf(stderr, "Error: Reshape sizes must match.\n");
        return NULL;
    }
    
    Tensor* result = _tensor_init_gpu(new_shape, new_ndim);
    // No data transfer is needed, just copy pointer and metadata
    result->data = t->data;
    return result;
}

Tensor* tensor_exp_gpu(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    exp_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_log_gpu(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    log_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_tanh_gpu(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    tanh_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_sigmoid_gpu(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    sigmoid_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_relu_gpu(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    relu_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_softplus_gpu(Tensor* t) {
    long long size = product(t->shape, t->ndim);
    Tensor* result = _tensor_init_gpu(t->shape, t->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    softplus_kernel<<<blocks_per_grid, threads_per_block>>>(t->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_maximum_gpu(Tensor* t1, Tensor* t2) {
    long long size = product(t1->shape, t1->ndim);
    Tensor* result = _tensor_init_gpu(t1->shape, t1->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    maximum_kernel<<<blocks_per_grid, threads_per_block>>>(t1->data, t2->data, result->data, size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor* tensor_logsumexp_gpu(Tensor* t, int axis) {
    // This is a complex reduction and would need a specialized kernel.
    fprintf(stderr, "LogSumExp not yet implemented in CUDA.\n");
    return NULL;
}

// Backward Pass
void tensor_add_backward_gpu(Tensor* grad_out, Tensor* grad_a, Tensor* grad_b) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    add_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, grad_a->data, grad_b->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_sub_backward_gpu(Tensor* grad_out, Tensor* grad_a, Tensor* grad_b) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    sub_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, grad_a->data, grad_b->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_mul_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    mul_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, t1->data, t2->data, grad_a->data, grad_b->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_div_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    div_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, t1->data, t2->data, grad_a->data, grad_b->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_pow_backward_gpu(Tensor* grad_out, Tensor* t, double p, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    pow_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, t->data, p, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_neg_backward_gpu(Tensor* grad_out, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    neg_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_matmul_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b) {
    int m_c = grad_out->shape[grad_out->ndim - 2];
    int n_c = t1->shape[t1->ndim - 1]; // Inner dim, common
    int k_c = grad_out->shape[grad_out->ndim - 1];
    
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid_A((n_c + threads_per_block.x - 1) / threads_per_block.x,
                           (m_c + threads_per_block.y - 1) / threads_per_block.y);
    dim3 blocks_per_grid_B((k_c + threads_per_block.x - 1) / threads_per_block.x,
                           (n_c + threads_per_block.y - 1) / threads_per_block.y);

    // This would require transposing t1 and t2 first.
    // For simplicity, we assume transposed copies are available.
    // This is a placeholder and would need a proper implementation with transpose.
    // For now, we'll just use the forward kernel with transposed inputs.
    // a_grad = out_grad @ t2.T
    // b_grad = t1.T @ out_grad
}

void tensor_sum_backward_gpu(Tensor* grad_out, Tensor* grad_in, int axis) {
    long long size_out = product(grad_out->shape, grad_out->ndim);
    long long size_in = product(grad_in->shape, grad_in->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size_in + threads_per_block.x - 1) / threads_per_block.x);
    sum_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, grad_in->data, size_out, size_in);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_transpose_backward_gpu(Tensor* grad_out, Tensor* grad_in) {
    if (grad_out->ndim != 2) {
        fprintf(stderr, "Error: Transpose backward only supported for 2D tensors.\n");
        return;
    }
    int M = grad_out->shape[0];
    int N = grad_out->shape[1];
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    transpose_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, grad_in->data, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_reshape_backward_gpu(Tensor* grad_out, Tensor* grad_in, const int* orig_shape, int orig_ndim) {
    long long size = product(grad_in->shape, grad_in->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    reshape_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_exp_backward_gpu(Tensor* grad_out, Tensor* out, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    exp_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, out->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_log_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    log_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, t->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_tanh_backward_gpu(Tensor* grad_out, Tensor* out, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    tanh_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, out->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_sigmoid_backward_gpu(Tensor* grad_out, Tensor* out, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    sigmoid_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, out->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_relu_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    relu_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, t->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_softplus_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    softplus_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, t->data, grad_in->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_maximum_backward_gpu(Tensor* grad_out, Tensor* t1, Tensor* t2, Tensor* grad_a, Tensor* grad_b) {
    long long size = product(grad_out->shape, grad_out->ndim);
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);
    maximum_backward_kernel<<<blocks_per_grid, threads_per_block>>>(grad_out->data, t1->data, t2->data, grad_a->data, grad_b->data, size);
    CUDA_CHECK(cudaGetLastError());
}

void tensor_logsumexp_backward_gpu(Tensor* grad_out, Tensor* t, Tensor* grad_in) {
    fprintf(stderr, "LogSumExp backward not yet implemented in CUDA.\n");
}




// ##########################################################################################################################
