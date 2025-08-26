// tensor_kernels.cu - CUDA kernel implementations for tensor operations

#include "tensor_lib.h"
#include <stdio.h>
#include <math.h>

// Kernel for element-wise addition
__global__ void add_kernel(double* out, const double* a, const double* b, long long size) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] + b[tid];
    }
}

// Kernel for element-wise multiplication
__global__ void mul_kernel(double* out, const double* a, const double* b, long long size) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] * b[tid];
    }
}

// Kernel for element-wise power
__global__ void pow_kernel(double* out, const double* in, double exp, long long size) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = pow(in[tid], exp);
    }
}

// Kernel for gradient unbroadcasting (summation)
__global__ void unbroadcast_kernel(double* out, const double* grad_data, const int* grad_shape, const int* target_shape, int ndim) {
    // This is a simplified placeholder and does not handle full broadcasting logic.
    // It assumes the target shape is a sub-shape of the grad_shape.
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < product(target_shape, ndim)) {
        out[tid] = grad_data[tid]; // Simple copy for now
    }
}

// Kernel for backward pass of add
__global__ void add_backward_kernel(double* grad_a, double* grad_b, const double* grad_out, long long size) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Placeholder: Add the incoming gradient
        // The Python side handles the unbroadcasting and accumulation.
        grad_a[tid] = grad_out[tid];
        grad_b[tid] = grad_out[tid];
    }
}

// Kernel for backward pass of multiply
__global__ void mul_backward_kernel(double* grad_a, const double* a_data, double* grad_b, const double* b_data, const double* grad_out, long long size) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Placeholder for multiplication gradient
        // Python side handles accumulation and unbroadcasting
        grad_a[tid] = grad_out[tid] * b_data[tid];
        grad_b[tid] = grad_out[tid] * a_data[tid];
    }
}

// Kernel for backward pass of power
__global__ void pow_backward_kernel(double* grad_in, const double* t_data, double exp, const double* grad_out, long long size) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Placeholder for power gradient
        grad_in[tid] = grad_out[tid] * exp * pow(t_data[tid], exp - 1);
    }
}


// --- Function wrappers for C++ and Python interop ---
static void launch_kernel(void (*kernel_func)(), long long size, const void** args) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    // Call the kernel using cudaLaunchKernel
    CUDA_CHECK(cudaLaunchKernel(
        (void*)kernel_func,
        dim3(blocks_per_grid),
        dim3(threads_per_block),
        (void**)args,
        0, // shared memory size
        0  // stream
    ));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Wrapper for element-wise addition
Tensor* tensor_add_gpu(const Tensor* a, const Tensor* b) {
    long long size = product(a->shape, a->ndim); // Simplified, assumes matching shape
    Tensor* out = _tensor_init_gpu(a->shape, a->ndim);
    
    const void* args[] = { &out->data, &a->data, &b->data, &size };
    launch_kernel((void*)add_kernel, size, args);
    return out;
}

// Wrapper for element-wise multiplication
Tensor* tensor_mul_gpu(const Tensor* a, const Tensor* b) {
    long long size = product(a->shape, a->ndim); // Simplified
    Tensor* out = _tensor_init_gpu(a->shape, a->ndim);

    const void* args[] = { &out->data, &a->data, &b->data, &size };
    launch_kernel((void*)mul_kernel, size, args);
    return out;
}

// Wrapper for element-wise power
Tensor* tensor_pow_gpu(const Tensor* t, double exp) {
    long long size = product(t->shape, t->ndim);
    Tensor* out = _tensor_init_gpu(t->shape, t->ndim);

    const void* args[] = { &out->data, &t->data, &exp, &size };
    launch_kernel((void*)pow_kernel, size, args);
    return out;
}

// Wrapper for unbroadcasting
Tensor* tensor_unbroadcast_gpu(const Tensor* grad, const Tensor* target_tensor, const int* target_shape, int target_ndim) {
    long long size = product(target_shape, target_ndim);
    Tensor* out = _tensor_init_gpu(target_shape, target_ndim);

    // This is a placeholder call. The actual implementation needs to handle summing.
    const void* args[] = { &out->data, &grad->data, &grad->shape, &target_shape, &target_ndim };
    launch_kernel((void*)unbroadcast_kernel, size, args);
    return out;
}
