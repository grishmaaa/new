#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Overwrites C with A*B (row-major). Returns 0 on success.
int matmul_cpu(const double* A, const double* B, double* C,
               size_t m, size_t k, size_t n);

// CUDA path: returns 0 on success, nonzero on error.
int matmul_cuda(const double* A, const double* B, double* C,
                size_t m, size_t k, size_t n);

#ifdef __cplusplus
}
#endif
#endif
