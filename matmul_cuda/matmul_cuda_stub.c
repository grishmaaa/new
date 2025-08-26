#include "matmul.h"
#include <stdio.h>
#include <stddef.h>

int matmul_cuda(const double* A, const double* B, double* C,
                size_t m, size_t k, size_t n) {
    (void)A; (void)B; (void)C; (void)m; (void)k; (void)n;
    fprintf(stderr, "CUDA not available in this build.\n");
    return 1;
}
    