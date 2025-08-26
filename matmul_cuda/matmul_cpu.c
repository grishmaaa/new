#include "matmul.h"
#include <string.h>
#include <time.h>
#include <stdio.h>

int matmul_cpu(const double* A, const double* B, double* C,
               size_t m, size_t k, size_t n)
{
    // C[i,j] = sum_p A[i,p] * B[p,j]
    // Zero C first.
    memset(C, 0, m * n * sizeof(double));

    clock_t start,end ;
    start = clock();

    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            double a_ip = A[i * k + p];
            const size_t bp = p * n;
            const size_t ci = i * n;
            for (size_t j = 0; j < n; ++j) {
                C[ci + j] += a_ip * B[bp + j];
            }
        }
    }
    end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", cpu_time);
    return 0;
}