// main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmul.h"
#include <cuda_runtime.h>
#include <sys/time.h>

static inline size_t idx(size_t r, size_t c, size_t cols) { return r * cols + c; }
static inline double dabs(double x) { return x < 0 ? -x : x; }

static double* alloc_matrix(size_t rows, size_t cols) {
    double *m = (double*)malloc(rows * cols * sizeof(double));
    if (!m) { fprintf(stderr, "Allocation failed\n"); exit(EXIT_FAILURE); }
    return m;
}

static void fill_random(double *M, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i)
        M[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
}

static double wall_ms(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

static double max_abs_diff(const double *X, const double *Y, size_t m, size_t n) {
    double maxd = 0.0;
    size_t N = m * n;
    for (size_t i = 0; i < N; ++i) {
        double d = dabs(X[i] - Y[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

int main(void) {
    typedef struct {
        size_t m, k, n;
    } TestCase;

    // Define a series of test cases with increasing dimensions
    TestCase tests[] = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 128, 256},
        {256, 256, 256},
        {512, 256, 512},
        {512, 512, 512},
        {1024, 512, 1024},
        {1024, 1024, 1024},
        {2048, 1024, 512},
        {2048, 2048, 1024},
        {2048, 2048, 2048},
        {4096, 2048, 4096},
        {4096, 4096, 4096},
        {1024, 8192, 512},
        {8192, 1024, 8192}
    };
    int num_tests = sizeof(tests) / sizeof(TestCase);

    srand((unsigned)time(NULL));

    // Print the header for our results table
    printf("| %-22s | %-14s | %-14s | %-10s | %-15s |\n", "Dimensions (m,k,n)", "CPU Time (ms)", "GPU Time (ms)", "Speedup", "Max Abs Error");
    printf("|------------------------|----------------|----------------|------------|-----------------|\n");

    // Loop through each test case
    for (int i = 0; i < num_tests; ++i) {
        size_t m = tests[i].m;
        size_t k = tests[i].k;
        size_t n = tests[i].n;

        char dim_str[25];
        snprintf(dim_str, sizeof(dim_str), "%zu x %zu x %zu", m, k, n);

        // Allocate matrices for the current dimensions
        double *A = alloc_matrix(m, k);
        double *B = alloc_matrix(k, n);
        double *C_cpu = alloc_matrix(m, n);
        double *C_gpu = alloc_matrix(m, n);

        // Gracefully handle memory allocation failures
        if (!A || !B || !C_cpu || !C_gpu) {
            fprintf(stderr, "Memory allocation failed for dimensions %s\n", dim_str);
            free(A); free(B); free(C_cpu); free(C_gpu);
            continue; // Skip to the next test case
        }

        fill_random(A, m, k);
        fill_random(B, k, n);

        // --- CPU Benchmark ---
        double t0 = wall_ms();
        int rc_cpu = matmul_cpu(A, B, C_cpu, m, k, n);
        double t1 = wall_ms();
        if (rc_cpu != 0) {
            fprintf(stderr, "CPU matmul failed for dimensions %s\n", dim_str);
        }

        // --- GPU Benchmark ---
        double g0 = wall_ms();
        int rc_gpu = matmul_cuda(A, B, C_gpu, m, k, n);
        double g1 = wall_ms();
        if (rc_gpu != 0) {
            fprintf(stderr, "CUDA matmul failed for dimensions %s\n", dim_str);
        }

        // --- Calculate and Print Results ---
        double cpu_ms = (rc_cpu == 0) ? (t1 - t0) : -1.0;
        double gpu_ms = (rc_gpu == 0) ? (g1 - g0) : -1.0;
        double speedup = (cpu_ms > 0 && gpu_ms > 0) ? (cpu_ms / gpu_ms) : 0.0;
        double max_err = (rc_cpu == 0 && rc_gpu == 0) ? max_abs_diff(C_cpu, C_gpu, m, n) : -1.0;

        printf("| %-22s | %-14.3f | %-14.3f | %-9.2fx | %-15.3e |\n",
               dim_str, cpu_ms, gpu_ms, speedup, max_err);

        // Free memory for the next iteration
        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
    }

    return EXIT_SUCCESS;
}
