#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "linear.h"
#include <cuda_runtime.h>
#include <math.h>

static double wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static float max_abs_diff(const float *X, const float *Y, int size) {
    float maxd = 0.0f;
    for (int i = 0; i < size; ++i) {
        float d = fabsf(X[i] - Y[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

int main(void) {
    int M = 2, K = 3, N = 4;

    float X[] = { 1,2,3, 4,5,6 }; // [2,3]
    float W[] = {
        0.1f, 0.2f, 0.3f,
       -0.5f, 0.4f,-0.1f,
        0.7f,-0.2f, 0.05f,
        0.0f, 1.0f,-1.0f
    }; // [4,3]
    float b[] = { 0.01f, -0.02f, 0.03f, 0.04f }; // [4]

    float *Y_cpu = (float*)calloc((size_t)M * N, sizeof(float));
    float *Y_gpu = (float*)calloc((size_t)M * N, sizeof(float));

    // CPU Forward Pass
    double t0 = wall_ms();
    linear_forward_cpu(X, W, b, Y_cpu, M, K, N);
    double cpu_time = wall_ms() - t0;

    // GPU Forward Pass
    double g0 = wall_ms();
    linear_forward_cuda(X, W, b, Y_gpu, M, K, N);
    double gpu_time = wall_ms() - g0;

    // Print results and timing
    printf("CPU Results:\n");
    for (int m = 0; m < M; ++m) {
        printf("Y_cpu[%d]:", m);
        for (int n = 0; n < N; ++n) printf(" %0.4f", Y_cpu[m*N + n]);
        printf("\n");
    }

    printf("\nGPU Results:\n");
    for (int m = 0; m < M; ++m) {
        printf("Y_gpu[%d]:", m);
        for (int n = 0; n < N; ++n) printf(" %0.4f", Y_gpu[m*N + n]);
        printf("\n");
    }

    float max_diff = max_abs_diff(Y_cpu, Y_gpu, M * N);
    printf("\nPerformance:\n");
    printf("CPU Time: %.3f ms\n", cpu_time);
    printf("GPU Time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Max Absolute Difference: %.6e\n", max_diff);

    free(Y_cpu);
    free(Y_gpu);
    return 0;
}
